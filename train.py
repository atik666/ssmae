from mae import MaskedAutoencoder
import torch
from torch import nn
from tqdm import tqdm
import os
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def create_mae_model(model_size='base', num_classes=1000):
    """Create MAE model with different sizes and classification capability"""
    configs = {
        'base': {
            'encoder_embed_dim': 768, 'encoder_depth': 12, 'encoder_num_heads': 12,
            'decoder_embed_dim': 512, 'decoder_depth': 8, 'decoder_num_heads': 16
        },
        'large': {
            'encoder_embed_dim': 1024, 'encoder_depth': 24, 'encoder_num_heads': 16,
            'decoder_embed_dim': 512, 'decoder_depth': 8, 'decoder_num_heads': 16
        },
        'huge': {
            'encoder_embed_dim': 1280, 'encoder_depth': 32, 'encoder_num_heads': 16,
            'decoder_embed_dim': 512, 'decoder_depth': 8, 'decoder_num_heads': 16
        }
    }
    
    config = configs[model_size]
    config['num_classes'] = num_classes
    return MaskedAutoencoder(**config)

class PseudoLabelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # list of file paths
        self.labels = labels            # list or tensor of ints
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def train_mae(model, dataloader, optimizer, device, num_epochs=100):
    """Training loop for MAE pretraining"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass in MAE mode
            loss, pred, mask = model(images, None, mode='mae')
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')

def train_classification(model, dataloader, optimizer, criterion, device, num_epochs=100):
    """Training loop for classification using pretrained MAE encoder"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass in classification mode
            logits = model(images, None, 'classify')
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

def evaluate_classification(model, eval_dataloader, criterion, device, 
                            unlabeled_dataloader=None, unlabeled_dataset=None, threshold=0.95,
                            already_pseudo_labeled_indices=None,
                            weak_transform=None, strong_transform=None,
                            ):
    """Helper function to evaluate model on classification task"""
    model.eval()  # Set model to evaluation mode
    total_eval_loss = 0
    eval_correct = 0
    eval_total = 0
    num_eval_batches = 0

    eval_progress_bar = tqdm(
        eval_dataloader,
        desc=f"Evaluating",
        total=len(eval_dataloader),
        leave=False # Keep the progress bar for evaluation nested
    )

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for images, labels in eval_progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images, None, 'classify')
            loss = criterion(logits, labels)
            
            total_eval_loss += loss.item()
            num_eval_batches += 1
            
            _, predicted = torch.max(logits.data, 1)
            eval_total += labels.size(0)
            eval_correct += (predicted == labels).sum().item()

    avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else 0
    epoch_eval_accuracy = 100. * eval_correct / eval_total if eval_total > 0 else 0

    # Print final evaluation metrics after the loop
    print(f"Evaluation: Avg Loss: {avg_eval_loss:.4f}, Accuracy: {epoch_eval_accuracy:.2f}%")

    new_pseudo_labeled_data = []
    new_pseudo_labeled_indices = set()
    if unlabeled_dataloader is not None and unlabeled_dataset is not None:
        print(f"Generating pseudo-labels with threshold {threshold}...")
        pseudo_label_progress_bar = tqdm(
            unlabeled_dataloader,
            desc="Generating Pseudo-Labels",
            total=len(unlabeled_dataloader),
            leave=False
        )
        model.eval() # Ensure model is in eval mode
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(pseudo_label_progress_bar):
                start_idx = batch_idx * unlabeled_dataloader.batch_size
                for i in range(images.size(0)):
                    global_idx = start_idx + i
                    if already_pseudo_labeled_indices and global_idx in already_pseudo_labeled_indices:
                        continue  # Skip already pseudo-labeled

                    img = images[i].cpu()
                    # Apply weak and strong augmentations
                    img_weak = weak_transform(img) if weak_transform else img
                    img_strong = strong_transform(img) if strong_transform else img

                    img_weak = img_weak.unsqueeze(0).to(device)
                    img_strong = img_strong.unsqueeze(0).to(device)

                    logits_weak = model(img_weak, None, 'classify')
                    logits_strong = model(img_strong, None, 'classify')

                    prob_weak = torch.softmax(logits_weak, dim=1)
                    prob_strong = torch.softmax(logits_strong, dim=1)

                    max_prob_weak, pred_label_weak = torch.max(prob_weak, dim=1)
                    max_prob_strong, pred_label_strong = torch.max(prob_strong, dim=1)

                    # Both must be confident and agree
                    if (
                        max_prob_weak.item() > threshold and
                        max_prob_strong.item() > threshold and
                        pred_label_weak.item() == pred_label_strong.item()
                        ):
                        img_path = unlabeled_dataset.samples[global_idx][0]
                        new_pseudo_labeled_data.append((img_path, pred_label_weak.item()))
                        new_pseudo_labeled_indices.add(global_idx)

        print(f"Generated {len(new_pseudo_labeled_data)} new pseudo-labels.")
    
    return avg_eval_loss, epoch_eval_accuracy, new_pseudo_labeled_data, new_pseudo_labeled_indices

def train_SSMAE_w_unlabeled(model, unlabeled_data_path, labeled_data_train_path, optimizer, device, 
                num_epochs=100, labeled_data_test_path=None, checkpoint_path='models/best_model.pth', confidence_threshold=0.95, **kwargs):
    """Training loop for SSMAE with unlabeled data"""

    # Load model if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading last saved model weights from {checkpoint_path} ...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Define transformations for labeled data
    # These should match the transformations used during training
    img_size = 224 # Example image size

    labeled_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Weak augmentation (minimal changes)
    weak_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Strong augmentation (heavy changes)
    strong_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    unlabeled_dataset = ImageFolder(root=unlabeled_data_path, transform=labeled_transform)
    labeled_dataset = ImageFolder(root=labeled_data_train_path, transform=labeled_transform)
    test_dataset = ImageFolder(root=labeled_data_test_path, transform=test_transform)

    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        batch_size=32,
        shuffle=True, # Shuffle for training
        num_workers=4,
        pin_memory=True,
        drop_last=True # Recommended if zipping with another dataloader
    )

    labeled_dataloader = DataLoader(
        labeled_dataset,
        batch_size=16,
        shuffle=True, # Shuffle for training
        num_workers=4,
        pin_memory=True,
        drop_last=True # Recommended if zipping with another dataloader
    )

    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=4,
        pin_memory=True,
        drop_last=False # No need to drop last batch for evaluation
    )

    model.train()
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0

    # For memory-efficient pseudo-labeling
    pseudo_image_paths = []
    pseudo_labels = []
    already_pseudo_labeled_indices = set() # Track indices of already pseudo-labeled samples

    original_labeled_dataset = labeled_dataloader.dataset

    # Store original dataloader parameters to recreate it
    original_dataloader_params = {
        'batch_size': labeled_dataloader.batch_size,
        'shuffle': True, # Shuffle should be true for training
        'num_workers': labeled_dataloader.num_workers,
        'pin_memory': labeled_dataloader.pin_memory,
    }

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        # Prepare pseudo-label dataset for this epoch
        if pseudo_image_paths:
            pseudo_dataset = PseudoLabelDataset(pseudo_image_paths, pseudo_labels, transform=labeled_transform)
            combined_dataset = ConcatDataset([original_labeled_dataset, pseudo_dataset])
            current_epoch_labeled_dataloader = DataLoader(combined_dataset, **original_dataloader_params)
            print(f"Epoch {epoch+1}: Training with {len(original_labeled_dataset)} original and {len(pseudo_dataset)} pseudo-labeled samples.")
        else:
            current_epoch_labeled_dataloader = labeled_dataloader
            print(f"Epoch {epoch+1}: Training with {len(original_labeled_dataset)} original labeled samples only.")

        # Wrap the zipped dataloaders with tqdm for a progress bar
        progress_bar = tqdm(
            zip(unlabeled_dataloader, current_epoch_labeled_dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=min(len(unlabeled_dataloader), len(current_epoch_labeled_dataloader)) 
        )
        model.train() # Ensure model is in training mode

        # Zip both dataloaders to process them together
        for batch_idx, ((unlabeled_images, _), (labeled_images, labels)) in enumerate(progress_bar):
            unlabeled_images = unlabeled_images.to(device)
            labeled_images = labeled_images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Concatenate both unlabeled and labeled images for reconstruction
            all_images = torch.cat([unlabeled_images, labeled_images], dim=0)

            # MAE reconstruction loss from concatenated images
            mae_loss, pred, mask = model(all_images, None, 'mae')

            # Classification loss from labeled data only (which now includes pseudo-labels)
            logits = model(labeled_images, None, 'classify')
            cls_loss = criterion(logits, labels)

            # Combine both losses
            combined_loss = mae_loss + cls_loss

            # Backpropagate the total combined loss
            combined_loss.mean().backward()
            optimizer.step()

            # Track metrics
            total_loss += combined_loss.mean().item()
            num_batches += 1

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_accuracy_batch = 100. * (predicted == labels).sum().item() / labels.size(0) if labels.size(0) > 0 else 0

            if batch_idx % 100 == 0:
                # Ensure total is not zero for accuracy calculation
                accuracy_so_far = 100. * correct / total if total > 0 else 0.0
                print(
                    f'Epoch {epoch+1}, Batch {batch_idx}/{min(len(unlabeled_dataloader), len(current_epoch_labeled_dataloader))}, '
                    f'Total Loss: {combined_loss.mean().item():.4f}, '
                    f'MAE Loss: {mae_loss.mean().item():.4f}, '
                    f'Cls Loss: {cls_loss.mean().item():.4f}, '
                    f'Acc (batch): {current_accuracy_batch:.2f}%, '
                    f'Acc (epoch): {accuracy_so_far:.2f}%'
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        print(f'Epoch {epoch+1} completed, Average Total Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%')

        if eval_dataloader is not None:
            # Evaluate on the labeled validation set and generate new pseudo-labels for the next epoch
            # Pass unlabeled_eval_dataloader for pseudo-label generation
            _, epoch_eval_accuracy, new_pseudo_labels_from_eval, new_pseudo_label_indices = evaluate_classification(
                model=model, 
                eval_dataloader=eval_dataloader, 
                criterion=criterion, 
                device=device,
                unlabeled_dataloader=unlabeled_dataloader, # Use this for pseudo-labeling
                unlabeled_dataset=unlabeled_dataset,
                threshold=confidence_threshold,
                already_pseudo_labeled_indices=already_pseudo_labeled_indices,
                weak_transform=weak_transform,
                strong_transform=strong_transform)

            # Save the model if validation accuracy on Labeled data improves
            if epoch_eval_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_eval_accuracy
                # Ensure the checkpoint directory exists
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}% to {checkpoint_path}")
            
            # Store file paths and labels
            for idx, (img_tensor, label_tensor) in zip(new_pseudo_label_indices, new_pseudo_labels_from_eval):
                if idx not in already_pseudo_labeled_indices: # Avoid duplicates
                    # Get file path from unlabeled_dataset
                    img_path = unlabeled_dataset.samples[idx][0]
                    pseudo_image_paths.append(img_path)
                    pseudo_labels.append(label_tensor.item())
                    already_pseudo_labeled_indices.add(idx)

            print(f"Total accumulated pseudo-labels: {len(already_pseudo_labeled_indices)}")

            model.train() # Ensure model is back in training mode after evaluation

def train_SSMAE(model, unlabeled_dataloader, labeled_dataloader, optimizer, device, 
                num_epochs=100, eval_dataloader=None, checkpoint_path='models/best_model.pth'):
    """Training loop for SSMAE pretraining"""

    model.train()
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        # Wrap the zipped dataloaders with tqdm for a progress bar
        progress_bar = tqdm(
            zip(unlabeled_dataloader, labeled_dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=min(len(unlabeled_dataloader), len(labeled_dataloader)) # Show progress based on the shorter dataloader
        )

        # Zip both dataloaders to process them together
        for batch_idx, ((unlabeled_images, _), (labeled_images, labels)) in enumerate(progress_bar):
            unlabeled_images = unlabeled_images.to(device)
            labeled_images = labeled_images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Concatenate both unlabeled and labeled images for reconstruction
            all_images = torch.cat([unlabeled_images, labeled_images], dim=0)

            # MAE reconstruction loss from concatenated images
            mae_loss, pred, mask = model(all_images, None, 'mae')

            # Classification loss from labeled data only
            logits = model(labeled_images, None, 'classify')
            cls_loss = criterion(logits, labels)

            # Combine both losses
            combined_loss = mae_loss + cls_loss

            # Backpropagate the total combined loss
            combined_loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += combined_loss.item()
            num_batches += 1

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                accuracy = 100. * correct / total
                print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                      f'Total Loss: {combined_loss.item():.4f}, '
                      f'MAE Loss: {mae_loss.item():.4f}, '
                      f'Classification Loss: {cls_loss.item():.4f}, '
                      f'Acc: {accuracy:.2f}%')
        
        avg_loss = total_loss / num_batches
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1} completed, Average Total Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if eval_dataloader is not None:
            _ , epoch_eval_accuracy = evaluate_classification(model=model, eval_dataloader=eval_dataloader, criterion=criterion, device=device)
            # Save the model if validation accuracy improves
            if epoch_eval_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_eval_accuracy
                torch.save(model.state_dict(), checkpoint_path)
                print(f"New best model saved with accuracy: {best_val_accuracy:.2f}% to {checkpoint_path}")
            model.train() # Ensure model is back in training mode after evaluation


def freeze_encoder_for_classification(model, freeze=True):
    """
    Freeze or unfreeze encoder weights for classification fine-tuning
    
    Args:
        model: MAE model
        freeze: If True, freeze encoder weights (only train classifier)
                If False, unfreeze encoder weights (fine-tune entire encoder)
    """
    # Freeze/unfreeze encoder components
    encoder_components = [
        model.patch_embed, model.cls_token, model.pos_embed,
        model.encoder_blocks, model.encoder_norm
    ]
    
    for component in encoder_components:
        if hasattr(component, 'parameters'):
            for param in component.parameters():
                param.requires_grad = not freeze
        elif hasattr(component, 'requires_grad'):  # For parameter tensors
            component.requires_grad = not freeze
    
    # Keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Keep decoder frozen (not used in classification)
    decoder_components = [
        model.decoder_embed, model.mask_token, model.decoder_pos_embed,
        model.decoder_blocks, model.decoder_norm, model.decoder_pred
    ]
    
    for component in decoder_components:
        if hasattr(component, 'parameters'):
            for param in component.parameters():
                param.requires_grad = False
        elif hasattr(component, 'requires_grad'):
            component.requires_grad = False
    
    print(f"Encoder weights {'frozen' if freeze else 'unfrozen'} for classification")

def load_pretrained_mae_for_classification(model, mae_checkpoint_path):
    """
    Load pretrained MAE weights into the model for classification
    Only loads encoder weights, skips decoder and classifier
    """
    checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
    
    # Filter out decoder weights and classifier weights
    encoder_state_dict = {}
    for key, value in checkpoint.items():
        if not key.startswith('decoder_') and not key.startswith('classifier'):
            encoder_state_dict[key] = value
    
    # Load encoder weights
    missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)
    
    print(f"Loaded pretrained MAE encoder weights")
    print(f"Missing keys (expected - classifier): {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    return model

def visualize_reconstruction(model, image, device):
    """Visualize original, masked, and reconstructed images"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        loss, pred, mask = model(image.unsqueeze(0).to(device))
        
        # Reconstruct image
        reconstructed = model.unpatchify(pred)
        
        # Create masked image for visualization
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2 * 3)
        original_patches = model.patchify(image.unsqueeze(0).to(device))
        masked_patches = original_patches * (1 - mask_expanded)
        masked_image = model.unpatchify(masked_patches)
        
        return image, masked_image.squeeze(0), reconstructed.squeeze(0)