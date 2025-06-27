from mae import MaskedAutoencoder
import torch
from torch import nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    
class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths."""
    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super().__getitem__(index)
        # The image file path
        path = self.samples[index][0]
        # Return (image, path)
        return original_tuple[0], path
    
def evaluate_classification(model, eval_dataloader, criterion, device):
    """
    Helper function to evaluate model on classification task
    
    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        tuple: (average loss, accuracy percentage)
    """
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
    
    return avg_eval_loss, epoch_eval_accuracy

def train_SSMAE_w_unlabeled(model, unlabeled_data_path, labeled_data_train_path, labeled_data_test_path, optimizer, device, 
                num_epochs=100, checkpoint_path='models/best_model_match.pth', 
                confidence_threshold=0.96, **kwargs):
    
    """
    Semi-supervised training with MAE and FixMatch.
    
    Args:
        model: The model to train
        unlabeled_data_path: Directory containing unlabeled data
        labeled_data_train_path: Directory containing labeled training data
        labeled_data_test_path: Directory containing labeled test data
        optimizer: Optimizer for training
        device: Device to run training on
        num_epochs: Number of training epochs
        checkpoint_path: Path to save best model checkpoint
        confidence_threshold: Threshold for pseudo-labeling confidence
        lambda_u: Weight for unsupervised loss component
        temperature: Temperature for sharpening pseudo-labels
        patience: Number of epochs to wait for improvement before stopping
        **kwargs: Additional arguments
    """

    # Load model if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading last saved model weights from {checkpoint_path} ...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found, starting training from scratch.")

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

    unlabeled_transform = transforms.Compose([
        transforms.ToPILImage(),
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
                           std=[0.229, 0.224, 0.225])])

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

    # unlabeled_dataset = ImageFolder(root=unlabeled_data_path, transform=labeled_transform)
    unlabeled_dataset = ImageFolderWithPaths(root=unlabeled_data_path, transform=labeled_transform)
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

    pseudo_label_memory = {}  # Dictionary to store pseudo-labels
    best_val_accuracy = 0.0
    lambda_u = 1.0  # Adjust this weight as needed
    temperature = 1.2 # To make the weak predictions less confident

    for epoch in range(num_epochs):
        total_loss = 0
        total_cls = 0
        total_mae = 0
        num_batches = 0
        correct = 0
        total = 0
        epoch_new_pseudo_labels = 0  # Track new pseudo-labels generated in this epoch

        # Wrap the zipped dataloaders with tqdm for a progress bar
        progress_bar = tqdm(
            zip(unlabeled_dataloader, labeled_dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=min(len(unlabeled_dataloader), len(labeled_dataloader)) )

        # Zip both dataloaders to process them together
        for batch_idx, ((unlabeled_images, unlabeled_paths), (labeled_images, labels)) in enumerate(progress_bar):

            # Prepare weak augmentations for unlabeled images
            weak_imgs = torch.stack([weak_transform(img) for img in unlabeled_images]).to(device)
            
            labeled_images = labeled_images.to(device)
            labels = labels.to(device)

            # Get model predictions for pseudo-labeling
            logits_weak = model(weak_imgs, None, 'classify')

            logits_weak = logits_weak / temperature

            probs_weak = torch.softmax(logits_weak, dim=1)
            max_probs_weak, preds_weak = torch.max(probs_weak, dim=1)

            # FixMatch: generate pseudo-labels from weak augmentations
            mask = max_probs_weak > confidence_threshold
            num_confident = mask.sum().item()

            if epoch >= 2 and num_confident > 0 and best_val_accuracy > 18.0:

                # Strong augmentations
                strong_imgs = torch.stack([strong_transform(img) for img in unlabeled_images]).to(device)
                logits_strong = model(strong_imgs, None, 'classify') # Get logits for strong augmentations

                # Create one-hot pseudo-labels from confident weak augmentation predictions
                pseudo_labels = torch.zeros(preds_weak.size(0), logits_weak.size(1), device=device)
                pseudo_labels[range(preds_weak.size(0)), preds_weak] = 1

                # Apply unsupervised loss only on confident examples
                fixmatch_loss = torch.nn.functional.cross_entropy(
                    logits_strong[mask], 
                    pseudo_labels[mask],
                    reduction='mean'
                )

                # Track newly generated pseudo-labels
                for i, img_path in enumerate(unlabeled_paths):
                    if max_probs_weak[i].item() > confidence_threshold:
                        if img_path not in pseudo_label_memory:
                            pseudo_label_memory[img_path] = preds_weak[i].item()
                            epoch_new_pseudo_labels += 1

            # MAE step: use all images (labeled + unlabeled)
            unlabeled_images = torch.stack([unlabeled_transform(img) for img in unlabeled_images]).to(device)
            all_mae_imgs = torch.cat([labeled_images, unlabeled_images], dim=0)
            mae_loss, _, _ = model(all_mae_imgs, None, 'mae')

            # Classification step: use labeled + pseudo-labeled
            logits = model(labeled_images, None, 'classify')
            cls_loss = criterion(logits, labels)

            # Combine losses: supervised + unsupervised + reconstruction
            # You can weight the unsupervised loss with a lambda parameter if needed

            combined_loss = cls_loss + mae_loss

            # Combine both losses
            if num_confident > 0:
                combined_loss = combined_loss + lambda_u * fixmatch_loss

            # Backpropagate the total combined loss
            optimizer.zero_grad()
            combined_loss.mean().backward()
            optimizer.step()

            # Track metrics
            total_loss += combined_loss.mean().item()
            total_mae += mae_loss.mean().item()
            total_cls += cls_loss.mean().item()
            num_batches += 1

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                acc = 100. * correct / total if total > 0 else 0.0
                print(f"Epoch {epoch+1}, Batch {batch_idx}, "
                      f"Total Loss: {combined_loss.mean().item():.4f}, "
                      f"MAE Loss: {mae_loss.mean().item():.4f}, "
                      f"Cls Loss: {cls_loss.mean().item():.4f}, "
                      f"Acc: {acc:.2f}%")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0
        avg_cls = total_cls / num_batches if num_batches > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0

        print(f"Epoch {epoch+1} completed, Avg Total Loss: {avg_loss:.4f}, "
              f"Avg MAE Loss: {avg_mae:.4f}, Avg Cls Loss: {avg_cls:.4f}, Accuracy: {accuracy:.2f}%")
        
        print(f"Epoch {epoch+1}: New pseudo-labels generated this epoch: {epoch_new_pseudo_labels}, Total pseudo-labeled generated: {len(pseudo_label_memory)}")

        # Evaluate on the labeled validation set and generate new pseudo-labels for the next epoch
        # Pass unlabeled_eval_dataloader for pseudo-label generation
        _, epoch_eval_accuracy = evaluate_classification(
            model=model, eval_dataloader=eval_dataloader, criterion=criterion, device=device)

        # Save the model if validation accuracy on Labeled data improves
        if epoch_eval_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_eval_accuracy
            # Ensure the checkpoint directory exists
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}% to {checkpoint_path}")

        model.train() # Ensure model is back in training mode after evaluation

