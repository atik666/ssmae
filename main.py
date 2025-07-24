from ssmae import MaskedAutoencoder
import torch
from torch import nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

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
    
def evaluate_classification(model, eval_dataloader, criterion, device, calc_high_confidence=True):
    """Helper function to evaluate model on classification task"""
    model.eval()  # Set model to evaluation mode
    total_eval_loss = 0
    eval_correct = 0
    eval_total = 0
    num_eval_batches = 0

    if calc_high_confidence:
        # Initialize high confidence counters for overall statistics
        high_conf_correct_total = 0
        high_conf_count_total = 0
        high_conf_accuracy = 0.0  # Initialize with default value

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

            if calc_high_confidence:

                """ Calculate probabilities and high-confidence metrics """

                probs = torch.softmax(logits, dim=1)
                max_probs, _ = torch.max(probs, dim=1)

                # Calculate high-confidence metrics
                high_conf_mask = max_probs > 0.95
                high_conf_count = high_conf_mask.sum().item()

                if high_conf_count > 0:
                    _, predicted = torch.max(logits.data, 1)
                    high_conf_correct = (predicted[high_conf_mask] == labels[high_conf_mask]).sum().item()
                    # Accumulate totals instead of printing
                    high_conf_correct_total += high_conf_correct
                    high_conf_count_total += high_conf_count

            loss = criterion(logits, labels)
            
            total_eval_loss += loss.item()
            num_eval_batches += 1
            
            _, predicted = torch.max(logits.data, 1)
            eval_total += labels.size(0)
            eval_correct += (predicted == labels).sum().item()

    avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else 0
    epoch_eval_accuracy = 100. * eval_correct / eval_total if eval_total > 0 else 0

    if calc_high_confidence:
        # Print high confidence statistics once after all batches
        if high_conf_count_total > 0:
            high_conf_accuracy = 100. * high_conf_correct_total / high_conf_count_total
            print(f"High confidence accuracy (prob > 0.95): {high_conf_accuracy:.2f}% ({high_conf_correct_total}/{high_conf_count_total})")
        else:
            print("No predictions with confidence > 0.95")

    # Print final evaluation metrics after the loop
    print(f"Evaluation: Avg Loss: {avg_eval_loss:.4f}, Accuracy: {epoch_eval_accuracy:.2f}%")

    if calc_high_confidence:
        return avg_eval_loss, epoch_eval_accuracy, high_conf_accuracy
    else:
        return avg_eval_loss, epoch_eval_accuracy

def train_SSMAE_w_unlabeled(model, unlabeled_data_path, labeled_data_train_path, labeled_data_test_path, optimizer, device, 
                num_epochs=100, checkpoint_path='models/best_model_clean.pth', 
                confidence_threshold=0.95, high_conf_acc_threshold=85.0, cls_loss_weight=0.5, **kwargs):
    
    """
    Semi-supervised training with MAE, labeled and pseudo-labeling.
    unlabeled_dataloader must yield (images, paths) where paths is a list of file paths.
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

    pseudo_label_memory = {}  # {img_path: pseudo_label}
    best_val_accuracy = 0.0
    enable_pseudo_labeling = False  # Initialize pseudo-labeling flag
    pseudo_label_start_epoch = 20  # Track the epoch when pseudo-labeling starts
    epoch_eval_accuracy = 0.0  # Initialize evaluation accuracy

    for epoch in range(num_epochs):
        total_loss = 0
        total_cls = 0
        total_mae = 0
        num_batches = 0
        correct = 0
        total = 0
        epoch_new_pseudo_labels = 0  # Counter for new pseudo-labels in this epoch

        # Wrap the zipped dataloaders with tqdm for a progress bar
        progress_bar = tqdm(
            zip(unlabeled_dataloader, labeled_dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=min(len(unlabeled_dataloader), len(labeled_dataloader)) )

        # Start with evaluation to determine if we should do pseudo-labeling this epoch
        if epoch >= pseudo_label_start_epoch:  # Skip first few epochs as we don't have high_conf_accuracy yet
            # Evaluate on the labeled validation set and generate new pseudo-labels for the next epoch
            # Pass unlabeled_eval_dataloader for pseudo-label generation
            _, epoch_eval_accuracy, high_conf_accuracy = evaluate_classification(
                model=model, eval_dataloader=eval_dataloader, criterion=criterion, device=device, calc_high_confidence=True)
            
            # Only enable pseudo-labeling if high confidence accuracy is good enough
            enable_pseudo_labeling = high_conf_accuracy >= high_conf_acc_threshold
            if enable_pseudo_labeling:
                print(f"High confidence accuracy {high_conf_accuracy:.2f}% exceeds threshold {high_conf_acc_threshold:.2f}% - Pseudo-labeling enabled")
            else:
                print(f"High confidence accuracy {high_conf_accuracy:.2f}% below threshold {high_conf_acc_threshold:.2f}% - Pseudo-labeling disabled")
        elif epoch > 0 and epoch < pseudo_label_start_epoch:
            _, epoch_eval_accuracy = evaluate_classification(
                model=model, eval_dataloader=eval_dataloader, criterion=criterion, device=device, calc_high_confidence=False)
        else: 
            print("Skipping evaluation in first epoch, pseudo-labeling not enabled yet")

        # Save the model if validation accuracy on Labeled data improves
        if epoch_eval_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_eval_accuracy
            # Ensure the checkpoint directory exists
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}% to {checkpoint_path}")

        # Zip both dataloaders to process them together
        for batch_idx, ((unlabeled_images, unlabeled_paths), (labeled_images, labels)) in enumerate(progress_bar):

            # Prepare weak and strong augmentations for unlabeled images
            weak_imgs = torch.stack([weak_transform(img) for img in unlabeled_images]).to(device)
            strong_imgs = torch.stack([strong_transform(img) for img in unlabeled_images]).to(device)
            labeled_images = labeled_images.to(device)
            labels = labels.to(device)

            # Get model predictions for pseudo-labeling
            logits_weak = model(weak_imgs, None, 'classify')
            logits_strong = model(strong_imgs, None, 'classify')

            probs_weak = torch.softmax(logits_weak, dim=1)
            probs_strong = torch.softmax(logits_strong, dim=1)

            max_probs_weak, preds_weak = torch.max(probs_weak, dim=1)
            max_probs_strong, preds_strong = torch.max(probs_strong, dim=1)

            # Select pseudo-labels for this batch
            pseudo_imgs = []
            pseudo_targets = []
            for i, img_path in enumerate(unlabeled_paths):
                if img_path in pseudo_label_memory:
                    # Already pseudo-labeled, use stored pseudo-label
                    pseudo_imgs.append(weak_imgs[i])
                    pseudo_targets.append(pseudo_label_memory[img_path])
                elif enable_pseudo_labeling and (
                    max_probs_weak[i].item() > confidence_threshold and
                    max_probs_strong[i].item() > confidence_threshold and
                    preds_weak[i].item() == preds_strong[i].item()
                ):
                    # New confident pseudo-label
                    pseudo_label = preds_weak[i].item()
                    pseudo_label_memory[img_path] = pseudo_label
                    pseudo_imgs.append(weak_imgs[i])
                    pseudo_targets.append(pseudo_label)
                    epoch_new_pseudo_labels += 1  # Increment counter for new pseudo-labels in this epoch

            # Combine labeled and pseudo-labeled data for supervised loss
            if pseudo_imgs:
                pseudo_imgs = torch.stack(pseudo_imgs)
                pseudo_targets = torch.tensor(pseudo_targets, dtype=torch.long, device=device)
                all_imgs = torch.cat([labeled_images, pseudo_imgs], dim=0)
                all_targets = torch.cat([labels, pseudo_targets], dim=0)
            else:
                all_imgs = labeled_images
                all_targets = labels

            # MAE step: use all images (labeled + unlabeled)
            unlabeled_images = torch.stack([unlabeled_transform(img) for img in unlabeled_images]).to(device)
            all_mae_imgs = torch.cat([labeled_images, unlabeled_images], dim=0)
            mae_loss, _, _ = model(all_mae_imgs, None, 'mae')

            # Classification step: use labeled + pseudo-labeled
            logits = model(all_imgs, None, 'classify')
            cls_loss = criterion(logits, all_targets)

            # Combine both losses
            combined_loss = mae_loss + cls_loss_weight * cls_loss

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
            total += all_targets.size(0)
            correct += (predicted == all_targets).sum().item()

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
        
        if epoch >= pseudo_label_start_epoch:
            print(f"Epoch {epoch+1}: New pseudo-labels generated this epoch: {epoch_new_pseudo_labels}, Total pseudo-labeled generated: {len(pseudo_label_memory)}")

        model.train() # Ensure model is back in training mode after evaluation

