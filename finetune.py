import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from tqdm import tqdm
from main import create_mae_model, evaluate_classification

def finetune_model(pretrained_model_path, train_data_path, test_data_path, device, 
                   num_classes=10, num_epochs=50, batch_size=32, learning_rate=1e-4,
                   checkpoint_path='models/finetuned_model.pth', model_size='base'):
    """
    Finetune a pretrained MAE model for classification
    
    Args:
        pretrained_model_path: Path to the pretrained model checkpoint
        train_data_path: Path to training data
        test_data_path: Path to test data
        device: Device to run on
        num_classes: Number of classes for classification
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        checkpoint_path: Path to save the best finetuned model
        model_size: Size of the model ('base', 'large', 'huge')
    """
    
    # Create model
    model = create_mae_model(model_size=model_size, num_classes=num_classes)
    model = model.to(device)
    
    # Load pretrained weights
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)  # Use strict=False in case of size mismatches
        print("Pretrained model loaded successfully")
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}")
        print("Training from scratch...")
    
    # Data transforms
    img_size = 224
    train_transform = transforms.Compose([
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
    
    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training variables
    best_val_accuracy = 0.0
    
    print(f"Starting finetuning for {num_epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass - only classification
            logits = model(images, None, 'classify')
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                acc = 100. * correct / total if total > 0 else 0.0
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{acc:.2f}%'
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_dataloader)
        train_accuracy = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss, val_accuracy = evaluate_classification(
            model=model, 
            eval_dataloader=test_dataloader, 
            criterion=criterion, 
            device=device,
            calc_high_confidence=False  # No need for high confidence in validation
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
        
        print("-" * 50)
    
    print(f"Finetuning completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best model saved to: {checkpoint_path}")
    
    return model, best_val_accuracy

def freeze_encoder_finetune(pretrained_model_path, train_data_path, test_data_path, device,
                           num_classes=10, num_epochs=30, batch_size=32, learning_rate=1e-3,
                           checkpoint_path='models/frozen_encoder_finetuned.pth', model_size='base'):
    """
    Finetune only the classifier head while freezing the encoder
    """
    # Create model
    model = create_mae_model(model_size=model_size, num_classes=num_classes)
    model = model.to(device)
    
    # Load pretrained weights
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("Pretrained model loaded successfully")
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}")

    # Freeze ALL components except classifier
    components_to_freeze = [
        # Encoder components
        'patch_embed', 'encoder_blocks', 'encoder_norm', 'pos_embed', 'cls_token',
        # Decoder components (these should be frozen for classification)
        'mask_token', 'decoder_pos_embed', 'decoder_embed', 'decoder_blocks', 
        'decoder_norm', 'decoder_pred'
    ]

    for component in components_to_freeze:
        if hasattr(model, component):
            print(f"Freezing {component} parameters...")
            component_module = getattr(model, component)
            if isinstance(component_module, nn.Parameter):
                component_module.requires_grad = False
            else:
                for param in component_module.parameters():
                    param.requires_grad = False

    # Only train classifier parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Print which parameters are trainable for verification
    print("Trainable parameter names:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} parameters")
    
    # Use the same finetuning process but with frozen encoder
    return finetune_model(
        pretrained_model_path=pretrained_model_path,  # Don't reload since we already loaded
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        device=device,
        num_classes=num_classes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
        model_size=model_size
    )

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    path = '/mnt/d/OneDrive - Rowan University/RA/Summer 25/SSMAE/'
    pretrained_model_path = "models/best_model_clean.pth"
    train_data_path = path+"data/cifar10/train/labeled_10"  # Use your labeled training data
    test_data_path = path+"data/cifar10/test"

    # Finetuning parameters
    num_classes = 10 # Adjust based on your dataset
    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    
    print("Starting full model finetuning...")
    model, accuracy = finetune_model(
        pretrained_model_path=pretrained_model_path,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        device=device,
        num_classes=num_classes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path='models/best_model_clean.pth'
    )
    print(f"Full finetuning accuracy: {accuracy:.2f}%")
    
    # print("\nStarting frozen encoder finetuning...")
    # frozen_model, frozen_accuracy = freeze_encoder_finetune(
    #     pretrained_model_path=pretrained_model_path,
    #     train_data_path=train_data_path,
    #     test_data_path=test_data_path,
    #     device=device,
    #     num_classes=num_classes,
    #     num_epochs=100,
    #     batch_size=batch_size,
    #     learning_rate=1e-3,
    #     checkpoint_path='models/frozen_encoder_finetuned.pth'
    # )
    
    # print(f"Frozen encoder finetuning accuracy: {frozen_accuracy:.2f}%")