from train import create_mae_model
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from train import train_SSMAE, train_SSMAE_w_unlabeled

if __name__ == "__main__":
    # Create model for ImageNet classification (1000 classes)
    model = create_mae_model('base', num_classes=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for name, p in model.named_parameters() 
                        if not name.startswith('decoder_') and not name.startswith('classifier'))
    classifier_params = (sum(p.numel() for p in model.module.classifier.parameters()) 
                        if torch.cuda.device_count() > 1 
                        else sum(p.numel() for p in model.classifier.parameters()))

    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")

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


    unlabeled_data = "./data/train/unlabeled"
    labeled_data_train = "./data/train/labeled"
    labeled_data_test = "./data/test"

    unlabeled_dataset = ImageFolder(root=unlabeled_data, transform=labeled_transform)
    labeled_dataset = ImageFolder(root=labeled_data_train, transform=labeled_transform)
    test_dataset = ImageFolder(root=labeled_data_test, transform=test_transform)

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

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=4,
        pin_memory=True,
        drop_last=False # No need to drop last batch for evaluation
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Learning rate
        weight_decay=0.05  # Weight decay for regularization
    )

    # TODO: need to pass the datapath (not dataloader) to the train function for doing the augmentation (transformations) later.

    # train_SSMAE(model, unlabeled_dataloader, labeled_dataloader, optimizer, device, num_epochs=100, eval_dataloader=test_dataloader)
    train_SSMAE_w_unlabeled(model, unlabeled_dataloader, labeled_dataloader, optimizer, device, num_epochs=100, eval_dataloader=test_dataloader)