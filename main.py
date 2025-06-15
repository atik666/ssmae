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

    unlabeled_data_path = "./data/train/unlabeled"
    labeled_data_train_path = "./data/train/labeled"
    labeled_data_test_path = "./data/test"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Learning rate
        weight_decay=0.05  # Weight decay for regularization
    )

    # TODO: need to pass the datapath (not dataloader) to the train function for doing the augmentation (transformations) later.

    # train_SSMAE(model, unlabeled_dataloader, labeled_dataloader, optimizer, device, num_epochs=100, eval_dataloader=test_dataloader)
    train_SSMAE_w_unlabeled(model, unlabeled_data_path, labeled_data_train_path, optimizer, device, num_epochs=100, labeled_data_test_path=labeled_data_test_path)