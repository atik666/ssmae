import torch
import timm
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Datasets and loaders
    train_dataset = datasets.ImageFolder("./data/train/labeled", transform=transform)
    val_dataset = datasets.ImageFolder("./data/test", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, 51):  # 50 epochs
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)
        print(f"Epoch {epoch} Train Loss: {total_loss/total:.4f} Acc: {correct/total:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += imgs.size(0)
        print(f"Epoch {epoch} Val Loss: {val_loss/val_total:.4f} Acc: {val_correct/val_total:.4f}")