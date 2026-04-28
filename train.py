import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from collections import Counter
import argparse
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./classifier_checkpoints', help='Directory to save outputs')
    parser.add_argument('--num_epochs', type=int, default=30, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Transforms with Moderate Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        # Moderate explicit Gaussian noise
        transforms.Lambda(lambda x: torch.clamp(x + 0.05 * torch.randn_like(x), 0, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Data Loaders using ImageFolder
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    # Windows CPU multiprocessing usually completely hangs PyTorch dataloaders
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_mem = True if torch.cuda.is_available() else False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    num_classes = len(train_ds.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 3. Class Imbalance Handling
    print("Computing class weights...")
    counts = Counter([label for _, label in train_ds.samples])
    # Protect against missing classes, though ImageFolder guarantees they are present if in classes
    weights = [1.0 / counts[i] if i in counts and counts[i] > 0 else 1.0 for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Model Definition & Freezing Strategy
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    
    # Freeze initial feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Modify classifier head with heavy dropout
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(0.5, inplace=True),
        nn.Linear(1024, num_classes)
    )
    
    model = model.to(device)

    # 5. Optimizer & Scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # 6. Training Loop with Early Stopping
    patience = args.patience
    best_loss = float('inf')
    counter = 0

    best_model_path = os.path.join(args.checkpoint_dir, 'best_classifier.pt')

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  [*] Best model saved!")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

if __name__ == '__main__':
    main()
