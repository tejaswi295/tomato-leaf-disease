import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation: blends pairs of samples to prevent memorization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for MixUp-blended targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0, path='best_classifier.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class TomatoDiseaseClassifier(nn.Module):
    """
    Advanced classifier supporting ResNet50 and EfficientNet_B0.
    Includes custom classification head with Dropout & BatchNorm.
    """
    def __init__(self, num_classes=10, model_name='efficientnet'):
        super(TomatoDiseaseClassifier, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Freeze base layers to retain pretrained features
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the last layer4 block for fine-tuning
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
                
            in_features = self.backbone.fc.in_features
            # Advanced prediction head
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == 'efficientnet':
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Freeze base layers
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the final convolutional stages
            for param in self.backbone.features[-2:].parameters():
                param.requires_grad = True
                
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        return self.backbone(x)


class PlantVillageDataset(torch.utils.data.Dataset):
    """Custom dataset for PlantVillage tomato disease images without split-specific logic."""
    def __init__(self, root_dir, disease_classes=None):
        self.root_dir = root_dir
        # Standard 10 PlantVillage tomato classes
        self.disease_classes = disease_classes or [
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.disease_classes)}
        
        search_dir = root_dir
        if os.path.exists(os.path.join(root_dir, 'PlantVillage')):
            search_dir = os.path.join(root_dir, 'PlantVillage')
            
        self.images = []
        self.labels = []
        
        for class_name in self.disease_classes:
            class_dir = os.path.join(search_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
                        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # We simply return path and label, transformations are applied via the Subset wrapper
        return self.images[idx], self.labels[idx]


class TransformSubset(torch.utils.data.Dataset):
    """Applies transformations to a generated subset dynamically."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        img_path, label = self.subset[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.subset)


def train_classifier(args):
    # Device setup & Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Model: {args.model_name.upper()}")
    
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'runs'))
    
    # 1. Moderated Real-World Field Data Augmentations
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.5))], p=0.2),
        transforms.ToTensor(),
        # Add slight random Gaussian noise directly to the tensor to simulate sensor noise
        transforms.Lambda(lambda x: x + 0.02 * torch.randn_like(x)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # RandomErasing forces model to not rely on any single region
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    
    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Dataset and T/V/T Spilt Logic
    base_dataset = PlantVillageDataset(args.data_dir)
    n_total = len(base_dataset)
    if n_total == 0:
        raise ValueError(f"No images found in {args.data_dir}. Check dataset existence.")
        
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val
    
    generator = torch.Generator().manual_seed(args.seed)
    train_ds_raw, val_ds_raw, test_ds_raw = torch.utils.data.random_split(
        base_dataset, [n_train, n_val, n_test], generator=generator
    )
    
    train_dataset = TransformSubset(train_ds_raw, transform=transform_train)
    val_dataset = TransformSubset(val_ds_raw, transform=transform_val_test)
    test_dataset = TransformSubset(test_ds_raw, transform=transform_val_test)
    
    print(f"Split distribution -> Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # Loaders
    num_workers = args.num_workers if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # 3. Handle Class Imbalance with Weighted Loss
    train_labels = [base_dataset.labels[train_ds_raw.indices[i]] for i in range(len(train_ds_raw))]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(base_dataset.disease_classes)
    
    class_weights = []
    for i in range(num_classes):
        # class_weight = total_samples / (num_classes * class_count)
        count = class_counts.get(i, 0)
        weight = total_samples / (num_classes * count) if count > 0 else 0
        class_weights.append(weight)
        
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Class weights calculated for CrossEntropyLoss.")

    # Model & Optimizers
    model = TomatoDiseaseClassifier(num_classes=num_classes, model_name=args.model_name).to(device)
    # Label smoothing (0.1) prevents overconfident predictions and combats memorization
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine Annealing scheduler generally works very well for fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    best_model_path = os.path.join(args.checkpoint_dir, 'best_classifier.pt')
    early_stopping = EarlyStopping(patience=args.patience, path=best_model_path)

    # 4. Training Loop
    print("\nStarting Training Phase...")
    train_losses, val_losses = [], []
    
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Apply MixUp augmentation to prevent memorization
            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            # Gradient clipping to stabilize training with aggressive augmentation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(val_targets, val_preds)
        
        tb_writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        tb_writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        tb_writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        tb_writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        scheduler.step()
        
        print(f"--> Epoch [{epoch+1}/{args.num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping hook
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered! Model training halted.")
            break
            
    # 5. Testing Evaluation Phase
    print("\nLoading best model for independent evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_preds, test_targets = [], []
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
    # Compute Metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_targets, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
    
    print("\n" + "="*30)
    print("FINAL TEST SET PERFORMANCE")
    print("="*30)
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    
    # Save Metrics JSON
    metrics_to_save = {
        'model': args.model_name,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }
    with open(os.path.join(args.checkpoint_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
        
    # Plot curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy)')
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_curves.png'))
    plt.close()
    
    # Confusion Matrix Output
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.replace('Tomato___', '').replace('_', ' ') for c in base_dataset.disease_classes],
                yticklabels=[c.replace('Tomato___', '').replace('_', ' ') for c in base_dataset.disease_classes])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Artifacts and tensorboard logs saved to '{args.checkpoint_dir}'")
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Training Pipeline for Tomato Leaf Diseases')
    parser.add_argument('--data_dir', type=str, default='./data/PlantVillage', help='Path to PlantVillage dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./classifier_checkpoints', help='Directory to save outputs')
    parser.add_argument('--model_name', type=str, default='efficientnet', choices=['resnet50', 'efficientnet'], help='Backbone model architecture')
    parser.add_argument('--num_epochs', type=int, default=30, help='Max number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training/validating/testing')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=20, help='Interval blocks for logging output during training')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader thread workers')
    
    args = parser.parse_args()
    train_classifier(args)
