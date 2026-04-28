import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import numpy as np

def main():
    checkpoint_dir = './classifier_checkpoints'
    model_path = os.path.join(checkpoint_dir, 'best_classifier.pt')
    test_dir = './dataset/test'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # No augmentation on test set
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    num_classes = len(test_ds.classes)
    class_names = test_ds.classes

    print("Loading model...")
    # Model definition MUST match train.py exactly
    model = mobilenet_v3_large()
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(0.5, inplace=True),
        nn.Linear(1024, num_classes)
    )
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}.")
        print("Please train the model first by running train.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    
    # Ensured explicit eval
    model.eval()

    all_preds = []
    all_targets = []
    all_confs = []
    
    correct_confs = []
    incorrect_confs = []

    print(f"Starting evaluation on the untouched test set ({len(test_ds)} images)...")
    
    # explicit no gradient tracking
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Use softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            confs_np = confs.cpu().numpy()
            
            all_preds.extend(preds_np)
            all_targets.extend(labels_np)
            all_confs.extend(confs_np)
            
            # Calibration check variables
            for c, p, l in zip(confs_np, preds_np, labels_np):
                if p == l:
                    correct_confs.append(c)
                else:
                    incorrect_confs.append(c)

    # 1. Metrics Calculation
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    print("\n" + "="*40)
    print("FINAL TEST SET METRICS")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # 2. Calibration Check Logging
    avg_conf_correct = np.mean(correct_confs) if correct_confs else 0.0
    avg_conf_incorrect = np.mean(incorrect_confs) if incorrect_confs else 0.0
    print("\n--- Calibration Check ---")
    print(f"Average confidence on CORRECT predictions: {avg_conf_correct:.4f}")
    print(f"Average confidence on INCORRECT predictions: {avg_conf_incorrect:.4f}")
    if avg_conf_incorrect > 0.8:
        print("Warning: The model is highly confident even when it is wrong (potential overfitting to features).")

    # 3. Save Metrics to JSON
    metrics_to_save = {
        'model': 'resnet50_finetuned',
        'test_accuracy': acc,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'avg_conf_correct': avg_conf_correct,
        'avg_conf_incorrect': avg_conf_incorrect
    }
    with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    # 4. Classification Report
    print("\n" + "="*40)
    print("CLASSIFICATION REPORT")
    print("="*40)
    # Strip the long "Tomato___" prefix for a cleaner display
    short_classes = [c.replace('Tomato___', '') for c in class_names]
    report = classification_report(all_targets, all_preds, target_names=short_classes)
    print(report)

    # 5. Confusion Matrix Saving
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_classes,
                yticklabels=short_classes)
    plt.title('Confusion Matrix - Strictly Untouched Test Set')
    plt.ylabel('True Disease Class')
    plt.xlabel('Predicted Disease Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'confusion_matrix.png'))
    plt.close()

    print(f"\nSaved confusion matrix and metrics to {checkpoint_dir}")

if __name__ == '__main__':
    main()
