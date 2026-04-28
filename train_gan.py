import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from generator import Generator
from discriminator import Discriminator


class TomatoDiseaseDataset(Dataset):
    """
    Custom dataset for tomato leaf disease images from PlantVillage dataset.
    Filters for tomato leaf disease classes only.
    """
    
    def __init__(self, root_dir, transform=None, disease_classes=['Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Septoria_leaf_spot']):
        """
        Args:
            root_dir: Path to PlantVillage dataset root or train/val split folder
            transform: Image transformations to apply
            disease_classes: List of tomato disease class names to include
        """
        self.root_dir = root_dir
        self.transform = transform
        self.disease_classes = disease_classes
        self.images = []
        self.labels = []
        
        # Handle nested PlantVillage folder structure
        search_dir = root_dir
        
        # Check for nested PlantVillage structure
        if os.path.exists(os.path.join(root_dir, 'PlantVillage')):
            search_dir = os.path.join(root_dir, 'PlantVillage')
        
        # Check for train/val split
        if os.path.exists(os.path.join(search_dir, 'PlantVillage')):
            search_dir = os.path.join(search_dir, 'PlantVillage', 'train')
        elif os.path.exists(os.path.join(search_dir, 'train')):
            search_dir = os.path.join(search_dir, 'train')
        
        # Collect image paths from disease classes
        for class_name in disease_classes:
            class_dir = os.path.join(search_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):
                        try:
                            # Verify image is readable
                            Image.open(img_path).verify()
                            self.images.append(img_path)
                            self.labels.append(class_name)
                        except (IOError, OSError, ValueError):
                            # Silently skip corrupted/invalid images
                            continue
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def train_dcgan(args):
    """
    Train the DCGAN on tomato leaf disease images.
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Dataset and DataLoader
    print(f"Loading dataset from: {args.data_dir}")
    dataset = TomatoDiseaseDataset(args.data_dir, transform=transform)
    
    if len(dataset) == 0:
        print(f"No images found in {args.data_dir}")
        print("Please ensure PlantVillage dataset is in the specified directory.")
        return
    
    print(f"Dataset size: {len(dataset)} images")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Model initialization
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    fixed_noise = torch.randn(16, args.latent_dim, device=device)
    
    g_losses = []
    d_losses = []
    
    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Labels for real and fake images
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # ============ Train Discriminator ============
            optimizer_d.zero_grad()
            
            # Real images
            d_real_output = discriminator(real_images)
            d_real_loss = criterion(d_real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise)
            d_fake_output = discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_d.step()
            
            # ============ Train Generator ============
            optimizer_g.zero_grad()
            
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise)
            d_fake_output = discriminator(fake_images)
            
            # Generator wants to fool discriminator (fake_labels = 1 for generator training)
            g_loss = criterion(d_fake_output, real_labels)
            g_loss.backward()
            optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"G_loss: {g_loss.item():.4f} D_loss: {d_loss.item():.4f}")
        
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Avg G_loss: {avg_g_loss:.4f} Avg D_loss: {avg_d_loss:.4f}")
        
        # Save generated images
        if (epoch + 1) % args.save_interval == 0:
            with torch.no_grad():
                generated = generator(fixed_noise)
            save_generated_images(generated, epoch + 1, args.output_dir)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
            }
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final models
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'generator_final.pt'))
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'discriminator_final.pt'))
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'training_losses.png'))
    print(f"Training losses plot saved to {os.path.join(args.output_dir, 'training_losses.png')}")
    
    print("Training complete!")


def save_generated_images(images, epoch, output_dir):
    """
    Save generated images as a grid.
    
    Args:
        images: Tensor of shape (batch_size, 3, 128, 128)
        epoch: Current epoch number
        output_dir: Directory to save images
    """
    # Denormalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Convert to numpy
    images_np = images.cpu().numpy()
    
    # Create grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        if idx < images_np.shape[0]:
            img = np.transpose(images_np[idx], (1, 2, 0))
            ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'generated_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Generated images saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DCGAN on tomato leaf disease images')
    parser.add_argument('--data_dir', type=str, default='./data/PlantVillage',
                        help='Path to PlantVillage dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate for optimizers')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Dimension of the latent noise vector')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size (128x128)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Interval for logging training progress')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for saving generated images')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    train_dcgan(args)
