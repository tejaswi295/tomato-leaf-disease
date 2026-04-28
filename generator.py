import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator for 128x128 tomato leaf disease images.
    Takes random noise as input and generates realistic tomato leaf images.
    """
    
    def __init__(self, latent_dim=100, num_channels=3):
        """
        Args:
            latent_dim: Dimension of the input noise vector
            num_channels: Number of output image channels (3 for RGB)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        
        # Initial dense layer: (batch_size, latent_dim) -> (batch_size, 512*8*8)
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        # Transposed convolution layers for upsampling
        # 8x8 -> 16x16
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 16x16 -> 32x32
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 32x32 -> 64x64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 64x64 -> 128x128
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final layer to generate RGB image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, num_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        """
        Forward pass through the generator.
        
        Args:
            z: Noise tensor of shape (batch_size, latent_dim)
            
        Returns:
            Generated image tensor of shape (batch_size, num_channels, 128, 128)
        """
        # Dense layer + reshape -> (batch_size, 512, 8, 8)
        x = self.fc(z)
        x = x.view(x.size(0), 512, 8, 8)
        
        # Deconvolution layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        
        # Final output
        x = self.final(x)
        
        return x
