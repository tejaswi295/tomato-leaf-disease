import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for 128x128 tomato leaf disease images.
    Classifies real vs. generated images.
    """
    
    def __init__(self, num_channels=3):
        """
        Args:
            num_channels: Number of input image channels (3 for RGB)
        """
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        
        # Convolutional layers for downsampling
        # 128x128 -> 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 64x64 -> 32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 32x32 -> 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 16x16 -> 8x8
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 8x8 -> 4x4
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final dense layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, img):
        """
        Forward pass through the discriminator.
        
        Args:
            img: Image tensor of shape (batch_size, num_channels, 128, 128)
            
        Returns:
            Classification probability of shape (batch_size, 1)
        """
        # Convolutional layers
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten and dense layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x
