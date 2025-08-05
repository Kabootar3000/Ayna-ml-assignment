import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)  # Add SEBlock
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ColorEmbedding(nn.Module):
    """Embed color names into a vector representation"""
    
    def __init__(self, num_colors, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_colors, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, color_indices):
        embedded = self.embedding(color_indices)
        return self.projection(embedded)


class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=3, bilinear=False, num_colors=10):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Color embedding
        self.color_embedding = ColorEmbedding(num_colors, embedding_dim=64)
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        # Color conditioning layers
        self.color_projection = nn.Linear(64, 1024)
        self.color_conv1 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.color_conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.color_conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.color_conv4 = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x, color_indices):
        # Encode color information
        color_emb = self.color_embedding(color_indices)
        color_proj = self.color_projection(color_emb)
        
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply color conditioning at bottleneck
        batch_size = x5.size(0)
        color_spatial = color_proj.view(batch_size, -1, 1, 1)
        x5 = x5 + self.color_conv1(color_spatial)
        
        # Decoder path with color conditioning
        x = self.up1(x5, x4)
        
        x = self.up2(x, x3)
        
        x = self.up3(x, x2)
        
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits


def create_model(n_channels=1, n_classes=3, num_colors=10):
    """Create and return the UNet model"""
    return ConditionalUNet(n_channels=n_channels, n_classes=n_classes, num_colors=num_colors) 