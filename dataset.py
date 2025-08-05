import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import random
from PIL import ImageFilter


def elastic_transform(image, alpha=34, sigma=4):
    # Simple elastic transform using PIL (for demonstration)
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))


class PolygonColorDataset(Dataset):
    def __init__(self, data_dir, split='training', transform=None, image_size=128):
        """
        Args:
            data_dir: Path to dataset directory
            split: 'training' or 'validation'
            transform: Optional transform to be applied
            image_size: Size to resize images to
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        
        # Load data.json
        json_path = os.path.join(data_dir, split, 'data.json')
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Define color mapping
        self.colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'magenta', 'cyan']
        self.color_to_idx = {color: idx for idx, color in enumerate(self.colors)}
        
        # Define transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input polygon image
        input_path = os.path.join(self.data_dir, self.split, 'inputs', item['input_polygon'])
        input_image = Image.open(input_path).convert('L')  # Convert to grayscale
        
        # Load output colored image
        output_path = os.path.join(self.data_dir, self.split, 'outputs', item['output_image'])
        output_image = Image.open(output_path).convert('RGB')
        
        # Get color index
        color = item['colour']
        color_idx = self.color_to_idx[color]
        
        # Apply transforms
        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)
        
        return {
            'input': input_image,
            'output': output_image,
            'color': color,
            'color_idx': torch.tensor(color_idx, dtype=torch.long)
        }


class PolygonColorDatasetWithAugmentation(Dataset):
    def __init__(self, data_dir, split='training', image_size=128, augment=True):
        """
        Enhanced dataset with augmentation for training
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == 'training'
        
        # Load data.json
        json_path = os.path.join(data_dir, split, 'data.json')
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Define color mapping
        self.colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'magenta', 'cyan']
        self.color_to_idx = {color: idx for idx, color in enumerate(self.colors)}
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(elastic_transform),  # FIXED: use function, not lambda
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input polygon image
        input_path = os.path.join(self.data_dir, self.split, 'inputs', item['input_polygon'])
        input_image = Image.open(input_path).convert('L')
        
        # Load output colored image
        output_path = os.path.join(self.data_dir, self.split, 'outputs', item['output_image'])
        output_image = Image.open(output_path).convert('RGB')
        
        # Apply augmentation if training
        if self.augment:
            # Apply same augmentation to both input and output
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            input_image = self.augment_transform(input_image)
            
            torch.manual_seed(seed)
            output_image = self.augment_transform(output_image)
        
        # Apply final transforms
        input_image = self.transform(input_image)
        output_image = self.transform(output_image)
        
        # Add random noise after ToTensor
        if self.augment:
            input_image = input_image + 0.05 * torch.randn_like(input_image)
            input_image = torch.clamp(input_image, 0, 1)
        
        # Get color index
        color = item['colour']
        color_idx = self.color_to_idx[color]
        
        return {
            'input': input_image,
            'output': output_image,
            'color': color,
            'color_idx': torch.tensor(color_idx, dtype=torch.long)
        }


def get_data_loaders(data_dir, batch_size=8, image_size=128, num_workers=2):
    """
    Create training and validation data loaders
    """
    # Training dataset with augmentation
    train_dataset = PolygonColorDatasetWithAugmentation(
        data_dir=data_dir,
        split='training',
        image_size=image_size,
        augment=True
    )
    
    # Validation dataset without augmentation
    val_dataset = PolygonColorDatasetWithAugmentation(
        data_dir=data_dir,
        split='validation',
        image_size=image_size,
        augment=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader