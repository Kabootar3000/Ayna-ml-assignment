#!/usr/bin/env python3
"""
Quick training script for demonstration purposes.
This script trains the model for a few epochs to show the training process.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import create_model
from dataset import get_data_loaders
from train import CombinedLoss, train_epoch, validate_epoch, save_sample_images


def main():
    """Quick training demonstration"""
    print("Starting quick training demonstration...")
    
    # Configuration
    config = {
        'batch_size': 4,
        'epochs': 5,  # Quick training
        'lr': 1e-4,
        'image_size': 128,
        'device': 'auto'
    }
    
    # Device setup
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    
    print(f"Using device: {device}")
    
    # Initialize wandb (optional)
    try:
        wandb.init(
            project="polygon-coloring-demo",
            config=config,
            mode="disabled"  # Disable wandb for demo
        )
        print("Wandb initialized (disabled mode)")
    except:
        print("Wandb not available, continuing without it")
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader = get_data_loaders(
        data_dir='dataset',
        batch_size=config['batch_size'],
        image_size=config['image_size']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(n_channels=1, n_classes=3, num_colors=8)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    
    # Create save directory
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("=" * 50)
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        # Train
        train_loss, train_l1, train_ssim, train_perceptual = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_l1, val_ssim, val_perceptual = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Log metrics
        if wandb.run is not None:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_l1': train_l1,
                'train_ssim': train_ssim,
                'train_perceptual': train_perceptual,
                'val_loss': val_loss,
                'val_l1': val_l1,
                'val_ssim': val_ssim,
                'val_perceptual': val_perceptual,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train L1: {train_l1:.4f}, Val L1: {val_l1:.4f}")
        print(f"Train SSIM: {train_ssim:.4f}, Val SSIM: {val_ssim:.4f}")
        print(f"Train Perceptual: {train_perceptual:.4f}, Val Perceptual: {val_perceptual:.4f}")
        
        # Save sample images every epoch
        save_sample_images(model, val_loader, device, epoch, 'samples')
    
    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, 'checkpoints/demo_model.pth')
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Final validation loss: {val_loss:.4f}")
    print("Model saved to 'checkpoints/demo_model.pth'")
    print("Sample images saved to 'samples/' directory")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()