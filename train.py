import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision import transforms as T

from model import create_model
from dataset import get_data_loaders


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights='IMAGENET1K_V1').features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def forward(self, pred, target):
        # pred/target: (B, 3, H, W), values in [0,1]
        pred = self.transform(pred)
        target = self.transform(target)
        return F.l1_loss(self.vgg(pred), self.vgg(target))

class CombinedLoss(nn.Module):
    """Combined loss function: L1 + SSIM-like perceptual loss"""
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        
    def forward(self, pred, target):
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # SSIM-like loss (simplified)
        # Calculate structural similarity
        mu_pred = torch.mean(pred, dim=[2, 3], keepdim=True)
        mu_target = torch.mean(target, dim=[2, 3], keepdim=True)
        
        sigma_pred = torch.var(pred, dim=[2, 3], keepdim=True)
        sigma_target = torch.var(target, dim=[2, 3], keepdim=True)
        
        sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target), dim=[2, 3], keepdim=True)
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred ** 2 + sigma_target ** 2 + c2))
        
        ssim_loss = 1 - torch.mean(ssim)
        perceptual = self.perceptual_loss(pred, target)
        total_loss = self.alpha * l1 + self.beta * ssim_loss + self.gamma * perceptual
        
        return total_loss, l1, ssim_loss, perceptual


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_l1 = 0
    total_ssim = 0
    total_perceptual = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        inputs = batch['input'].to(device)
        targets = batch['output'].to(device)
        color_indices = batch['color_idx'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs, color_indices)
        loss, l1_loss, ssim_loss, perceptual = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_ssim += ssim_loss.item()
        total_perceptual += perceptual.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'L1': f'{l1_loss.item():.4f}',
            'SSIM': f'{ssim_loss.item():.4f}',
            'Perceptual': f'{perceptual.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_l1 = total_l1 / len(train_loader)
    avg_ssim = total_ssim / len(train_loader)
    avg_perceptual = total_perceptual / len(train_loader)
    
    return avg_loss, avg_l1, avg_ssim, avg_perceptual


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_l1 = 0
    total_ssim = 0
    total_perceptual = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            color_indices = batch['color_idx'].to(device)
            
            outputs = model(inputs, color_indices)
            loss, l1_loss, ssim_loss, perceptual = criterion(outputs, targets)
            
            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_ssim += ssim_loss.item()
            total_perceptual += perceptual.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_l1 = total_l1 / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    avg_perceptual = total_perceptual / len(val_loader)
    
    return avg_loss, avg_l1, avg_ssim, avg_perceptual


def save_sample_images(model, val_loader, device, epoch, save_dir='samples'):
    """Save sample images for visualization"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        inputs = batch['input'].to(device)
        targets = batch['output'].to(device)
        color_indices = batch['color_idx'].to(device)
        colors = batch['color']
        
        outputs = model(inputs, color_indices)
        
        # Save first 4 samples
        for i in range(min(4, len(inputs))):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Input (grayscale)
            input_img = inputs[i].cpu().squeeze().numpy()
            axes[0].imshow(input_img, cmap='gray')
            axes[0].set_title(f'Input ({colors[i]})')
            axes[0].axis('off')
            
            # Target
            target_img = targets[i].cpu().permute(1, 2, 0).numpy()
            target_img = np.clip(target_img, 0, 1)
            axes[1].imshow(target_img)
            axes[1].set_title('Target')
            axes[1].axis('off')
            
            # Prediction
            pred_img = outputs[i].cpu().permute(1, 2, 0).numpy()
            pred_img = np.clip(pred_img, 0, 1)
            axes[2].imshow(pred_img)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_sample_{i}.png'))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train UNet for polygon coloring')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--wandb_project', type=str, default='polygon-coloring-unet', help='Wandb project name')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'image_size': args.image_size,
            'device': str(device),
        }
    )
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Create model
    model = create_model(n_channels=1, n_classes=3, num_colors=8)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_l1 = 0
        total_ssim = 0
        total_perceptual = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            color_indices = batch['color_idx'].to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, color_indices)
                    loss, l1_loss, ssim_loss, perceptual = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs, color_indices)
                loss, l1_loss, ssim_loss, perceptual = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_ssim += ssim_loss.item()
            total_perceptual += perceptual.item()
        
        # Validate
        val_loss, val_l1, val_ssim, val_perceptual = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': total_loss / len(train_loader),
            'train_l1': total_l1 / len(train_loader),
            'train_ssim': total_ssim / len(train_loader),
            'train_perceptual': total_perceptual / len(train_loader),
            'val_loss': val_loss,
            'val_l1': val_l1,
            'val_ssim': val_ssim,
            'val_perceptual': val_perceptual,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Save sample images every 10 epochs
        if epoch % 10 == 0:
            save_sample_images(model, val_loader, device, epoch)
        
        print(f'Epoch {epoch}: Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    wandb.finish()


if __name__ == '__main__':
    main()