#!/usr/bin/env python3
"""
Test script to verify the UNet model implementation and dataset loading.
"""

import torch
import numpy as np
from model import create_model
from dataset import get_data_loaders
import matplotlib.pyplot as plt


def test_model_forward_pass():
    """Test that the model can perform forward pass with dummy data"""
    print("Testing model forward pass...")
    
    # Create model
    model = create_model(n_channels=1, n_classes=3, num_colors=8)
    
    # Create dummy input
    batch_size = 2
    input_images = torch.randn(batch_size, 1, 128, 128)
    color_indices = torch.randint(0, 8, (batch_size,))
    
    # Forward pass
    outputs = model(input_images, color_indices)
    
    print(f"Input shape: {input_images.shape}")
    print(f"Color indices: {color_indices}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    
    assert outputs.shape == (batch_size, 3, 128, 128), f"Expected shape (2, 3, 128, 128), got {outputs.shape}"
    print("✅ Model forward pass test passed!")


def test_dataset_loading():
    """Test that the dataset can be loaded correctly"""
    print("\nTesting dataset loading...")
    
    try:
        # Load data loaders
        train_loader, val_loader = get_data_loaders(
            data_dir='dataset',
            batch_size=4,
            image_size=128
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Test a batch
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {batch['input'].shape}")
        print(f"Output shape: {batch['output'].shape}")
        print(f"Color indices shape: {batch['color_idx'].shape}")
        print(f"Sample colors: {batch['color']}")
        
        print("✅ Dataset loading test passed!")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    return True


def test_end_to_end():
    """Test end-to-end model with real data"""
    print("\nTesting end-to-end with real data...")
    
    try:
        # Load data
        train_loader, val_loader = get_data_loaders(
            data_dir='dataset',
            batch_size=2,
            image_size=128
        )
        
        # Create model
        model = create_model(n_channels=1, n_classes=3, num_colors=8)
        
        # Get a batch
        batch = next(iter(train_loader))
        inputs = batch['input']
        targets = batch['output']
        color_indices = batch['color_idx']
        
        # Forward pass
        outputs = model(inputs, color_indices)
        
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Color indices: {color_indices}")
        
        # Check output range
        print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
        print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
        
        print("✅ End-to-end test passed!")
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False
    
    return True


def visualize_sample():
    """Visualize a sample from the dataset"""
    print("\nVisualizing sample data...")
    
    try:
        # Load data
        train_loader, val_loader = get_data_loaders(
            data_dir='dataset',
            batch_size=1,
            image_size=128
        )
        
        # Get a sample
        batch = next(iter(train_loader))
        input_img = batch['input'][0].squeeze().numpy()
        output_img = batch['output'][0].permute(1, 2, 0).numpy()
        color = batch['color'][0]
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title(f'Input Polygon')
        axes[0].axis('off')
        
        axes[1].imshow(output_img)
        axes[1].set_title(f'Output ({color})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Sample visualization saved to 'sample_visualization.png'")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")


def main():
    """Run all tests"""
    print("Running UNet Polygon Coloring Tests")
    print("=" * 40)
    
    # Run tests
    test_model_forward_pass()
    dataset_ok = test_dataset_loading()
    end_to_end_ok = test_end_to_end()
    
    if dataset_ok:
        visualize_sample()
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    print("✅ Model forward pass: PASSED")
    print(f"{'✅' if dataset_ok else '❌'} Dataset loading: {'PASSED' if dataset_ok else 'FAILED'}")
    print(f"{'✅' if end_to_end_ok else '❌'} End-to-end test: {'PASSED' if end_to_end_ok else 'FAILED'}")
    
    if dataset_ok and end_to_end_ok:
        print("\n🎉 All tests passed! The implementation is ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 