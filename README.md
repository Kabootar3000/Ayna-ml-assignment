# UNet Polygon Coloring - ML Assignment

This project implements a conditional UNet model for generating colored polygon images from grayscale polygon inputs and color specifications.

## Problem Statement

The task is to train a UNet model that takes two inputs:
1. An image of a polygon (e.g., triangle, square, octagon)
2. The name of a color (e.g., "blue", "red", "yellow")

The model outputs an image of the input polygon filled with the specified color.


## Project Structure

```
├── model.py                # UNet model implementation
├── dataset.py              # Custom dataset classes
├── train.py                # Training script with wandb integration
├── quick_train.py          # Fast training script for quick experiments
├── test_model.py           # Script for model evaluation/testing
├── inference.ipynb         # Jupyter notebook for inference
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── sample_visualization.png# Example output visualization
├── checkpoints/            # Saved model checkpoints (.pth)
├── samples/                # Sample outputs generated during training
├── dataset/                # Dataset directory
│   ├── training/
│   │   ├── inputs/         # Grayscale polygon images
│   │   ├── outputs/        # Colored polygon images
│   │   └── data.json       # Training data mapping
│   └── validation/
│       ├── inputs/         # Validation polygon images
│       ├── outputs/        # Validation colored images
│       └── data.json       # Validation data mapping
└── wandb/                  # Weights & Biases experiment logs
```

## Additional Files & Directories

- **quick_train.py**: Script for rapid prototyping or debugging with fewer epochs or a smaller dataset.
- **test_model.py**: Script to evaluate the trained model on the validation set or custom images.
- **samples/**: Contains sample outputs generated at different epochs during training for qualitative assessment.
- **sample_visualization.png**: Example visualization of model output.
- **checkpoints/**: Stores model checkpoint files (e.g., `best_model.pth`).


## Setup Instructions

1. **Python Version:**
   - Recommended: Python 3.8+
   - OS: Windows, Linux, or macOS

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Wandb (Optional but Recommended):**
   - Login using the terminal (recommended, keeps your API key safe):
     ```bash
     wandb login
     ```
   - **Do not** put your API key in your code or share it publicly.

4. **Train the Model:**
   ```bash
   python train.py --epochs 100 --batch_size 8 --lr 1e-4
   ```

5. **Run Inference:**
## Visualization

You can view a sample output in `sample_visualization.png`. The `samples/` directory contains outputs generated at various epochs to help track model progress.

   ```bash
   jupyter notebook inference.ipynb
   ```

## Model Architecture

### Conditional UNet Design

- **Encoder Path:** Standard UNet encoder with skip connections
- **Color Conditioning:** Color names are embedded and projected to spatial features
- **Decoder Path:** UNet decoder with color conditioning at multiple scales
- **Output:** 3-channel RGB image (128x128)

### Key Components

1. **ColorEmbedding:** Embeds color names into 64-dimensional vectors
2. **ConditionalUNet:** Main model with color conditioning at bottleneck and decoder layers
3. **CombinedLoss:** L1 + SSIM-like perceptual loss for better image quality

## Training Details

- **Learning Rate:** 1e-4 (AdamW optimizer)
- **Batch Size:** 8
- **Image Size:** 128x128
- **Epochs:** 100
- **Loss Function:** Combined L1 + SSIM loss (α=0.7)
- **Data Augmentation:** Random rotation, crop, and horizontal flip

## Security & Collaboration Best Practices

- **Never share your `.env` file or API keys.**
- Use a `.gitignore` file to exclude sensitive files and large artifacts:
  ```
  # .gitignore
  __pycache__/
  *.pyc
  *.pyo
  *.pyd
  *.log
  .env
  wandb/
  *.pth
  *.ckpt
  *.pt
  checkpoints/
  *.h5
  .ipynb_checkpoints/
  .DS_Store
  Thumbs.db
  .vscode/
  data/
  dataset/
  *.zip
  *.tar
  *.tar.gz
  *.npz
  *.npy
  ```
- If using environment variables (like `WANDB_API_KEY`), keep them in a `.env` file and **never commit it**.

## Usage Examples

### Training
```bash
python train.py --epochs 100 --batch_size 8 --lr 1e-4 --wandb_project my-polygon-unet
```

### Inference
```python
# Load model
model = create_model(n_channels=1, n_classes=3, num_colors=8)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
input_image = preprocess_image('triangle.png')
prediction = predict_colorized_polygon(model, input_image, 'red', device)
```

## Deliverables

- ✅ UNet model implementation (`model.py`)
- ✅ Training script with wandb integration (`train.py`)
- ✅ Custom dataset classes (`dataset.py`)
- ✅ Inference notebook (`inference.ipynb`)
- ✅ Comprehensive README with insights
- 🔄 Wandb project link (to be shared after training)

**Note:**  
This implementation focuses on a clean, well-documented solution that demonstrates understanding of conditional image generation, UNet architecture, and proper ML practices including experiment tracking.