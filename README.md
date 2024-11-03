# Image Processing Pipeline with PyTorch
A modular pipeline for handling image datasets, with features for loading, augmentation, and visualization using PyTorch.

## Project Structure
```
project/
├── src/
│   ├── data/
│   │   ├── augmentation_manager.py   # Image augmentation utilities
│   │   ├── image_data_manager.py     # Data loading pipeline
│   │   └── setup_test_dataset.py     # Test dataset creation
│   └── utils/
│       ├── logger.py                 # Logging utilities
│       └── image_visualizer.py       # Image visualization tools
└── tests/
    └── test_image_pipeline.py        # Pipeline tests
```

## Features
- **Data Loading**: Efficient loading of image datasets using PyTorch DataLoader
- **Image Augmentation**: Configurable augmentation pipeline with multiple strength levels
- **Visualization**: Tools for displaying and inspecting image batches
- **Test Dataset**: Automated setup of medical image test data

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.4.0
```

## Quick Start

### Setup Environment
```bash
pip install torch torchvision numpy matplotlib
```

### Basic Usage
```python
from src.data.image_data_manager import ImageDataManager
from src.data.augmentation_manager import AugmentationManager

# Initialize data manager
data_manager = ImageDataManager(
    data_dir="path/to/images",
    image_size=(128, 128),
    batch_size=32
)

# Get data loaders
train_loader, test_loader = data_manager.create_dataloaders()

# Setup augmentation
aug_manager = AugmentationManager(
    image_size=(128, 128),
    augmentation_strength='medium'
)
```

## Components

### ImageDataManager
Handles data loading and preprocessing:
- Configurable batch size and image size
- Multi-worker data loading
- Automatic normalization using ImageNet statistics

### AugmentationManager
Provides image augmentation functionality:
- Multiple augmentation strength levels (light/medium/strong)
- Common augmentations (rotation, flips, color jitter)
- Configurable parameters

### ImageVisualizer
Tools for visualizing loaded images:
- Batch visualization
- Automatic denormalization
- Class label display

## Testing
Run the test pipeline:
```bash
python tests/test_image_pipeline.py
```

The test script:
- Creates a sample medical image dataset
- Tests data loading
- Demonstrates augmentation
- Shows visualization capabilities

## Directory Structure Requirements
Your image dataset should be organized as:
```
data_dir/
├── train/
│   ├── class1/
│   │   └── images...
│   └── class2/
│       └── images...
└── test/
    ├── class1/
    │   └── images...
    └── class2/
        └── images...
```