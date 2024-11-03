"""
Test script for image processing pipeline.

Tests the complete workflow of dataset setup, loading, augmentation,
and visualization using sample medical images.

Created: 11-03-2024
Author: Suresh Kandru <hello@sureshkandru.com>
Version: 1.0.0

Changes:
--------
11-03-2024: Initial version - Added pipeline testing functionality
"""

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
import logging
import shutil
import sys
# Add parent directory to system path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.setup_test_dataset import setup_test_dataset
from src.data.image_data_manager import ImageDataManager
from src.data.augmentation_manager import AugmentationManager
from src.utils.image_visualizer  import ImageVisualizer




def test_image_pipeline():
    """Tests the complete image processing pipeline.
   
    Executes the following steps:
    1. Sets up test dataset with medical images
    2. Initializes data loading pipeline
    3. Tests basic image loading and visualization
    4. Tests data augmentation
    5. Cleans up test data
    
    Raises:
        Exception: If any step in the pipeline fails
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup test dataset
    data_dir = setup_test_dataset()
    
    # Initialize data manager
    data_manager = ImageDataManager(
        data_dir=str(data_dir),
        image_size=(128, 128),
        batch_size=2  # Small batch size for testing
    )
    
    try:
        # Create dataloaders
        train_loader, test_loader = data_manager.create_dataloaders()
        
        # Initialize visualizer
        visualizer = ImageVisualizer()
        
        # Display sample images
        print("Displaying sample images from training set:")
        visualizer.show_batch(train_loader, num_images=2)
        
        # Test data augmentation
        print("\nDisplaying augmented images:")
        aug_manager = AugmentationManager(
            image_size=(128, 128),
            augmentation_strength='medium'
        )
        
        # Create dataset with augmentation
        aug_dataset = datasets.ImageFolder(
            str(data_dir / 'train'),
            transform=aug_manager.get_transforms(train=True)
        )
        
        aug_loader = DataLoader(
            aug_dataset,
            batch_size=2,
            shuffle=True
        )
        
        # Show augmented images
        visualizer.show_batch(aug_loader, num_images=2)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        raise
    finally:
        # Cleanup (optional)
        logging.info(f"Cleanup Data Directory")
        """if data_dir.exists():
            shutil.rmtree(data_dir)"""

if __name__ == "__main__":
    test_image_pipeline()