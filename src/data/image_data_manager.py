"""
Image data manager for handling training and test datasets.

Created: 11-03-2024
Author: Suresh Kandru <hello@sureshkandru.com>
Version: 1.0.0

Changes:
--------
11-03-2024: Initial version - Added basic data loading functionality
"""
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
import logging

class ImageDataManager:
    """Manages image data loading and preprocessing for training and testing.
   
    Args:
        data_dir (str): Root directory containing train and test folders
        image_size (tuple): Target image dimensions (height, width). Default: (128, 128)
        batch_size (int): Number of samples per batch. Default: 32
        num_workers (int): Number of subprocesses for data loading. Default: 4
    """
    def __init__(
        self,
        data_dir: str,
        image_size: tuple = (128, 128),
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Basic transformations
        self.base_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def create_dataloaders(self):
        """Creates train and test data loaders.

        Returns:
            tuple: (train_loader, test_loader) pair of DataLoader objects
            
        Raises:
            Exception: If data loading fails or paths are invalid
        """
        try:
            # Create datasets
            train_dataset = datasets.ImageFolder(
                str(self.data_dir / 'train'),
                transform=self.base_transforms
            )
            
            test_dataset = datasets.ImageFolder(
                str(self.data_dir / 'test'),
                transform=self.base_transforms
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            logging.info(f"Created dataloaders with {len(train_dataset)} training images "
                        f"and {len(test_dataset)} test images")
            
            return train_loader, test_loader
            
        except Exception as e:
            logging.error(f"Error creating dataloaders: {str(e)}")
            raise