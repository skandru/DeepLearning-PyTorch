
"""
Utility class for visualizing batches of images from data loaders.

Created: 11-03-2024
Author: Suresh Kandru <hello@sureshkandru.com>
Version: 1.0.0

Changes:
--------
11-03-2024: Initial version - Added basic image visualization
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

class ImageVisualizer:
    """Provides utilities for visualizing images from data loaders with proper denormalization."""

    @staticmethod
    def denormalize(tensor):
        """Denormalizes image tensors using ImageNet statistics.
       
        Args:
            tensor (torch.Tensor): Normalized image tensor
            
        Returns:
            torch.Tensor: Denormalized image tensor
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    @staticmethod
    def show_batch(dataloader, num_images=5):
        """Displays a batch of images from the dataloader.
       
        Args:
            dataloader (DataLoader): PyTorch dataloader containing images
            num_images (int): Number of images to display. Default: 5
        """
        images, labels = next(iter(dataloader))
        images = images[:num_images]
        
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        
        for i, (img, ax) in enumerate(zip(images, axes)):
            # Convert to displayable format
            img = ImageVisualizer.denormalize(img)
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Class: {labels[i].item()}')
        
        plt.tight_layout()
        plt.show()