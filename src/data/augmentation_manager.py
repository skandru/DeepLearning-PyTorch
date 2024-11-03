"""
augmentation_manager.py
Created on: 11-03-2024
Author: Suresh Kandru (hello@sureshkandru.com)

This module provides functionality for managing image augmentation transformations
in the image processing pipeline.
"""

import torch
from torchvision import transforms
from typing import Dict, Any, Optional

class AugmentationManager:
    """
    Manages image augmentation transformations with configurable strength levels.
    
    Attributes:
        image_size (tuple): Target size for images (height, width)
        strength (str): Augmentation strength ('light', 'medium', 'strong')
    """
    def __init__(
        self,
        image_size: tuple = (128, 128),
        augmentation_strength: str = 'medium'
    ):
        """
        Initialize the AugmentationManager.
        
        Args:
            image_size (tuple): Target size for images
            augmentation_strength (str): Strength of augmentations
        """
        self.image_size = image_size
        self.strength = augmentation_strength
        
    def get_transforms(self, train: bool = True):
        """Get transforms based on training/testing phase"""
        if not train:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        augmentation_params = {
            'light': {
                'rotate_degrees': 15,
                'brightness': 0.1,
                'contrast': 0.1,
                'saturation': 0.1,
                'hue': 0.05
            },
            'medium': {
                'rotate_degrees': 30,
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            },
            'strong': {
                'rotate_degrees': 45,
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.3,
                'hue': 0.15
            }
        }
        
        params = augmentation_params[self.strength]
        
        return transforms.Compose([
            # Geometric transformations
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.8, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(params['rotate_degrees']),
            
            # Color transformations
            transforms.ColorJitter(
                brightness=params['brightness'],
                contrast=params['contrast'],
                saturation=params['saturation'],
                hue=params['hue']
            ),
            
            # Optional advanced augmentations
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])