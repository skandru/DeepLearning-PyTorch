"""
Test dataset setup utility for medical image classification.

Downloads and organizes sample chest X-ray images into train/test splits
for demonstration purposes. Uses a subset of the COVID-19 chest X-ray dataset.

Created: 11-03-2024
Author: Suresh Kandru <hello@sureshkandru.com>
Version: 1.0.0

Changes:
--------
11-03-2024: Initial version - Basic dataset setup functionality
"""


import os
import requests
from pathlib import Path
import zipfile
import shutil

def setup_test_dataset():
    """Creates and populates a test dataset structure with medical images.
   
    Downloads chest X-ray images and organizes them into the following structure:
    medical_images_test/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
    
    Returns:
        Path: Base directory path containing the organized dataset
        
    Raises:
        Exception: If image download or directory creation fails
    """
    # Create base directories
    base_dir = Path("medical_images_test")
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    
    # Create subdirectories for classes
    classes = ['NORMAL', 'PNEUMONIA']
    for split_dir in [train_dir, test_dir]:
        for class_name in classes:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Download sample images
    urls = {
        'NORMAL': [
            'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-001.jpg',
            'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/1-s2.0-S1684118220300608-main.pdf-002.jpg'
        ],
        'PNEUMONIA': [
            'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/covid-19-caso-70-1-PA.jpg',
            'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/covid-19-pneumonia-15-PA.jpg'
        ]
    }
    
    for class_name, image_urls in urls.items():
        for i, url in enumerate(image_urls):
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Save to both train and test (for demonstration)
                for split_dir in [train_dir, test_dir]:
                    save_path = split_dir / class_name / f"image_{i}.jpg"
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                print(f"Downloaded {class_name} image {i}")
                
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
    
    return base_dir