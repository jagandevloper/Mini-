"""
Advanced Data Augmentation for Kidney Stone Detection
====================================================

This module provides specialized augmentation techniques optimized for medical
X-ray images, particularly KUB (Kidney-Ureter-Bladder) X-ray images for
kidney stone detection. The augmentations are designed to improve model
robustness while maintaining medical accuracy.

Key Features:
- Medical image-specific augmentations
- YOLOv8 compatible transformations
- Anatomical structure preservation
- Realistic augmentation parameters
- Batch processing support

Author: [Your Name]
Date: 2024
License: MIT
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from typing import Dict, List, Tuple, Optional, Union
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageAugmentation:
    """
    Medical Image Augmentation Pipeline
    
    This class provides augmentation techniques specifically designed for
    medical X-ray images, ensuring that augmentations are realistic and
    preserve anatomical structures while improving model generalization.
    """
    
    def __init__(self, 
                 image_size: int = 640,
                 augmentation_strength: str = 'medium',
                 preserve_anatomy: bool = True):
        """
        Initialize medical image augmentation pipeline.
        
        Args:
            image_size: Target image size for augmentation
            augmentation_strength: Strength level ('light', 'medium', 'strong')
            preserve_anatomy: Whether to preserve anatomical structures
        """
        self.image_size = image_size
        self.augmentation_strength = augmentation_strength
        self.preserve_anatomy = preserve_anatomy
        
        # Define augmentation parameters based on strength
        self._set_augmentation_parameters()
        
        logger.info(f"MedicalImageAugmentation initialized with strength: {augmentation_strength}")
    
    def _set_augmentation_parameters(self):
        """Set augmentation parameters based on strength level."""
        
        if self.augmentation_strength == 'light':
            # Light augmentation - minimal changes for medical accuracy
            self.params = {
                'brightness_limit': 0.1,
                'contrast_limit': 0.1,
                'hue_shift_limit': 5,
                'sat_shift_limit': 10,
                'val_shift_limit': 10,
                'rotation_limit': 5,
                'scale_limit': 0.05,
                'noise_var_limit': (5.0, 15.0),
                'blur_limit': 2,
                'elastic_alpha': 50,
                'elastic_sigma': 5
            }
        elif self.augmentation_strength == 'medium':
            # Medium augmentation - balanced approach
            self.params = {
                'brightness_limit': 0.2,
                'contrast_limit': 0.2,
                'hue_shift_limit': 10,
                'sat_shift_limit': 20,
                'val_shift_limit': 20,
                'rotation_limit': 10,
                'scale_limit': 0.1,
                'noise_var_limit': (10.0, 30.0),
                'blur_limit': 3,
                'elastic_alpha': 100,
                'elastic_sigma': 10
            }
        else:  # strong
            # Strong augmentation - maximum generalization
            self.params = {
                'brightness_limit': 0.3,
                'contrast_limit': 0.3,
                'hue_shift_limit': 15,
                'sat_shift_limit': 30,
                'val_shift_limit': 30,
                'rotation_limit': 15,
                'scale_limit': 0.15,
                'noise_var_limit': (15.0, 50.0),
                'blur_limit': 5,
                'elastic_alpha': 150,
                'elastic_sigma': 15
            }
    
    def create_training_augmentation(self) -> A.Compose:
        """
        Create comprehensive training augmentation pipeline.
        
        Returns:
            Albumentations Compose object with training augmentations
        """
        transforms = []
        
        # Geometric transformations
        transforms.extend([
            # Resize to target size
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR),
            
            # Horizontal flip (common in medical imaging)
            A.HorizontalFlip(p=0.5),
            
            # Vertical flip (less common but useful for augmentation)
            A.VerticalFlip(p=0.2),
            
            # Random rotation (small angles to preserve anatomy)
            A.Rotate(
                limit=self.params['rotation_limit'],
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.3
            ),
            
            # Random scale (slight scaling)
            A.RandomScale(
                scale_limit=self.params['scale_limit'],
                interpolation=cv2.INTER_LINEAR,
                p=0.3
            ),
        ])
        
        # Photometric transformations
        transforms.extend([
            # Brightness and contrast adjustment
            A.RandomBrightnessContrast(
                brightness_limit=self.params['brightness_limit'],
                contrast_limit=self.params['contrast_limit'],
                p=0.5
            ),
            
            # HSV color space adjustments
            A.HueSaturationValue(
                hue_shift_limit=self.params['hue_shift_limit'],
                sat_shift_limit=self.params['sat_shift_limit'],
                val_shift_limit=self.params['val_shift_limit'],
                p=0.3
            ),
            
            # Gamma correction (simulates different X-ray exposures)
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            
            # Channel shuffle (simulates different X-ray machines)
            A.ChannelShuffle(p=0.1),
        ])
        
        # Noise and blur (simulates real-world conditions)
        transforms.extend([
            # Gaussian noise
            A.GaussNoise(
                var_limit=self.params['noise_var_limit'],
                p=0.2
            ),
            
            # Motion blur (simulates patient movement)
            A.MotionBlur(
                blur_limit=self.params['blur_limit'],
                p=0.1
            ),
            
            # Gaussian blur
            A.Blur(
                blur_limit=self.params['blur_limit'],
                p=0.1
            ),
        ])
        
        # Advanced augmentations
        if not self.preserve_anatomy:
            # Elastic deformation (use carefully for medical images)
            transforms.append(
                A.ElasticTransform(
                    alpha=self.params['elastic_alpha'],
                    sigma=self.params['elastic_sigma'],
                    alpha_affine=50,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.1
                )
            )
        
        # Cutout and mixup (advanced techniques)
        transforms.extend([
            # Cutout (random rectangular regions)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.1
            ),
            
            # Grid distortion (subtle)
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.1
            ),
        ])
        
        # Final transforms
        transforms.extend([
            # Normalize for YOLOv8
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # Convert to tensor
            ToTensorV2()
        ])
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def create_validation_augmentation(self) -> A.Compose:
        """
        Create validation augmentation pipeline (minimal augmentation).
        
        Returns:
            Albumentations Compose object with validation transforms
        """
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def create_test_augmentation(self) -> A.Compose:
        """
        Create test augmentation pipeline (no augmentation).
        
        Returns:
            Albumentations Compose object with test transforms
        """
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


class YOLOv8Augmentation:
    """
    YOLOv8 Specific Augmentation Pipeline
    
    This class provides augmentations specifically designed for YOLOv8 training,
    including Mosaic, MixUp, and other YOLOv8-specific techniques.
    """
    
    def __init__(self, 
                 image_size: int = 640,
                 mosaic_prob: float = 1.0,
                 mixup_prob: float = 0.0,
                 copy_paste_prob: float = 0.0):
        """
        Initialize YOLOv8 augmentation pipeline.
        
        Args:
            image_size: Target image size
            mosaic_prob: Probability of applying Mosaic augmentation
            mixup_prob: Probability of applying MixUp augmentation
            copy_paste_prob: Probability of applying Copy-Paste augmentation
        """
        self.image_size = image_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        
        logger.info(f"YOLOv8Augmentation initialized with mosaic_prob={mosaic_prob}")
    
    def create_mosaic_augmentation(self) -> A.Compose:
        """
        Create Mosaic augmentation pipeline.
        
        Mosaic combines 4 images into one, improving detection of small objects
        and providing more diverse training examples.
        
        Returns:
            Albumentations Compose object with Mosaic augmentation
        """
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def create_mixup_augmentation(self) -> A.Compose:
        """
        Create MixUp augmentation pipeline.
        
        MixUp blends two images and their labels, creating smooth interpolations
        between classes and improving model generalization.
        
        Returns:
            Albumentations Compose object with MixUp augmentation
        """
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))


def create_augmentation_pipeline(config: Dict) -> Dict:
    """
    Create complete augmentation pipeline based on configuration.
    
    Args:
        config: Configuration dictionary from data.yaml
        
    Returns:
        Dictionary containing augmentation pipelines
    """
    # Extract configuration
    img_size = config.get('img_size', 640)
    mosaic_prob = config.get('mosaic', 1.0)
    mixup_prob = config.get('mixup', 0.0)
    copy_paste_prob = config.get('copy_paste', 0.0)
    augmentation_strength = config.get('augmentation_strength', 'medium')
    
    # Create augmentation pipelines
    medical_aug = MedicalImageAugmentation(
        image_size=img_size,
        augmentation_strength=augmentation_strength,
        preserve_anatomy=True
    )
    
    yolo_aug = YOLOv8Augmentation(
        image_size=img_size,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        copy_paste_prob=copy_paste_prob
    )
    
    return {
        'medical_training': medical_aug.create_training_augmentation(),
        'medical_validation': medical_aug.create_validation_augmentation(),
        'medical_test': medical_aug.create_test_augmentation(),
        'yolo_mosaic': yolo_aug.create_mosaic_augmentation(),
        'yolo_mixup': yolo_aug.create_mixup_augmentation()
    }


class AugmentationVisualizer:
    """
    Visualization utilities for augmentation results.
    
    This class provides methods to visualize augmentation effects on
    medical images, helping to understand and tune augmentation parameters.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize augmentation visualizer.
        
        Args:
            save_path: Path to save visualization results
        """
        self.save_path = Path(save_path) if save_path else None
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
    
    def visualize_augmentations(self, 
                              image: np.ndarray,
                              augmentations: List[A.Compose],
                              titles: List[str],
                              num_samples: int = 4) -> None:
        """
        Visualize multiple augmentation pipelines on the same image.
        
        Args:
            image: Input image
            augmentations: List of augmentation pipelines
            titles: List of titles for each augmentation
            num_samples: Number of samples to generate per augmentation
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(augmentations), num_samples, 
                                figsize=(num_samples * 3, len(augmentations) * 3))
        
        if len(augmentations) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (aug, title) in enumerate(zip(augmentations, titles)):
            for j in range(num_samples):
                # Apply augmentation
                augmented = aug(image=image)['image']
                
                # Convert tensor to numpy for visualization
                if isinstance(augmented, torch.Tensor):
                    augmented = augmented.permute(1, 2, 0).numpy()
                    augmented = augmented * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    augmented = np.clip(augmented, 0, 1)
                
                # Plot
                axes[i, j].imshow(augmented)
                axes[i, j].set_title(f"{title} - Sample {j+1}")
                axes[i, j].axis('off')
        
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(self.save_path / 'augmentation_visualization.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test the augmentation pipeline
    import matplotlib.pyplot as plt
    
    # Create sample medical image
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Initialize augmentation
    aug_pipeline = MedicalImageAugmentation(
        image_size=640,
        augmentation_strength='medium',
        preserve_anatomy=True
    )
    
    # Create augmentation pipelines
    train_aug = aug_pipeline.create_training_augmentation()
    val_aug = aug_pipeline.create_validation_augmentation()
    
    # Test augmentations
    augmented_train = train_aug(image=sample_image)['image']
    augmented_val = val_aug(image=sample_image)['image']
    
    print(f"Original shape: {sample_image.shape}")
    print(f"Train augmented shape: {augmented_train.shape}")
    print(f"Val augmented shape: {augmented_val.shape}")
    
    logger.info("Augmentation pipeline test completed successfully!")

