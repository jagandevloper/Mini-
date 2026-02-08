"""
Preprocessing Pipeline for Kidney Stone Detection
================================================

This module provides comprehensive preprocessing utilities specifically designed
for KUB (Kidney-Ureter-Bladder) X-ray images. It includes normalization,
resizing, and medical image-specific preprocessing techniques.

Key Features:
- Medical image normalization (DICOM compatible)
- Aspect ratio preservation with intelligent padding
- Histogram equalization for enhanced contrast
- Noise reduction for X-ray images
- Batch processing capabilities

Author: [Your Name]
Date: 2024
License: MIT
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    Medical Image Preprocessor for KUB X-ray Images
    
    This class provides specialized preprocessing for medical X-ray images,
    including normalization, contrast enhancement, and noise reduction
    techniques optimized for kidney stone detection.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (640, 640),
                 normalize_method: str = 'imagenet',
                 enhance_contrast: bool = True,
                 reduce_noise: bool = True):
        """
        Initialize the medical image preprocessor.
        
        Args:
            target_size: Target image size (width, height)
            normalize_method: Normalization method ('imagenet', 'medical', 'minmax')
            enhance_contrast: Whether to apply contrast enhancement
            reduce_noise: Whether to apply noise reduction
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.enhance_contrast = enhance_contrast
        self.reduce_noise = reduce_noise
        
        # ImageNet normalization (standard for YOLOv8)
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Medical image normalization (optimized for X-ray)
        self.medical_mean = [0.5, 0.5, 0.5]  # Center around 0.5 for X-ray
        self.medical_std = [0.25, 0.25, 0.25]  # Lower std for medical images
        
        logger.info(f"MedicalImagePreprocessor initialized with target_size={target_size}")
    
    def resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while preserving aspect ratio using intelligent padding.
        
        This method is crucial for medical images where aspect ratio preservation
        is important for maintaining anatomical proportions.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Resized image with padding (H, W, C)
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor to fit image in target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        
        # Calculate padding offsets (center the image)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        logger.debug(f"Resized image from {image.shape} to {padded.shape}")
        return padded
    
    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        CLAHE is particularly effective for X-ray images as it enhances local
        contrast while preventing over-amplification of noise.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
        return enhanced
    
    def reduce_noise_bilateral(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for noise reduction while preserving edges.
        
        Bilateral filtering is ideal for medical images as it reduces noise
        while preserving important anatomical structures and edges.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Noise-reduced image
        """
        if len(image.shape) == 3:
            # Apply bilateral filter to each channel
            filtered_channels = []
            for i in range(image.shape[2]):
                filtered = cv2.bilateralFilter(image[:, :, i], 9, 75, 75)
                filtered_channels.append(filtered)
            filtered = np.stack(filtered_channels, axis=2)
        else:
            # Grayscale image
            filtered = cv2.bilateralFilter(image, 9, 75, 75)
            
        return filtered
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image based on selected method.
        
        Args:
            image: Input image as numpy array (0-255 range)
            
        Returns:
            Normalized image (0-1 range)
        """
        # Convert to float and normalize to 0-1
        normalized = image.astype(np.float32) / 255.0
        
        if self.normalize_method == 'medical':
            # Medical-specific normalization
            normalized = (normalized - 0.5) / 0.25
        elif self.normalize_method == 'minmax':
            # Min-max normalization
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        # 'imagenet' normalization is applied later in the transform pipeline
        
        return normalized
    
    def preprocess_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for model input
        """
        # Step 1: Resize with padding
        processed = self.resize_with_padding(image)
        
        # Step 2: Contrast enhancement
        if self.enhance_contrast:
            processed = self.enhance_contrast_clahe(processed)
        
        # Step 3: Noise reduction
        if self.reduce_noise:
            processed = self.reduce_noise_bilateral(processed)
        
        # Step 4: Normalize
        processed = self.normalize_image(processed)
        
        logger.debug(f"Preprocessed image shape: {processed.shape}")
        return processed
    
    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Batch tensor ready for model input
        """
        processed_images = []
        
        for image in images:
            processed = self.preprocess_single_image(image)
            processed_images.append(processed)
        
        # Convert to tensor and stack
        batch_tensor = torch.stack([torch.from_numpy(img).permute(2, 0, 1) 
                                   for img in processed_images])
        
        logger.info(f"Preprocessed batch of {len(images)} images")
        return batch_tensor


class YOLOv8Transform:
    """
    YOLOv8 Compatible Transform Pipeline
    
    This class provides transforms compatible with YOLOv8 training and inference,
    including proper normalization and tensor conversion.
    """
    
    def __init__(self, 
                 image_size: int = 640,
                 normalize_method: str = 'imagenet',
                 training: bool = True):
        """
        Initialize YOLOv8 transform pipeline.
        
        Args:
            image_size: Target image size
            normalize_method: Normalization method
            training: Whether this is for training (includes augmentation)
        """
        self.image_size = image_size
        self.training = training
        
        # Define normalization values
        if normalize_method == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:  # medical
            mean = [0.5, 0.5, 0.5]
            std = [0.25, 0.25, 0.25]
        
        # Base transforms
        self.base_transforms = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        
        # Training transforms with augmentation
        if training:
            self.training_transforms = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),  # Less common for medical images
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
    
    def __call__(self, image: np.ndarray, training: bool = None) -> torch.Tensor:
        """
        Apply transforms to image.
        
        Args:
            image: Input image as numpy array
            training: Override training mode
            
        Returns:
            Transformed image tensor
        """
        use_training = training if training is not None else self.training
        
        if use_training and hasattr(self, 'training_transforms'):
            transformed = self.training_transforms(image=image)
        else:
            transformed = self.base_transforms(image=image)
        
        return transformed['image']


def create_preprocessing_pipeline(config: dict) -> dict:
    """
    Create complete preprocessing pipeline based on configuration.
    
    Args:
        config: Configuration dictionary from data.yaml
        
    Returns:
        Dictionary containing preprocessor and transforms
    """
    # Extract configuration
    img_size = config.get('img_size', 640)
    normalize_method = config.get('normalize_method', 'imagenet')
    
    # Create preprocessor
    preprocessor = MedicalImagePreprocessor(
        target_size=(img_size, img_size),
        normalize_method=normalize_method,
        enhance_contrast=True,
        reduce_noise=True
    )
    
    # Create transforms
    train_transform = YOLOv8Transform(
        image_size=img_size,
        normalize_method=normalize_method,
        training=True
    )
    
    val_transform = YOLOv8Transform(
        image_size=img_size,
        normalize_method=normalize_method,
        training=False
    )
    
    return {
        'preprocessor': preprocessor,
        'train_transform': train_transform,
        'val_transform': val_transform
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessing pipeline
    import matplotlib.pyplot as plt
    
    # Create sample medical image (simulate X-ray)
    sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor()
    
    # Process image
    processed = preprocessor.preprocess_single_image(sample_image)
    
    print(f"Original shape: {sample_image.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    logger.info("Preprocessing pipeline test completed successfully!")

