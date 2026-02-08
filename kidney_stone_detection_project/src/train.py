#!/usr/bin/env python3
"""
Simple CUDA Training Script for Kidney Stone Detection
=====================================================

This script provides a simplified training pipeline for kidney stone detection
using YOLOv8-nano with CUDA support.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime
import logging

# YOLOv8 imports
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_cuda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function with CUDA support."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! Please check your GPU setup.")
        return False
    
    logger.info(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Configuration
    config_path = "data/data.yaml"
    model_size = "nano"  # nano, small, medium, large, xlarge
    device = "cuda"
    
    # Create timestamp for unique run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"kidney_stone_detection_cuda_{timestamp}"
    
    # Setup directories
    project_dir = Path(f"runs/{project_name}")
    project_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training outputs will be saved to: {project_dir}")
    
    try:
        # Initialize YOLOv8 model
        model_name = f"yolov8{model_size[0]}.pt"  # yolov8n.pt
        logger.info(f"Loading model: {model_name}")
        
        model = YOLO(model_name)
        model.to(device)
        
        logger.info(f"YOLOv8-{model_size} model initialized on {device}")
        
        # Training parameters
        train_args = {
            'data': config_path,
            'epochs': 50,  # Reduced for testing
            'batch': 16,
            'imgsz': 640,
            'device': device,
            'project': str(project_dir),
            'name': 'train',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'patience': 20,
            'save_period': 10,
            'val': True,
            'plots': True,
            'save': True,
            'save_txt': True,
            'save_conf': True,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'visualize': False,
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'conf': 0.25,
            'iou': 0.45,
            'max_det': 1000,
            'half': False,  # Disable half precision for stability
            'dnn': False,
            'verbose': True,
        }
        
        logger.info("Starting training with CUDA...")
        logger.info(f"Training parameters: epochs={train_args['epochs']}, batch={train_args['batch']}, device={device}")
        
        # Start training
        import time
        start_time = time.time()
        
        results = model.train(**train_args)
        
        training_time = time.time() - start_time
        
        logger.info(f"Training completed successfully in {training_time:.2f} seconds")
        logger.info(f"Results saved to: {project_dir}")
        
        # Test GPU memory usage
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")
        sys.exit(1)
