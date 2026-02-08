"""
YOLOv8-nano Training Script for Kidney Stone Detection
=====================================================

This script provides a comprehensive training pipeline for kidney stone detection
using YOLOv8-nano architecture. It includes early stopping, model checkpointing,
comprehensive logging, and medical image-specific optimizations.

Key Features:
- YOLOv8-nano for lightweight inference
- Early stopping with patience
- Comprehensive evaluation metrics
- Medical image-specific preprocessing
- Real-time training monitoring
- Model checkpointing and resume capability

Author: [Your Name]
Date: 2024
License: MIT
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
import json
import time

# YOLOv8 imports
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.data import build_dataloader

# Custom imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import create_preprocessing_pipeline
from utils.augmentation import create_augmentation_pipeline
from utils.visualization import TrainingVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KidneyStoneTrainer:
    """
    Comprehensive trainer for kidney stone detection using YOLOv8-nano.
    
    This class provides a complete training pipeline with medical image-specific
    optimizations, early stopping, and comprehensive evaluation.
    """
    
    def __init__(self, 
                 config_path: str,
                 model_size: str = 'nano',
                 device: str = 'auto',
                 project_name: str = 'kidney_stone_detection'):
        """
        Initialize the kidney stone trainer.
        
        Args:
            config_path: Path to data.yaml configuration file
            model_size: YOLOv8 model size ('nano', 'small', 'medium', 'large', 'xlarge')
            device: Device to use for training ('auto', 'cpu', 'cuda', 'mps')
            project_name: Name of the training project
        """
        self.config_path = Path(config_path)
        self.model_size = model_size
        self.device = device
        self.project_name = project_name
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize paths
        self._setup_paths()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize training parameters
        self._setup_training_params()
        
        # Initialize visualizer
        self.visualizer = TrainingVisualizer(save_path=self.results_dir)
        
        logger.info(f"KidneyStoneTrainer initialized with model_size={model_size}")
    
    def _load_config(self) -> Dict:
        """Load and validate configuration from data.yaml."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ['train', 'val', 'test', 'nc', 'names']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in config")
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_paths(self):
        """Setup directory paths for training outputs."""
        # Create timestamp for unique run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories
        self.project_dir = Path(f"runs/{self.project_name}_{timestamp}")
        self.results_dir = self.project_dir / "results"
        self.models_dir = self.project_dir / "models"
        self.logs_dir = self.project_dir / "logs"
        
        # Create directories
        for dir_path in [self.results_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training outputs will be saved to: {self.project_dir}")
    
    def _initialize_model(self) -> YOLO:
        """Initialize YOLOv8 model with specified size."""
        try:
            # Create model name
            model_name = f"yolov8{self.model_size[0]}.pt"  # yolov8n.pt, yolov8s.pt, etc.
            
            # Initialize model
            model = YOLO(model_name)
            
            # Set device
            if self.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device
            
            model.to(device)
            
            logger.info(f"YOLOv8-{self.model_size} model initialized on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _setup_training_params(self):
        """Setup training parameters from configuration."""
        # Extract training parameters from config
        self.training_params = {
            'epochs': self.config.get('epochs', 100),
            'batch_size': self.config.get('batch_size', 16),
            'learning_rate': self.config.get('learning_rate', 0.01),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'momentum': self.config.get('momentum', 0.937),
            'patience': self.config.get('patience', 50),
            'save_period': self.config.get('save_period', 10),
            'val_period': self.config.get('val_period', 1),
            'img_size': self.config.get('img_size', 640),
            'conf_threshold': self.config.get('conf_threshold', 0.25),
            'iou_threshold': self.config.get('iou_threshold', 0.45),
        }
        
        # Augmentation parameters
        self.augmentation_params = {
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.0),
            'copy_paste': self.config.get('copy_paste', 0.0),
            'augment': self.config.get('augment', True),
        }
        
        logger.info("Training parameters configured")
    
    def train(self, 
              resume: bool = False,
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the kidney stone detection model.
        
        Args:
            resume: Whether to resume training from checkpoint
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting kidney stone detection training...")
        
        try:
            # Prepare training arguments
            train_args = {
                'data': str(self.config_path),
                'epochs': self.training_params['epochs'],
                'batch': self.training_params['batch_size'],
                'imgsz': self.training_params['img_size'],
                'device': self.device,
                'project': str(self.project_dir),
                'name': 'train',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': self.training_params['learning_rate'],
                'weight_decay': self.training_params['weight_decay'],
                'momentum': self.training_params['momentum'],
                'patience': self.training_params['patience'],
                'save_period': self.training_params['save_period'],
                'val': True,
                'plots': True,
                'save': True,
                'save_txt': True,
                'save_conf': True,
                'save_crop': False,
                'show_labels': True,
                'show_conf': True,
                'visualize': False,
                'augment': self.augmentation_params['augment'],
                'mosaic': self.augmentation_params['mosaic'],
                'mixup': self.augmentation_params['mixup'],
                'copy_paste': self.augmentation_params['copy_paste'],
                'conf': self.training_params['conf_threshold'],
                'iou': self.training_params['iou_threshold'],
                'max_det': 1000,
                'half': False,  # Disable half precision for stability
                'dnn': False,
                'plots': True,
                'source': None,
                'vid_stride': 1,
                'stream_buffer': False,
                'shape': None,
                'shuffle': True,
                'save_dir': str(self.results_dir),
                'verbose': True,
            }
            
            # Add resume parameters if needed
            if resume and resume_from:
                train_args['resume'] = resume_from
                logger.info(f"Resuming training from: {resume_from}")
            
            # Start training
            start_time = time.time()
            results = self.model.train(**train_args)
            training_time = time.time() - start_time
            
            # Log training completion
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save training summary
            self._save_training_summary(results, training_time)
            
            # Generate training visualizations
            self._generate_training_plots(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _save_training_summary(self, results: Any, training_time: float):
        """Save comprehensive training summary."""
        try:
            # Extract metrics from results
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            else:
                metrics = {}
            
            # Create training summary
            summary = {
                'training_info': {
                    'start_time': datetime.now().isoformat(),
                    'training_time_seconds': training_time,
                    'model_size': self.model_size,
                    'device': self.device,
                    'config_path': str(self.config_path),
                    'project_dir': str(self.project_dir)
                },
                'training_parameters': self.training_params,
                'augmentation_parameters': self.augmentation_params,
                'dataset_info': {
                    'num_classes': self.config['nc'],
                    'class_names': self.config['names'],
                    'train_path': self.config['train'],
                    'val_path': self.config['val'],
                    'test_path': self.config['test']
                },
                'final_metrics': metrics,
                'model_paths': {
                    'best_model': str(self.results_dir / 'weights' / 'best.pt'),
                    'last_model': str(self.results_dir / 'weights' / 'last.pt')
                }
            }
            
            # Save summary to JSON
            summary_path = self.results_dir / 'training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Training summary saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")
    
    def _generate_training_plots(self, results: Any):
        """Generate comprehensive training plots."""
        try:
            # Create plots directory
            plots_dir = self.results_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Generate training curves
            self.visualizer.plot_training_curves(results, save_path=plots_dir)
            
            # Generate confusion matrix
            self.visualizer.plot_confusion_matrix(results, save_path=plots_dir)
            
            # Generate PR curves
            self.visualizer.plot_pr_curves(results, save_path=plots_dir)
            
            logger.info(f"Training plots saved to: {plots_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate training plots: {e}")
    
    def evaluate(self, 
                 weights_path: Optional[str] = None,
                 test_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            weights_path: Path to model weights
            test_data: Path to test data (if different from config)
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting model evaluation...")
        
        try:
            # Use best weights if not specified
            if weights_path is None:
                weights_path = self.results_dir / 'weights' / 'best.pt'
            
            # Use test data from config if not specified
            if test_data is None:
                test_data = self.config['test']
            
            # Load model with weights
            model = YOLO(str(weights_path))
            
            # Run validation
            results = model.val(
                data=str(self.config_path),
                split='test',
                imgsz=self.training_params['img_size'],
                batch=self.training_params['batch_size'],
                conf=self.training_params['conf_threshold'],
                iou=self.training_params['iou_threshold'],
                device=self.device,
                plots=True,
                save_json=True,
                save_dir=str(self.results_dir / 'evaluation')
            )
            
            # Save evaluation results
            self._save_evaluation_results(results)
            
            logger.info("Model evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _save_evaluation_results(self, results: Any):
        """Save detailed evaluation results."""
        try:
            eval_dir = self.results_dir / 'evaluation'
            eval_dir.mkdir(exist_ok=True)
            
            # Extract metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            else:
                metrics = {}
            
            # Save metrics to JSON
            metrics_path = eval_dir / 'evaluation_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Evaluation results saved to: {eval_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def export_model(self, 
                    weights_path: Optional[str] = None,
                    formats: List[str] = ['onnx', 'torchscript']) -> Dict[str, str]:
        """
        Export trained model to different formats.
        
        Args:
            weights_path: Path to model weights
            formats: List of export formats
            
        Returns:
            Dictionary mapping format to exported model path
        """
        logger.info(f"Exporting model to formats: {formats}")
        
        try:
            # Use best weights if not specified
            if weights_path is None:
                weights_path = self.results_dir / 'weights' / 'best.pt'
            
            # Load model
            model = YOLO(str(weights_path))
            
            # Export to different formats
            exported_models = {}
            export_dir = self.results_dir / 'exports'
            export_dir.mkdir(exist_ok=True)
            
            for format_type in formats:
                try:
                    exported_path = model.export(
                        format=format_type,
                        imgsz=self.training_params['img_size'],
                        optimize=True,
                        half=False,
                        int8=False,
                        dynamic=False,
                        simplify=True,
                        opset=None,
                        workspace=4,
                        nms=True
                    )
                    
                    # Move to export directory
                    export_filename = f"kidney_stone_detection.{format_type}"
                    final_path = export_dir / export_filename
                    
                    if Path(exported_path).exists():
                        Path(exported_path).rename(final_path)
                        exported_models[format_type] = str(final_path)
                        logger.info(f"Model exported to {format_type}: {final_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to export to {format_type}: {e}")
            
            return exported_models
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train YOLOv8-nano for Kidney Stone Detection')
    parser.add_argument('--config', type=str, default='data/data.yaml',
                       help='Path to data.yaml configuration file')
    parser.add_argument('--model-size', type=str, default='nano',
                       choices=['nano', 'small', 'medium', 'large', 'xlarge'],
                       help='YOLOv8 model size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training')
    parser.add_argument('--project-name', type=str, default='kidney_stone_detection',
                       help='Name of the training project')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after training')
    parser.add_argument('--export', action='store_true',
                       help='Export model after training')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = KidneyStoneTrainer(
            config_path=args.config,
            model_size=args.model_size,
            device=args.device,
            project_name=args.project_name
        )
        
        # Train model
        logger.info("Starting training...")
        results = trainer.train(
            resume=args.resume,
            resume_from=args.resume_from
        )
        
        # Evaluate if requested
        if args.evaluate:
            logger.info("Running evaluation...")
            eval_results = trainer.evaluate()
        
        # Export if requested
        if args.export:
            logger.info("Exporting model...")
            exported_models = trainer.export_model()
            logger.info(f"Exported models: {exported_models}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
