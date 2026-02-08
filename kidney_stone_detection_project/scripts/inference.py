"""
Batch Inference Script for Kidney Stone Detection
=================================================

This script provides efficient batch inference capabilities for kidney stone
detection on large datasets. It includes optimized processing, progress tracking,
and comprehensive result analysis.

Key Features:
- Efficient batch processing
- Progress tracking and logging
- Result analysis and statistics
- Export capabilities (JSON, CSV)
- Memory optimization
- Parallel processing support

Author: [Your Name]
Date: 2024
License: MIT
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import argparse
from datetime import datetime
import time
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# YOLOv8 imports
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Custom imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import MedicalImagePreprocessor
from utils.visualization import TrainingVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchInference:
    """
    Efficient batch inference for kidney stone detection.
    
    This class provides optimized batch processing capabilities with
    progress tracking, result analysis, and export functionality.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = 'auto',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 batch_size: int = 8,
                 num_workers: int = 4):
        """
        Initialize batch inference system.
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to data.yaml configuration
            device: Device to use for inference
            confidence_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            batch_size: Batch size for processing
            num_workers: Number of worker processes
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize preprocessor
        self.preprocessor = MedicalImagePreprocessor()
        
        # Results storage
        self.results = []
        self.statistics = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'total_detections': 0,
            'processing_time': 0,
            'average_inference_time': 0
        }
        
        logger.info(f"BatchInference initialized with batch_size={batch_size}, num_workers={num_workers}")
    
    def _load_config(self) -> Dict:
        """Load configuration from data.yaml."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_model(self) -> YOLO:
        """Load trained model with optimizations."""
        try:
            model = YOLO(str(self.model_path))
            
            # Set device
            if self.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device
            
            model.to(device)
            
            # Optimize for batch inference
            if device == 'cuda':
                model.model.half()  # Use half precision for speed
            
            logger.info(f"Model loaded on {device} with batch optimizations")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_dataset(self, 
                       dataset_path: str,
                       output_dir: str = None,
                       save_annotations: bool = True,
                       save_results: bool = True,
                       export_formats: List[str] = ['json', 'csv']) -> Dict[str, Any]:
        """
        Process entire dataset with batch inference.
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory for results
            save_annotations: Whether to save annotated images
            save_results: Whether to save detection results
            export_formats: Formats to export results
            
        Returns:
            Comprehensive processing results
        """
        logger.info(f"Starting batch processing of dataset: {dataset_path}")
        
        try:
            # Setup output directory
            if output_dir is None:
                output_dir = f"batch_inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find all images
            image_paths = self._find_images(dataset_path)
            self.statistics['total_images'] = len(image_paths)
            
            if not image_paths:
                logger.error(f"No images found in dataset: {dataset_path}")
                return {}
            
            logger.info(f"Found {len(image_paths)} images to process")
            
            # Process images in batches
            start_time = time.time()
            self.results = []
            
            # Create progress bar
            with tqdm(total=len(image_paths), desc="Processing images") as pbar:
                for i in range(0, len(image_paths), self.batch_size):
                    batch_paths = image_paths[i:i + self.batch_size]
                    batch_results = self._process_batch(batch_paths)
                    self.results.extend(batch_results)
                    
                    # Update progress
                    pbar.update(len(batch_paths))
                    
                    # Update statistics
                    self.statistics['processed_images'] += len(batch_results)
                    self.statistics['failed_images'] += len(batch_paths) - len(batch_results)
            
            # Calculate final statistics
            processing_time = time.time() - start_time
            self.statistics['processing_time'] = processing_time
            self.statistics['average_inference_time'] = processing_time / len(image_paths) if image_paths else 0
            self.statistics['total_detections'] = sum(len(result.get('detections', [])) for result in self.results)
            
            # Save results
            if save_results:
                self._save_results(output_path, export_formats)
            
            # Save annotated images
            if save_annotations:
                self._save_annotated_images(output_path)
            
            # Generate analysis report
            analysis_report = self._generate_analysis_report()
            
            # Save analysis report
            report_path = output_path / 'analysis_report.json'
            with open(report_path, 'w') as f:
                json.dump(analysis_report, f, indent=2, default=str)
            
            logger.info(f"Batch processing completed. Results saved to: {output_path}")
            return analysis_report
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def _find_images(self, dataset_path: str) -> List[str]:
        """Find all images in dataset directory."""
        try:
            dataset_dir = Path(dataset_path)
            if not dataset_dir.exists():
                logger.error(f"Dataset directory does not exist: {dataset_path}")
                return []
            
            # Supported image extensions
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            image_paths = []
            for ext in image_extensions:
                # Search recursively
                image_paths.extend(dataset_dir.rglob(f'*{ext}'))
                image_paths.extend(dataset_dir.rglob(f'*{ext.upper()}'))
            
            # Convert to strings and sort
            image_paths = sorted([str(p) for p in image_paths])
            
            logger.info(f"Found {len(image_paths)} images in dataset")
            return image_paths
            
        except Exception as e:
            logger.error(f"Failed to find images: {e}")
            return []
    
    def _process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of images."""
        try:
            batch_results = []
            
            # Load images
            images = []
            valid_paths = []
            
            for path in image_paths:
                try:
                    image = cv2.imread(path)
                    if image is not None:
                        images.append(image)
                        valid_paths.append(path)
                    else:
                        logger.warning(f"Failed to load image: {path}")
                except Exception as e:
                    logger.warning(f"Error loading image {path}: {e}")
            
            if not images:
                return batch_results
            
            # Run batch inference
            try:
                results = self.model(
                    images,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    stream=False
                )
                
                # Process results
                for i, (image, result, path) in enumerate(zip(images, results, valid_paths)):
                    try:
                        detections = self._extract_detections(result)
                        
                        batch_result = {
                            'image_path': path,
                            'image_shape': image.shape,
                            'detections': detections,
                            'detection_count': len(detections),
                            'processing_time': time.time(),
                            'model_confidence': self.confidence_threshold,
                            'model_iou': self.iou_threshold
                        }
                        
                        batch_results.append(batch_result)
                        
                    except Exception as e:
                        logger.error(f"Failed to process result for {path}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                # Fallback to individual processing
                for image, path in zip(images, valid_paths):
                    try:
                        result = self.model(
                            image,
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            verbose=False
                        )
                        
                        detections = self._extract_detections(result[0])
                        
                        batch_result = {
                            'image_path': path,
                            'image_shape': image.shape,
                            'detections': detections,
                            'detection_count': len(detections),
                            'processing_time': time.time(),
                            'model_confidence': self.confidence_threshold,
                            'model_iou': self.iou_threshold
                        }
                        
                        batch_results.append(batch_result)
                        
                    except Exception as e:
                        logger.error(f"Individual inference failed for {path}: {e}")
                        continue
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []
    
    def _extract_detections(self, result) -> List[Dict[str, Any]]:
        """Extract detection information from YOLOv8 result."""
        try:
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i].tolist(),
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.config.get('names', ['kidney_stone'])[class_ids[i]] if class_ids[i] < len(self.config.get('names', [])) else f'Class_{class_ids[i]}'
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Failed to extract detections: {e}")
            return []
    
    def _save_results(self, output_path: Path, export_formats: List[str]):
        """Save detection results in specified formats."""
        try:
            # Save JSON format
            if 'json' in export_formats:
                json_path = output_path / 'detection_results.json'
                with open(json_path, 'w') as f:
                    json.dump(self.results, f, indent=2, default=str)
                logger.info(f"Results saved in JSON format: {json_path}")
            
            # Save CSV format
            if 'csv' in export_formats:
                csv_data = self._prepare_csv_data()
                csv_path = output_path / 'detection_results.csv'
                csv_data.to_csv(csv_path, index=False)
                logger.info(f"Results saved in CSV format: {csv_path}")
            
            # Save statistics
            stats_path = output_path / 'processing_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(self.statistics, f, indent=2, default=str)
            logger.info(f"Statistics saved: {stats_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _prepare_csv_data(self) -> pd.DataFrame:
        """Prepare data for CSV export."""
        try:
            csv_rows = []
            
            for result in self.results:
                image_path = result['image_path']
                detections = result.get('detections', [])
                
                if detections:
                    for detection in detections:
                        row = {
                            'image_path': image_path,
                            'image_shape': str(result.get('image_shape', '')),
                            'bbox_x1': detection['bbox'][0],
                            'bbox_y1': detection['bbox'][1],
                            'bbox_x2': detection['bbox'][2],
                            'bbox_y2': detection['bbox'][3],
                            'confidence': detection['confidence'],
                            'class_id': detection['class_id'],
                            'class_name': detection['class_name'],
                            'processing_time': result.get('processing_time', ''),
                            'model_confidence': result.get('model_confidence', ''),
                            'model_iou': result.get('model_iou', '')
                        }
                        csv_rows.append(row)
                else:
                    # No detections
                    row = {
                        'image_path': image_path,
                        'image_shape': str(result.get('image_shape', '')),
                        'bbox_x1': '',
                        'bbox_y1': '',
                        'bbox_x2': '',
                        'bbox_y2': '',
                        'confidence': '',
                        'class_id': '',
                        'class_name': '',
                        'processing_time': result.get('processing_time', ''),
                        'model_confidence': result.get('model_confidence', ''),
                        'model_iou': result.get('model_iou', '')
                    }
                    csv_rows.append(row)
            
            return pd.DataFrame(csv_rows)
            
        except Exception as e:
            logger.error(f"Failed to prepare CSV data: {e}")
            return pd.DataFrame()
    
    def _save_annotated_images(self, output_path: Path):
        """Save annotated images."""
        try:
            annotated_dir = output_path / 'annotated_images'
            annotated_dir.mkdir(exist_ok=True)
            
            logger.info(f"Saving annotated images to: {annotated_dir}")
            
            for result in tqdm(self.results, desc="Saving annotated images"):
                try:
                    # Load original image
                    image_path = result['image_path']
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        continue
                    
                    # Annotate image
                    annotated_image = self._annotate_image(image, result['detections'])
                    
                    # Save annotated image
                    output_filename = f"annotated_{Path(image_path).name}"
                    output_filepath = annotated_dir / output_filename
                    cv2.imwrite(str(output_filepath), annotated_image)
                    
                except Exception as e:
                    logger.error(f"Failed to save annotated image for {result['image_path']}: {e}")
                    continue
            
            logger.info(f"Annotated images saved to: {annotated_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save annotated images: {e}")
    
    def _annotate_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Annotate image with detection results."""
        try:
            annotated_image = image.copy()
            
            # Colors for different classes
            colors = {
                'kidney_stone': (0, 255, 0),  # Green
                'default': (255, 0, 0)  # Red
            }
            
            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Get color for class
                color = colors.get(class_name, colors['default'])
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f'{class_name}: {confidence:.2f}'
                label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                
                # Draw label background
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                          font, font_scale, (255, 255, 255), font_thickness)
            
            return annotated_image
            
        except Exception as e:
            logger.error(f"Failed to annotate image: {e}")
            return image
    
    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        try:
            report = {
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'model_path': str(self.model_path),
                    'config_path': str(self.config_path),
                    'device': self.device,
                    'batch_size': self.batch_size,
                    'num_workers': self.num_workers,
                    'confidence_threshold': self.confidence_threshold,
                    'iou_threshold': self.iou_threshold
                },
                'statistics': self.statistics,
                'detection_analysis': self._analyze_detections(),
                'performance_analysis': self._analyze_performance(),
                'dataset_analysis': self._analyze_dataset()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")
            return {}
    
    def _analyze_detections(self) -> Dict[str, Any]:
        """Analyze detection patterns."""
        try:
            if not self.results:
                return {}
            
            # Collect detection data
            all_detections = []
            detection_counts = []
            confidence_scores = []
            
            for result in self.results:
                detections = result.get('detections', [])
                detection_counts.append(len(detections))
                
                for detection in detections:
                    all_detections.append(detection)
                    confidence_scores.append(detection['confidence'])
            
            # Calculate statistics
            analysis = {
                'total_detections': len(all_detections),
                'images_with_detections': sum(1 for count in detection_counts if count > 0),
                'images_without_detections': sum(1 for count in detection_counts if count == 0),
                'average_detections_per_image': np.mean(detection_counts) if detection_counts else 0,
                'max_detections_per_image': max(detection_counts) if detection_counts else 0,
                'confidence_statistics': {
                    'mean': np.mean(confidence_scores) if confidence_scores else 0,
                    'std': np.std(confidence_scores) if confidence_scores else 0,
                    'min': np.min(confidence_scores) if confidence_scores else 0,
                    'max': np.max(confidence_scores) if confidence_scores else 0
                },
                'class_distribution': self._analyze_class_distribution(all_detections)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze detections: {e}")
            return {}
    
    def _analyze_class_distribution(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze class distribution of detections."""
        try:
            class_counts = {}
            
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            return class_counts
            
        except Exception as e:
            logger.error(f"Failed to analyze class distribution: {e}")
            return {}
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze processing performance."""
        try:
            performance = {
                'total_processing_time': self.statistics['processing_time'],
                'average_time_per_image': self.statistics['average_inference_time'],
                'images_per_second': self.statistics['processed_images'] / self.statistics['processing_time'] if self.statistics['processing_time'] > 0 else 0,
                'success_rate': self.statistics['processed_images'] / self.statistics['total_images'] if self.statistics['total_images'] > 0 else 0,
                'failure_rate': self.statistics['failed_images'] / self.statistics['total_images'] if self.statistics['total_images'] > 0 else 0
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            return {}
    
    def _analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        try:
            if not self.results:
                return {}
            
            # Collect image data
            image_shapes = []
            image_sizes = []
            
            for result in self.results:
                shape = result.get('image_shape', (0, 0, 0))
                image_shapes.append(shape)
                image_sizes.append(shape[0] * shape[1] if len(shape) >= 2 else 0)
            
            # Calculate statistics
            dataset_analysis = {
                'total_images': len(image_shapes),
                'image_shape_statistics': {
                    'common_width': max(set(shape[1] for shape in image_shapes if len(shape) >= 2), key=lambda x: list(shape[1] for shape in image_shapes if len(shape) >= 2).count(x)) if image_shapes else 0,
                    'common_height': max(set(shape[0] for shape in image_shapes if len(shape) >= 2), key=lambda x: list(shape[0] for shape in image_shapes if len(shape) >= 2).count(x)) if image_shapes else 0,
                    'average_width': np.mean([shape[1] for shape in image_shapes if len(shape) >= 2]) if image_shapes else 0,
                    'average_height': np.mean([shape[0] for shape in image_shapes if len(shape) >= 2]) if image_shapes else 0
                },
                'image_size_statistics': {
                    'mean': np.mean(image_sizes) if image_sizes else 0,
                    'std': np.std(image_sizes) if image_sizes else 0,
                    'min': np.min(image_sizes) if image_sizes else 0,
                    'max': np.max(image_sizes) if image_sizes else 0
                }
            }
            
            return dataset_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze dataset: {e}")
            return {}


def main():
    """Main batch inference function."""
    parser = argparse.ArgumentParser(description='Batch Inference for Kidney Stone Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='data/data.yaml',
                       help='Path to data.yaml configuration file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for inference')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of worker processes')
    parser.add_argument('--save-annotations', action='store_true',
                       help='Save annotated images')
    parser.add_argument('--export-formats', nargs='+', default=['json', 'csv'],
                       help='Export formats for results')
    
    args = parser.parse_args()
    
    try:
        # Initialize batch inference
        batch_inference = BatchInference(
            model_path=args.model,
            config_path=args.config,
            device=args.device,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Process dataset
        logger.info("Starting batch inference...")
        results = batch_inference.process_dataset(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            save_annotations=args.save_annotations,
            save_results=True,
            export_formats=args.export_formats
        )
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH INFERENCE - PROCESSING SUMMARY")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Device: {args.device}")
        print(f"Batch Size: {args.batch_size}")
        print("-"*60)
        
        statistics = batch_inference.statistics
        print(f"Total Images: {statistics['total_images']}")
        print(f"Processed Images: {statistics['processed_images']}")
        print(f"Failed Images: {statistics['failed_images']}")
        print(f"Total Detections: {statistics['total_detections']}")
        print(f"Processing Time: {statistics['processing_time']:.2f}s")
        print(f"Average Time per Image: {statistics['average_inference_time']:.3f}s")
        
        if 'detection_analysis' in results:
            detection_analysis = results['detection_analysis']
            print(f"Images with Detections: {detection_analysis.get('images_with_detections', 0)}")
            print(f"Average Detections per Image: {detection_analysis.get('average_detections_per_image', 0):.2f}")
            
            confidence_stats = detection_analysis.get('confidence_statistics', {})
            print(f"Average Confidence: {confidence_stats.get('mean', 0):.3f}")
        
        if 'performance_analysis' in results:
            performance = results['performance_analysis']
            print(f"Images per Second: {performance.get('images_per_second', 0):.2f}")
            print(f"Success Rate: {performance.get('success_rate', 0)*100:.1f}%")
        
        print("="*60)
        
        logger.info("Batch inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()






