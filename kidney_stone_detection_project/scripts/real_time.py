"""
Real-time Kidney Stone Detection Pipeline
=========================================

This module provides real-time inference capabilities for kidney stone detection
from webcam feeds, video files, and live medical imaging streams. It includes
optimized preprocessing, efficient inference, and real-time visualization.

Key Features:
- Real-time webcam detection
- Video file processing
- Live medical imaging stream support
- Optimized inference pipeline
- Real-time visualization with annotations
- Performance monitoring
- Clinical workflow integration

Author: [Your Name]
Date: 2024
License: MIT
"""

import os
import sys
import cv2
import torch
import numpy as np
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import argparse
from datetime import datetime
import json
import yaml
from collections import deque
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
        logging.FileHandler('realtime_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealTimeDetector:
    """
    Real-time kidney stone detection system.
    
    This class provides optimized real-time inference for kidney stone detection
    with support for webcam feeds, video files, and medical imaging streams.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = 'auto',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 max_fps: int = 30,
                 buffer_size: int = 5):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to data.yaml configuration
            device: Device to use for inference
            confidence_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            max_fps: Maximum FPS for processing
            buffer_size: Size of frame buffer
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_fps = max_fps
        self.buffer_size = buffer_size
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize preprocessor
        self.preprocessor = MedicalImagePreprocessor()
        
        # Performance monitoring
        self.performance_stats = {
            'fps': deque(maxlen=30),
            'inference_times': deque(maxlen=30),
            'frame_times': deque(maxlen=30),
            'detection_counts': deque(maxlen=30)
        }
        
        # Detection history
        self.detection_history = deque(maxlen=100)
        
        # Initialize visualization
        self._setup_visualization()
        
        logger.info(f"RealTimeDetector initialized with model: {model_path}")
    
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
            
            # Optimize for inference
            if device == 'cuda':
                model.model.half()  # Use half precision for speed
            
            logger.info(f"Model loaded on {device} with optimizations")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_visualization(self):
        """Setup visualization parameters."""
        self.class_names = self.config.get('names', ['kidney_stone'])
        self.colors = {
            'kidney_stone': (0, 255, 0),  # Green for kidney stones
            'background': (255, 255, 255),  # White background
            'text': (0, 0, 0),  # Black text
            'info': (255, 0, 0)  # Red for info
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        logger.info("Visualization setup completed")
    
    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], Dict[str, Any]]:
        """
        Detect kidney stones in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (annotated_frame, detections, performance_info)
        """
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                stream=False
            )
            
            # Extract detections
            detections = self._extract_detections(results[0])
            
            # Annotate frame
            annotated_frame = self._annotate_frame(frame, detections)
            
            # Calculate performance metrics
            inference_time = time.time() - start_time
            performance_info = self._update_performance_stats(inference_time, len(detections))
            
            # Store detection history
            self._store_detection_history(detections, inference_time)
            
            return annotated_frame, detections, performance_info
            
        except Exception as e:
            logger.error(f"Frame detection failed: {e}")
            return frame, [], {'error': str(e)}
    
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
                        'class_name': self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else f'Class_{class_ids[i]}',
                        'timestamp': time.time()
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Failed to extract detections: {e}")
            return []
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Annotate frame with detection results."""
        try:
            annotated_frame = frame.copy()
            
            # Draw detections
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Get color for class
                color = self.colors.get(class_name, self.colors['kidney_stone'])
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f'{class_name}: {confidence:.2f}'
                label_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
                
                # Draw label background
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                          self.font, self.font_scale, self.colors['text'], self.font_thickness)
            
            # Draw performance info
            annotated_frame = self._draw_performance_info(annotated_frame)
            
            # Draw detection history
            annotated_frame = self._draw_detection_history(annotated_frame)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Failed to annotate frame: {e}")
            return frame
    
    def _draw_performance_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance information on frame."""
        try:
            # Calculate current FPS
            current_fps = len(self.performance_stats['fps'])
            if current_fps > 0:
                avg_fps = sum(self.performance_stats['fps']) / current_fps
            else:
                avg_fps = 0
            
            # Calculate average inference time
            if self.performance_stats['inference_times']:
                avg_inference_time = sum(self.performance_stats['inference_times']) / len(self.performance_stats['inference_times'])
            else:
                avg_inference_time = 0
            
            # Draw performance info
            info_text = [
                f"FPS: {avg_fps:.1f}",
                f"Inference: {avg_inference_time*1000:.1f}ms",
                f"Device: {self.device}",
                f"Confidence: {self.confidence_threshold}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(frame, text, (10, y_offset), 
                          self.font, self.font_scale, self.colors['info'], self.font_thickness)
                y_offset += 25
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to draw performance info: {e}")
            return frame
    
    def _draw_detection_history(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection history visualization."""
        try:
            if not self.detection_history:
                return frame
            
            # Create detection count over time
            recent_detections = list(self.detection_history)[-30:]  # Last 30 frames
            
            if len(recent_detections) > 1:
                # Draw detection count graph
                graph_width = 200
                graph_height = 100
                graph_x = frame.shape[1] - graph_width - 10
                graph_y = 10
                
                # Draw graph background
                cv2.rectangle(frame, (graph_x, graph_y), 
                            (graph_x + graph_width, graph_y + graph_height), 
                            (50, 50, 50), -1)
                
                # Draw detection counts
                max_detections = max(recent_detections) if recent_detections else 1
                for i, count in enumerate(recent_detections):
                    if i > 0:
                        x1 = graph_x + int((i - 1) * graph_width / len(recent_detections))
                        x2 = graph_x + int(i * graph_width / len(recent_detections))
                        y1 = graph_y + graph_height - int(count * graph_height / max_detections)
                        y2 = graph_y + graph_height - int(count * graph_height / max_detections)
                        
                        cv2.line(frame, (x1, y1), (x2, y2), self.colors['kidney_stone'], 2)
                
                # Draw graph label
                cv2.putText(frame, "Detections", (graph_x, graph_y - 5), 
                          self.font, 0.5, self.colors['info'], 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to draw detection history: {e}")
            return frame
    
    def _update_performance_stats(self, inference_time: float, detection_count: int) -> Dict[str, Any]:
        """Update performance statistics."""
        try:
            current_time = time.time()
            
            # Update FPS
            if hasattr(self, '_last_frame_time'):
                fps = 1.0 / (current_time - self._last_frame_time)
                self.performance_stats['fps'].append(fps)
            
            self._last_frame_time = current_time
            
            # Update other stats
            self.performance_stats['inference_times'].append(inference_time)
            self.performance_stats['detection_counts'].append(detection_count)
            
            # Calculate averages
            avg_fps = sum(self.performance_stats['fps']) / len(self.performance_stats['fps']) if self.performance_stats['fps'] else 0
            avg_inference_time = sum(self.performance_stats['inference_times']) / len(self.performance_stats['inference_times']) if self.performance_stats['inference_times'] else 0
            avg_detections = sum(self.performance_stats['detection_counts']) / len(self.performance_stats['detection_counts']) if self.performance_stats['detection_counts'] else 0
            
            return {
                'fps': avg_fps,
                'inference_time': avg_inference_time,
                'detection_count': detection_count,
                'avg_detections': avg_detections,
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Failed to update performance stats: {e}")
            return {}
    
    def _store_detection_history(self, detections: List[Dict], inference_time: float):
        """Store detection history for analysis."""
        try:
            self.detection_history.append(len(detections))
            
            # Store detailed detection info
            if detections:
                detection_info = {
                    'timestamp': time.time(),
                    'detection_count': len(detections),
                    'inference_time': inference_time,
                    'detections': detections
                }
                
                # Keep only recent history
                if len(self.detection_history) > 1000:
                    self.detection_history.popleft()
            
        except Exception as e:
            logger.error(f"Failed to store detection history: {e}")
    
    def run_webcam(self, camera_index: int = 0, save_output: bool = False, output_path: str = None):
        """
        Run real-time detection on webcam feed.
        
        Args:
            camera_index: Camera index (0 for default camera)
            save_output: Whether to save output video
            output_path: Path to save output video
        """
        logger.info(f"Starting webcam detection with camera index: {camera_index}")
        
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_index}")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, self.max_fps)
            
            # Initialize video writer if saving
            writer = None
            if save_output:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_size = (640, 480)
                output_path = output_path or f"webcam_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
                logger.info(f"Output video will be saved to: {output_path}")
            
            # Main detection loop
            frame_count = 0
            start_time = time.time()
            
            print("Starting webcam detection...")
            print("Press 'q' to quit, 's' to save screenshot, 'r' to reset stats")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Run detection
                annotated_frame, detections, performance_info = self.detect_frame(frame)
                
                # Display frame
                cv2.imshow('Kidney Stone Detection - Real-time', annotated_frame)
                
                # Save frame if writer is available
                if writer is not None:
                    writer.write(annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    # Reset performance stats
                    self.performance_stats = {
                        'fps': deque(maxlen=30),
                        'inference_times': deque(maxlen=30),
                        'frame_times': deque(maxlen=30),
                        'detection_counts': deque(maxlen=30)
                    }
                    logger.info("Performance stats reset")
                
                frame_count += 1
                
                # Print periodic stats
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    logger.info(f"Processed {frame_count} frames, Current FPS: {current_fps:.1f}")
            
            # Cleanup
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            logger.info(f"Webcam detection completed. Total frames: {frame_count}, Average FPS: {avg_fps:.1f}")
            
        except Exception as e:
            logger.error(f"Webcam detection failed: {e}")
            raise
    
    def run_video(self, video_path: str, save_output: bool = False, output_path: str = None):
        """
        Run detection on video file.
        
        Args:
            video_path: Path to input video file
            save_output: Whether to save output video
            output_path: Path to save output video
        """
        logger.info(f"Starting video detection: {video_path}")
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties: {width}x{height}, {fps} FPS, {frame_count_total} frames")
            
            # Initialize video writer if saving
            writer = None
            if save_output:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path = output_path or f"video_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                logger.info(f"Output video will be saved to: {output_path}")
            
            # Process video
            frame_count = 0
            start_time = time.time()
            
            print(f"Processing video: {video_path}")
            print("Press 'q' to quit, 'p' to pause/resume")
            
            paused = False
            
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run detection
                    annotated_frame, detections, performance_info = self.detect_frame(frame)
                    
                    # Display frame
                    cv2.imshow('Kidney Stone Detection - Video', annotated_frame)
                    
                    # Save frame if writer is available
                    if writer is not None:
                        writer.write(annotated_frame)
                    
                    frame_count += 1
                    
                    # Print progress
                    if frame_count % 30 == 0:
                        progress = (frame_count / frame_count_total) * 100
                        elapsed_time = time.time() - start_time
                        current_fps = frame_count / elapsed_time
                        logger.info(f"Progress: {progress:.1f}%, FPS: {current_fps:.1f}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"Video {'paused' if paused else 'resumed'}")
            
            # Cleanup
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            logger.info(f"Video processing completed. Processed {frame_count}/{frame_count_total} frames, Average FPS: {avg_fps:.1f}")
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
    
    def run_image_batch(self, image_paths: List[str], save_output: bool = False, output_dir: str = None):
        """
        Run detection on batch of images.
        
        Args:
            image_paths: List of image paths
            save_output: Whether to save annotated images
            output_dir: Directory to save annotated images
        """
        logger.info(f"Starting batch image detection: {len(image_paths)} images")
        
        try:
            # Create output directory if saving
            if save_output:
                output_dir = Path(output_dir) if output_dir else Path(f"batch_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Annotated images will be saved to: {output_dir}")
            
            # Process images
            start_time = time.time()
            results = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.warning(f"Failed to load image: {image_path}")
                        continue
                    
                    # Run detection
                    annotated_image, detections, performance_info = self.detect_frame(image)
                    
                    # Save annotated image if requested
                    if save_output:
                        output_path = output_dir / f"annotated_{Path(image_path).name}"
                        cv2.imwrite(str(output_path), annotated_image)
                    
                    # Store results
                    result = {
                        'image_path': image_path,
                        'detections': detections,
                        'performance': performance_info,
                        'timestamp': time.time()
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {e}")
                    continue
            
            # Save results summary
            if save_output:
                results_path = output_dir / 'detection_results.json'
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Detection results saved to: {results_path}")
            
            # Print summary
            total_time = time.time() - start_time
            avg_time_per_image = total_time / len(image_paths) if image_paths else 0
            total_detections = sum(len(result['detections']) for result in results)
            
            logger.info(f"Batch processing completed:")
            logger.info(f"  Images processed: {len(results)}/{len(image_paths)}")
            logger.info(f"  Total detections: {total_detections}")
            logger.info(f"  Average time per image: {avg_time_per_image:.3f}s")
            logger.info(f"  Total processing time: {total_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch image processing failed: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            summary = {
                'fps_stats': {
                    'current': list(self.performance_stats['fps'])[-1] if self.performance_stats['fps'] else 0,
                    'average': sum(self.performance_stats['fps']) / len(self.performance_stats['fps']) if self.performance_stats['fps'] else 0,
                    'min': min(self.performance_stats['fps']) if self.performance_stats['fps'] else 0,
                    'max': max(self.performance_stats['fps']) if self.performance_stats['fps'] else 0
                },
                'inference_stats': {
                    'average': sum(self.performance_stats['inference_times']) / len(self.performance_stats['inference_times']) if self.performance_stats['inference_times'] else 0,
                    'min': min(self.performance_stats['inference_times']) if self.performance_stats['inference_times'] else 0,
                    'max': max(self.performance_stats['inference_times']) if self.performance_stats['inference_times'] else 0
                },
                'detection_stats': {
                    'total_detections': sum(self.performance_stats['detection_counts']),
                    'average_per_frame': sum(self.performance_stats['detection_counts']) / len(self.performance_stats['detection_counts']) if self.performance_stats['detection_counts'] else 0,
                    'frames_with_detections': sum(1 for count in self.performance_stats['detection_counts'] if count > 0)
                },
                'model_info': {
                    'model_path': str(self.model_path),
                    'device': self.device,
                    'confidence_threshold': self.confidence_threshold,
                    'iou_threshold': self.iou_threshold
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}


def main():
    """Main real-time inference function."""
    parser = argparse.ArgumentParser(description='Real-time Kidney Stone Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='data/data.yaml',
                       help='Path to data.yaml configuration file')
    parser.add_argument('--source', type=str, default='0',
                       help='Source: camera index (0), video path, or image directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for inference')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold')
    parser.add_argument('--max-fps', type=int, default=30,
                       help='Maximum FPS for processing')
    parser.add_argument('--save-output', action='store_true',
                       help='Save output video/images')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Output path for saved files')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = RealTimeDetector(
            model_path=args.model,
            config_path=args.config,
            device=args.device,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            max_fps=args.max_fps
        )
        
        # Determine source type and run appropriate detection
        source = args.source
        
        if source.isdigit():
            # Camera input
            camera_index = int(source)
            logger.info(f"Starting webcam detection with camera {camera_index}")
            detector.run_webcam(
                camera_index=camera_index,
                save_output=args.save_output,
                output_path=args.output_path
            )
        
        elif Path(source).is_file():
            # Video file input
            logger.info(f"Starting video detection: {source}")
            detector.run_video(
                video_path=source,
                save_output=args.save_output,
                output_path=args.output_path
            )
        
        elif Path(source).is_dir():
            # Image directory input
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(Path(source).glob(f'*{ext}'))
                image_paths.extend(Path(source).glob(f'*{ext.upper()}'))
            
            if image_paths:
                logger.info(f"Starting batch image detection: {len(image_paths)} images")
                detector.run_image_batch(
                    image_paths=[str(p) for p in image_paths],
                    save_output=args.save_output,
                    output_dir=args.output_path
                )
            else:
                logger.error(f"No images found in directory: {source}")
        
        else:
            logger.error(f"Invalid source: {source}")
            return
        
        # Print final performance summary
        performance_summary = detector.get_performance_summary()
        print("\n" + "="*60)
        print("REAL-TIME DETECTION - PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Source: {source}")
        print(f"Device: {args.device}")
        print("-"*60)
        
        fps_stats = performance_summary.get('fps_stats', {})
        print(f"Average FPS: {fps_stats.get('average', 0):.1f}")
        print(f"FPS Range: {fps_stats.get('min', 0):.1f} - {fps_stats.get('max', 0):.1f}")
        
        inference_stats = performance_summary.get('inference_stats', {})
        print(f"Average Inference Time: {inference_stats.get('average', 0)*1000:.1f}ms")
        
        detection_stats = performance_summary.get('detection_stats', {})
        print(f"Total Detections: {detection_stats.get('total_detections', 0)}")
        print(f"Average Detections per Frame: {detection_stats.get('average_per_frame', 0):.1f}")
        
        print("="*60)
        
        logger.info("Real-time detection completed successfully!")
        
    except Exception as e:
        logger.error(f"Real-time detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()






