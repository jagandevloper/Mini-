"""
Grad-CAM Explainability for Kidney Stone Detection
==================================================

This module provides comprehensive explainability analysis for kidney stone
detection models using Grad-CAM (Gradient-weighted Class Activation Mapping)
and other interpretability techniques. It helps radiologists understand where
the model focuses when making predictions.

Key Features:
- Grad-CAM visualization for YOLOv8 models
- Attention heatmap generation
- Multi-layer analysis
- Clinical interpretability
- Batch processing capabilities
- Publication-ready visualizations

Author: [Your Name]
Date: 2024
License: MIT
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import argparse
from PIL import Image, ImageDraw, ImageFont
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# YOLOv8 imports
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Grad-CAM imports
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    logger.warning("pytorch_grad_cam not installed. Install with: pip install grad-cam")
    GradCAM = None

# Custom imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.preprocessing import MedicalImagePreprocessor
from utils.visualization import TrainingVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('explainability.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class YOLOv8GradCAM:
    """
    Grad-CAM implementation for YOLOv8 models.
    
    This class provides Grad-CAM visualization specifically adapted for
    YOLOv8 object detection models, enabling explainability analysis
    for kidney stone detection.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = 'auto',
                 target_layers: Optional[List[str]] = None):
        """
        Initialize YOLOv8 Grad-CAM analyzer.
        
        Args:
            model_path: Path to trained YOLOv8 model
            config_path: Path to data.yaml configuration
            device: Device to use for computation
            target_layers: List of target layer names for Grad-CAM
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = device
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        
        # Setup target layers
        self.target_layers = target_layers or self._get_default_target_layers()
        
        # Initialize Grad-CAM
        self._setup_gradcam()
        
        # Initialize preprocessor
        self.preprocessor = MedicalImagePreprocessor()
        
        logger.info(f"YOLOv8GradCAM initialized with target layers: {self.target_layers}")
    
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
        """Load YOLOv8 model."""
        try:
            model = YOLO(str(self.model_path))
            
            # Set device
            if self.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device
            
            model.to(device)
            logger.info(f"Model loaded on {device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_default_target_layers(self) -> List[str]:
        """Get default target layers for Grad-CAM."""
        # YOLOv8-nano default target layers
        return [
            "model.22.cv2.conv",  # Last detection layer
            "model.22.cv3.conv",  # Last detection layer
            "model.21.cv2.conv",  # Second to last detection layer
            "model.21.cv3.conv", # Second to last detection layer
        ]
    
    def _setup_gradcam(self):
        """Setup Grad-CAM for YOLOv8 model."""
        try:
            if GradCAM is None:
                logger.error("pytorch_grad_cam not available. Please install with: pip install grad-cam")
                return
            
            # Get the actual model (not the YOLO wrapper)
            model = self.model.model
            
            # Find target layers in the model
            self.gradcam_layers = []
            for layer_name in self.target_layers:
                try:
                    layer = self._find_layer_by_name(model, layer_name)
                    if layer is not None:
                        self.gradcam_layers.append(layer)
                        logger.info(f"Found target layer: {layer_name}")
                    else:
                        logger.warning(f"Layer not found: {layer_name}")
                except Exception as e:
                    logger.warning(f"Failed to find layer {layer_name}: {e}")
            
            if not self.gradcam_layers:
                logger.error("No valid target layers found for Grad-CAM")
                return
            
            # Initialize Grad-CAM
            self.gradcam = GradCAM(
                model=model,
                target_layers=self.gradcam_layers,
                use_cuda=(self.device == 'cuda')
            )
            
            logger.info(f"Grad-CAM initialized with {len(self.gradcam_layers)} target layers")
            
        except Exception as e:
            logger.error(f"Failed to setup Grad-CAM: {e}")
    
    def _find_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Find layer by name in the model."""
        try:
            # Split the layer name by dots
            parts = layer_name.split('.')
            current_module = model
            
            for part in parts:
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                else:
                    return None
            
            return current_module
        except Exception as e:
            logger.error(f"Error finding layer {layer_name}: {e}")
            return None
    
    def generate_gradcam(self, 
                        image: np.ndarray,
                        class_idx: int = 0,
                        use_eigen_smooth: bool = False,
                        aug_smooth: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for an image.
        
        Args:
            image: Input image as numpy array
            class_idx: Class index for Grad-CAM
            use_eigen_smooth: Whether to use eigen smoothing
            aug_smooth: Whether to use augmentation smoothing
            
        Returns:
            Dictionary of Grad-CAM heatmaps for each target layer
        """
        try:
            if not hasattr(self, 'gradcam'):
                logger.error("Grad-CAM not properly initialized")
                return {}
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Generate Grad-CAM
            gradcam_maps = {}
            
            for i, layer in enumerate(self.gradcam_layers):
                try:
                    # Create Grad-CAM for this layer
                    gradcam_layer = GradCAM(
                        model=self.model.model,
                        target_layers=[layer],
                        use_cuda=(self.device == 'cuda')
                    )
                    
                    # Generate heatmap
                    heatmap = gradcam_layer(
                        input_tensor=processed_image,
                        targets=[ClassifierOutputTarget(class_idx)],
                        eigen_smooth=use_eigen_smooth,
                        aug_smooth=aug_smooth
                    )
                    
                    # Convert to numpy
                    heatmap = heatmap[0]  # Remove batch dimension
                    gradcam_maps[f'layer_{i}_{layer.__class__.__name__}'] = heatmap
                    
                except Exception as e:
                    logger.error(f"Failed to generate Grad-CAM for layer {i}: {e}")
                    continue
            
            logger.info(f"Generated Grad-CAM for {len(gradcam_maps)} layers")
            return gradcam_maps
            
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM: {e}")
            return {}
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for Grad-CAM."""
        try:
            # Resize image
            resized = cv2.resize(image, (640, 640))
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            # Convert to tensor
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            
            if self.device == 'cuda':
                tensor = tensor.cuda()
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def visualize_gradcam(self, 
                         image: np.ndarray,
                         gradcam_maps: Dict[str, np.ndarray],
                         alpha: float = 0.4,
                         save_path: Optional[Path] = None,
                         show: bool = False) -> np.ndarray:
        """
        Visualize Grad-CAM heatmaps overlaid on the original image.
        
        Args:
            image: Original image
            gradcam_maps: Dictionary of Grad-CAM heatmaps
            alpha: Transparency for heatmap overlay
            save_path: Path to save visualization
            show: Whether to display the visualization
            
        Returns:
            Visualization image
        """
        try:
            # Resize original image to match Grad-CAM size
            resized_image = cv2.resize(image, (640, 640))
            
            # Create visualization
            num_layers = len(gradcam_maps)
            if num_layers == 0:
                logger.warning("No Grad-CAM maps to visualize")
                return resized_image
            
            # Create subplot layout
            cols = min(3, num_layers)
            rows = (num_layers + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            if num_layers == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Plot each Grad-CAM map
            for i, (layer_name, heatmap) in enumerate(gradcam_maps.items()):
                if i >= len(axes):
                    break
                
                # Normalize heatmap
                heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                
                # Apply colormap
                heatmap_colored = plt.cm.jet(heatmap_normalized)[:, :, :3]
                heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
                
                # Overlay on original image
                overlay = cv2.addWeighted(resized_image, 1-alpha, heatmap_colored, alpha, 0)
                
                # Plot
                axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[i].set_title(f'Grad-CAM: {layer_name}', fontsize=12, fontweight='bold')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_layers, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle('Grad-CAM Visualization - Kidney Stone Detection', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save visualization
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Grad-CAM visualization saved to: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return overlay
            
        except Exception as e:
            logger.error(f"Failed to visualize Grad-CAM: {e}")
            return image
    
    def analyze_attention_patterns(self, 
                                 gradcam_maps: Dict[str, np.ndarray],
                                 image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze attention patterns from Grad-CAM heatmaps.
        
        Args:
            gradcam_maps: Dictionary of Grad-CAM heatmaps
            image: Original image
            
        Returns:
            Analysis results
        """
        try:
            analysis = {
                'attention_regions': [],
                'confidence_scores': {},
                'spatial_distribution': {},
                'clinical_relevance': {}
            }
            
            for layer_name, heatmap in gradcam_maps.items():
                # Find attention peaks
                peaks = self._find_attention_peaks(heatmap)
                analysis['attention_regions'].extend(peaks)
                
                # Calculate confidence scores
                confidence = np.mean(heatmap)
                analysis['confidence_scores'][layer_name] = confidence
                
                # Analyze spatial distribution
                spatial_stats = self._analyze_spatial_distribution(heatmap)
                analysis['spatial_distribution'][layer_name] = spatial_stats
                
                # Assess clinical relevance
                clinical_relevance = self._assess_clinical_relevance(heatmap, image)
                analysis['clinical_relevance'][layer_name] = clinical_relevance
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze attention patterns: {e}")
            return {}
    
    def _find_attention_peaks(self, heatmap: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """Find attention peaks in the heatmap."""
        try:
            # Find local maxima
            from scipy.ndimage import maximum_filter
            local_maxima = maximum_filter(heatmap, size=20) == heatmap
            
            # Apply threshold
            peaks = []
            peak_coords = np.where((local_maxima) & (heatmap > threshold))
            
            for i in range(len(peak_coords[0])):
                y, x = peak_coords[0][i], peak_coords[1][i]
                intensity = heatmap[y, x]
                
                peaks.append({
                    'x': int(x),
                    'y': int(y),
                    'intensity': float(intensity),
                    'confidence': float(intensity)
                })
            
            return peaks
            
        except Exception as e:
            logger.error(f"Failed to find attention peaks: {e}")
            return []
    
    def _analyze_spatial_distribution(self, heatmap: np.ndarray) -> Dict[str, float]:
        """Analyze spatial distribution of attention."""
        try:
            # Calculate spatial statistics
            h, w = heatmap.shape
            
            # Center of mass
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            total_mass = np.sum(heatmap)
            
            if total_mass > 0:
                center_x = np.sum(x_coords * heatmap) / total_mass
                center_y = np.sum(y_coords * heatmap) / total_mass
            else:
                center_x, center_y = w/2, h/2
            
            # Spread
            spread_x = np.sqrt(np.sum(((x_coords - center_x) ** 2) * heatmap) / total_mass) if total_mass > 0 else 0
            spread_y = np.sqrt(np.sum(((y_coords - center_y) ** 2) * heatmap) / total_mass) if total_mass > 0 else 0
            
            return {
                'center_x': float(center_x),
                'center_y': float(center_y),
                'spread_x': float(spread_x),
                'spread_y': float(spread_y),
                'total_attention': float(total_mass),
                'max_intensity': float(np.max(heatmap)),
                'mean_intensity': float(np.mean(heatmap))
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze spatial distribution: {e}")
            return {}
    
    def _assess_clinical_relevance(self, heatmap: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
        """Assess clinical relevance of attention patterns."""
        try:
            # Analyze attention in different anatomical regions
            h, w = heatmap.shape
            
            # Define anatomical regions (simplified)
            regions = {
                'left_kidney': (0, 0, w//2, h//2),
                'right_kidney': (w//2, 0, w, h//2),
                'bladder': (w//4, h//2, 3*w//4, h),
                'ureters': (w//4, h//4, 3*w//4, 3*h//4)
            }
            
            region_attention = {}
            for region_name, (x1, y1, x2, y2) in regions.items():
                region_heatmap = heatmap[y1:y2, x1:x2]
                region_attention[region_name] = {
                    'mean_attention': float(np.mean(region_heatmap)),
                    'max_attention': float(np.max(region_heatmap)),
                    'attention_coverage': float(np.sum(region_heatmap > 0.5) / region_heatmap.size)
                }
            
            # Overall clinical assessment
            max_region = max(region_attention.items(), key=lambda x: x[1]['mean_attention'])
            
            return {
                'region_attention': region_attention,
                'primary_attention_region': max_region[0],
                'clinical_confidence': max_region[1]['mean_attention'],
                'attention_focus': 'focused' if max_region[1]['attention_coverage'] < 0.3 else 'diffuse'
            }
            
        except Exception as e:
            logger.error(f"Failed to assess clinical relevance: {e}")
            return {}


class ExplainabilityAnalyzer:
    """
    Comprehensive explainability analyzer for kidney stone detection.
    
    This class provides batch processing, comparison analysis, and
    clinical interpretation of explainability results.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 output_dir: str = 'explainability_results',
                 device: str = 'auto'):
        """
        Initialize explainability analyzer.
        
        Args:
            model_path: Path to trained model
            config_path: Path to data.yaml configuration
            output_dir: Output directory for results
            device: Device to use for computation
        """
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Grad-CAM analyzer
        self.gradcam_analyzer = YOLOv8GradCAM(
            model_path=model_path,
            config_path=config_path,
            device=device
        )
        
        # Initialize visualizer
        self.visualizer = TrainingVisualizer(save_path=self.output_dir)
        
        logger.info(f"ExplainabilityAnalyzer initialized with output_dir: {output_dir}")
    
    def analyze_batch(self, 
                     image_paths: List[str],
                     class_idx: int = 0,
                     save_individual: bool = True,
                     generate_summary: bool = True) -> Dict[str, Any]:
        """
        Analyze explainability for a batch of images.
        
        Args:
            image_paths: List of image paths to analyze
            class_idx: Class index for analysis
            save_individual: Whether to save individual results
            generate_summary: Whether to generate summary analysis
            
        Returns:
            Batch analysis results
        """
        try:
            logger.info(f"Starting batch analysis of {len(image_paths)} images")
            
            batch_results = {
                'individual_results': [],
                'summary_statistics': {},
                'attention_patterns': {},
                'clinical_insights': {}
            }
            
            for i, image_path in enumerate(image_paths):
                try:
                    logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.warning(f"Failed to load image: {image_path}")
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Generate Grad-CAM
                    gradcam_maps = self.gradcam_analyzer.generate_gradcam(
                        image=image,
                        class_idx=class_idx
                    )
                    
                    if not gradcam_maps:
                        logger.warning(f"No Grad-CAM maps generated for: {image_path}")
                        continue
                    
                    # Analyze attention patterns
                    attention_analysis = self.gradcam_analyzer.analyze_attention_patterns(
                        gradcam_maps=gradcam_maps,
                        image=image
                    )
                    
                    # Create individual result
                    individual_result = {
                        'image_path': image_path,
                        'gradcam_maps': gradcam_maps,
                        'attention_analysis': attention_analysis,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    batch_results['individual_results'].append(individual_result)
                    
                    # Save individual results if requested
                    if save_individual:
                        self._save_individual_result(individual_result, i)
                    
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {e}")
                    continue
            
            # Generate summary analysis
            if generate_summary:
                batch_results['summary_statistics'] = self._generate_summary_statistics(
                    batch_results['individual_results']
                )
                batch_results['attention_patterns'] = self._analyze_attention_patterns(
                    batch_results['individual_results']
                )
                batch_results['clinical_insights'] = self._generate_clinical_insights(
                    batch_results['individual_results']
                )
            
            # Save batch results
            self._save_batch_results(batch_results)
            
            logger.info(f"Batch analysis completed. Processed {len(batch_results['individual_results'])} images")
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return {}
    
    def _save_individual_result(self, result: Dict[str, Any], index: int):
        """Save individual analysis result."""
        try:
            # Create individual result directory
            individual_dir = self.output_dir / f'image_{index:03d}'
            individual_dir.mkdir(exist_ok=True)
            
            # Save Grad-CAM visualizations
            image = cv2.imread(result['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            visualization_path = individual_dir / 'gradcam_visualization.png'
            self.gradcam_analyzer.visualize_gradcam(
                image=image,
                gradcam_maps=result['gradcam_maps'],
                save_path=visualization_path,
                show=False
            )
            
            # Save analysis data
            analysis_path = individual_dir / 'attention_analysis.json'
            with open(analysis_path, 'w') as f:
                json.dump(result['attention_analysis'], f, indent=2)
            
            logger.debug(f"Individual result saved to: {individual_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save individual result: {e}")
    
    def _generate_summary_statistics(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from individual results."""
        try:
            if not individual_results:
                return {}
            
            # Collect statistics
            confidence_scores = []
            attention_regions = []
            clinical_confidences = []
            
            for result in individual_results:
                analysis = result.get('attention_analysis', {})
                
                # Collect confidence scores
                for layer_name, confidence in analysis.get('confidence_scores', {}).items():
                    confidence_scores.append(confidence)
                
                # Collect attention regions
                attention_regions.extend(analysis.get('attention_regions', []))
                
                # Collect clinical confidences
                for layer_name, clinical in analysis.get('clinical_relevance', {}).items():
                    clinical_confidences.append(clinical.get('clinical_confidence', 0))
            
            # Calculate summary statistics
            summary = {
                'total_images': len(individual_results),
                'confidence_statistics': {
                    'mean': float(np.mean(confidence_scores)) if confidence_scores else 0,
                    'std': float(np.std(confidence_scores)) if confidence_scores else 0,
                    'min': float(np.min(confidence_scores)) if confidence_scores else 0,
                    'max': float(np.max(confidence_scores)) if confidence_scores else 0
                },
                'attention_statistics': {
                    'total_regions': len(attention_regions),
                    'avg_regions_per_image': len(attention_regions) / len(individual_results) if individual_results else 0,
                    'avg_intensity': float(np.mean([r['intensity'] for r in attention_regions])) if attention_regions else 0
                },
                'clinical_statistics': {
                    'mean_clinical_confidence': float(np.mean(clinical_confidences)) if clinical_confidences else 0,
                    'std_clinical_confidence': float(np.std(clinical_confidences)) if clinical_confidences else 0
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary statistics: {e}")
            return {}
    
    def _analyze_attention_patterns(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze attention patterns across all images."""
        try:
            if not individual_results:
                return {}
            
            # Collect spatial distributions
            spatial_distributions = []
            region_attentions = []
            
            for result in individual_results:
                analysis = result.get('attention_analysis', {})
                
                # Collect spatial distributions
                for layer_name, spatial in analysis.get('spatial_distribution', {}).items():
                    spatial_distributions.append(spatial)
                
                # Collect region attentions
                for layer_name, clinical in analysis.get('clinical_relevance', {}).items():
                    region_attentions.append(clinical.get('region_attention', {}))
            
            # Analyze patterns
            patterns = {
                'spatial_patterns': {
                    'avg_center_x': float(np.mean([s.get('center_x', 0) for s in spatial_distributions])) if spatial_distributions else 0,
                    'avg_center_y': float(np.mean([s.get('center_y', 0) for s in spatial_distributions])) if spatial_distributions else 0,
                    'avg_spread': float(np.mean([s.get('spread_x', 0) + s.get('spread_y', 0) for s in spatial_distributions])) if spatial_distributions else 0
                },
                'region_preferences': self._analyze_region_preferences(region_attentions),
                'attention_consistency': self._analyze_attention_consistency(individual_results)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze attention patterns: {e}")
            return {}
    
    def _analyze_region_preferences(self, region_attentions: List[Dict]) -> Dict[str, float]:
        """Analyze which anatomical regions receive most attention."""
        try:
            region_totals = {}
            region_counts = {}
            
            for region_attention in region_attentions:
                for region_name, stats in region_attention.items():
                    if region_name not in region_totals:
                        region_totals[region_name] = 0
                        region_counts[region_name] = 0
                    
                    region_totals[region_name] += stats.get('mean_attention', 0)
                    region_counts[region_name] += 1
            
            # Calculate averages
            region_preferences = {}
            for region_name in region_totals:
                if region_counts[region_name] > 0:
                    region_preferences[region_name] = region_totals[region_name] / region_counts[region_name]
            
            return region_preferences
            
        except Exception as e:
            logger.error(f"Failed to analyze region preferences: {e}")
            return {}
    
    def _analyze_attention_consistency(self, individual_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze consistency of attention patterns across images."""
        try:
            # Collect attention intensities for each layer
            layer_intensities = {}
            
            for result in individual_results:
                analysis = result.get('attention_analysis', {})
                confidence_scores = analysis.get('confidence_scores', {})
                
                for layer_name, confidence in confidence_scores.items():
                    if layer_name not in layer_intensities:
                        layer_intensities[layer_name] = []
                    layer_intensities[layer_name].append(confidence)
            
            # Calculate consistency metrics
            consistency = {}
            for layer_name, intensities in layer_intensities.items():
                if len(intensities) > 1:
                    consistency[layer_name] = {
                        'coefficient_of_variation': float(np.std(intensities) / np.mean(intensities)) if np.mean(intensities) > 0 else 0,
                        'consistency_score': float(1 - np.std(intensities)) if np.std(intensities) <= 1 else 0
                    }
            
            return consistency
            
        except Exception as e:
            logger.error(f"Failed to analyze attention consistency: {e}")
            return {}
    
    def _generate_clinical_insights(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate clinical insights from explainability analysis."""
        try:
            insights = {
                'model_reliability': {},
                'anatomical_focus': {},
                'clinical_recommendations': []
            }
            
            # Analyze model reliability
            confidence_scores = []
            clinical_confidences = []
            
            for result in individual_results:
                analysis = result.get('attention_analysis', {})
                
                for layer_name, confidence in analysis.get('confidence_scores', {}).items():
                    confidence_scores.append(confidence)
                
                for layer_name, clinical in analysis.get('clinical_relevance', {}).items():
                    clinical_confidences.append(clinical.get('clinical_confidence', 0))
            
            # Model reliability assessment
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            avg_clinical_confidence = np.mean(clinical_confidences) if clinical_confidences else 0
            
            insights['model_reliability'] = {
                'average_confidence': float(avg_confidence),
                'average_clinical_confidence': float(avg_clinical_confidence),
                'reliability_score': float((avg_confidence + avg_clinical_confidence) / 2),
                'assessment': 'High' if avg_confidence > 0.8 else 'Medium' if avg_confidence > 0.6 else 'Low'
            }
            
            # Anatomical focus analysis
            region_preferences = self._analyze_region_preferences([
                result.get('attention_analysis', {}).get('clinical_relevance', {}).get('region_attention', {})
                for result in individual_results
            ])
            
            insights['anatomical_focus'] = {
                'primary_focus_region': max(region_preferences.items(), key=lambda x: x[1])[0] if region_preferences else 'Unknown',
                'region_distribution': region_preferences,
                'focus_consistency': 'Consistent' if len(region_preferences) <= 2 else 'Distributed'
            }
            
            # Clinical recommendations
            if avg_confidence > 0.8:
                insights['clinical_recommendations'].append("Model shows high confidence - suitable for clinical use")
            elif avg_confidence > 0.6:
                insights['clinical_recommendations'].append("Model shows moderate confidence - use with caution")
            else:
                insights['clinical_recommendations'].append("Model shows low confidence - requires improvement")
            
            if avg_clinical_confidence > 0.7:
                insights['clinical_recommendations'].append("Attention patterns align well with clinical expectations")
            else:
                insights['clinical_recommendations'].append("Attention patterns may need clinical validation")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate clinical insights: {e}")
            return {}
    
    def _save_batch_results(self, batch_results: Dict[str, Any]):
        """Save batch analysis results."""
        try:
            # Save main results
            results_path = self.output_dir / 'batch_analysis_results.json'
            with open(results_path, 'w') as f:
                json.dump(batch_results, f, indent=2, default=str)
            
            # Generate summary report
            self._generate_summary_report(batch_results)
            
            logger.info(f"Batch results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")
    
    def _generate_summary_report(self, batch_results: Dict[str, Any]):
        """Generate human-readable summary report."""
        try:
            report_path = self.output_dir / 'summary_report.txt'
            
            with open(report_path, 'w') as f:
                f.write("KIDNEY STONE DETECTION - EXPLAINABILITY ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Summary statistics
                summary_stats = batch_results.get('summary_statistics', {})
                f.write(f"Total Images Analyzed: {summary_stats.get('total_images', 0)}\n")
                
                confidence_stats = summary_stats.get('confidence_statistics', {})
                f.write(f"Average Confidence: {confidence_stats.get('mean', 0):.3f}\n")
                f.write(f"Confidence Std Dev: {confidence_stats.get('std', 0):.3f}\n")
                
                # Clinical insights
                clinical_insights = batch_results.get('clinical_insights', {})
                model_reliability = clinical_insights.get('model_reliability', {})
                f.write(f"\nModel Reliability Assessment: {model_reliability.get('assessment', 'Unknown')}\n")
                f.write(f"Reliability Score: {model_reliability.get('reliability_score', 0):.3f}\n")
                
                # Recommendations
                recommendations = clinical_insights.get('clinical_recommendations', [])
                f.write("\nClinical Recommendations:\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"Summary report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")


def main():
    """Main explainability analysis function."""
    parser = argparse.ArgumentParser(description='Explainability Analysis for Kidney Stone Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='data/data.yaml',
                       help='Path to data.yaml configuration file')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                       help='Paths to images for analysis')
    parser.add_argument('--output-dir', type=str, default='explainability_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for computation')
    parser.add_argument('--class-idx', type=int, default=0,
                       help='Class index for analysis')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ExplainabilityAnalyzer(
            model_path=args.model,
            config_path=args.config,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # Run batch analysis
        logger.info("Starting explainability analysis...")
        results = analyzer.analyze_batch(
            image_paths=args.images,
            class_idx=args.class_idx,
            save_individual=True,
            generate_summary=True
        )
        
        # Print summary
        print("\n" + "="*60)
        print("KIDNEY STONE DETECTION - EXPLAINABILITY ANALYSIS")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Images Analyzed: {len(args.images)}")
        print(f"Output Directory: {args.output_dir}")
        print("-"*60)
        
        summary_stats = results.get('summary_statistics', {})
        print(f"Total Images Processed: {summary_stats.get('total_images', 0)}")
        
        confidence_stats = summary_stats.get('confidence_statistics', {})
        print(f"Average Confidence: {confidence_stats.get('mean', 0):.3f}")
        print(f"Confidence Range: {confidence_stats.get('min', 0):.3f} - {confidence_stats.get('max', 0):.3f}")
        
        clinical_insights = results.get('clinical_insights', {})
        model_reliability = clinical_insights.get('model_reliability', {})
        print(f"Model Reliability: {model_reliability.get('assessment', 'Unknown')}")
        print(f"Reliability Score: {model_reliability.get('reliability_score', 0):.3f}")
        
        print("-"*60)
        print("Analysis completed successfully!")
        print("Check the output directory for detailed results and visualizations.")
        print("="*60)
        
        logger.info("Explainability analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Explainability analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()






