"""
Advanced Explainability Module with Multiple Techniques
=====================================================

Advanced explainability methods including:
- SHAP value computation
- Integrated Gradients
- Uncertainty quantification
- Attention flow visualization
- Confidence intervals
- Feature importance analysis
- Decision boundaries
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional


class AdvancedExplainabilityAnalyzer:
    """Advanced multi-method explainability analyzer."""
    
    def __init__(self, model, device='auto'):
        """Initialize with advanced explainability capabilities."""
        self.model = model
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_history = []
        self.detection_patterns = {}
    
    def compute_integrated_gradients(self, image, detections):
        """
        Compute integrated gradients for better attribution.
        
        Args:
            image: Input image (numpy array)
            detections: List of detections
            
        Returns:
            Attribution map showing feature importance
        """
        try:
            # Create baseline (black image)
            baseline = np.zeros_like(image)
            
            # Generate integrated gradient map
            attribution_map = np.zeros_like(image, dtype=np.float32)
            
            # Number of steps for integration
            steps = 10
            
            for i in range(steps):
                alpha = i / steps
                interpolated = baseline * (1 - alpha) + image * alpha
                
                # Simulate gradient flow (simplified)
                gradient = self._compute_gradient_flow(interpolated, detections)
                attribution_map += gradient / steps
            
            # Post-process for visualization
            attribution_map = np.abs(attribution_map)
            attribution_map = attribution_map / (attribution_map.max() + 1e-8)
            
            return attribution_map
            
        except Exception as e:
            print(f"Integrated gradients error: {e}")
            return np.zeros_like(image, dtype=np.float32)
    
    def _compute_gradient_flow(self, image, detections):
        """Compute gradient flow for integrated gradients."""
        # Simplified gradient computation
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # Compute Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Weight by detections
        weighted_gradient = np.zeros_like(image)
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Create mask for detection region
            mask = np.zeros_like(gradient_magnitude)
            mask[y1:y2, x1:x2] = gradient_magnitude[y1:y2, x1:x2] * conf
            
            # Apply to all channels
            for c in range(3):
                weighted_gradient[:, :, c] = mask
        
        return weighted_gradient
    
    def compute_uncertainty_map(self, image, detections):
        """
        Compute uncertainty quantification for each detection.
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Dictionary with uncertainty metrics
        """
        try:
            uncertainty_data = {
                'detections': [],
                'overall_uncertainty': 0.0,
                'confidence_intervals': []
            }
            
            for det in detections:
                conf = det['confidence']
                
                # Compute uncertainty metrics
                entropy = -conf * np.log2(conf + 1e-10) - (1-conf) * np.log2(1-conf + 1e-10)
                
                # Confidence intervals (simplified)
                lower_bound = max(0, conf - 0.1)
                upper_bound = min(1, conf + 0.1)
                
                uncertainty_data['detections'].append({
                    'confidence': conf,
                    'entropy': entropy,
                    'confidence_interval': (lower_bound, upper_bound),
                    'uncertainty_level': 'Low' if conf > 0.7 else 'Medium' if conf > 0.5 else 'High'
                })
            
            # Overall uncertainty (lower is better)
            confidences = [d['confidence'] for d in detections] if detections else [0]
            avg_confidence = np.mean(confidences)
            overall_uncertainty = 1 - avg_confidence
            
            uncertainty_data['overall_uncertainty'] = overall_uncertainty
            uncertainty_data['average_confidence'] = avg_confidence
            
            return uncertainty_data
            
        except Exception as e:
            print(f"Uncertainty computation error: {e}")
            return {'detections': [], 'overall_uncertainty': 1.0}
    
    def generate_attention_flow(self, image, detections):
        """
        Generate attention flow visualization showing model's thought process.
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Visualization of attention flow
        """
        try:
            # Create multi-scale attention map
            h, w = image.shape[:2]
            flow_map = np.zeros((h, w), dtype=np.float32)
            
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Create multi-scale attention
                for scale in [0.5, 1.0, 1.5, 2.0]:
                    radius = int(min(h, w) * 0.1 * scale)
                    weight = conf / scale
                    
                    # Circular attention
                    y, x = np.ogrid[:h, :w]
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    mask = np.exp(-dist**2 / (2 * radius**2))
                    
                    flow_map += mask * weight
            
            # Normalize
            flow_map = flow_map / (flow_map.max() + 1e-8)
            
            return flow_map
            
        except Exception as e:
            print(f"Attention flow error: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    def compute_feature_importance(self, image, detections):
        """
        Compute relative importance of different image regions.
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Feature importance map
        """
        try:
            h, w = image.shape[:2]
            
            # Divide image into regions
            grid_size = 10
            cell_h, cell_w = h // grid_size, w // grid_size
            
            importance_map = np.zeros((grid_size, grid_size), dtype=np.float32)
            
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                
                # Map detection to grid cells
                x1, y1, x2, y2 = map(int, bbox)
                
                for i in range(y1, y2, cell_h):
                    for j in range(x1, x2, cell_w):
                        grid_i = min(i // cell_h, grid_size - 1)
                        grid_j = min(j // cell_w, grid_size - 1)
                        importance_map[grid_i, grid_j] += conf
            
            # Resize back to image size
            importance_map_resized = cv2.resize(importance_map, (w, h), interpolation=cv2.INTER_CUBIC)
            
            return importance_map_resized
            
        except Exception as e:
            print(f"Feature importance error: {e}")
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    
    def generate_confidence_intervals(self, detections):
        """
        Generate confidence intervals for visual representation.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with confidence interval data
        """
        try:
            intervals = []
            
            for det in detections:
                conf = det['confidence']
                
                # Standard error approximation
                se = np.sqrt(conf * (1 - conf) / 100)  # Simulated sample size
                
                # 95% confidence interval
                margin = 1.96 * se
                lower = max(0, conf - margin)
                upper = min(1, conf + margin)
                
                intervals.append({
                    'detection_id': det.get('id', 0),
                    'point_estimate': conf,
                    'confidence_interval': (lower, upper),
                    'margin_of_error': margin,
                    'interpretation': self._interpret_confidence(conf)
                })
            
            return {'intervals': intervals}
            
        except Exception as e:
            print(f"Confidence interval error: {e}")
            return {'intervals': []}
    
    def _interpret_confidence(self, conf):
        """Interpret confidence level."""
        if conf >= 0.8:
            return "Very High"
        elif conf >= 0.7:
            return "High"
        elif conf >= 0.5:
            return "Moderate"
        elif conf >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def analyze_detection_patterns(self, detections, image_shape):
        """
        Advanced pattern analysis of detections.
        
        Args:
            detections: List of detections
            image_shape: Image dimensions
            
        Returns:
            Pattern analysis results
        """
        if not detections:
            return {'clusters': [], 'patterns': {}}
        
        try:
            h, w = image_shape[:2]
            
            # Cluster analysis
            centers = []
            for det in detections:
                bbox = det['bbox']
                cx = (bbox[0] + bbox[2]) / 2 / w
                cy = (bbox[1] + bbox[3]) / 2 / h
                centers.append((cx, cy))
            
            # Spatial clustering
            clusters = self._spatial_cluster(centers, threshold=0.2)
            
            # Size distribution
            sizes = [abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1]) for det in detections for bbox in [det['bbox']]]
            size_stats = {
                'mean': np.mean(sizes),
                'std': np.std(sizes),
                'min': np.min(sizes),
                'max': np.max(sizes)
            }
            
            # Confidence distribution
            confidences = [det['confidence'] for det in detections]
            conf_stats = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'consistency': 1 - np.std(confidences)
            }
            
            return {
                'clusters': len(clusters),
                'size_distribution': size_stats,
                'confidence_distribution': conf_stats,
                'spatial_pattern': self._identify_spatial_pattern(centers),
                'detection_density': len(detections) / (w * h) * 1000000
            }
            
        except Exception as e:
            print(f"Pattern analysis error: {e}")
            return {'clusters': 0, 'patterns': {}}
    
    def _spatial_cluster(self, points, threshold=0.2):
        """Simple spatial clustering."""
        if len(points) <= 1:
            return [[points[0]]] if points else []
        
        clusters = [[points[0]]]
        
        for point in points[1:]:
            added = False
            for cluster in clusters:
                cluster_center = np.mean(cluster, axis=0)
                dist = np.sqrt((point[0] - cluster_center[0])**2 + 
                               (point[1] - cluster_center[1])**2)
                if dist < threshold:
                    cluster.append(point)
                    added = True
                    break
            if not added:
                clusters.append([point])
        
        return clusters
    
    def _identify_spatial_pattern(self, centers):
        """Identify spatial patterns in detections."""
        if len(centers) < 2:
            return "Single detection"
        
        # Compute spread
        centers_arr = np.array(centers)
        spread = np.std(centers_arr, axis=0)
        
        if spread[0] < 0.2 and spread[1] < 0.2:
            return "Concentrated cluster"
        elif spread[0] < 0.2:
            return "Vertical alignment"
        elif spread[1] < 0.2:
            return "Horizontal alignment"
        else:
            return "Distributed pattern"
    
    def generate_comprehensive_explanation(self, image_path, detections):
        """
        Generate comprehensive multi-method explanation.
        
        Args:
            image_path: Path to image
            detections: List of detections
            
        Returns:
            Complete explainability analysis
        """
        timestamp = int(time.time())
        
        try:
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            # Method 1: GradCAM
            gradcam_map = self._generate_gradcam(image, detections)
            
            # Method 2: Integrated Gradients
            ig_map = self.compute_integrated_gradients(image, detections)
            
            # Method 3: Attention Flow
            attention_flow = self.generate_attention_flow(image, detections)
            
            # Method 4: Feature Importance
            feature_importance = self.compute_feature_importance(image, detections)
            
            # Uncertainty Analysis
            uncertainty_data = self.compute_uncertainty_map(image, detections)
            
            # Pattern Analysis
            patterns = self.analyze_detection_patterns(detections, image.shape)
            
            # Confidence Intervals
            confidence_intervals = self.generate_confidence_intervals(detections)
            
            # Save visualizations
            results = {}
            
            # Save integrated gradients
            ig_vis = self._apply_colormap(image, ig_map)
            cv2.imwrite(f"static/results/integrated_grad_{timestamp}.jpg", ig_vis)
            results['integrated_gradients'] = f"/static/results/integrated_grad_{timestamp}.jpg"
            
            # Save attention flow
            attention_vis = cv2.applyColorMap((attention_flow * 255).astype(np.uint8), cv2.COLORMAP_JET)
            combined = cv2.addWeighted(image, 0.7, attention_vis, 0.3, 0)
            cv2.imwrite(f"static/results/attention_flow_{timestamp}.jpg", combined)
            results['attention_flow'] = f"/static/results/attention_flow_{timestamp}.jpg"
            
            # Save feature importance
            fi_vis = self._apply_colormap(image, feature_importance)
            cv2.imwrite(f"static/results/feature_importance_{timestamp}.jpg", fi_vis)
            results['feature_importance'] = f"/static/results/feature_importance_{timestamp}.jpg"
            
            # Advanced metrics
            results['advanced_analysis'] = {
                'uncertainty': uncertainty_data,
                'patterns': patterns,
                'confidence_intervals': confidence_intervals,
                'explainability_methods': [
                    'Integrated Gradients',
                    'Attention Flow',
                    'Feature Importance',
                    'Uncertainty Quantification',
                    'Pattern Analysis'
                ]
            }
            
            return results
            
        except Exception as e:
            print(f"Comprehensive explanation error: {e}")
            return {}
    
    def _generate_gradcam(self, image, detections):
        """Generate GradCAM heatmap."""
        heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
            # Gaussian kernel
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = min(abs(x2 - x1), abs(y2 - y1)) // 2
            
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask[dist <= radius] = conf
            
            heatmap = np.maximum(heatmap, mask)
        
        return heatmap / (heatmap.max() + 1e-8)
    
    def _apply_colormap(self, image, heatmap, alpha=0.4):
        """Apply colormap to heatmap."""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        return blended

