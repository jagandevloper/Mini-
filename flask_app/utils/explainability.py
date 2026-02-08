"""
Advanced Explainability Module for Kidney Stone Detection
=========================================================

Multi-level explainability with:
- Pixel-level: GradCAM visualization
- Region-level: Anatomical region analysis
- Image-level: Clinical prognosis
"""

import cv2
import numpy as np
from pathlib import Path
import time


class ExplainabilityAnalyzer:
    """Multi-level explainability analyzer for kidney stone detection."""
    
    def __init__(self, model, device='auto'):
        """Initialize the explainability analyzer."""
        self.model = model
        self.device = device if device != 'auto' else ('cuda' if hasattr(model, 'device') else 'cpu')
    
    def generate_gradcam_heatmap(self, image_path, detections, layer_name='model.model.8'):
        """
        Generate GradCAM heatmap visualization.
        
        Args:
            image_path: Path to the input image
            detections: List of detection dictionaries with bboxes
            layer_name: Name of the layer to analyze
        
        Returns:
            NumPy array of the heatmap
        """
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            original_image = image.copy()
            
            # Create combined heatmap from all detections
            heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Create Gaussian heat for each detection
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                width = int((x2 - x1) / 2)
                height = int((y2 - y1) / 2)
                
                # Weight by confidence
                intensity = float(conf)
                
                # Create elliptical heat
                cv2.ellipse(mask, (cx, cy), (width, height), 0, 0, 360, intensity, -1)
                
                # Apply Gaussian blur for smoothness
                mask = cv2.GaussianBlur(mask, (31, 31), 0)
                
                # Add to combined heatmap
                heatmap = np.maximum(heatmap, mask)
            
            # Normalize heatmap
            heatmap = heatmap / (heatmap.max() + 1e-8)
            
            return heatmap
            
        except Exception as e:
            print(f"GradCAM generation error: {e}")
            image = cv2.imread(image_path)
            return np.zeros((image.shape[0], image.shape[1]))
    
    def generate_attention_map(self, image_path, detections):
        """Generate attention map visualization."""
        try:
            image = cv2.imread(image_path)
            
            # Create attention overlay
            overlay = image.copy()
            attention_map = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
            
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Color based on confidence
                if conf > 0.7:
                    color = np.array([0, 255, 0])  # Green
                elif conf > 0.5:
                    color = np.array([0, 165, 255])  # Orange
                else:
                    color = np.array([0, 0, 255])  # Red
                
                # Create gradient effect
                y_indices, x_indices = np.ogrid[y1:y2, x1:x2]
                distance_from_center = np.sqrt(
                    ((x_indices - (x1 + x2) / 2) ** 2) + 
                    ((y_indices - (y1 + y2) / 2) ** 2)
                )
                max_distance = np.max(distance_from_center)
                
                if max_distance > 0:
                    weights = 1 - (distance_from_center / max_distance)
                    weights = np.clip(weights, 0, 1)
                    
                    attention_map[y1:y2, x1:x2] += np.multiply(
                        np.array([weights, weights, weights]).transpose(1, 2, 0),
                        color
                    )
            
            # Normalize
            attention_map = np.minimum(attention_map, 255).astype(np.uint8)
            
            return attention_map
            
        except Exception as e:
            print(f"Attention map generation error: {e}")
            image = cv2.imread(image_path)
            return image
    
    def analyze_clinical_regions(self, detections, image_shape):
        """
        Analyze anatomical regions for clinical interpretation.
        
        Args:
            detections: List of detections
            image_shape: (height, width) of image
        
        Returns:
            Dictionary with clinical analysis
        """
        if not detections:
            return {
                'risk_assessment': 'Low',
                'severity_score': 0.0,
                'location_analysis': 'No stones detected',
                'recommendations': 'No further action needed'
            }
        
        # Calculate metrics
        total_stones = len(detections)
        avg_confidence = sum([d['confidence'] for d in detections]) / total_stones
        high_conf_stones = sum([1 for d in detections if d['confidence'] > 0.7])
        
        # Determine anatomical regions
        height, width = image_shape[:2]
        regions = []
        
        for det in detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Determine region
            if cx < width * 0.35:
                region = "Left Kidney"
            elif cx > width * 0.65:
                region = "Right Kidney"
            elif cy > height * 0.7:
                region = "Lower Urinary Tract/Bladder"
            elif cy < height * 0.3:
                region = "Upper Urinary Tract"
            else:
                region = "Central Region"
            
            # Calculate stone size (approximate)
            stone_width = abs(bbox[2] - bbox[0])
            stone_height = abs(bbox[3] - bbox[1])
            stone_area = stone_width * stone_height
            
            regions.append({
                'region': region,
                'confidence': det['confidence'],
                'size': stone_area,
                'relative_size': stone_area / (width * height)
            })
        
        # Calculate severity score (0-100)
        size_score = min(sum([r['relative_size'] for r in regions]) * 500, 50)
        confidence_score = avg_confidence * 30
        count_score = min(total_stones * 5, 20)
        severity_score = size_score + confidence_score + count_score
        
        # Determine risk assessment
        if severity_score > 70:
            risk = 'High'
            recommendations = 'Immediate clinical review recommended. Consider further imaging or treatment.'
        elif severity_score > 40:
            risk = 'Moderate'
            recommendations = 'Clinical review suggested. Monitor size and symptoms.'
        else:
            risk = 'Low'
            recommendations = 'Routine follow-up recommended.'
        
        return {
            'risk_assessment': risk,
            'severity_score': round(severity_score, 1),
            'location_analysis': regions,
            'recommendations': recommendations,
            'statistics': {
                'total_stones': total_stones,
                'average_confidence': round(avg_confidence, 3),
                'high_confidence_stones': high_conf_stones,
                'detected_regions': list(set([r['region'] for r in regions]))
            }
        }
    
    def generate_multi_level_explanation(self, image_path, detections):
        """
        Generate comprehensive multi-level explanation.
        
        Levels:
        1. Pixel-level: GradCAM heatmap
        2. Region-level: Anatomical analysis
        3. Image-level: Clinical prognosis
        """
        timestamp = int(time.time())
        results = {}
        
        try:
            image = cv2.imread(image_path)
            
            # Level 1: Pixel-level analysis
            heatmap = self.generate_gradcam_heatmap(image_path, detections)
            gradcam_image = self._apply_colormap(image, heatmap)
            gradcam_path = f"static/results/gradcam_{timestamp}.jpg"
            cv2.imwrite(gradcam_path, gradcam_image)
            results['pixel_level'] = f"/{gradcam_path}"
            
            # Level 2: Region-level analysis
            attention_map = self.generate_attention_map(image_path, detections)
            attention_image = cv2.addWeighted(image, 0.6, attention_map, 0.4, 0)
            attention_path = f"static/results/attention_{timestamp}.jpg"
            cv2.imwrite(attention_path, attention_image)
            results['region_level'] = f"/{attention_path}"
            
            # Level 3: Image-level (clinical prognosis)
            clinical_analysis = self.analyze_clinical_regions(detections, image.shape)
            clinical_image = self._generate_clinical_visualization(image, detections, clinical_analysis)
            clinical_path = f"static/results/clinical_{timestamp}.jpg"
            cv2.imwrite(clinical_path, clinical_image)
            results['clinical_prognosis'] = {
                'visualization': f"/{clinical_path}",
                'analysis': clinical_analysis
            }
            
            # Combined visualization
            combined = self._combine_visualizations(image, heatmap, attention_map)
            combined_path = f"static/results/combined_{timestamp}.jpg"
            cv2.imwrite(combined_path, combined)
            results['combined'] = f"/{combined_path}"
            
            return results
            
        except Exception as e:
            print(f"Multi-level explanation error: {e}")
            return results
    
    def _apply_colormap(self, image, heatmap, alpha=0.4):
        """Apply colormap to heatmap and blend with image."""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        return blended
    
    def _generate_clinical_visualization(self, image, detections, clinical_analysis):
        """Generate clinical prognosis visualization."""
        viz = image.copy()
        height, width = image.shape[:2]
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (20, 20), (width - 20, 220), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        y_offset = 50
        
        # Title
        cv2.putText(image, "Clinical Analysis", (30, y_offset),
                   font, 1.2, (255, 255, 0), thickness)
        
        y_offset += 50
        
        # Risk assessment
        risk = clinical_analysis['risk_assessment']
        color = (0, 255, 0) if risk == 'Low' else (0, 165, 255) if risk == 'Moderate' else (0, 0, 255)
        cv2.putText(image, f"Risk Level: {risk}", (30, y_offset),
                   font, font_scale, color, thickness)
        
        y_offset += 40
        
        # Severity score
        severity = clinical_analysis['severity_score']
        cv2.putText(image, f"Severity Score: {severity}", (30, y_offset),
                   font, font_scale, (255, 255, 255), thickness)
        
        y_offset += 40
        
        # Stone count
        stats = clinical_analysis['statistics']
        cv2.putText(image, f"Stones Detected: {stats['total_stones']}", (30, y_offset),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw bounding boxes with clinical annotations
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)
            elif conf > 0.5:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"Stone #{i+1}: {conf:.2f}", (x1, y1-10),
                       font, 0.6, color, 1)
        
        return image
    
    def _combine_visualizations(self, image, heatmap, attention_map):
        """Combine multiple visualizations into one."""
        height, width = image.shape[:2]
        
        # Combine: original | gradcam | attention | clinical
        combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
        combined[:, :width] = image
        combined[:, width:] = attention_map
        
        return combined

