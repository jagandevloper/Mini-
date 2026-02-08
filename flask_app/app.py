#!/usr/bin/env python3
"""
Kidney Stone Detection Flask Web Application - Enhanced with Advanced Explainability
====================================================================================

A production-ready Flask application for kidney stone detection using YOLOv8
with advanced multi-level explainability and clinical prognosis.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import io
import base64
import json
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging

# Import advanced explainability modules
from utils.explainability import ExplainabilityAnalyzer
try:
    from utils.advanced_explainability import AdvancedExplainabilityAnalyzer
    ADVANCED_EXPLAINABILITY = True
except ImportError:
    ADVANCED_EXPLAINABILITY = False
    AdvancedExplainabilityAnalyzer = None

# Import clinical relevance analyzer
try:
    from utils.clinical_relevance import ClinicalRelevanceAnalyzer
    CLINICAL_RELEVANCE = True
except ImportError:
    CLINICAL_RELEVANCE = False
    ClinicalRelevanceAnalyzer = None

# Import production features
try:
    from production import setup_production, prod_logger, perf_monitor, timing_middleware
    PRODUCTION_MODE = True
except ImportError:
    PRODUCTION_MODE = False
    timing_middleware = lambda f: f

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Global variables
detector = None
explainability_analyzer = None
advanced_explainability_analyzer = None

# Initialize clinical relevance analyzer if available
if CLINICAL_RELEVANCE:
    clinical_relevance_analyzer = ClinicalRelevanceAnalyzer()
else:
    clinical_relevance_analyzer = None


class LightweightKidneyStoneDetector:
    """Lightweight and optimized kidney stone detector with explainability."""
    
    def __init__(self, model_path: str, device: str = 'auto', enable_half_precision: bool = True):
        """Initialize the detector with optimizations."""
        self.model_path = model_path
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_half_precision = enable_half_precision and (self.device == 'cuda')
        self.load_model()
        self.inference_stats = {
            'total_detections': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
    
    def load_model(self):
        """Load and optimize the YOLOv8 model."""
        try:
            logger.info(f"Loading optimized model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Move to device
            self.model.to(self.device)
            
            # Enable half precision for faster inference on GPU
            # BUT convert inputs properly to avoid dtype mismatch
            if self.enable_half_precision and self.device == 'cuda':
                try:
                    # Convert model to half precision
                    self.model.model = self.model.model.half()
                    logger.info("✅ Enabled FP16 (half precision) for faster inference")
                except Exception as e:
                    logger.warning(f"Failed to enable FP16, using FP32: {e}")
                    self.enable_half_precision = False
            
            # Optimize model for inference
            if self.device == 'cuda':
                torch.backends.cudnn.benchmark = True
                logger.info(f"✅ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("⚠️ Using CPU")
            
            logger.info("✅ Model loaded and optimized successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def optimize_for_realtime(self):
        """Apply real-time optimizations."""
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.set_num_threads(4)
    
    def detect_image(self, image_path: str, conf_threshold: float = 0.25):
        """Detect kidney stones in an image with real-time optimizations."""
        try:
            start_time = time.time()
            
            # For FP16 models, ensure dtype consistency
            # YOLOv8 handles this internally, but we ensure proper device placement
            results = self.model(
                image_path, 
                conf=conf_threshold, 
                device=self.device, 
                imgsz=640,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            result = results[0]
            detections = result.boxes
            
            detection_data = {
                'inference_time': round(inference_time * 1000, 1),
                'detections': [],
                'image_size': result.orig_shape,
                'model_confidence': conf_threshold
            }
            
            if detections is not None and len(detections) > 0:
                for i, detection in enumerate(detections):
                    conf = detection.conf.item()
                    bbox = detection.xyxy[0].cpu().numpy()
                    
                    detection_data['detections'].append({
                        'id': i + 1,
                        'bbox': bbox.tolist(),
                        'confidence': round(conf, 3),
                        'class': 'kidney_stone'
                    })
            
            # Save result image
            result_image_path = save_result_image(result)
            
            detection_data['result_image'] = result_image_path
            detection_data['success'] = True
            
            # Update stats
            self.inference_stats['total_detections'] += 1
            self.inference_stats['total_time'] += inference_time
            self.inference_stats['avg_time'] = self.inference_stats['total_time'] / self.inference_stats['total_detections']
            
            return detection_data
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class KidneyStoneDetector:
    """Backward compatibility wrapper."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize the detector."""
        self.detector = LightweightKidneyStoneDetector(model_path, device)
        self.model = self.detector.model
        self.model_path = self.detector.model_path
        self.device = self.detector.device
    
    def detect_image(self, image_path: str, conf_threshold: float = 0.25):
        """Detect kidney stones in an image."""
        return self.detector.detect_image(image_path, conf_threshold)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_result_image(result):
    """Save result image and return path."""
    try:
        timestamp = int(time.time())
        filename = f"result_{timestamp}.jpg"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        result.save(filepath)
        return f"/static/results/{filename}"
    except Exception as e:
        logger.error(f"Failed to save result image: {e}")
        return None


# Routes
@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests."""
    return '', 204  # Return 204 No Content to suppress favicon 404 errors

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
@timing_middleware
def upload():
    """Upload and detect kidney stones."""
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    confidence = float(request.form.get('confidence', 0.25))
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run detection
        try:
            results = detector.detect_image(filepath, confidence)
            
            # Record metrics
            if PRODUCTION_MODE:
                perf_monitor.record_detection()
            
            # Remove uploaded file
            os.remove(filepath)
            
            return jsonify(results)
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/explain', methods=['POST'])
@timing_middleware
def explain():
    """Advanced multi-level explainability analysis with enhanced methods."""
    global explainability_analyzer, advanced_explainability_analyzer
    
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run detection
            results = detector.detect_image(filepath, conf_threshold=0.25)
            
            # Initialize explainability analyzers if not already done
            if explainability_analyzer is None:
                explainability_analyzer = ExplainabilityAnalyzer(detector.model, detector.device)
            
            # Initialize advanced explainability if available
            if ADVANCED_EXPLAINABILITY and advanced_explainability_analyzer is None:
                advanced_explainability_analyzer = AdvancedExplainabilityAnalyzer(
                    detector.model, detector.device
                )
            
            # Generate advanced multi-level explainability
            if results['success']:
                try:
                    # Standard explainability
                    multi_level_explanation = explainability_analyzer.generate_multi_level_explanation(
                        filepath, results['detections']
                    )
                    
                    # Advanced explainability (if available)
                    advanced_explanation = {}
                    if ADVANCED_EXPLAINABILITY and advanced_explainability_analyzer:
                        try:
                            advanced_explanation = advanced_explainability_analyzer.generate_comprehensive_explanation(
                                filepath, results['detections']
                            )
                        except Exception as e:
                            logger.warning(f"Advanced explainability failed: {e}")
                    
                    # Clinical relevance analysis (if available)
                    clinical_relevance = {}
                    clinical_report = {}
                    if CLINICAL_RELEVANCE and clinical_relevance_analyzer is not None:
                        try:
                            # Get image for shape
                            import cv2
                            image = cv2.imread(filepath)
                            if image is not None:
                                image_shape = image.shape
                                
                                clinical_relevance = clinical_relevance_analyzer.analyze_clinical_relevance(
                                    results['detections'], image_shape
                                )
                                
                                clinical_report = clinical_relevance_analyzer.generate_clinical_report(
                                    results['detections'], image_shape
                                )
                        except Exception as e:
                            logger.warning(f"Clinical relevance analysis failed: {e}")
                    else:
                        logger.info("Clinical relevance analyzer not available")
                    
                    # Record explainability usage
                    if PRODUCTION_MODE:
                        perf_monitor.record_explainability()
                    
                    # Combine with original results
                    explainability_data = {
                        'success': True,
                        'filename': filename,
                        'detection_results': results,
                        'multi_level_explanation': multi_level_explanation,
                        'advanced_explanation': advanced_explanation if advanced_explanation else None,
                        'clinical_relevance': clinical_relevance if clinical_relevance else None,
                        'clinical_report': clinical_report if clinical_report else None,
                        'explainability_methods_used': (
                            'advanced' if advanced_explanation else 'standard'
                        )
                    }
                except Exception as explain_error:
                    logger.warning(f"Explainability generation failed: {explain_error}, falling back to basic analysis")
                    explainability_data = {
                        'success': True,
                        'filename': filename,
                        'detection_results': results,
                        'multi_level_explanation': {},
                        'advanced_explanation': None,
                        'explainability_methods_used': 'basic'
                    }
            else:
                explainability_data = {
                    'success': False,
                    'error': results.get('error', 'Detection failed')
                }
            
            # Remove uploaded file
            os.remove(filepath)
            
            return jsonify(explainability_data)
            
        except Exception as e:
            logger.error(f"Explainability error: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


def generate_explainability(image_path, detection_results):
    """Generate explainability analysis."""
    try:
        # Read image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Extract detection data
        detections = detection_results.get('detections', [])
        image_size = detection_results.get('image_size', [height, width])
        
        # Generate anatomical focus analysis
        anatomical_focus = analyze_anatomical_regions(detections, image_size)
        
        # Calculate confidence scores
        confidence_scores = calculate_confidence_metrics(detections)
        
        # Generate all visualizations
        attention_image = generate_attention_overlay(image, detections)
        heatmap_image = generate_heatmap_visualization(image, detections)
        clinical_image = generate_clinical_analysis(detections, image)
        
        return {
            'success': True,
            'filename': os.path.basename(image_path),
            'analysis': {
                'detection_count': len(detections),
                'confidence_avg': sum([d['confidence'] for d in detections]) / len(detections) if detections else 0,
                'anatomical_focus': anatomical_focus,
                'confidence_scores': confidence_scores
            },
            'visualizations': {
                'attention_overlay': attention_image,
                'attention_analysis': heatmap_image,
                'clinical_relevance': clinical_image
            }
        }
        
    except Exception as e:
        logger.error(f"Explainability generation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def analyze_anatomical_regions(detections, image_size):
    """Analyze anatomical regions where detections occurred."""
    if not detections:
        return {
            'primary_focus_region': 'No detections',
            'focus_score': 0.0,
            'focus_quality': 'Low',
            'clinical_relevance': 'No kidney stones detected'
        }
    
    # Calculate center of detections
    centers = []
    for det in detections:
        bbox = det['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        centers.append((cx, cy))
    
    if centers:
        avg_x = sum([c[0] for c in centers]) / len(centers)
        avg_y = sum([c[1] for c in centers]) / len(centers)
        
        # Determine region based on position
        width, height = image_size[1], image_size[0]
        
        if avg_x < width * 0.4:
            region = "Left Kidney Region"
        elif avg_x > width * 0.6:
            region = "Right Kidney Region"
        else:
            region = "Central/Bladder Region"
        
        # Calculate average confidence
        avg_conf = sum([d['confidence'] for d in detections]) / len(detections)
        
        # Determine focus quality
        if avg_conf > 0.7:
            quality = "High"
        elif avg_conf > 0.5:
            quality = "Moderate"
        else:
            quality = "Low"
        
        # Determine clinical relevance
        relevance = "High" if avg_conf > 0.6 else "Moderate" if avg_conf > 0.4 else "Needs Review"
        
        return {
            'primary_focus_region': region,
            'focus_score': avg_conf,
            'focus_quality': quality,
            'clinical_relevance': relevance,
            'detection_count': len(detections)
        }
    
    return {
        'primary_focus_region': 'Unknown',
        'focus_score': 0.0,
        'focus_quality': 'Low',
        'clinical_relevance': 'Unknown'
    }


def calculate_confidence_metrics(detections):
    """Calculate confidence metrics for different layers."""
    if not detections:
        return {}
    
    confidences = [d['confidence'] for d in detections]
    
    return {
        'detection_layer': sum(confidences) / len(confidences) if confidences else 0,
        'post_processing': max(confidences) if confidences else 0,
        'overall_confidence': sum(confidences) / len(confidences) if confidences else 0
    }


def generate_attention_overlay(image, detections):
    """Generate attention overlay visualization."""
    try:
        overlay = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            if conf > 0.7:
                color = (0, 255, 0)
                thickness = 4
            elif conf > 0.5:
                color = (0, 165, 255)
                thickness = 3
            else:
                color = (0, 0, 255)
                thickness = 2
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            bar_height = int((y2 - y1) * conf)
            cv2.rectangle(overlay, (x1, y2 - bar_height), (x1 + 10, y2), color, -1)
            
            label = f"Stone: {conf:.1%}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        timestamp = int(time.time())
        filename = f"attention_{timestamp}.jpg"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        cv2.imwrite(filepath, overlay)
        
        return f"/static/results/{filename}"
        
    except Exception as e:
        logger.error(f"Failed to generate attention overlay: {e}")
        return None


def generate_heatmap_visualization(image, detections):
    """Generate heatmap visualization showing attention regions."""
    try:
        heatmap = np.zeros_like(image, dtype=np.float32)
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            width = int((x2 - x1) / 2)
            height = int((y2 - y1) / 2)
            
            cv2.ellipse(mask, (cx, cy), (width, height), 0, 0, 360, conf, -1)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            mask_region = mask[y1:y2, x1:x2]
            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], mask_region[:, :, np.newaxis] if len(image.shape) == 3 else mask_region)
        
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        heatmap_colored = cv2.applyColorMap((heatmap[:, :, 0] * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        
        timestamp = int(time.time())
        filename = f"heatmap_{timestamp}.jpg"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        cv2.imwrite(filepath, blended)
        
        return f"/static/results/{filename}"
        
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}")
        return None


def generate_clinical_analysis(detections, image):
    """Generate clinical analysis visualization."""
    try:
        analysis_img = image.copy()
        height, width = image.shape[:2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        num_detections = len(detections)
        avg_conf = sum([d['confidence'] for d in detections]) / len(detections) if detections else 0
        high_conf_count = sum([1 for d in detections if d['confidence'] > 0.7])
        
        y_offset = 30
        line_height = 40
        
        cv2.putText(analysis_img, "Clinical Analysis", (20, y_offset), 
                   font, 1.2, (0, 255, 255), thickness)
        
        y_offset += line_height * 2
        
        cv2.putText(analysis_img, f"Stones Detected: {num_detections}", (20, y_offset), 
                   font, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        cv2.putText(analysis_img, f"Avg Confidence: {avg_conf:.1%}", (20, y_offset), 
                   font, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        cv2.putText(analysis_img, f"High Confidence: {high_conf_count}", (20, y_offset), 
                   font, font_scale, (0, 255, 0), thickness)
        
        timestamp = int(time.time())
        filename = f"clinical_{timestamp}.jpg"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        cv2.imwrite(filepath, analysis_img)
        
        return f"/static/results/{filename}"
        
    except Exception as e:
        logger.error(f"Failed to generate clinical analysis: {e}")
        return None


@app.route('/explainability_info')
def explainability_info():
    """Get explainability information."""
    return jsonify({
        'available': True,
        'standard_features': [
            'Attention overlay visualization',
            'Anatomical region analysis',
            'Confidence score breakdown',
            'Clinical relevance assessment'
        ],
        'advanced_features': [
            'Integrated Gradients',
            'Attention Flow Analysis',
            'Feature Importance Maps',
            'Uncertainty Quantification',
            'Pattern Recognition'
        ],
        'advanced_available': ADVANCED_EXPLAINABILITY
    })


@app.route('/batch', methods=['POST'])
def batch():
    """Batch detection."""
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    confidence = float(request.form.get('confidence', 0.25))
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                result = detector.detect_image(filepath, confidence)
                result['filename'] = filename
                results.append(result)
            except Exception as e:
                logger.error(f"Batch detection error for {filename}: {e}")
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    return jsonify({
        'success': True,
        'results': results
    })


@app.route('/model_info')
def model_info():
    """Get model information with performance metrics."""
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    
    model_size = os.path.getsize(detector.model_path) / (1024 * 1024)
    stats = getattr(detector.detector, 'inference_stats', {}) if hasattr(detector, 'detector') else {}
    
    return jsonify({
        'device': detector.device,
        'cuda_available': cuda_available,
        'gpu_name': gpu_name,
        'model_size': round(model_size, 2),
        'model_path': str(detector.model_path),
        'status': 'Ready',
        'optimizations': {
            'half_precision': getattr(detector.detector, 'enable_half_precision', False) if hasattr(detector, 'detector') else False,
            'real_time_optimized': True
        },
        'performance': {
            'avg_inference_time_ms': round(stats.get('avg_time', 0) * 1000, 1),
            'total_detections': stats.get('total_detections', 0)
        },
        'explainability': {
            'standard_available': True,
            'advanced_available': ADVANCED_EXPLAINABILITY,
            'methods_count': 8 if ADVANCED_EXPLAINABILITY else 4
        }
    })


if __name__ == '__main__':
    # Setup production features if available
    if PRODUCTION_MODE:
        app = setup_production(app)
        logger.info("✅ Production mode enabled")
    
    # Get model path from environment or use default
    model_path = os.environ.get('MODEL_PATH')
    
    # Try multiple possible locations
    if not model_path or not os.path.exists(model_path):
        possible_paths = [
            'models/best.pt',
            '../runs/kidney_stone_cuda_success/weights/best.pt',
            'runs/kidney_stone_cuda_success/weights/best.pt',
            '../kidney_stone_detection_project/experiments/kidney_stone_cuda_test/weights/best.pt',
            'experiments/kidney_stone_cuda_test/weights/best.pt'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"✅ Found model at: {path}")
                break
        else:
            logger.error("Could not find model file!")
            print("❌ Model file not found. Please ensure best.pt is in models/ folder")
            model_path = None
    
    try:
        # Initialize detector
        if model_path and os.path.exists(model_path):
            detector = KidneyStoneDetector(model_path)
            logger.info("Detector initialized successfully")
        else:
            logger.error("Model file not found")
            detector = None
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        print(f"❌ Error loading model: {e}")
        detector = None
    
    print("\n🌐 Starting Kidney Stone Detection Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    if detector:
        print("✅ Model loaded successfully")
        print(f"🔧 Device: {detector.device}")
        print(f"📁 Model: {model_path}")
        if ADVANCED_EXPLAINABILITY:
            print("🧠 Advanced explainability available!")
    else:
        print("⚠️  Model not loaded - some features may not work")
    
    try:
        print("\n⏳ Starting web server (this may take a moment)...")
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False, 
            use_reloader=False, 
            threaded=True,
            use_evalex=False
        )
    except KeyboardInterrupt:
        print("\n✅ Application stopped gracefully")
    except Exception as e:
        logger.error(f"Error running Flask app: {e}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
