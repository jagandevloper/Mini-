# Kidney Stone Detection - API Documentation
## Comprehensive API Reference

### 📚 Overview

This document provides detailed API documentation for all modules and functions in the Kidney Stone Detection project. Each function includes parameters, return values, examples, and usage notes.

---

## 🏗️ Core Modules

### 1. Training Module (`scripts/train.py`)

#### `KidneyStoneTrainer`

Main training class for kidney stone detection models.

```python
class KidneyStoneTrainer:
    def __init__(self, config_path: str, model_size: str = 'nano', 
                 device: str = 'auto', project_name: str = 'kidney_stone_detection')
```

**Parameters:**
- `config_path` (str): Path to data.yaml configuration file
- `model_size` (str): YOLOv8 model size ('nano', 'small', 'medium', 'large', 'xlarge')
- `device` (str): Device to use for training ('auto', 'cpu', 'cuda', 'mps')
- `project_name` (str): Name of the training project

**Example:**
```python
trainer = KidneyStoneTrainer(
    config_path='data/data.yaml',
    model_size='nano',
    device='cuda',
    project_name='my_kidney_stone_model'
)
```

#### `train()`

Train the kidney stone detection model.

```python
def train(self, resume: bool = False, resume_from: Optional[str] = None) -> Dict[str, Any]
```

**Parameters:**
- `resume` (bool): Whether to resume training from checkpoint
- `resume_from` (str, optional): Path to checkpoint to resume from

**Returns:**
- `Dict[str, Any]`: Training results dictionary

**Example:**
```python
results = trainer.train(resume=False)
print(f"Training completed with mAP@0.5: {results['mAP@0.5']}")
```

#### `evaluate()`

Evaluate the trained model on test data.

```python
def evaluate(self, weights_path: Optional[str] = None, 
            test_data: Optional[str] = None) -> Dict[str, Any]
```

**Parameters:**
- `weights_path` (str, optional): Path to model weights
- `test_data` (str, optional): Path to test data

**Returns:**
- `Dict[str, Any]`: Evaluation results dictionary

---

### 2. Evaluation Module (`scripts/evaluate.py`)

#### `KidneyStoneEvaluator`

Comprehensive evaluator for kidney stone detection models.

```python
class KidneyStoneEvaluator:
    def __init__(self, model_path: str, config_path: str, device: str = 'auto',
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45)
```

**Parameters:**
- `model_path` (str): Path to trained model weights
- `config_path` (str): Path to data.yaml configuration
- `device` (str): Device to use for evaluation
- `confidence_threshold` (float): Confidence threshold for predictions
- `iou_threshold` (float): IoU threshold for NMS

#### `evaluate_dataset()`

Evaluate model on specified dataset split.

```python
def evaluate_dataset(self, split: str = 'test', save_results: bool = True,
                    output_dir: Optional[str] = None) -> Dict[str, Any]
```

**Parameters:**
- `split` (str): Dataset split to evaluate ('test', 'val', 'train')
- `save_results` (bool): Whether to save detailed results
- `output_dir` (str, optional): Directory to save results

**Returns:**
- `Dict[str, Any]`: Comprehensive evaluation results

**Example:**
```python
evaluator = KidneyStoneEvaluator('models/best.pt', 'data/data.yaml')
results = evaluator.evaluate_dataset(split='test', save_results=True)
print(f"mAP@0.5: {results['performance_metrics']['mAP@0.5']}")
```

---

### 3. Real-time Detection Module (`scripts/real_time.py`)

#### `RealTimeDetector`

Real-time kidney stone detection system.

```python
class RealTimeDetector:
    def __init__(self, model_path: str, config_path: str, device: str = 'auto',
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                 max_fps: int = 30, buffer_size: int = 5)
```

**Parameters:**
- `model_path` (str): Path to trained model weights
- `config_path` (str): Path to data.yaml configuration
- `device` (str): Device to use for inference
- `confidence_threshold` (float): Confidence threshold for predictions
- `iou_threshold` (float): IoU threshold for NMS
- `max_fps` (int): Maximum FPS for processing
- `buffer_size` (int): Size of frame buffer

#### `detect_frame()`

Detect kidney stones in a single frame.

```python
def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], Dict[str, Any]]
```

**Parameters:**
- `frame` (np.ndarray): Input frame as numpy array

**Returns:**
- `Tuple[np.ndarray, List[Dict], Dict[str, Any]]`: (annotated_frame, detections, performance_info)

**Example:**
```python
detector = RealTimeDetector('models/best.pt', 'data/data.yaml')
annotated_frame, detections, perf_info = detector.detect_frame(frame)
print(f"Found {len(detections)} kidney stones")
```

#### `run_webcam()`

Run real-time detection on webcam feed.

```python
def run_webcam(self, camera_index: int = 0, save_output: bool = False, 
              output_path: str = None)
```

**Parameters:**
- `camera_index` (int): Camera index (0 for default camera)
- `save_output` (bool): Whether to save output video
- `output_path` (str): Path to save output video

**Example:**
```python
detector.run_webcam(camera_index=0, save_output=True, output_path='output.avi')
```

#### `run_video()`

Run detection on video file.

```python
def run_video(self, video_path: str, save_output: bool = False, 
             output_path: str = None)
```

**Parameters:**
- `video_path` (str): Path to input video file
- `save_output` (bool): Whether to save output video
- `output_path` (str): Path to save output video

---

### 4. Explainability Module (`scripts/explainability.py`)

#### `YOLOv8GradCAM`

Grad-CAM implementation for YOLOv8 models.

```python
class YOLOv8GradCAM:
    def __init__(self, model_path: str, config_path: str, device: str = 'auto',
                 target_layers: Optional[List[str]] = None)
```

**Parameters:**
- `model_path` (str): Path to trained YOLOv8 model
- `config_path` (str): Path to data.yaml configuration
- `device` (str): Device to use for computation
- `target_layers` (List[str], optional): List of target layer names for Grad-CAM

#### `generate_gradcam()`

Generate Grad-CAM heatmaps for an image.

```python
def generate_gradcam(self, image: np.ndarray, class_idx: int = 0,
                    use_eigen_smooth: bool = False, aug_smooth: bool = False) -> Dict[str, np.ndarray]
```

**Parameters:**
- `image` (np.ndarray): Input image as numpy array
- `class_idx` (int): Class index for Grad-CAM
- `use_eigen_smooth` (bool): Whether to use eigen smoothing
- `aug_smooth` (bool): Whether to use augmentation smoothing

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary of Grad-CAM heatmaps for each target layer

**Example:**
```python
gradcam = YOLOv8GradCAM('models/best.pt', 'data/data.yaml')
heatmaps = gradcam.generate_gradcam(image, class_idx=0)
```

#### `ExplainabilityAnalyzer`

Comprehensive explainability analyzer for kidney stone detection.

```python
class ExplainabilityAnalyzer:
    def __init__(self, model_path: str, config_path: str, 
                 output_dir: str = 'explainability_results', device: str = 'auto')
```

#### `analyze_batch()`

Analyze explainability for a batch of images.

```python
def analyze_batch(self, image_paths: List[str], class_idx: int = 0,
                 save_individual: bool = True, generate_summary: bool = True) -> Dict[str, Any]
```

**Parameters:**
- `image_paths` (List[str]): List of image paths to analyze
- `class_idx` (int): Class index for analysis
- `save_individual` (bool): Whether to save individual results
- `generate_summary` (bool): Whether to generate summary analysis

**Returns:**
- `Dict[str, Any]`: Batch analysis results

---

### 5. Preprocessing Module (`utils/preprocessing.py`)

#### `MedicalImagePreprocessor`

Medical image preprocessor for KUB X-ray images.

```python
class MedicalImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (640, 640),
                 normalize_method: str = 'imagenet', enhance_contrast: bool = True,
                 reduce_noise: bool = True)
```

**Parameters:**
- `target_size` (Tuple[int, int]): Target image size (width, height)
- `normalize_method` (str): Normalization method ('imagenet', 'medical', 'minmax')
- `enhance_contrast` (bool): Whether to apply contrast enhancement
- `reduce_noise` (bool): Whether to apply noise reduction

#### `preprocess_single_image()`

Complete preprocessing pipeline for a single image.

```python
def preprocess_single_image(self, image: np.ndarray) -> np.ndarray
```

**Parameters:**
- `image` (np.ndarray): Input image as numpy array

**Returns:**
- `np.ndarray`: Preprocessed image ready for model input

**Example:**
```python
preprocessor = MedicalImagePreprocessor(target_size=(640, 640))
processed_image = preprocessor.preprocess_single_image(image)
```

#### `YOLOv8Transform`

YOLOv8 compatible transform pipeline.

```python
class YOLOv8Transform:
    def __init__(self, image_size: int = 640, normalize_method: str = 'imagenet',
                 training: bool = True)
```

---

### 6. Augmentation Module (`utils/augmentation.py`)

#### `MedicalImageAugmentation`

Medical image augmentation pipeline.

```python
class MedicalImageAugmentation:
    def __init__(self, image_size: int = 640, augmentation_strength: str = 'medium',
                 preserve_anatomy: bool = True)
```

**Parameters:**
- `image_size` (int): Target image size for augmentation
- `augmentation_strength` (str): Strength level ('light', 'medium', 'strong')
- `preserve_anatomy` (bool): Whether to preserve anatomical structures

#### `create_training_augmentation()`

Create comprehensive training augmentation pipeline.

```python
def create_training_augmentation(self) -> A.Compose
```

**Returns:**
- `A.Compose`: Albumentations Compose object with training augmentations

---

### 7. Visualization Module (`utils/visualization.py`)

#### `TrainingVisualizer`

Comprehensive visualization tools for training monitoring and analysis.

```python
class TrainingVisualizer:
    def __init__(self, save_path: Optional[Union[str, Path]] = None)
```

#### `plot_training_curves()`

Plot comprehensive training curves including loss, metrics, and learning rate.

```python
def plot_training_curves(self, results: Any, save_path: Optional[Path] = None,
                        show: bool = False) -> None
```

#### `plot_performance_metrics()`

Plot comprehensive performance metrics as bar charts.

```python
def plot_performance_metrics(self, metrics: Dict[str, float], 
                           save_path: Optional[Path] = None, show: bool = False) -> None
```

---

### 8. Batch Inference Module (`scripts/inference.py`)

#### `BatchInference`

Efficient batch inference for kidney stone detection.

```python
class BatchInference:
    def __init__(self, model_path: str, config_path: str, device: str = 'auto',
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                 batch_size: int = 8, num_workers: int = 4)
```

#### `process_dataset()`

Process entire dataset with batch inference.

```python
def process_dataset(self, dataset_path: str, output_dir: str = None,
                   save_annotations: bool = True, save_results: bool = True,
                   export_formats: List[str] = ['json', 'csv']) -> Dict[str, Any]
```

**Parameters:**
- `dataset_path` (str): Path to dataset directory
- `output_dir` (str): Output directory for results
- `save_annotations` (bool): Whether to save annotated images
- `save_results` (bool): Whether to save detection results
- `export_formats` (List[str]): Formats to export results

**Returns:**
- `Dict[str, Any]`: Comprehensive processing results

---

## 🔧 Utility Functions

### Configuration Functions

#### `create_preprocessing_pipeline()`

Create complete preprocessing pipeline based on configuration.

```python
def create_preprocessing_pipeline(config: dict) -> dict
```

#### `create_augmentation_pipeline()`

Create complete augmentation pipeline based on configuration.

```python
def create_augmentation_pipeline(config: Dict) -> Dict
```

---

## 📊 Data Structures

### Detection Result Format

```python
detection = {
    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
    'confidence': 0.95,        # Confidence score
    'class_id': 0,             # Class ID
    'class_name': 'kidney_stone',  # Class name
    'timestamp': 1234567890.123  # Detection timestamp
}
```

### Performance Metrics Format

```python
metrics = {
    'mAP@0.5': 0.85,
    'mAP@0.5:0.95': 0.72,
    'precision': 0.90,
    'recall': 0.80,
    'f1_score': 0.85,
    'sensitivity': 0.80,
    'specificity': 0.90,
    'clinical_accuracy': 0.85
}
```

### Grad-CAM Result Format

```python
gradcam_result = {
    'layer_0_Conv2d': np.ndarray,  # Heatmap for layer 0
    'layer_1_Conv2d': np.ndarray,  # Heatmap for layer 1
    # ... more layers
}
```

---

## 🚀 Usage Examples

### Complete Training Pipeline

```python
from scripts.train import KidneyStoneTrainer

# Initialize trainer
trainer = KidneyStoneTrainer(
    config_path='data/data.yaml',
    model_size='nano',
    device='cuda'
)

# Train model
results = trainer.train()

# Evaluate model
eval_results = trainer.evaluate()

# Export model
exported_models = trainer.export_model()
```

### Real-time Detection

```python
from scripts.real_time import RealTimeDetector

# Initialize detector
detector = RealTimeDetector(
    model_path='models/best.pt',
    config_path='data/data.yaml',
    device='cuda'
)

# Run webcam detection
detector.run_webcam(camera_index=0, save_output=True)

# Process video file
detector.run_video('input_video.mp4', save_output=True)
```

### Explainability Analysis

```python
from scripts.explainability import ExplainabilityAnalyzer

# Initialize analyzer
analyzer = ExplainabilityAnalyzer(
    model_path='models/best.pt',
    config_path='data/data.yaml'
)

# Analyze batch of images
results = analyzer.analyze_batch(
    image_paths=['image1.jpg', 'image2.jpg'],
    class_idx=0,
    save_individual=True
)
```

### Batch Processing

```python
from scripts.inference import BatchInference

# Initialize batch inference
batch_inference = BatchInference(
    model_path='models/best.pt',
    config_path='data/data.yaml',
    batch_size=16
)

# Process dataset
results = batch_inference.process_dataset(
    dataset_path='data/test/images',
    save_annotations=True,
    export_formats=['json', 'csv']
)
```

---

## ⚠️ Important Notes

1. **Memory Management**: Large batch sizes may cause out-of-memory errors. Reduce batch size if needed.

2. **Device Selection**: Use 'auto' for automatic device selection, or specify 'cuda'/'cpu' explicitly.

3. **File Paths**: Always use absolute paths or ensure relative paths are correct from the project root.

4. **Model Compatibility**: Ensure model weights are compatible with the specified model size.

5. **Data Format**: Input images should be in standard formats (JPG, PNG) and labels in YOLOv8 format.

---

## 📝 Error Handling

All functions include comprehensive error handling and logging. Check log files for detailed error information:

- `training.log` - Training-related errors
- `evaluation.log` - Evaluation-related errors
- `realtime_inference.log` - Real-time inference errors
- `explainability.log` - Explainability analysis errors
- `batch_inference.log` - Batch processing errors

For additional help, refer to the troubleshooting section in INSTALLATION.md.


