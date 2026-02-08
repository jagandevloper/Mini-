# Kidney Stone Detection Project - Installation Guide
## Complete Setup Instructions

### 🚀 Quick Start

1. **Clone the repository:**
```bash
git clone <repository-url>
cd kidney_stone_detection_project
```

2. **Create virtual environment:**
```bash
# Using conda (recommended)
conda create -n kidney_stone python=3.9
conda activate kidney_stone

# Or using venv
python -m venv kidney_stone_env
source kidney_stone_env/bin/activate  # Linux/Mac
# or
kidney_stone_env\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Prepare your dataset:**
   - Place KUB X-ray images in `data/train/images/`, `data/valid/images/`, `data/test/images/`
   - Place corresponding YOLOv8 format labels in respective `labels/` folders
   - Update `data/data.yaml` with your dataset paths

5. **Train the model:**
```bash
python scripts/train.py --config data/data.yaml --model-size nano
```

### 📋 Detailed Installation

#### System Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (for GPU acceleration)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+

#### Step-by-Step Installation

##### 1. Environment Setup

**Option A: Using Conda (Recommended)**
```bash
# Create conda environment
conda create -n kidney_stone python=3.9 -y
conda activate kidney_stone

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv kidney_stone_env
source kidney_stone_env/bin/activate  # Linux/Mac
# or
kidney_stone_env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose appropriate version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

##### 2. Verify Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test YOLOv8 installation
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"

# Test OpenCV installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

##### 3. Dataset Preparation

**Dataset Structure:**
```
data/
├── train/
│   ├── images/          # Training images (.jpg, .png)
│   └── labels/          # YOLOv8 format labels (.txt)
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels
├── test/
│   ├── images/          # Test images
│   └── labels/          # Test labels
└── data.yaml            # Dataset configuration
```

**Label Format (YOLOv8):**
```
# Each line in label file: class_id center_x center_y width height
# All coordinates normalized (0-1)
0 0.5 0.5 0.1 0.1
```

**data.yaml Configuration:**
```yaml
train: data/train/images
val: data/valid/images
test: data/test/images

nc: 1
names: ['kidney_stone']
```

##### 4. Model Training

**Basic Training:**
```bash
python scripts/train.py --config data/data.yaml --model-size nano
```

**Advanced Training Options:**
```bash
python scripts/train.py \
    --config data/data.yaml \
    --model-size nano \
    --device cuda \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.01 \
    --patience 50
```

##### 5. Model Evaluation

```bash
python scripts/evaluate.py \
    --model runs/kidney_stone_detection_*/weights/best.pt \
    --config data/data.yaml \
    --split test
```

##### 6. Real-time Inference

**Webcam Detection:**
```bash
python scripts/real_time.py \
    --model runs/kidney_stone_detection_*/weights/best.pt \
    --source 0 \
    --device cuda
```

**Video File Processing:**
```bash
python scripts/real_time.py \
    --model runs/kidney_stone_detection_*/weights/best.pt \
    --source path/to/video.mp4 \
    --save-output
```

##### 7. Explainability Analysis

```bash
python scripts/explainability.py \
    --model runs/kidney_stone_detection_*/weights/best.pt \
    --images data/test/images/*.jpg \
    --output-dir explainability_results
```

### 🔧 Troubleshooting

#### Common Issues and Solutions

**Issue 1: CUDA Out of Memory**
```bash
# Solution: Reduce batch size
python scripts/train.py --batch-size 8  # Instead of 16

# Or use CPU
python scripts/train.py --device cpu
```

**Issue 2: Import Errors**
```bash
# Solution: Reinstall dependencies
pip uninstall ultralytics
pip install ultralytics

# Or install specific version
pip install ultralytics==8.0.0
```

**Issue 3: Dataset Loading Errors**
```bash
# Check data.yaml paths
python -c "import yaml; print(yaml.safe_load(open('data/data.yaml')))"

# Verify image and label files exist
ls data/train/images/ | wc -l
ls data/train/labels/ | wc -l
```

**Issue 4: Webcam Not Working**
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened())"

# Try different camera indices
python scripts/real_time.py --source 1  # Try camera 1
```

#### Performance Optimization

**GPU Optimization:**
```bash
# Use mixed precision training
python scripts/train.py --half

# Use TensorRT optimization (if available)
python scripts/train.py --trt
```

**CPU Optimization:**
```bash
# Use multiple workers
python scripts/train.py --workers 8

# Use smaller model
python scripts/train.py --model-size nano
```

### 📊 Usage Examples

#### Example 1: Complete Training Pipeline

```bash
# 1. Train model
python scripts/train.py --config data/data.yaml --model-size nano --epochs 100

# 2. Evaluate model
python scripts/evaluate.py --model runs/kidney_stone_detection_*/weights/best.pt

# 3. Run explainability analysis
python scripts/explainability.py --model runs/kidney_stone_detection_*/weights/best.pt --images data/test/images/*.jpg

# 4. Test real-time detection
python scripts/real_time.py --model runs/kidney_stone_detection_*/weights/best.pt --source 0
```

#### Example 2: Batch Processing

```bash
# Process entire test dataset
python scripts/inference.py \
    --model runs/kidney_stone_detection_*/weights/best.pt \
    --dataset data/test/images \
    --save-annotations \
    --export-formats json csv
```

#### Example 3: Custom Configuration

```bash
# Train with custom parameters
python scripts/train.py \
    --config data/data.yaml \
    --model-size nano \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.005 \
    --patience 100 \
    --device cuda \
    --project-name custom_kidney_stone
```

### 🐳 Docker Installation (Optional)

**Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/train.py", "--config", "data/data.yaml"]
```

**Build and Run:**
```bash
# Build image
docker build -t kidney-stone-detection .

# Run container
docker run --gpus all -v $(pwd)/data:/app/data kidney-stone-detection
```

### 📝 Additional Notes

- **Memory Requirements**: Training requires 8-16GB RAM, inference requires 4-8GB RAM
- **Storage**: Each training run creates ~2GB of logs and checkpoints
- **GPU**: CUDA-compatible GPU recommended for training, optional for inference
- **Dataset Size**: Minimum 100 images per class recommended for good performance

### 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs in `training.log`, `evaluation.log`, etc.
3. Verify your dataset format matches YOLOv8 requirements
4. Ensure all dependencies are correctly installed
5. Check GPU memory usage and reduce batch size if needed

### 📚 Next Steps

After successful installation:

1. **Train your first model** using the provided dataset
2. **Evaluate performance** using the evaluation script
3. **Analyze explainability** with Grad-CAM visualizations
4. **Test real-time detection** with webcam or video files
5. **Customize parameters** for your specific use case

For more detailed information, see the main README.md file.


