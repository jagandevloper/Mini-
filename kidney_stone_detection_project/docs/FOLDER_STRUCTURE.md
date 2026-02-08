# Kidney Stone Detection Project - Organized Structure

## 📁 Final Folder Organization

```
kidney_stone_detection_project/
│
├── 📄 README.md                          # Quick start guide
├── 📄 requirements.txt                   # Dependencies  
├── 📄 .gitignore                         # Git ignore rules
├── 📄 FOLDER_ORGANIZATION_PLAN.md        # This file
│
├── 📁 src/                               # ✅ Source code (NEW)
│   ├── __init__.py
│   ├── web_app.py                        # Flask web application
│   ├── real_time_detector.py            # Real-time detection
│   └── train.py                          # Training script
│
├── 📁 models/                            # ✅ Models directory (NEW)
│   ├── pretrained/                       # Pre-trained YOLO models
│   │   ├── yolov8n.pt
│   │   └── yolo11n.pt
│   └── trained/                          # Symlinks to experiments/
│
├── 📁 data/                              # Dataset (UNCHANGED)
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── README files
│
├── 📁 experiments/                       # ✅ Training runs (NEW)
│   ├── kidney_stone_cuda_test/
│   └── kidney_stone_cuda_success/
│
├── 📁 outputs/                           # ✅ Generated outputs (NEW)
│   ├── detections/
│   ├── evaluations/
│   ├── explainability/
│   └── visualizations/
│
├── 📁 scripts/                           # ✅ Utility scripts (EXISTS)
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── real_time.py
│   └── explainability.py
│
├── 📁 utils/                             # ✅ Utility modules (EXISTS)
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   └── visualization.py
│
├── 📁 web/                               # ✅ Web app files (NEW)
│   ├── static/
│   │   ├── results/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   │   └── index.html
│   └── uploads/
│
├── 📁 docs/                              # ✅ Documentation (NEW)
│   ├── README.md
│   ├── INSTALLATION.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── PROJECT_SUMMARY.md
│   ├── API_DOCUMENTATION.md
│   └── RESEARCH_PAPER_TEMPLATE.md
│
├── 📁 notebooks/                          # ✅ Jupyter notebooks (NEW)
│
├── 📁 tests/                             # ✅ Unit tests (NEW)
│
└── 📁 logs/                              # ✅ Log files (NEW)
```

## ✅ Changes Made:

1. **Created new directories**:
   - `src/` - Source code
   - `models/` - Model files
   - `experiments/` - Training runs
   - `outputs/` - Generated outputs
   - `web/` - Web application files
   - `docs/` - Documentation
   - `notebooks/` - Jupyter notebooks
   - `tests/` - Unit tests
   - `logs/` - Log files

2. **Moved files**:
   - `web_app.py` → `src/web_app.py`
   - `real_time_detector.py` → `src/real_time_detector.py`
   - `simple_train_cuda.py` → `src/train.py`
   - Documentation → `docs/`
   - Web files → `web/`
   - Training runs → `experiments/`
   - Pretrained models → `models/pretrained/`

3. **Cleaned up**:
   - Created `.gitignore`
   - Added `__init__.py` to modules
   - Organized static files

## 🎯 Next Steps:

### Update import paths:

Since files moved, update imports:

**In `src/web_app.py`**:
```python
# Change:
from explainability_simple import KidneyStoneExplainability

# To:
import sys
sys.path.insert(0, '..')
from explainability_simple import KidneyStoneExplainability
```

**To run the application**:
```bash
cd kidney_stone_detection_project
python src/web_app.py  # Instead of python web_app.py
```

### Update model paths:

**In scripts**, update model paths:
```python
# Change:
model_path = 'runs/kidney_stone_cuda_test/weights/best.pt'

# To:
model_path = 'experiments/kidney_stone_cuda_test/weights/best.pt'
```

### Update data paths:

```python
# Change:
data_path = 'data/data.yaml'

# Keep as is, or:
data_path = os.path.join(os.path.dirname(__file__), '../data/data.yaml')
```

## 📝 Files Still at Root (Intentionally):

- `web_app.py` - Might need for compatibility
- `real_time_detector.py` - Might need for compatibility
- `requirements.txt` - Dependency list
- `*.log` files - Will move to logs/ soon
- `runs/` - Will move to experiments/ soon
- `scripts/` - Already organized
- `utils/` - Already organized
- `data/` - Already organized
- `templates/`, `static/`, `uploads/` - Old structure

## 🧹 Cleanup Needed:

Run these commands:
```bash
# Move remaining files
mv logs/* logs/ 2>/dev/null || mkdir logs
mv *.log logs/ 2>/dev/null

# Remove old directories
rm -rf __pycache__/
rm detection_result_*.jpg

# Update .gitignore
# (Already created above)
```

## ✅ Current Status:

**✅ Well Organized**:
- Source code in `src/`
- Documentation in `docs/`
- Models in `models/`
- Experiments in `experiments/`

**⚠️ Needs Attention**:
- Import paths in moved files
- Some files still in old locations
- Virtual environment should be excluded

## 🚀 Usage After Reorganization:

```bash
# Navigate to project
cd kidney_stone_detection_project

# Run web app
python src/web_app.py

# Run training
python src/train.py

# Run evaluation
python scripts/evaluate.py

# Run real-time detection
python src/real_time_detector.py
```

