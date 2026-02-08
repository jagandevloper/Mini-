# 📁 Kidney Stone Detection Project - Folder Organization Plan

## Current Issues:
1. Files scattered in root directory
2. Duplicate data folders (root + project)
3. Mixed training scripts at root level
4. Virtual environment in project folder (should be excluded)
5. No clear separation of concerns
6. Outputs mixed with source code

## Proposed Structure:

```
kidney_stone_detection_project/
│
├── 📄 README.md                          # Main documentation
├── 📄 requirements.txt                   # Dependencies
├── 📄 .gitignore                         # Git ignore file
│
├── 📁 src/                               # Source code
│   ├── __init__.py
│   ├── detector.py                      # Main detection logic
│   ├── web_app.py                        # Flask web application
│   ├── real_time_detector.py            # Real-time detection
│   ├── train.py                          # Training script
│   └── evaluate.py                       # Evaluation script
│
├── 📁 models/                             # Models & weights
│   ├── pretrained/
│   │   ├── yolov8n.pt
│   │   └── yolo11n.pt
│   ├── trained/
│   │   └── best.pt (symlink to runs)
│   └── README.md
│
├── 📁 data/                              # Dataset
│   ├── data.yaml                         # Dataset config
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── README.md
│
├── 📁 experiments/                       # Training runs
│   ├── kidney_stone_cuda_test/
│   └── kidney_stone_cuda_success/
│
├── 📁 outputs/                           # Generated outputs
│   ├── detections/                       # Detection results
│   ├── evaluations/                      # Evaluation plots
│   ├── explainability/                   # Grad-CAM results
│   └── visualizations/                  # Charts & graphs
│
├── 📁 scripts/                           # Utility scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── real_time.py
│   └── explainability.py
│
├── 📁 utils/                             # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   └── visualization.py
│
├── 📁 web/                               # Web application files
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── results/
│   ├── templates/
│   │   └── index.html
│   └── uploads/
│
├── 📁 docs/                              # Documentation
│   ├── README.md
│   ├── INSTALLATION.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── API_DOCUMENTATION.md
│   └── PROJECT_SUMMARY.md
│
├── 📁 notebooks/                         # Jupyter notebooks
│   ├── exploration.ipynb
│   └── analysis.ipynb
│
├── 📁 tests/                             # Unit tests
│   ├── test_detector.py
│   └── test_evaluation.py
│
└── 📁 logs/                              # Log files
    ├── training.log
    └── inference.log
```

## Actions Needed:
1. Move `web_app.py` → `src/web_app.py`
2. Move `real_time_detector.py` → `src/real_time_detector.py`
3. Move `simple_train_cuda.py` → `src/train.py`
4. Move scripts → `scripts/` (keep organized)
5. Move utils → `utils/` (add __init__.py)
6. Move docs → `docs/` directory
7. Create `outputs/` for all generated content
8. Move models → `models/` directory
9. Create `experiments/` for training runs
10. Clean up root directory

## Files to Delete:
- `detection_result_*.jpg` (temporary outputs)
- `training.log`, `training_cuda.log` (move to logs/)
- Virtual environment (exclude from project)

## Files to Keep in Root:
- README.md
- requirements.txt
- .gitignore
- setup.py (if exists)
