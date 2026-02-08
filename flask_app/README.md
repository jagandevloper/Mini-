# 🏥 Lightweight Multi-Level Explainable Kidney Stone Detection

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

# Or use launchers:
# Windows: start.bat
# Linux/Mac: ./start.sh
```

**Open:** http://localhost:5000

## ✨ Features

### 1. Lightweight Real-Time Detection
- Model size: ~6 MB (YOLOv8-nano)
- GPU inference: 8-10 ms
- CPU inference: 80-150 ms
- FP16 support for 2x speedup

### 2. Multi-Level Explainability

#### Level 1: Pixel-Level (GradCAM)
- Visualizes pixel-level attention
- Heatmap shows where model focuses
- Color-coded intensity map

#### Level 2: Region-Level (Attention Maps)
- Anatomical region analysis
- Confidence-weighted visualizations
- Spatial distribution patterns

#### Level 3: Image-Level (Clinical Prognosis)
- **Risk Assessment**: Low/Moderate/High
- **Severity Score**: 0-100 scale
- **Treatment Recommendations**: Automated suggestions

### 3. Clinical Features
- Automated risk assessment
- Anatomical location identification
- Statistical analysis
- Treatment recommendations

## 📁 Project Structure

```
flask_app/
├── app.py                      # Main Flask application
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── utils/
│   ├── __init__.py
│   └── explainability.py      # Advanced explainability module
├── templates/
│   └── index.html              # Web interface
├── models/
│   └── best.pt                 # Trained model
└── static/results/             # Visualizations
```

## 🔬 Technical Details

### Model
- **Architecture**: YOLOv8-nano
- **Input**: 640x640 pixels
- **Output**: Bounding boxes + confidence scores
- **Optimizations**: FP16, CUDA acceleration

### Explainability Methods
- **GradCAM**: Gradient-weighted Class Activation Mapping
- **Attention Maps**: Confidence-weighted regional analysis
- **Clinical Analysis**: Risk + severity scoring

### Performance

| Device | Inference | Throughput |
|--------|-----------|------------|
| GPU (FP16) | 8-10 ms | 100+ FPS |
| GPU (FP32) | 15-20 ms | 50-70 FPS |
| CPU | 80-150 ms | 6-12 FPS |

## 📖 Usage

### Web Interface
1. Upload KUB X-ray image
2. Click "Detect Kidney Stones"
3. Click "Explain Detection" for multi-level analysis

### API Endpoints

**Detection:**
```
POST /upload
```

**Explainability:**
```
POST /explain
```

**Batch Processing:**
```
POST /batch
```

**Model Info:**
```
GET /model_info
```

## ⚠️ Important Notes

- Research tool only (not FDA-approved)
- Not for medical diagnosis
- Always verify with healthcare professionals
- Images processed locally, no external uploads

## 📚 Documentation

- `README_PROJECT.md`: Full documentation
- `QUICK_START.md`: Quick setup guide
- `PROJECT_SUMMARY.md`: Technical summary

## 🎯 Clinical Interpretation

### Severity Scoring
```
Severity = Size Score (50%) + Confidence (30%) + Count (20%)
```

### Risk Levels
- **Low** (<40): Routine follow-up
- **Moderate** (40-70): Clinical review
- **High** (>70): Immediate attention

## 🛠️ Configuration

Edit `config.py` to customize:
- Model path
- Device (auto/cuda/cpu)
- Confidence threshold
- Upload limits

## 🐛 Troubleshooting

**Model not found:**
```bash
# Place model at: flask_app/models/best.pt
```

**CUDA issues:**
```python
# Edit config.py
DEVICE = 'cpu'
```

## 📄 License

MIT License

---
**For medical use: Always consult qualified healthcare professionals.**
