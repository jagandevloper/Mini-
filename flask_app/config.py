"""
Configuration settings for Flask application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'kidney-stone-detection-secret-key-2024')

# Upload settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = BASE_DIR / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Results folder
RESULTS_FOLDER = BASE_DIR / 'static' / 'results'

# Model configuration
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    str(BASE_DIR.parent / 'kidney_stone_detection_project' / 'experiments' / 'kidney_stone_cuda_test' / 'weights' / 'best.pt')
)

# Device configuration
DEVICE = 'auto'  # 'auto', 'cuda', or 'cpu'

# Detection settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 0.9

# Application settings
DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
PORT = int(os.environ.get('PORT', 5000))
HOST = os.environ.get('HOST', '0.0.0.0')

