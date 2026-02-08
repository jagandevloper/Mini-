#!/usr/bin/env python3
"""Persistent Flask server runner"""
import sys
import os
import logging

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Flask app and detector class
import app as app_module
from app import app

def initialize_detector():
    """Initialize the detector before starting the server."""
    model_path = None
    
    # Try multiple possible locations
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
    
    if not model_path:
        logger.error("❌ Could not find model file!")
        return None
    
    try:
        # Initialize detector using the lightweight detector class
        from app import LightweightKidneyStoneDetector
        detector = LightweightKidneyStoneDetector(model_path)
        logger.info("✅ Detector initialized successfully")
        return detector
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 KIDNEY STONE DETECTION - Starting Persistent Server")
    print("=" * 70)
    
    # Initialize detector
    detector = initialize_detector()
    if detector:
        # Set the global detector in the app module
        app_module.detector = detector
        print(f"✅ Model loaded: {detector.model_path}")
        print(f"✅ Device: {detector.device}")
    else:
        print("⚠️  Warning: Model not loaded - detection will not work")
    
    print(f"✅ Server will run on: http://localhost:5000")
    print(f"✅ Press Ctrl+C to stop the server")
    print("=" * 70)
    
    try:
        # Run Flask with minimal configuration
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n✅ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
