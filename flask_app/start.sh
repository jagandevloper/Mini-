#!/bin/bash
# Start script for Flask application

echo "🚀 Starting Kidney Stone Detection Flask Application..."
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p static/results
mkdir -p models

# Check if model exists
if [ ! -f "models/best.pt" ]; then
    echo "⚠️  Model file not found in models/best.pt"
    echo "💡 The app will look for the model in the parent directory"
fi

# Start Flask application
echo "🌐 Starting Flask application..."
echo "📱 Open your browser: http://localhost:5000"
echo ""
python app.py

