#!/usr/bin/env python3
import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'OK'

@app.route('/health')
def health():
    return 'HEALTHY'

if __name__ == '__main__':
    print(f"Python: {sys.version}")
    print(f"Working dir: {os.getcwd()}")
    print("Starting Flask...")
    sys.stdout.flush()
    
    # Run the app
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    
    print("App.run() returned!")
    sys.stdout.flush()
