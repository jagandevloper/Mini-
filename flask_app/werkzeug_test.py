#!/usr/bin/env python3
import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from flask import Flask
from werkzeug.serving import run_simple

app = Flask(__name__)

@app.route('/')
def index():
    return 'OK'

if __name__ == '__main__':
    print("Starting with Werkzeug run_simple...")
    sys.stdout.flush()
    
    run_simple(
        'localhost',
        5000,
        app,
        use_debugger=False,
        use_reloader=False,
        threaded=True
    )
    
    print("run_simple() returned!")
    sys.stdout.flush()
