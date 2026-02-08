#!/usr/bin/env python3
"""Minimal Flask app test"""
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    print("Starting app on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    print("App stopped")
