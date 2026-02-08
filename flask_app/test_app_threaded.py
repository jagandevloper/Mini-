#!/usr/bin/env python3
"""Minimal Flask app with threading to keep server alive"""
from flask import Flask
import time
import sys
import threading

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

def keep_alive():
    """Keep the app alive by preventing exit"""
    while True:
        time.sleep(1)

if __name__ == '__main__':
    print("Starting app on port 5000...")
    # Run Flask in a separate thread
    server_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True),
        daemon=False
    )
    server_thread.start()
    
    # Keep main thread alive
    try:
        keep_alive()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
