#!/usr/bin/env python3
"""Debug wrapper to understand why Flask is exiting"""
import sys
import subprocess
import time
import os

print("=" * 60)
print("KIDNEY STONE DETECTION - Flask Server Debug Wrapper")
print("=" * 60)
print()

# Run Flask in subprocess so we can capture all output
proc = subprocess.Popen(
    [sys.executable, 'app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,  # Line buffered
    encoding='utf-8',
    errors='replace'  # Replace invalid chars instead of crashing
)

print(f"Subprocess PID: {proc.pid}")
print("Reading output...")
print("-" * 60)

# Read all output line by line
try:
    for line in proc.stdout:
        print(line, end='')
        sys.stdout.flush()
except KeyboardInterrupt:
    print("\nInterrupted by user")
    proc.terminate()
except UnicodeDecodeError as e:
    print(f"Unicode error: {e}")

print("-" * 60)
print(f"Process exited with code: {proc.returncode}")
sys.exit(proc.returncode)
