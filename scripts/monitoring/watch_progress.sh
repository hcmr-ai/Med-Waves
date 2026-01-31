#!/bin/bash
# Watch training progress in real-time

echo "=== Training Speed Monitor ==="
echo "Watching training output..."
echo ""

# Find the terminal file or stdout
MAIN_PID=$(pgrep -f "dnn_trainer.py" | head -1)

if [ -z "$MAIN_PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

echo "Monitoring PID: $MAIN_PID"
echo "Press Ctrl+C to stop"
echo ""

# Method 1: Watch the process output using strace (if available)
if command -v strace &> /dev/null; then
    echo "Tip: Look for 'it/s' to see training speed"
    echo ""
    # This might require sudo
    # sudo strace -e write -s 200 -p $MAIN_PID 2>&1 | grep --line-buffered "it/s"
    echo "Run: sudo strace -e write -s 200 -p $MAIN_PID 2>&1 | grep --line-buffered 'it/s'"
fi

# Method 2: Check terminal files
TERM_DIR="/home/ubuntu/.cursor/projects/home-ubuntu-Med-WAV/terminals"
if [ -d "$TERM_DIR" ]; then
    echo "Watching terminal files for progress..."
    watch -n 2 "ls -lht $TERM_DIR/*.txt | head -5 && echo '' && tail -30 $TERM_DIR/*.txt | grep -E 'Epoch|it/s|loss|━' | tail -10"
else
    echo "Terminal directory not found"
fi

