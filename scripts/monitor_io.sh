#!/bin/bash
# Monitor disk I/O for training processes

echo "=== Disk I/O Monitor ==="
echo "Press Ctrl+C to stop"
echo ""

# Get all training PIDs
PIDS=$(pgrep -f dnn_trainer | tr '\n' ',' | sed 's/,$//')

if [ -z "$PIDS" ]; then
    echo "❌ No training processes found"
    exit 1
fi

# Use iotop if available, otherwise use pidstat
if command -v iotop &> /dev/null; then
    echo "Using iotop (requires sudo)..."
    sudo iotop -p $PIDS -o -b -n 10 -d 1
elif command -v pidstat &> /dev/null; then
    echo "Using pidstat..."
    pidstat -d 1 -p $PIDS
else
    echo "Installing pidstat..."
    sudo apt-get update && sudo apt-get install -y sysstat
    pidstat -d 1 -p $PIDS
fi

