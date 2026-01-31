#!/bin/bash
# Quick training status check

echo "=== Training Status Check ==="
date
echo ""

# Find main process
MAIN_PID=$(pgrep -f "dnn_trainer.py" | head -1)
if [ -z "$MAIN_PID" ]; then
    echo "❌ No training running"
    exit 1
fi

# Get elapsed time
ETIME=$(ps -p $MAIN_PID -o etime= | tr -d ' ')
echo "✓ Training running for: $ETIME"
echo ""

# Process states
echo "Process States:"
ps -eo pid,state,%cpu,%mem,cmd | grep dnn_trainer | grep -v grep | \
    awk '{print "  PID " $1 ": " ($2=="R"?"🟢 Running":"🟡 I/O Wait") " (CPU: " $3 "%, Mem: " $4 "%)"}'
echo ""

# GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{print "  GPU Util: " $1 ", Memory: " $2 " / " $3}'
echo ""

# I/O throughput (bytes read since start)
echo "Data Loaded (since start):"
for pid in $(pgrep -f dnn_trainer | grep -v $MAIN_PID); do
    if [ -f /proc/$pid/io ]; then
        bytes=$(grep "read_bytes:" /proc/$pid/io | awk '{print $2}')
        gb=$(echo "scale=2; $bytes/1024/1024/1024" | bc)
        echo "  Worker $pid: ${gb} GB"
    fi
done
echo ""

# Check if progressing
echo "Is training progressing?"
if [ -f /tmp/training_last_io ]; then
    OLD_IO=$(cat /tmp/training_last_io)
    NEW_IO=$(grep "read_bytes:" /proc/$pid/io 2>/dev/null | awk '{print $2}' || echo "0")
    if [ "$NEW_IO" -gt "$OLD_IO" ]; then
        echo "  ✓ YES - I/O activity detected"
    else
        echo "  ⚠️  NO I/O change (might be in GPU compute phase)"
    fi
    echo "$NEW_IO" > /tmp/training_last_io
else
    echo "$NEW_IO" > /tmp/training_last_io
    echo "  → Run again in 10s to check progress"
fi

echo ""
echo "Tip: Run 'bash /home/ubuntu/check_training.sh' again to monitor"

