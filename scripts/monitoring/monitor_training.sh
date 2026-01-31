#!/bin/bash
# Monitor training progress

echo "=== Training Process Monitor ==="
echo ""

# Find the main training process
MAIN_PID=$(pgrep -f "dnn_trainer.py" | head -1)

if [ -z "$MAIN_PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

echo "Main PID: $MAIN_PID"
echo ""

# Show all related processes
echo "=== All Training Processes ==="
ps aux | grep -E "dnn_trainer|PID" | grep -v grep
echo ""

# Show process states
echo "=== Process States ==="
echo "D = Uninterruptible sleep (I/O)"
echo "R = Running"
echo "S = Sleeping"
echo ""
ps -eo pid,state,pcpu,pmem,rss,cmd | grep dnn_trainer | grep -v grep
echo ""

# Show I/O stats
echo "=== I/O Statistics (per process) ==="
for pid in $(pgrep -f dnn_trainer); do
    if [ -f /proc/$pid/io ]; then
        echo "PID $pid:"
        cat /proc/$pid/io | grep -E "read_bytes|write_bytes"
    fi
done
echo ""

# Show memory usage
echo "=== Memory Usage ==="
free -h
echo ""

# Check if log file exists and show recent progress
if [ -f /home/ubuntu/Med-WAV/training.log ]; then
    echo "=== Recent Training Output ==="
    tail -20 /home/ubuntu/Med-WAV/training.log
fi

