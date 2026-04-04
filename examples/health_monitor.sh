#!/bin/bash
# Monitor server health and GPU status.
# Usage: ./health_monitor.sh [interval_seconds]

INTERVAL=${1:-5}
BASE="http://localhost:8080"

while true; do
    health=$(curl -s "$BASE/health" 2>/dev/null)
    gpu=$(nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw,pstate --format=csv,noheader 2>/dev/null)

    qwen=$(echo "$health" | python3 -c "import sys,json; print('ON' if json.load(sys.stdin)['active']['qwen'] else 'off')" 2>/dev/null)
    whisper=$(echo "$health" | python3 -c "import sys,json; print('ON' if json.load(sys.stdin)['active']['whisper'] else 'off')" 2>/dev/null)
    timesfm=$(echo "$health" | python3 -c "import sys,json; print('ON' if json.load(sys.stdin)['active']['timesfm'] else 'off')" 2>/dev/null)

    echo "$(date +%H:%M:%S) | GPU: $gpu | qwen=$qwen whisper=$whisper timesfm=$timesfm"
    sleep "$INTERVAL"
done
