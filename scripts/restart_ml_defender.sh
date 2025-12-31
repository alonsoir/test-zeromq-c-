#!/bin/bash
# ML Defender Restart Script - Day 30
# Restart every 72h to mitigate memory leak (31 MB/h)
# Authors: Alonso Isidoro Roman + Claude (Anthropic)

LOG_FILE="/vagrant/logs/lab/restart_ml_defender.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$TIMESTAMP] Starting ML Defender restart procedure" | tee -a "$LOG_FILE"

# 1. Stop ml-detector gracefully
echo "[$TIMESTAMP] Stopping ml-detector..." | tee -a "$LOG_FILE"
pkill -SIGTERM ml-detector
sleep 5

# Verificar que se detuvo
if pgrep -x ml-detector > /dev/null; then
    echo "[$TIMESTAMP] Force killing ml-detector..." | tee -a "$LOG_FILE"
    pkill -SIGKILL ml-detector
    sleep 2
fi

# 2. Restart ml-detector
echo "[$TIMESTAMP] Starting ml-detector..." | tee -a "$LOG_FILE"
cd /vagrant/ml-detector/build
nohup ./ml-detector --config ../config/ml_detector_config.json >> /vagrant/logs/lab/detector.log 2>&1 &

sleep 5

# 3. Verify restart
if pgrep -x ml-detector > /dev/null; then
    PID=$(pgrep -x ml-detector)
    echo "[$TIMESTAMP] ✅ ml-detector restarted successfully (PID: $PID)" | tee -a "$LOG_FILE"
else
    echo "[$TIMESTAMP] ❌ ml-detector failed to restart" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[$TIMESTAMP] Restart procedure completed" | tee -a "$LOG_FILE"
