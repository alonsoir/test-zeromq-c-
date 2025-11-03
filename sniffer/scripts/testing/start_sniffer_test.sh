#!/bin/bash
echo "🚀 Starting Long-Running Sniffer Test..."

# Kill any existing sniffer
sudo pkill -9 sniffer 2>/dev/null

# Clean old logs
rm -f /tmp/sniffer_test_*.log 2>/dev/null

# Start time
START_TIME=$(date +%s)
echo "$START_TIME" > /tmp/sniffer_start_time.txt

# Start sniffer with nohup
cd /vagrant/sniffer/build
sudo nohup ./sniffer -c ../config/sniffer.json -i eth0 -vv \
    > /tmp/sniffer_test_output.log 2>&1 &

SNIFFER_PID=$!
echo "$SNIFFER_PID" > /tmp/sniffer.pid

echo "✅ Sniffer started with PID: $SNIFFER_PID"
echo "📝 Logs: /tmp/sniffer_test_output.log"
echo "⏰ Started at: $(date)"

# Monitor script
cat > /tmp/monitor_sniffer.sh << 'EOFMON'
#!/bin/bash
while true; do
    if ! ps -p $(cat /tmp/sniffer.pid 2>/dev/null) > /dev/null 2>&1; then
        echo "[$(date)] ❌ SNIFFER DIED!" | tee -a /tmp/sniffer_monitor.log
        break
    fi
    
    MEM=$(ps -p $(cat /tmp/sniffer.pid) -o rss= 2>/dev/null | tr -d ' ')
    CPU=$(ps -p $(cat /tmp/sniffer.pid) -o %cpu= 2>/dev/null | tr -d ' ')
    UPTIME=$(ps -p $(cat /tmp/sniffer.pid) -o etime= 2>/dev/null | tr -d ' ')
    
    echo "[$(date)] UP: $UPTIME | MEM: $MEM KB | CPU: $CPU%" | tee -a /tmp/sniffer_monitor.log
    
    sleep 300  # Every 5 minutes
done
EOFMON

chmod +x /tmp/monitor_sniffer.sh
nohup /tmp/monitor_sniffer.sh > /tmp/sniffer_monitor_output.log 2>&1 &
echo "✅ Monitoring started"

echo ""
echo "🔍 Quick checks:"
echo "  tail -f /tmp/sniffer_test_output.log     # Live sniffer output"
echo "  tail -f /tmp/sniffer_monitor.log         # Live monitoring"
echo "  cat /tmp/sniffer.pid                     # Get PID"
echo ""
