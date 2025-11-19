#!/bin/bash
# Traffic generator for stress test - FIXED VERSION

RATE_PPS=$1
SLEEP_INTERVAL=$(echo "scale=3; 1.0 / $RATE_PPS" | bc)

echo "🌐 Traffic generator started (${RATE_PPS} pps)"

# Clean up function
cleanup_connections() {
    pkill -f "curl.*example.com" 2>/dev/null || true
    pkill -f "dig.*google" 2>/dev/null || true
    pkill -f "ping.*8.8.8.8" 2>/dev/null || true
}

# Cleanup every 60 seconds
LAST_CLEANUP=$(date +%s)

while true; do
    # Cleanup old connections periodically
    CURRENT=$(date +%s)
    if [ $((CURRENT - LAST_CLEANUP)) -ge 60 ]; then
        cleanup_connections
        LAST_CLEANUP=$CURRENT
    fi

    RAND=$((RANDOM % 100))

    # Normal traffic (80%)
    if [ $RAND -lt 80 ]; then
        # HTTP requests (alternate destinations)
        if [ $((RANDOM % 2)) -eq 0 ]; then
            timeout 0.5s curl -s http://example.com > /dev/null 2>&1 &
        else
            timeout 0.5s curl -s http://google.com > /dev/null 2>&1 &
        fi

    # DDoS-like patterns (10%) - SYN flood to public DNS
    elif [ $RAND -lt 90 ]; then
        # Use hping3 if available, otherwise nc
        timeout 0.1s nc -zv 8.8.8.8 80 > /dev/null 2>&1 &

    # DNS queries (10%)
    else
        dig +short google.com @8.8.8.8 > /dev/null 2>&1 &
    fi

    sleep ${SLEEP_INTERVAL}
done