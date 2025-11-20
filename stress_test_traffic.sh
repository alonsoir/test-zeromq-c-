#!/bin/bash
# Simple traffic generator - Real traffic only (no telnet garbage)

RATE_PPS=$1
SLEEP_INTERVAL=$(echo "scale=3; 1.0 / $RATE_PPS" | bc)

echo "🌐 Traffic generator started (${RATE_PPS} pps - SIMPLE MODE)"

# Cleanup function for background processes
cleanup() {
    jobs -p | xargs -r kill 2>/dev/null
}
trap cleanup EXIT

while true; do
    RAND=$((RANDOM % 100))

    # 60% ICMP ping (rápido, bajo overhead)
    if [ $RAND -lt 60 ]; then
        ping -c 1 -W 1 8.8.8.8 > /dev/null 2>&1 &

    # 30% DNS queries (útil para detección)
    elif [ $RAND -lt 90 ]; then
        dig +short @8.8.8.8 google.com > /dev/null 2>&1 &

    # 10% HTTP (tráfico real)
    else
        timeout 0.5s curl -s http://example.com > /dev/null 2>&1 &
    fi

    sleep ${SLEEP_INTERVAL}

    # Cleanup cada 100 iteraciones (evitar acumulación)
    if [ $((RANDOM % 100)) -eq 0 ]; then
        jobs -p | xargs -r kill 2>/dev/null
    fi
done