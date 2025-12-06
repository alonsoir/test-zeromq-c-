#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# ML Defender - Gateway Mode Dashboard
# ══════════════════════════════════════════════════════════════════════════════
# Purpose: Real-time monitoring dashboard for dual-NIC gateway operation
# Location: /vagrant/scripts/gateway/defender/gateway_dashboard.sh
# Usage: ./gateway_dashboard.sh
# Exit: Press Ctrl+C
# ══════════════════════════════════════════════════════════════════════════════

# Function to cleanup on exit
cleanup() {
    tput cnorm  # Show cursor
    clear
    echo "Dashboard stopped."
    exit 0
}

trap cleanup EXIT INT TERM

# Hide cursor
tput civis

echo "Starting ML Defender Gateway Dashboard..."
sleep 1

while true; do
    # Get current time
    TIMESTAMP=$(date '+%H:%M:%S')

    # Get network statistics
    ETH1_PACKETS=$(cat /sys/class/net/eth1/statistics/rx_packets 2>/dev/null || echo "0")
    ETH3_PACKETS=$(cat /sys/class/net/eth3/statistics/rx_packets 2>/dev/null || echo "0")
    ETH1_DROPS=$(cat /sys/class/net/eth1/statistics/rx_dropped 2>/dev/null || echo "0")
    ETH3_DROPS=$(cat /sys/class/net/eth3/statistics/rx_dropped 2>/dev/null || echo "0")

    # Get sniffer events
    if [ -f /tmp/sniffer_output.log ]; then
        ETH3_EVENTS=$(tail -1000 /tmp/sniffer_output.log | grep -c "ifindex=5" 2>/dev/null || echo "0")
        ETH1_EVENTS=$(tail -1000 /tmp/sniffer_output.log | grep -c "ifindex=3" 2>/dev/null || echo "0")
    else
        ETH3_EVENTS="0"
        ETH1_EVENTS="0"
    fi

    # Get sniffer CPU usage
    if [ -f /tmp/sniffer.pid ]; then
        SNIFFER_PID=$(cat /tmp/sniffer.pid)
        if ps -p $SNIFFER_PID > /dev/null 2>&1; then
            CPU_USAGE=$(ps -p $SNIFFER_PID -o %cpu 2>/dev/null | tail -1 | tr -d ' ' || echo "0.0")
            MEM_USAGE=$(ps -p $SNIFFER_PID -o %mem 2>/dev/null | tail -1 | tr -d ' ' || echo "0.0")
            SNIFFER_STATUS="✅ Running"
        else
            CPU_USAGE="0.0"
            MEM_USAGE="0.0"
            SNIFFER_STATUS="❌ Stopped"
        fi
    else
        CPU_USAGE="0.0"
        MEM_USAGE="0.0"
        SNIFFER_STATUS="❌ Not Started"
    fi

    # Clear screen and display dashboard
    clear
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           ML DEFENDER - GATEWAY MODE DASHBOARD                   ║"
    echo "║           Time: $TIMESTAMP                                       ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                                                                  ║"
    echo "║  INTERFACE STATISTICS                                            ║"
    echo "║  ────────────────────────────────────────────────────────────    ║"
    printf "║  %-15s │ Packets: %8s │ Events: %5s │ Drops: %4s ║\n" \
        "eth1 (WAN)" "$ETH1_PACKETS" "$ETH1_EVENTS" "$ETH1_DROPS"
    printf "║  %-15s │ Packets: %8s │ Events: %5s │ Drops: %4s ║\n" \
        "eth3 (Gateway)" "$ETH3_PACKETS" "$ETH3_EVENTS" "$ETH3_DROPS"
    echo "║                                                                  ║"
    echo "║  SNIFFER STATUS                                                  ║"
    echo "║  ────────────────────────────────────────────────────────────    ║"
    printf "║  Status: %-20s  CPU: %5s%%  Memory: %5s%%     ║\n" \
        "$SNIFFER_STATUS" "$CPU_USAGE" "$MEM_USAGE"
    echo "║                                                                  ║"
    echo "║  GATEWAY MODE VALIDATION                                         ║"
    echo "║  ────────────────────────────────────────────────────────────    ║"

    if [ "$ETH3_EVENTS" -gt 0 ]; then
        echo "║                                                                  ║"
        echo "║  ✅ ✅ ✅  GATEWAY MODE VALIDATED  ✅ ✅ ✅                   ║"
        echo "║                                                                  ║"
        printf "║  Gateway events captured: %-6s                                ║\n" "$ETH3_EVENTS"
        echo "║  Phase 1: COMPLETE 🎆                                            ║"
    else
        echo "║                                                                  ║"
        echo "║  🔄 Waiting for gateway traffic...                              ║"
        echo "║                                                                  ║"
        echo "║  Actions:                                                        ║"
        echo "║  1. Start client VM: vagrant up client                          ║"
        echo "║  2. Generate traffic from client:                                ║"
        echo "║     vagrant ssh client                                           ║"
        echo "║     /vagrant/scripts/gateway/client/generate_traffic.sh         ║"
    fi

    echo "║                                                                  ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  COMMANDS                                                        ║"
    echo "║  • Validate:  /vagrant/scripts/gateway/defender/validate_gateway.sh ║"
    echo "║  • Logs:      tail -f /tmp/sniffer_output.log                   ║"
    echo "║  • Exit:      Press Ctrl+C                                       ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    # Update every 2 seconds
    sleep 2
done