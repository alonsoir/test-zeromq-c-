#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# ML Defender - Validate Gateway Mode
# ══════════════════════════════════════════════════════════════════════════════
# Purpose: Verify that gateway mode is capturing traffic (ifindex=5 events)
# Location: /vagrant/scripts/gateway/defender/validate_gateway.sh
# Usage: ./validate_gateway.sh
# Exit codes: 0 = SUCCESS, 1 = FAIL
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ML Defender - Gateway Mode Validation                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Verify sniffer is running
if [ ! -f /tmp/sniffer.pid ]; then
    echo "❌ Sniffer is not running"
    echo "   Start with: /vagrant/scripts/gateway/defender/start_gateway_test.sh"
    exit 1
fi

PID=$(cat /tmp/sniffer.pid)
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Sniffer is not active (PID: $PID)"
    exit 1
fi

echo "✅ Sniffer is running (PID: $PID)"
echo ""

# Analyze logs for gateway events
LOG_FILE="/tmp/sniffer_output.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No log file found"
    exit 1
fi

# Search for ifindex=5 events (eth3 in gateway mode)
ETH3_EVENTS=$(grep -c "ifindex=5" "$LOG_FILE" 2>/dev/null || echo "0")
ETH1_EVENTS=$(grep -c "ifindex=3" "$LOG_FILE" 2>/dev/null || echo "0")

echo "═══════════════════════════════════════════════════════════════"
echo "  CAPTURED EVENTS ANALYSIS"
echo "═══════════════════════════════════════════════════════════════"
echo "  eth1 (ifindex=3, host-based): $ETH1_EVENTS events"
echo "  eth3 (ifindex=5, gateway):    $ETH3_EVENTS events"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Decision
if [ "$ETH3_EVENTS" -gt 0 ]; then
    echo "✅ ✅ ✅ GATEWAY MODE VALIDATED ✅ ✅ ✅"
    echo ""
    echo "   $ETH3_EVENTS events captured on eth3 (gateway mode)"
    echo ""

    # Show sample event
    echo "📋 Sample gateway event:"
    grep "ifindex=5" "$LOG_FILE" | head -3 | while read line; do
        echo "   $line"
    done

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  VALIDATION SUCCESSFUL - PHASE 1 COMPLETE 🎆"
    echo "═══════════════════════════════════════════════════════════════"

    exit 0
else
    echo "❌ GATEWAY MODE NOT VALIDATED"
    echo ""
    echo "   No events captured on eth3 (ifindex=5)"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "   1. Verify client VM is running: vagrant status"
    echo "   2. Verify client is generating traffic:"
    echo "      vagrant ssh client"
    echo "      /vagrant/scripts/gateway/client/generate_traffic.sh"
    echo "   3. Check XDP attachment:"
    echo "      sudo bpftool net show"
    echo "   4. Check interface configuration:"
    echo "      sudo bpftool map dump name iface_configs"
    echo "   5. Check network connectivity:"
    echo "      sudo tcpdump -i eth3 -c 10"
    echo ""

    exit 1
fi