#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# ML Defender - Start Gateway Test
# ══════════════════════════════════════════════════════════════════════════════
# Purpose: Launch sniffer with dual-NIC gateway configuration (uses sniffer.json)
# Location: /vagrant/scripts/gateway/defender/start_gateway_test.sh
# Usage: ./start_gateway_test.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ML Defender - Gateway Mode Test Startup                   ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Kill previous instances
echo "🔧 Stopping previous sniffer instances..."
sudo pkill -9 sniffer 2>/dev/null || true
sleep 1

# Verify sniffer binary exists
if [ ! -f /vagrant/sniffer/build/sniffer ]; then
    echo "❌ ERROR: Sniffer binary not found"
    echo "   Build with: cd /vagrant/sniffer && make"
    exit 1
fi

# Verify config exists (using YOUR existing sniffer.json)
CONFIG_FILE="/vagrant/sniffer/config/sniffer.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Verify interfaces exist
echo "🔍 Verifying interfaces..."
for iface in eth1 eth3; do
    if ! ip link show $iface >/dev/null 2>&1; then
        echo "❌ ERROR: Interface $iface not found"
        exit 1
    fi
    echo "  ✅ $iface: $(ip addr show $iface | grep 'inet ' | awk '{print $2}')"
done

# Start sniffer (using YOUR sniffer.json with dual-NIC config)
echo ""
echo "🚀 Starting sniffer in dual-NIC mode..."
echo "   Config: /vagrant/sniffer/config/sniffer.json"
echo "   Profile: dual_nic (deployment.mode = dual)"
cd /vagrant/sniffer/build

sudo ./sniffer -c config/sniffer.json > /tmp/sniffer_output.log 2>&1 &
SNIFFER_PID=$!

# Wait for initialization
sleep 3

# Verify it's running
if ps -p $SNIFFER_PID > /dev/null; then
    echo "✅ Sniffer started successfully (PID: $SNIFFER_PID)"
    echo "$SNIFFER_PID" > /tmp/sniffer.pid

    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Dual-NIC Mode ACTIVE                                      ║"
    echo "╠════════════════════════════════════════════════════════════╣"
    echo "║  eth1 (192.168.56.20): Host-Based IDS (ifindex=3)         ║"
    echo "║  eth3 (192.168.100.1): Gateway Mode (ifindex=5)           ║"
    echo "║                                                            ║"
    echo "║  Monitor: tail -f /tmp/sniffer_output.log                 ║"
    echo "║  Validate: test-gateway                                    ║"
    echo "║  Dashboard: gateway-dash                                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"

    # Show initial output
    sleep 1
    echo ""
    echo "📋 Initial output:"
    echo "───────────────────────────────────────────────────────────"
    head -30 /tmp/sniffer_output.log
    echo "───────────────────────────────────────────────────────────"

else
    echo "❌ ERROR: Sniffer failed to start"
    echo ""
    cat /tmp/sniffer_output.log
    exit 1
fi