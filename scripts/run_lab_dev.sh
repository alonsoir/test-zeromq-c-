#!/bin/bash
# ML Defender - Launch Full Lab in Development Mode
# Launches: Firewall → Detector → Sniffer (in background)

set -e

PROJECT_ROOT="/vagrant"
LOG_DIR="$PROJECT_ROOT/logs/lab"
PID_DIR="$PROJECT_ROOT/logs/pids"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# Function to check if component is running
is_running() {
    local name=$1
    pgrep -f "$name" > /dev/null 2>&1
}

# Function to wait for port
wait_for_port() {
    local port=$1
    local timeout=$2
    local elapsed=0

    while ! ss -tlnp | grep -q ":$port "; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            return 1
        fi
    done
    return 0
}

# Kill existing processes
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ML Defender Lab - Development Mode                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🧹 Cleaning up existing processes..."
sudo pkill -9 firewall-acl-agent 2>/dev/null || true
pkill -9 ml-detector 2>/dev/null || true
sudo pkill -9 sniffer 2>/dev/null || true
sleep 2

# Check binaries exist
echo ""
echo "📦 Checking binaries..."
if [ ! -f "$PROJECT_ROOT/firewall-acl-agent/build/firewall-acl-agent" ]; then
    echo -e "${RED}❌ Firewall binary not found. Run: make firewall${NC}"
    exit 1
fi
if [ ! -f "$PROJECT_ROOT/ml-detector/build/ml-detector" ]; then
    echo -e "${RED}❌ Detector binary not found. Run: make detector${NC}"
    exit 1
fi
if [ ! -f "$PROJECT_ROOT/sniffer/build/sniffer" ]; then
    echo -e "${RED}❌ Sniffer binary not found. Run: make sniffer${NC}"
    exit 1
fi
echo -e "${GREEN}✅ All binaries found${NC}"

# ============================================================================
# STEP 1: Launch Firewall ACL Agent (SUB - must start first)
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 STEP 1/3: Starting Firewall ACL Agent"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Role: ZMQ SUB (tcp://localhost:5572)"
echo "   Action: Receives detections and blocks IPs via IPSet/IPTables"
echo ""

cd "$PROJECT_ROOT/firewall-acl-agent/build"
sudo ./firewall-acl-agent -c ../config/firewall.json \
    > "$LOG_DIR/firewall.log" 2>&1 &
FIREWALL_PID=$!
echo $FIREWALL_PID > "$PID_DIR/firewall.pid"

echo -n "   Waiting for firewall to initialize"
sleep 3
for i in {1..5}; do
    echo -n "."
    sleep 1
done
echo ""

if is_running "firewall-acl-agent"; then
    echo -e "${GREEN}✅ Firewall ACL Agent running (PID: $FIREWALL_PID)${NC}"
else
    echo -e "${RED}❌ Firewall failed to start. Check logs: tail $LOG_DIR/firewall.log${NC}"
    exit 1
fi

# ============================================================================
# STEP 2: Launch ML Detector (PUB - must start after firewall)
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 STEP 2/3: Starting ML Detector"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Input:  PULL tcp://127.0.0.1:5571 (from Sniffer)"
echo "   Output: PUB tcp://0.0.0.0:5572 (to Firewall)"
echo "   ML: 4-layer RandomForest detection pipeline"
echo ""

cd "$PROJECT_ROOT/ml-detector/build"
./ml-detector -c ../config/ml_detector_config.json \
    > "$LOG_DIR/detector.log" 2>&1 &
DETECTOR_PID=$!
echo $DETECTOR_PID > "$PID_DIR/detector.pid"

echo -n "   Waiting for detector to initialize"
sleep 2
for i in {1..3}; do
    echo -n "."
    sleep 1
done
echo ""

# Wait for port 5572 (PUB socket)
echo -n "   Waiting for port 5572 (PUB)"
if wait_for_port 5572 10; then
    echo -e " ${GREEN}✅${NC}"
else
    echo -e " ${RED}❌ Timeout${NC}"
fi

if is_running "ml-detector"; then
    echo -e "${GREEN}✅ ML Detector running (PID: $DETECTOR_PID)${NC}"
else
    echo -e "${RED}❌ Detector failed to start. Check logs: tail $LOG_DIR/detector.log${NC}"
    exit 1
fi

# ============================================================================
# STEP 3: Launch Sniffer (PUSH - starts last)
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📡 STEP 3/3: Starting Sniffer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Capture: eth0 (eBPF/XDP kernel space)"
echo "   Output: PUSH tcp://127.0.0.1:5571 (to Detector)"
echo "   Features: 4 groups (RF, Ransomware, Internal, DDoS)"
echo ""

cd "$PROJECT_ROOT/sniffer/build"
sudo ./sniffer -c ../config/sniffer.json \
    > "$LOG_DIR/sniffer.log" 2>&1 &
SNIFFER_PID=$!
echo $SNIFFER_PID > "$PID_DIR/sniffer.pid"

echo -n "   Waiting for sniffer to initialize"
sleep 2
for i in {1..3}; do
    echo -n "."
    sleep 1
done
echo ""

if is_running "sniffer"; then
    echo -e "${GREEN}✅ Sniffer running (PID: $SNIFFER_PID)${NC}"
else
    echo -e "${RED}❌ Sniffer failed to start. Check logs: tail $LOG_DIR/sniffer.log${NC}"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ ML Defender Lab Running                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Component Status:"
echo -e "   Firewall: ${GREEN}✅ PID $FIREWALL_PID${NC}"
echo -e "   Detector: ${GREEN}✅ PID $DETECTOR_PID${NC}"
echo -e "   Sniffer:  ${GREEN}✅ PID $SNIFFER_PID${NC}"
echo ""
echo "📋 Logs:"
echo "   Firewall: $LOG_DIR/firewall.log"
echo "   Detector: $LOG_DIR/detector.log"
echo "   Sniffer:  $LOG_DIR/sniffer.log"
echo ""
echo "🎯 Pipeline Flow:"
echo "   Sniffer → Detector → Firewall"
echo "   (5571)     (5572)     (IPSet/IPTables)"
echo ""
echo "📊 Monitoring:"
echo "   bash scripts/monitor_lab.sh    # Live monitoring"
echo "   make logs-lab                   # From host"
echo ""
echo "🛑 Stop Lab:"
echo "   make kill-lab                   # From host"
echo "   kill-lab                        # From VM"
echo ""

# Launch monitor automatically
echo "🚀 Starting live monitor in 3 seconds..."
sleep 3
exec bash "$PROJECT_ROOT/scripts/monitor_lab.sh"