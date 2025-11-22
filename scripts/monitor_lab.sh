#!/bin/bash
# ML Defender - Lab Monitoring Script
# Shows: CPU, RAM, ZMQ ports, IPSet stats, combined logs

PROJECT_ROOT="/vagrant"
LOG_DIR="$PROJECT_ROOT/logs/lab"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to get process stats
get_process_stats() {
    local proc_name=$1
    local pid=$(pgrep -f "$proc_name" | head -1)

    if [ -z "$pid" ]; then
        echo -e "${RED}❌ Not running${NC}"
        return
    fi

    # Get CPU and MEM from ps
    local stats=$(ps -p $pid -o %cpu,%mem,rss --no-headers 2>/dev/null)
    if [ -z "$stats" ]; then
        echo -e "${RED}❌ Process died${NC}"
        return
    fi

    local cpu=$(echo "$stats" | awk '{print $1}')
    local mem=$(echo "$stats" | awk '{print $2}')
    local rss=$(echo "$stats" | awk '{print $3}')
    local rss_mb=$((rss / 1024))

    echo -e "${GREEN}✅ PID $pid${NC} - CPU: ${CYAN}${cpu}%${NC} MEM: ${CYAN}${mem}%${NC} (${rss_mb}MB)"
}

# Function to check ZMQ ports
check_zmq_ports() {
    local port=$1
    local listening=$(ss -tlnp 2>/dev/null | grep ":$port " | wc -l)

    if [ $listening -gt 0 ]; then
        local established=$(ss -tnp 2>/dev/null | grep ":$port " | grep ESTAB | wc -l)
        echo -e "${GREEN}✅ Listening${NC} (${established} connections)"
    else
        echo -e "${RED}❌ Not listening${NC}"
    fi
}

# Function to get IPSet stats
get_ipset_stats() {
    local entries=$(sudo ipset list ml_defender_blacklist 2>/dev/null | grep -c "^[0-9]")
    local size=$(sudo ipset list ml_defender_blacklist 2>/dev/null | grep "Size in memory" | awk '{print $4}')

    if [ -z "$size" ]; then
        echo -e "${RED}❌ IPSet not found${NC}"
    else
        echo -e "${GREEN}✅ Active${NC} - Entries: ${CYAN}${entries}${NC} - Memory: ${CYAN}${size}B${NC}"
    fi
}

# Main monitoring loop
clear
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Header
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ML Defender Lab - Live Monitoring                         ║"
    echo "║  $TIMESTAMP                                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # ========================================
    # Component Status
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}📊 Component Status${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo -n "🔥 Firewall:  "
    get_process_stats "firewall-acl-agent"

    echo -n "🤖 Detector:  "
    get_process_stats "ml-detector"

    echo -n "📡 Sniffer:   "
    get_process_stats "sniffer"

    echo ""

    # ========================================
    # ZMQ Ports
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}🔌 ZMQ Ports${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo -n "Port 5571 (Sniffer → Detector): "
    check_zmq_ports 5571

    echo -n "Port 5572 (Detector → Firewall): "
    check_zmq_ports 5572

    echo ""

    # ========================================
    # IPSet Status
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}🔥 IPSet Blacklist${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo -n "ml_defender_blacklist: "
    get_ipset_stats

    # Show last 5 blocked IPs if any
    local blocked_ips=$(sudo ipset list ml_defender_blacklist 2>/dev/null | grep "^[0-9]" | tail -5)
    if [ ! -z "$blocked_ips" ]; then
        echo ""
        echo -e "${YELLOW}Recent blocked IPs:${NC}"
        echo "$blocked_ips" | while read ip; do
            echo "  • $ip"
        done
    fi

    echo ""

    # ========================================
    # Recent Logs (last 5 lines per component)
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}📋 Recent Logs (last 5 lines)${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "$LOG_DIR/firewall.log" ]; then
        echo -e "${MAGENTA}🔥 Firewall:${NC}"
        tail -5 "$LOG_DIR/firewall.log" 2>/dev/null | sed 's/^/  /'
        echo ""
    fi

    if [ -f "$LOG_DIR/detector.log" ]; then
        echo -e "${MAGENTA}🤖 Detector:${NC}"
        tail -5 "$LOG_DIR/detector.log" 2>/dev/null | grep -E "Stats|ERROR|WARNING|Detection" | tail -3 | sed 's/^/  /'
        echo ""
    fi

    if [ -f "$LOG_DIR/sniffer.log" ]; then
        echo -e "${MAGENTA}📡 Sniffer:${NC}"
        tail -5 "$LOG_DIR/sniffer.log" 2>/dev/null | grep -E "ESTADÍSTICAS|Paquetes|Tasa" | tail -3 | sed 's/^/  /'
        echo ""
    fi

    # ========================================
    # Footer
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${YELLOW}Press Ctrl+C to exit${NC} | Refreshing every 2 seconds..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sleep 2
done