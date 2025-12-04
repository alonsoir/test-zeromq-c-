#!/bin/bash
# ML Defender - Lab Monitoring Script (Enhanced v2.2)
# Shows: CPU, RAM, ZMQ ports, IPSet stats, config files, uptime, logs
# Includes: etcd-server, RAG, Dual-NIC monitoring
# scripts/monitor_lab.sh

PROJECT_ROOT="/vagrant"
LOG_DIR="$PROJECT_ROOT/logs/lab"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
ORANGE='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to get process stats with uptime
get_process_stats() {
    local proc_name=$1
    local pid=$(pgrep -f "$proc_name" | head -1)

    if [ -z "$pid" ]; then
        echo -e "${RED}❌ Not running${NC}"
        return 1
    fi

    # Get CPU and MEM from ps
    local stats=$(ps -p $pid -o %cpu,%mem,rss,etimes --no-headers 2>/dev/null)
    if [ -z "$stats" ]; then
        echo -e "${RED}❌ Process died${NC}"
        return 1
    fi

    local cpu=$(echo "$stats" | awk '{print $1}')
    local mem=$(echo "$stats" | awk '{print $2}')
    local rss=$(echo "$stats" | awk '{print $3}')
    local uptime_sec=$(echo "$stats" | awk '{print $4}')

    local rss_mb=$((rss / 1024))

    # Convert uptime to human readable
    local uptime_str=$(format_uptime $uptime_sec)

    echo -e "${GREEN}✅ PID $pid${NC} - CPU: ${CYAN}${cpu}%${NC} MEM: ${CYAN}${mem}%${NC} (${rss_mb}MB) - Uptime: ${YELLOW}${uptime_str}${NC}"
    return 0
}

# Format uptime seconds to human readable
format_uptime() {
    local seconds=$1
    local days=$((seconds / 86400))
    local hours=$(((seconds % 86400) / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))

    if [ $days -gt 0 ]; then
        echo "${days}d ${hours}h ${minutes}m"
    elif [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m ${secs}s"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}m ${secs}s"
    else
        echo "${secs}s"
    fi
}

# Function to get config file for a component
get_config_file() {
    local component=$1
    local config_file=""
    local found=0

    case $component in
        "firewall")
            config_file="/vagrant/firewall-acl-agent/config/firewall.json"
            if [ -f "$config_file" ]; then
                echo -e "${CYAN}$(basename $config_file)${NC}"
                found=1
            fi
            ;;
        "detector")
            config_file="/vagrant/ml-detector/config/ml_detector_config.json"
            if [ -f "$config_file" ]; then
                echo -e "${CYAN}$(basename $config_file)${NC}"
                found=1
            fi
            ;;
        "sniffer")
            config_file="/vagrant/sniffer/config/sniffer.json"
            if [ -f "$config_file" ]; then
                echo -e "${CYAN}$(basename $config_file)${NC}"
                found=1
            fi
            ;;
        "etcd")
            # Buscar en múltiples ubicaciones posibles
            if [ -f "/vagrant/etcd-server/config/etcd.conf" ]; then
                echo -e "${CYAN}etcd.conf${NC}"
                found=1
            elif [ -f "/vagrant/etcd-server/etcd.conf" ]; then
                echo -e "${CYAN}etcd.conf${NC}"
                found=1
            else
                echo -e "${YELLOW}No config file found${NC}"
            fi
            ;;
        "rag")
            config_file="/vagrant/rag/config/rag-config.json"
            if [ -f "$config_file" ]; then
                echo -e "${CYAN}$(basename $config_file)${NC}"
                found=1
            else
                echo -e "${YELLOW}No config file found${NC}"
            fi
            ;;
    esac
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
    local entries=$(sudo ipset list ml_defender_blacklist_test 2>/dev/null | grep -c "^[0-9]")
    local size=$(sudo ipset list ml_defender_blacklist_test 2>/dev/null | grep "Size in memory" | awk '{print $4}')

    if [ -z "$size" ]; then
        echo -e "${RED}❌ IPSet not found${NC}"
    else
        echo -e "${GREEN}✅ Active${NC} - Entries: ${CYAN}${entries}${NC} - Memory: ${CYAN}${size}B${NC}"
    fi
}

# Function to get etcd cluster status
get_etcd_status() {
    if pgrep -f "etcd-server" > /dev/null; then
        # Try to get etcd cluster health
        if command -v etcdctl > /dev/null; then
            local health=$(etcdctl endpoint health --endpoints=127.0.0.1:2379 2>/dev/null)
            if echo "$health" | grep -q "healthy"; then
                echo -e "${GREEN}✅ Healthy${NC}"
            else
                echo -e "${YELLOW}⚠️  Running${NC}"
            fi
        else
            echo -e "${GREEN}✅ Running${NC}"
        fi
    else
        echo -e "${RED}❌ Not running${NC}"
    fi
}

# Function to get sniffer dual-nic info
get_sniffer_info() {
    local config_file="/vagrant/sniffer/config/sniffer.json"
    if [ -f "$config_file" ]; then
        local profile=$(grep -o '"profile"[[:space:]]*:[[:space:]]*"[^"]*"' "$config_file" | cut -d'"' -f4)
        local interface=$(grep -o '"capture_interface"[[:space:]]*:[[:space:]]*"[^"]*"' "$config_file" | head -1 | cut -d'"' -f4)

        echo -e "${CYAN}Profile: ${profile}${NC}"
        echo -e "  ${YELLOW}Interface: ${interface}${NC}"
    fi
}

# Function to get RAG status
get_rag_status() {
    if pgrep -f "rag" > /dev/null; then
        # Check if RAG is responding (assuming HTTP port 8080)
        if command -v curl > /dev/null; then
            if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/health 2>/dev/null | grep -q "200\|OK"; then
                echo -e "${GREEN}✅ Healthy${NC}"
            else
                echo -e "${YELLOW}⚠️  Running${NC}"
            fi
        else
            echo -e "${GREEN}✅ Running${NC}"
        fi
    else
        echo -e "${RED}❌ Not running${NC}"
    fi
}

# Function to get system stats (sin bc)
get_system_stats() {
    echo -n "CPU: "
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}' | cut -d'.' -f1)
    if [ -z "$cpu_usage" ]; then
        cpu_usage=$(top -bn1 | grep "%Cpu" | awk '{print $2 + $4}' | cut -d'.' -f1)
    fi

    if [ -n "$cpu_usage" ]; then
        if [ "$cpu_usage" -gt 80 ]; then
            echo -e "${RED}${cpu_usage}%${NC}"
        elif [ "$cpu_usage" -gt 50 ]; then
            echo -e "${YELLOW}${cpu_usage}%${NC}"
        else
            echo -e "${GREEN}${cpu_usage}%${NC}"
        fi
    else
        echo -e "${RED}N/A${NC}"
    fi

    echo -n "RAM: "
    local mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ -n "$mem_usage" ]; then
        if [ "$mem_usage" -gt 80 ]; then
            echo -e "${RED}${mem_usage}%${NC}"
        elif [ "$mem_usage" -gt 50 ]; then
            echo -e "${YELLOW}${mem_usage}%${NC}"
        else
            echo -e "${GREEN}${mem_usage}%${NC}"
        fi
    else
        echo -e "${RED}N/A${NC}"
    fi

    echo -n "Disk: "
    local disk_usage=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ -n "$disk_usage" ]; then
        if [ "$disk_usage" -gt 80 ]; then
            echo -e "${RED}${disk_usage}%${NC}"
        elif [ "$disk_usage" -gt 50 ]; then
            echo -e "${YELLOW}${disk_usage}%${NC}"
        else
            echo -e "${GREEN}${disk_usage}%${NC}"
        fi
    else
        echo -e "${RED}N/A${NC}"
    fi
}

# Function to show tail -f commands
show_log_commands() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}📋 Log File Commands${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    declare -A log_files=(
        ["Firewall"]="$LOG_DIR/firewall.log"
        ["Detector"]="$LOG_DIR/detector.log"
        ["Sniffer"]="$LOG_DIR/sniffer.log"
        ["etcd"]="$LOG_DIR/etcd-server.log"
        ["RAG"]="$LOG_DIR/rag.log"
    )

    for component in "${!log_files[@]}"; do
        if [ -f "${log_files[$component]}" ]; then
            echo -e "${WHITE}${component}:${NC}  tail -f ${log_files[$component]}"
        fi
    done

    echo -e "${WHITE}All logs:${NC}   tail -f $LOG_DIR/*.log"
}

# Function to get dual-nic configuration from sniffer log
get_dual_nic_config() {
    local log_file="$LOG_DIR/sniffer.log"
    if [ ! -f "$log_file" ]; then
        return 1
    fi

    # Buscar la sección de configuración dual-nic
    local start_line=$(grep -n "Dual-NIC Deployment Configuration" "$log_file" | head -1 | cut -d: -f1)

    if [ -z "$start_line" ]; then
        return 1
    fi

    # Extraer desde la línea de inicio hasta la próxima línea de cierre
    local end_line=$(tail -n +$start_line "$log_file" | grep -n "╚════════════════════════════════════════════════════════════╝" | head -1 | cut -d: -f1)

    if [ -z "$end_line" ]; then
        # Si no encontramos el cierre, tomar las siguientes 12 líneas
        tail -n +$start_line "$log_file" | head -12
    else
        # Tomar desde la línea de inicio hasta la línea de cierre
        tail -n +$start_line "$log_file" | head -$((end_line))
    fi
}

# Main monitoring loop
clear
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    SYSTEM_UPTIME=$(uptime -p | sed 's/up //')

    # Header
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ML Defender Lab - Live Monitoring (Enhanced v2.2)         ║"
    echo "║  $TIMESTAMP                                ║"
    echo "║  System Uptime: $SYSTEM_UPTIME                            "
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # ========================================
    # System Stats
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}📈 System Statistics${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    get_system_stats
    echo ""

    # ========================================
    # Component Status
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}📊 Component Status & Configuration${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo -n "🔥 Firewall:  "
    if get_process_stats "firewall-acl-agent"; then
        echo -n "   Config: "
        get_config_file "firewall"
    fi

    echo ""
    echo -n "🤖 Detector:  "
    if get_process_stats "ml-detector"; then
        echo -n "   Config: "
        get_config_file "detector"
    fi

    echo ""
    echo -n "📡 Sniffer:   "
    if get_process_stats "sniffer"; then
        echo ""
        get_sniffer_info
    fi

    echo ""
    echo -n "🗄️  etcd-server: "
    if get_process_stats "etcd-server"; then
        echo -n "   Status: "
        get_etcd_status
        echo -n "          Config: "
        get_config_file "etcd"
    fi

    echo ""
    echo -n "🧠 RAG Engine: "
    if get_process_stats "rag"; then
        echo -n "   Status: "
        get_rag_status
        echo -n "          Config: "
        get_config_file "rag"
    fi

    echo ""

    # ========================================
    # Network Ports
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}🔌 Communication Channels${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo -n "Port 5571 (Sniffer → Detector): "
    check_zmq_ports 5571

    echo -n "Port 5572 (Detector → Firewall): "
    check_zmq_ports 5572

    echo -n "Port 2379 (etcd client): "
    check_zmq_ports 2379

    echo -n "Port 2380 (etcd peer): "
    check_zmq_ports 2380

    echo -n "Port 8080 (RAG API): "
    check_zmq_ports 8080

    echo ""

    # ========================================
    # IPSet Status
    # ========================================
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}🔥 IPSet Blacklist Status${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo -n "ml_defender_blacklist: "
    get_ipset_stats

    # Show last 5 blocked IPs if any
    blocked_ips=$(sudo ipset list ml_defender_blacklist_test 2>/dev/null | grep "^[0-9]" | tail -5)
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
    echo -e "${BLUE}📋 Recent Activity (last 5 lines)${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    declare -A log_files=(
        ["🔥 Firewall"]="$LOG_DIR/firewall.log"
        ["🤖 Detector"]="$LOG_DIR/detector.log"
        ["📡 Sniffer"]="$LOG_DIR/sniffer.log"
        ["🗄️  etcd"]="$LOG_DIR/etcd-server.log"
        ["🧠 RAG"]="$LOG_DIR/rag.log"
    )

    for component in "${!log_files[@]}"; do
        if [ -f "${log_files[$component]}" ]; then
            echo -e "${MAGENTA}$component:${NC}"

            # Get base component name for pattern matching
            base_comp=$(echo "$component" | sed 's/[^a-zA-Z0-9 ]//g' | xargs)

            case $base_comp in
                "Firewall")
                    tail -5 "${log_files[$component]}" 2>/dev/null | sed 's/^/  /'
                    ;;
                "Detector")
                    tail -50 "${log_files[$component]}" 2>/dev/null | grep -E "Stats|ERROR|WARNING|Detection|Processed" | tail -3 | sed 's/^/  /'
                    ;;
                "Sniffer")
                    tail -50 "${log_files[$component]}" 2>/dev/null | grep -E "Paquetes procesados:|Paquetes enviados:|Tasa:|Dual-NIC|eth[0-9]" | tail -5 | sed 's/^/  /'
                    ;;
                "etcd")
                    tail -5 "${log_files[$component]}" 2>/dev/null | sed 's/^/  /'
                    ;;
                "RAG")
                    tail -5 "${log_files[$component]}" 2>/dev/null | sed 's/^/  /'
                    ;;
            esac
            echo ""
        fi
    done

    # ========================================
    # Dual-NIC Specific Information
    # ========================================
    if pgrep -f "sniffer" > /dev/null && [ -f "$LOG_DIR/sniffer.log" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "${BLUE}🔧 Dual-NIC Deployment Status${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Obtener información dual-nic
        dual_nic_info=$(get_dual_nic_config)

        if [ -n "$dual_nic_info" ]; then
            echo "$dual_nic_info"
        else
            echo -e "  ${YELLOW}No dual-nic configuration found in logs${NC}"
        fi
        echo ""
    fi

    # ========================================
    # Log Commands
    # ========================================
    show_log_commands

    # ========================================
    # Footer
    # ========================================
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${YELLOW}Press Ctrl+C to exit${NC} | Refreshing every 3 seconds..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sleep 3
done