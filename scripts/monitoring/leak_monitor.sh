#!/usr/bin/env bash

INTERVAL=10
LOGFILE="/vagrant/logs/lab/leak_monitor.log"

# Obtiene memoria RSS real (KB)
mem_kb() {
    local pid=$1
    grep -i "Rss:" /proc/$pid/smaps_rollup 2>/dev/null | awk '{print $2}'
}

# Encuentra PID real evitando procesos sudo/grep
find_pid() {
    local name="$1"
    # Busca proceso que contiene el nombre exacto en la línea de comando
    pgrep -f "$name" | head -1
}

resolve_pids() {
    FW=$(find_pid "firewall-acl-agent")
    DT=$(find_pid "ml-detector")
    SN=$(find_pid "sniffer")

    if [[ -z "$FW" || -z "$DT" || -z "$SN" ]]; then
        echo "⚠️  Warning: Algunos procesos no encontrados."
        echo "   Firewall PID: ${FW:-NOT FOUND}"
        echo "   Detector PID: ${DT:-NOT FOUND}"
        echo "   Sniffer PID:  ${SN:-NOT FOUND}"
        echo ""
        echo "   Continuando con los procesos disponibles..."
        echo ""
    fi
}

print_header() {
    echo "=============================================================="
    echo "        ML Defender - Leak Monitor ($(date))"
    echo "        Interval: ${INTERVAL}s"
    echo "        Logging to: $LOGFILE"
    echo "=============================================================="
    echo ""
    echo "PIDs: Firewall=$FW | Detector=$DT | Sniffer=$SN"
    echo ""
    echo " Time         | Firewall KB | Detector KB | Sniffer KB | Δ Total"
    echo "------------- |-------------|-------------|------------|--------"
}

log_stats() {
    local now=$(date +"%H:%M:%S")

    local fw="${FW:+$(mem_kb $FW)}"
    local dt="${DT:+$(mem_kb $DT)}"
    local sn="${SN:+$(mem_kb $SN)}"
    
    # Calcular total y delta
    local total=$((${fw:-0} + ${dt:-0} + ${sn:-0}))
    local delta=""
    
    if [[ -n "$PREV_TOTAL" ]]; then
        local diff=$((total - PREV_TOTAL))
        if [[ $diff -gt 0 ]]; then
            delta="+$diff"
        elif [[ $diff -lt 0 ]]; then
            delta="$diff"
        else
            delta="0"
        fi
    fi
    
    PREV_TOTAL=$total

    printf "%-13s | %-11s | %-11s | %-10s | %s\n" \
        "$now" "${fw:--}" "${dt:--}" "${sn:--}" "${delta:-N/A}"
    
    echo "$now,${fw:-0},${dt:-0},${sn:-0},$total,$delta" >> "$LOGFILE"
}

main_loop() {
    print_header
    while true; do
        log_stats
        sleep $INTERVAL
    done
}

# ----------------- EXEC -----------------
resolve_pids
main_loop
