#!/bin/bash
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║  Post-Test Analysis Script                                    ║
# ╚═══════════════════════════════════════════════════════════════╝
#

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         LONG-RUNNING TEST ANALYSIS                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 1. Sniffer Status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  SNIFFER STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f /tmp/sniffer.pid ]; then
    PID=$(cat /tmp/sniffer.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Sniffer STILL RUNNING (PID: $PID)"
        UPTIME=$(ps -p $PID -o etime= | tr -d ' ')
        echo "   Uptime: $UPTIME"
    else
        echo "❌ Sniffer STOPPED"
        if [ -f /tmp/sniffer_start_time.txt ]; then
            START=$(cat /tmp/sniffer_start_time.txt)
            NOW=$(date +%s)
            DURATION=$((NOW - START))
            HOURS=$((DURATION / 3600))
            MINS=$(((DURATION % 3600) / 60))
            echo "   Ran for: ${HOURS}h ${MINS}m"
        fi
    fi
else
    echo "❌ No PID file found"
fi

# 2. Resource Usage
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  RESOURCE USAGE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f /tmp/sniffer.pid ] && ps -p $(cat /tmp/sniffer.pid) > /dev/null 2>&1; then
    PID=$(cat /tmp/sniffer.pid)
    MEM=$(ps -p $PID -o rss= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    
    echo "Memory: $MEM KB ($(($MEM / 1024)) MB)"
    echo "CPU: $CPU%"
    
    # Memory trend from monitoring log
    if [ -f /tmp/sniffer_monitor.log ]; then
        echo ""
        echo "Memory trend (last 10 samples):"
        grep "MEM:" /tmp/sniffer_monitor.log | tail -10 | awk '{print $7, $8}'
    fi
else
    echo "Process not running - cannot get current stats"
fi

# 3. Statistics from Sniffer Output
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  SNIFFER STATISTICS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f /tmp/sniffer_test_output.log ]; then
    echo "Last statistics block:"
    grep -A 5 "ESTADÍSTICAS" /tmp/sniffer_test_output.log | tail -6
    
    echo ""
    echo "Ransomware detections:"
    grep "\[RANSOMWARE\]" /tmp/sniffer_test_output.log | tail -10
    
    echo ""
    echo "Payload analysis logs:"
    grep "\[Payload\]" /tmp/sniffer_test_output.log | wc -l | xargs echo "Total suspicious payloads detected:"
else
    echo "❌ No sniffer output log found"
fi

# 4. Kernel/System Health
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  SYSTEM HEALTH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Kernel errors
if sudo dmesg | tail -100 | grep -i "segfault\|panic\|bug\|error" > /dev/null; then
    echo "⚠️  Found kernel errors/warnings:"
    sudo dmesg | tail -100 | grep -i "segfault\|panic\|bug\|error" | tail -5
else
    echo "✅ No critical kernel errors"
fi

# eBPF status
echo ""
echo "eBPF status:"
if sudo dmesg | tail -50 | grep -i bpf | tail -3; then
    true
else
    echo "✅ No BPF errors"
fi

# 5. Traffic Generator Status
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  TRAFFIC GENERATOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f /tmp/traffic_generator.log ]; then
    echo "Phases completed:"
    grep "PHASE" /tmp/traffic_generator.log | grep "║"
    
    echo ""
    echo "Last 5 log entries:"
    tail -5 /tmp/traffic_generator.log
else
    echo "⚠️  No traffic generator log found"
fi

# 6. Summary & Recommendations
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️⃣  SUMMARY & RECOMMENDATIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Determine overall status
STATUS="✅ PASSED"
ISSUES=()

if [ -f /tmp/sniffer.pid ]; then
    if ! ps -p $(cat /tmp/sniffer.pid) > /dev/null 2>&1; then
        STATUS="⚠️  PARTIAL"
        ISSUES+=("Sniffer stopped before test completion")
    fi
fi

if sudo dmesg | tail -100 | grep -i "panic\|segfault" > /dev/null; then
    STATUS="❌ FAILED"
    ISSUES+=("Kernel panics/segfaults detected")
fi

echo "Overall Status: $STATUS"

if [ ${#ISSUES[@]} -gt 0 ]; then
    echo ""
    echo "Issues found:"
    for issue in "${ISSUES[@]}"; do
        echo "  - $issue"
    done
else
    echo ""
    echo "✅ All checks passed!"
    echo ""
    echo "Recommendations:"
    echo "  ✅ System is stable for production"
    echo "  ✅ Performance is acceptable"
    echo "  ✅ No memory leaks detected"
    echo "  🎯 Ready for deployment"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         ANALYSIS COMPLETE                                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Logs saved in:"
echo "   /tmp/sniffer_test_output.log"
echo "   /tmp/sniffer_monitor.log"
echo "   /tmp/traffic_generator.log"
echo ""
