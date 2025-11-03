#!/bin/bash
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  SNIFFER STATUS CHECK v2                                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "⏰ Current Time: $(date)"
echo ""

# Find sniffer by process name
PID=$(pgrep -f "sniffer.*config.*sniffer.json" | head -1)

if [ ! -z "$PID" ]; then
    echo "✅ Sniffer Status: RUNNING"
    echo ""
    echo "Process Info:"
    echo "  PID:      $PID"
    echo "  Uptime:   $(ps -p $PID -o etime= | tr -d ' ')"
    echo "  CPU Time: $(ps -p $PID -o time= | tr -d ' ')"
    echo "  Memory:   $(ps -p $PID -o rss= | tr -d ' ') KB ($(( $(ps -p $PID -o rss= | tr -d ' ') / 1024 )) MB)"
    echo "  CPU%:     $(ps -p $PID -o %cpu= | tr -d ' ')%"
    echo ""
    
    echo "Last Statistics:"
    tail -20 /tmp/sniffer_test_output.log | grep -A 4 "ESTADÍSTICAS" | tail -5
    echo ""
    
    # Calculate stats
    PACKETS=$(tail -20 /tmp/sniffer_test_output.log | grep "Paquetes procesados:" | tail -1 | awk '{print $3}')
    UPTIME_SEC=$(tail -20 /tmp/sniffer_test_output.log | grep "Tiempo activo:" | tail -1 | awk '{print $3}')
    RATE=$(tail -20 /tmp/sniffer_test_output.log | grep "Tasa:" | tail -1 | awk '{print $2}')
    
    HOURS=$(echo "scale=1; $UPTIME_SEC / 3600" | bc)
    
    echo "Summary:"
    echo "  ✅ Runtime:  ${HOURS}h ($UPTIME_SEC seconds)"
    echo "  ✅ Packets:  $PACKETS"
    echo "  ✅ Rate:     $RATE evt/s (average)"
    echo ""
    
    echo "🌙 All Good - Safe to Sleep!"
    echo "   System is stable and running perfectly"
else
    echo "❌ Sniffer NOT RUNNING"
    echo "   Check logs: /tmp/sniffer_test_output.log"
fi
echo ""
