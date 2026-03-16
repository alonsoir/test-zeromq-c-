#!/bin/bash
clear
while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ML DEFENDER STRESS TEST - LIVE MONITORING                     ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Tiempo transcurrido
    TEST_DIR=$(ls -td /vagrant/stress_test_* 2>/dev/null | head -1)
    if [ -z "$TEST_DIR" ]; then
        echo "❌ No active test found"
        exit 1
    fi
    
    START_TIME=$(stat -c %Y "$TEST_DIR")
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    
    echo "⏱️  Elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "📁 Test dir: $TEST_DIR"
    echo ""
    
    # Procesos
    echo "🔄 Processes:"
    SNIFFER_PID=$(pgrep -f "sniffer.*sniffer.json" || echo "N/A")
    ML_PID=$(pgrep -f "ml-detector.*ml_detector" || echo "N/A")
    echo "   Sniffer: PID ${SNIFFER_PID}"
    echo "   ML-Detector: PID ${ML_PID}"
    echo ""
    
    # Memoria
    echo "💾 Memory Usage:"
    if [ -f "${TEST_DIR}/monitoring/memory.csv" ]; then
        tail -1 "${TEST_DIR}/monitoring/memory.csv" | awk -F',' '{printf "   Total: %s MB (Sniffer: %s MB, ML-Detector: %s MB)\n", $6, $2, $4}'
    else
        echo "   Waiting for data..."
    fi
    echo ""
    
    # CPU
    echo "⚡ CPU Usage:"
    if [ -f "${TEST_DIR}/monitoring/cpu.csv" ]; then
        tail -1 "${TEST_DIR}/monitoring/cpu.csv" | awk -F',' '{printf "   Sniffer: %s%%, ML-Detector: %s%%, System: %s%%\n", $2, $3, $4}'
    else
        echo "   Waiting for data..."
    fi
    echo ""
    
    # Últimas detecciones
    echo "🛡️  Last ML Detections:"
    if [ -f "${TEST_DIR}/logs/sniffer.log" ]; then
        grep "ML Defender Embedded Detectors:" "${TEST_DIR}/logs/sniffer.log" -A 5 | tail -6 | sed 's/^/   /'
    else
        echo "   Waiting for data..."
    fi
    echo ""
    
    # Stats generales
    echo "📊 Sniffer Stats:"
    if [ -f "${TEST_DIR}/logs/sniffer.log" ]; then
        grep "Events processed:" "${TEST_DIR}/logs/sniffer.log" | tail -1 | sed 's/^/   /'
        grep "Avg ML detection time:" "${TEST_DIR}/logs/sniffer.log" | tail -1 | sed 's/^/   /'
    else
        echo "   Waiting for data..."
    fi
    echo ""
    
    echo "Press Ctrl+C to exit monitor (test continues running)"
    echo "Refreshing every 5 seconds..."
    
    sleep 5
done
