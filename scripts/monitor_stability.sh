#!/bin/bash

# Monitoreo en tiempo real del test de estabilidad (VERSION CORREGIDA)

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Monitor de Estabilidad - Actualización cada 15s          ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    date
    echo ""
    
    # Uptime de procesos (FIX: tomar solo el primer PID)
    echo "🏃 PROCESOS ACTIVOS:"
    SNIFFER_PID=$(pgrep -f "build/sniffer" | head -1)
    DETECTOR_PID=$(pgrep -f "build/ml-detector" | head -1)
    
    if [ -n "$SNIFFER_PID" ]; then
        SNIFFER_TIME=$(ps -p $SNIFFER_PID -o etime= 2>/dev/null | xargs)
        SNIFFER_CPU=$(ps -p $SNIFFER_PID -o %cpu= 2>/dev/null | xargs)
        SNIFFER_MEM=$(ps -p $SNIFFER_PID -o %mem= 2>/dev/null | xargs)
        SNIFFER_RSS=$(ps -p $SNIFFER_PID -o rss= 2>/dev/null | awk '{print int($1/1024)}')
        echo "  ✅ Sniffer (PID $SNIFFER_PID)"
        echo "     Uptime: $SNIFFER_TIME | CPU: ${SNIFFER_CPU}% | MEM: ${SNIFFER_MEM}% (${SNIFFER_RSS}MB)"
    else
        echo "  ❌ Sniffer: NO RUNNING"
    fi
    
    if [ -n "$DETECTOR_PID" ]; then
        DETECTOR_TIME=$(ps -p $DETECTOR_PID -o etime= 2>/dev/null | xargs)
        DETECTOR_CPU=$(ps -p $DETECTOR_PID -o %cpu= 2>/dev/null | xargs)
        DETECTOR_MEM=$(ps -p $DETECTOR_PID -o %mem= 2>/dev/null | xargs)
        DETECTOR_RSS=$(ps -p $DETECTOR_PID -o rss= 2>/dev/null | awk '{print int($1/1024)}')
        echo "  ✅ ml-detector (PID $DETECTOR_PID)"
        echo "     Uptime: $DETECTOR_TIME | CPU: ${DETECTOR_CPU}% | MEM: ${DETECTOR_MEM}% (${DETECTOR_RSS}MB)"
    else
        echo "  ❌ ml-detector: NO RUNNING"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Stats del sniffer
    echo "📦 SNIFFER:"
    SNIFFER_STATS=$(tail -5 /tmp/sniffer.log 2>/dev/null | grep "ESTADÍSTICAS" -A 2 | tail -3)
    if [ -n "$SNIFFER_STATS" ]; then
        echo "$SNIFFER_STATS" | sed 's/^/  /'
    else
        echo "  (Sin estadísticas aún)"
    fi
    
    # Últimos errores ZMQ
    ZMQ_ERRORS=$(grep -c "ZMQ send falló" /tmp/sniffer.log 2>/dev/null || echo 0)
    if [ $ZMQ_ERRORS -eq 0 ]; then
        echo "  ✅ Sin errores ZMQ"
    else
        RECENT_ZMQ=$(tail -100 /tmp/sniffer.log | grep -c "ZMQ send falló")
        if [ $RECENT_ZMQ -eq 0 ]; then
            echo "  ✅ Sin errores ZMQ recientes (total histórico: $ZMQ_ERRORS)"
        else
            echo "  ⚠️  Errores ZMQ recientes: $RECENT_ZMQ (total: $ZMQ_ERRORS)"
        fi
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Stats del ml-detector
    echo "🤖 ML-DETECTOR:"
    TOTAL_ATTACKS=$(grep -c "Attack detected" /tmp/ml-detector.log 2>/dev/null || echo 0)
    RECENT_ATTACKS=$(tail -100 /tmp/ml-detector.log 2>/dev/null | grep -c "Attack detected")
    echo "  Total attacks: $TOTAL_ATTACKS"
    echo "  Últimos 100 líneas: $RECENT_ATTACKS attacks"
    
    # Últimas detecciones
    echo "  Últimas 3 detecciones:"
    grep "Attack detected" /tmp/ml-detector.log 2>/dev/null | tail -3 | while read line; do
        TIME=$(echo "$line" | grep -oP '\[\K[^\]]+')
        MSG=$(echo "$line" | cut -d']' -f3- | cut -c 2-)
        echo "    [$TIME] $MSG"
    done
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Memoria total del sistema
    echo "💾 SISTEMA:"
    free -h | grep Mem | awk '{print "  RAM: " $3 " usado de " $2 " (" int($3/$2*100) "%)"}'
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | xargs)
    echo "  Load average: $LOAD"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Test de estabilidad status
    if [ -f /tmp/stability.pid ]; then
        STAB_PID=$(cat /tmp/stability.pid)
        if kill -0 $STAB_PID 2>/dev/null; then
            echo "🧪 Test de estabilidad: ✅ CORRIENDO (PID $STAB_PID)"
        else
            echo "🧪 Test de estabilidad: ⚠️  NO CORRIENDO"
        fi
    else
        echo "🧪 Test de estabilidad: ❓ Estado desconocido"
    fi
    
    echo ""
    echo "🏛️  Construyendo fundaciones para mil años de runtime..."
    echo ""
    echo "Presiona Ctrl+C para salir"
    
    sleep 15
done
