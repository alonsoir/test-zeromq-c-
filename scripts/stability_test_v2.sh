#!/bin/bash

╔════════════════════════════════════════════════════════════╗
║  Test de Estabilidad - Fundaciones Romanas                 ║
║  Duración: 2 horas                                         ║
║  Objetivo: Validar resistencia bajo carga sostenida        ║
╚════════════════════════════════════════════════════════════╝

START_TIME=$(date +%s)
END_TIME=$((START_TIME + 7200))  # 2 horas
DURATION_HOURS=2

echo "🏛️  Iniciando test de estabilidad estilo Via Appia"
echo ""
echo "⏰ Inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "⏰ Fin estimado: $(date -d @$END_TIME '+%Y-%m-%d %H:%M:%S')"
echo "⏱️  Duración: $DURATION_HOURS horas"
echo ""
echo "📊 Patrón de tráfico (cada 10 segundos):"
echo "   - 60% SSH (puerto 22)    → Filtrado en kernel"
echo "   - 30% App (puerto 8000)  → Capturado"
echo "   - 10% Misc (puerto 9999) → Default capture"
echo ""
echo "🎯 Criterios de éxito:"
echo "   ✅ Procesos vivos durante 2 horas"
echo "   ✅ ml-detector recibe solo tráfico relevante"
echo "   ✅ Sin crashes o memory leaks"
echo "   ✅ Sin saturación ZMQ"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Guardar estado inicial
ATTACKS_START=$(grep -c "Attack detected" /tmp/ml-detector.log 2>/dev/null || echo 0)
SNIFFER_PID=$(pgrep -f "sniffer -c")
DETECTOR_PID=$(pgrep -f "ml-detector -c")

if [ -z "$SNIFFER_PID" ] || [ -z "$DETECTOR_PID" ]; then
    echo "❌ ERROR: Sniffer o ml-detector no están corriendo"
    echo "   Sniffer PID: ${SNIFFER_PID:-NOT RUNNING}"
    echo "   Detector PID: ${DETECTOR_PID:-NOT RUNNING}"
    exit 1
fi

echo "✅ Procesos iniciales:"
echo "   Sniffer PID: $SNIFFER_PID"
echo "   ml-detector PID: $DETECTOR_PID"
echo "   Attacks iniciales: $ATTACKS_START"
echo ""

ITERATION=0
CHECKPOINTS=0
TOTAL_PACKETS_GENERATED=0

while [ $(date +%s) -lt $END_TIME ]; do
    ITERATION=$((ITERATION + 1))
    ELAPSED=$(($(date +%s) - START_TIME))
    REMAINING=$((END_TIME - $(date +%s)))
    PROGRESS=$((ELAPSED * 100 / 7200))
    
    # Progress bar
    BAR_LENGTH=50
    FILLED=$((PROGRESS * BAR_LENGTH / 100))
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')
    EMPTY=$(printf "%$((BAR_LENGTH - FILLED))s" | tr ' ' '░')
    
    echo -ne "\r[$(date +%H:%M:%S)] [$BAR$EMPTY] ${PROGRESS}% | Iter: $ITERATION | Restante: $((REMAINING/60))m   "
    
    # Generar tráfico mixto
    # 60% SSH (puerto 22) - será filtrado
    for i in {1..60}; do
        timeout 0.02 nc -vz localhost 22 >/dev/null 2>&1 &
    done
    TOTAL_PACKETS_GENERATED=$((TOTAL_PACKETS_GENERATED + 60))
    
    # 30% Aplicación (puerto 8000) - será capturado
    for i in {1..30}; do
        timeout 0.02 nc -vz localhost 8000 >/dev/null 2>&1 &
    done
    TOTAL_PACKETS_GENERATED=$((TOTAL_PACKETS_GENERATED + 30))
    
    # 10% Misceláneo (puerto 9999) - default action
    for i in {1..10}; do
        timeout 0.02 nc -vz localhost 9999 >/dev/null 2>&1 &
    done
    TOTAL_PACKETS_GENERATED=$((TOTAL_PACKETS_GENERATED + 10))
    
    # Esperar antes de siguiente iteración
    sleep 10
    
    # Checkpoint cada 10 iteraciones (~1.6 minutos)
    if [ $((ITERATION % 10)) -eq 0 ]; then
        CHECKPOINTS=$((CHECKPOINTS + 1))
        
        # Verificar que procesos siguen vivos
        if ! kill -0 $SNIFFER_PID 2>/dev/null; then
            echo -e "\n\n❌ FALLO CRÍTICO: Sniffer murió en iteración $ITERATION"
            exit 1
        fi
        
        if ! kill -0 $DETECTOR_PID 2>/dev/null; then
            echo -e "\n\n❌ FALLO CRÍTICO: ml-detector murió en iteración $ITERATION"
            exit 1
        fi
        
        # Stats rápidas
        ATTACKS_NOW=$(grep -c "Attack detected" /tmp/ml-detector.log 2>/dev/null || echo 0)
        ATTACKS_DELTA=$((ATTACKS_NOW - ATTACKS_START))
        
        echo -e "\n"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Checkpoint #$CHECKPOINTS ($(date '+%H:%M:%S'))"
        echo "   Tiempo transcurrido: $((ELAPSED / 60)) min"
        echo "   Paquetes generados: $TOTAL_PACKETS_GENERATED"
        echo "   Attacks detectados: $ATTACKS_DELTA nuevos (total: $ATTACKS_NOW)"
        echo "   Procesos: Sniffer ✅ | ml-detector ✅"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
done

# Test completado
ATTACKS_END=$(grep -c "Attack detected" /tmp/ml-detector.log 2>/dev/null || echo 0)
ATTACKS_TOTAL=$((ATTACKS_END - ATTACKS_START))
ELAPSED_FINAL=$(($(date +%s) - START_TIME))

echo -e "\n\n"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  TEST DE ESTABILIDAD COMPLETADO                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Duración real: $((ELAPSED_FINAL / 60)) minutos ($((ELAPSED_FINAL / 3600))h $((ELAPSED_FINAL % 3600 / 60))m)"
echo "🔄 Iteraciones: $ITERATION"
echo "📦 Paquetes generados: $TOTAL_PACKETS_GENERATED"
echo "🎯 Attacks detectados: $ATTACKS_TOTAL"
echo ""

# Verificar procesos finales
if kill -0 $SNIFFER_PID 2>/dev/null && kill -0 $DETECTOR_PID 2>/dev/null; then
    echo "✅ Ambos procesos siguen vivos"
    echo ""
    echo "🏛️  FUNDACIONES VALIDADAS - Digno de la Via Appia"
else
    echo "⚠️  Al menos un proceso murió durante el test"
fi

echo ""
echo "📊 Ejecuta el análisis post-test:"
echo "   bash /tmp/analyze_stability_results.sh"
echo ""
