#!/bin/bash
# Script CORREGIDO para aplicar configuración estable a ml-detector.json
# NO cambia topología (PULL funciona correctamente)
# Solo ajusta: HWM, buffers, threads

set -e

DETECTOR_JSON="/vagrant/ml-detector/config/ml_detector_config.json"
BACKUP_JSON="${DETECTOR_JSON}.backup.$(date +%Y%m%d_%H%M%S)"

echo "🔧 Aplicando Configuración Estable al ML-Detector (CORREGIDO)"
echo "============================================================="
echo ""

if [ ! -f "$DETECTOR_JSON" ]; then
    echo "❌ Error: No se encuentra $DETECTOR_JSON"
    exit 1
fi

echo "📦 Creando backup..."
cp "$DETECTOR_JSON" "$BACKUP_JSON"
echo "✓ Backup creado: $BACKUP_JSON"
echo ""

if command -v jq &> /dev/null; then
    echo "✓ jq encontrado - usando método automático"
    echo ""
    echo "🔄 Aplicando cambios (sin cambiar topología PULL)..."
    
    jq '
    # NO cambiar socket_type - funciona con PULL
    # Solo ajustar valores para balancear con sniffer
    
    # 1. Aumentar HWM para match con sniffer
    .network.input_socket.high_water_mark = 1000 |
    .network.output_socket.high_water_mark = 1000 |
    
    # 2. Ajustar ZMQ settings para balance con sniffer
    .zmq.connection_settings.sndhwm = 1000 |
    .zmq.connection_settings.rcvhwm = 1000 |
    .zmq.connection_settings.linger_ms = 100 |
    .zmq.connection_settings.rcvbuf = 262144 |
    .zmq.connection_settings.sndbuf = 262144 |
    
    # 3. Reducir threads ligeramente
    .threading.worker_threads = 2 |
    .threading.ml_inference_threads = 2 |
    .threading.feature_extractor_threads = 1 |
    .threading.total_worker_threads = 7 |
    
    # 4. Desactivar compresión (overhead)
    .transport.compression.enabled = false |
    
    # 5. Actualizar profile lab
    .profiles.lab.worker_threads = 2
    ' "$DETECTOR_JSON" > "${DETECTOR_JSON}.tmp"
    
    if jq empty "${DETECTOR_JSON}.tmp" 2>/dev/null; then
        mv "${DETECTOR_JSON}.tmp" "$DETECTOR_JSON"
        echo "✅ Cambios aplicados exitosamente"
    else
        echo "❌ Error: JSON inválido"
        cp "$BACKUP_JSON" "$DETECTOR_JSON"
        rm -f "${DETECTOR_JSON}.tmp"
        exit 1
    fi
else
    echo "⚠️  jq no encontrado - aplicar manualmente"
    exit 0
fi

echo ""
echo "📊 Resumen de cambios:"
echo "  ✓ Socket: PULL (NO CAMBIÓ - funciona correctamente)"
echo "  ✓ HWM input: 500 → 1000 (balanceado con sniffer)"
echo "  ✓ rcvhwm: 500 → 1000 (balanceado)"
echo "  ✓ rcvbuf: 128KB → 256KB (balanceado)"
echo "  ✓ Linger: 0 → 100ms (evitar pérdida)"
echo "  ✓ Threads: 10 → 7 (reducir overhead)"
echo "  ✓ Compresión: ON → OFF (reducir overhead)"
echo ""
echo "💾 Backup: $BACKUP_JSON"
