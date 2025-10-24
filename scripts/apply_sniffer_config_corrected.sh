#!/bin/bash
# Script CORREGIDO para aplicar configuración estable a sniffer.json
# NO cambia topología (PUB funciona correctamente)
# Solo ajusta: HWM, buffers, threads, timeouts

set -e

SNIFFER_JSON="/vagrant/sniffer/config/sniffer.json"
BACKUP_JSON="${SNIFFER_JSON}.backup.$(date +%Y%m%d_%H%M%S)"

echo "🔧 Aplicando Configuración Estable al Sniffer (CORREGIDO)"
echo "========================================================="
echo ""

if [ ! -f "$SNIFFER_JSON" ]; then
    echo "❌ Error: No se encuentra $SNIFFER_JSON"
    exit 1
fi

echo "📦 Creando backup..."
cp "$SNIFFER_JSON" "$BACKUP_JSON"
echo "✓ Backup creado: $BACKUP_JSON"
echo ""

if command -v jq &> /dev/null; then
    echo "✓ jq encontrado - usando método automático"
    echo ""
    echo "🔄 Aplicando cambios (sin cambiar topología PUB)..."
    
    jq '
    # NO cambiar socket_type - funciona con PUB
    # Solo ajustar valores problemáticos
    
    # 1. Balancear HWM
    .network.output_socket.address = "127.0.0.1" |
    .network.output_socket.high_water_mark = 1000 |
    
    # 2. Reducir threads conservadoramente
    .threading.ring_consumer_threads = 1 |
    .threading.feature_processor_threads = 1 |
    .threading.zmq_sender_threads = 1 |
    .threading.statistics_collector_threads = 1 |
    .threading.total_worker_threads = 4 |
    
    # 3. Ajustar ZMQ settings
    .zmq.worker_threads = 1 |
    .zmq.io_thread_pools = 1 |
    .zmq.queue_management.internal_queues = 2 |
    .zmq.queue_management.queue_size = 300 |
    .zmq.connection_settings.sndhwm = 1000 |
    .zmq.connection_settings.rcvhwm = 1000 |
    .zmq.connection_settings.linger_ms = 100 |
    .zmq.connection_settings.send_timeout_ms = 250 |
    .zmq.connection_settings.recv_timeout_ms = 500 |
    .zmq.connection_settings.sndbuf = 262144 |
    .zmq.connection_settings.rcvbuf = 262144 |
    
    # 4. Reducir buffers
    .buffers.ring_buffer_entries = 65536 |
    .buffers.user_processing_queue_depth = 300 |
    .buffers.zmq_send_buffer_size = 262144 |
    .buffers.flow_state_buffer_entries = 100000 |
    .buffers.statistics_buffer_entries = 10000 |
    
    # 5. Añadir filtro puerto 22
    .capture.filter_expression = "not port 22"
    ' "$SNIFFER_JSON" > "${SNIFFER_JSON}.tmp"
    
    if jq empty "${SNIFFER_JSON}.tmp" 2>/dev/null; then
        mv "${SNIFFER_JSON}.tmp" "$SNIFFER_JSON"
        echo "✅ Cambios aplicados exitosamente"
    else
        echo "❌ Error: JSON inválido"
        cp "$BACKUP_JSON" "$SNIFFER_JSON"
        rm -f "${SNIFFER_JSON}.tmp"
        exit 1
    fi
else
    echo "⚠️  jq no encontrado - aplicar manualmente"
    exit 0
fi

echo ""
echo "📊 Resumen de cambios:"
echo "  ✓ Socket: PUB (NO CAMBIÓ - funciona correctamente)"
echo "  ✓ HWM: 2000 → 1000"
echo "  ✓ sndhwm: 10000 → 1000 (balanceado con detector)"
echo "  ✓ Buffers: 1MB → 256KB (balanceado)"
echo "  ✓ Threads: 9 → 4 (reducir context switching)"
echo "  ✓ Timeouts: 100ms → 250-500ms"
echo "  ✓ Linger: 0 → 100ms (evitar pérdida)"
echo "  ✓ Queue: 10K → 300 (conservador)"
echo "  ✓ Filtro puerto 22 añadido"
echo ""
echo "💾 Backup: $BACKUP_JSON"
