#!/bin/bash
CAPTURE_DIR="/tmp/zeromq_captures"
mkdir -p "$CAPTURE_DIR"

INTERFACE=${1:-eth2}
DURATION=${2:-60}
OUTPUT="$CAPTURE_DIR/zeromq_$(date +%Y%m%d_%H%M%S).pcap"

echo "=== Captura de Tráfico ZeroMQ ==="
echo "Interfaz: $INTERFACE"
echo "Duración: ${DURATION}s"
echo "Output: $OUTPUT"
echo ""

sudo tcpdump -i "$INTERFACE" -w "$OUTPUT" \
    'port 5555 or port 2379 or port 2380 or port 5571' -v &

TCPDUMP_PID=$!
echo "Captura iniciada (PID: $TCPDUMP_PID)"
sleep "$DURATION"
sudo kill $TCPDUMP_PID 2>/dev/null
wait $TCPDUMP_PID 2>/dev/null || true

echo ""
echo "Captura finalizada: $OUTPUT"
echo "Primeros 20 paquetes:"
sudo tcpdump -r "$OUTPUT" -n | head -20
