#!/bin/bash
set -e

echo "=== Test eBPF Sniffer en eth2 ==="
echo ""

# 1. Verificar interfaz
ETH2_IP=$(ip -4 addr show eth2 | grep inet | awk '{print $2}' | cut -d'/' -f1)
if [ -z "$ETH2_IP" ]; then
    echo "ERROR: eth2 no tiene IP"
    exit 1
fi
echo "✓ eth2 IP: $ETH2_IP"

# 2. Verificar servicios
echo ""
echo "Verificando servicios..."
if ! sudo ss -tulpn | grep -q ':5555'; then
    echo "⚠ Puerto 5555 (ZeroMQ) no está escuchando"
    echo "  Ejecuta: make lab-start"
    exit 1
fi
echo "✓ Servicio ZeroMQ detectado en puerto 5555"

# 3. Test de captura básica
echo ""
echo "Test de captura en eth2 (10 seg)..."
timeout 10 sudo tcpdump -i eth2 -n port 5555 -c 5 2>&1 | tail -8

echo ""
echo "=== Listo para ejecutar sniffer ==="
echo "  sudo ./sniffer --verbose"