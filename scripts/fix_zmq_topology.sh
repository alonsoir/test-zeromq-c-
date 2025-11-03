#!/bin/bash
set -euo pipefail

SNIFFER_CONFIG="/vagrant/sniffer/config/sniffer.json"
DETECTOR_CONFIG="/vagrant/ml-detector/config/ml_detector_config.json"
BACKUP_DIR="/home/vagrant/config_backups"

echo "======================================"
echo "ZMQ Topology Fix: PUB→PULL to PUSH→PULL"
echo "======================================"
echo ""

# Crear backup
mkdir -p "$BACKUP_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)
cp "$SNIFFER_CONFIG" "$BACKUP_DIR/sniffer.json.$timestamp"
cp "$DETECTOR_CONFIG" "$BACKUP_DIR/ml_detector_config.json.$timestamp"
echo "✅ Backups creados en $BACKUP_DIR/"

# Cambiar sniffer: PUB → PUSH
echo ""
echo "📝 Cambiando sniffer de PUB a PUSH..."
sed -i 's/"socket_type": "PUB"/"socket_type": "PUSH"/g' "$SNIFFER_CONFIG"

# Verificar cambio
if grep -q '"socket_type": "PUSH"' "$SNIFFER_CONFIG"; then
    echo "✅ Sniffer cambiado a PUSH correctamente"
else
    echo "❌ ERROR: No se pudo cambiar socket_type"
    exit 1
fi

# Verificar detector
echo ""
echo "🔍 Verificando detector (debe ser PULL)..."
if grep -q '"socket_type": "PULL"' "$DETECTOR_CONFIG"; then
    echo "✅ Detector ya usa PULL (correcto)"
else
    echo "❌ WARNING: Detector no usa PULL"
fi

echo ""
echo "======================================"
echo "✅ CAMBIO COMPLETADO"
echo "======================================"
echo ""
echo "Topología corregida:"
echo "  Sniffer:     PUSH bind 127.0.0.1:5571"
echo "  ML-Detector: PULL connect 127.0.0.1:5571"
echo ""
echo "Próximos pasos:"
echo "  1. pkill -9 sniffer; pkill -9 ml_detector"
echo "  2. cd /vagrant/ml-detector/build && ./ml-detector --verbose 2>&1 | tee /tmp/detector_post_fix.log"
echo "  3. cd /vagrant/sniffer/build && sudo ./sniffer --verbose 2>&1 | tee /tmp/sniffer_post_fix.log"
echo ""
