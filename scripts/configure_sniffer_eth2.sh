#!/bin/bash
set -e

SNIFFER_CONFIG="sniffer/config/sniffer.json"

echo "Configurando sniffer para eth2..."

# Backup
cp "$SNIFFER_CONFIG" "${SNIFFER_CONFIG}.bak"

# Actualizar interfaz a eth2
jq '.interface = "eth2"' "$SNIFFER_CONFIG" > "${SNIFFER_CONFIG}.tmp"
mv "${SNIFFER_CONFIG}.tmp" "$SNIFFER_CONFIG"

echo "✓ Configuración actualizada:"
jq '.interface' "$SNIFFER_CONFIG"

echo ""
echo "Siguiente paso: Iniciar servicios y sniffer"
echo "  make lab-start"
echo "  sudo ./sniffer/zeromq_sniffer"