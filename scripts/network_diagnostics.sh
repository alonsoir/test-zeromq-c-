#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== DIAGNÓSTICO DE RED - ZeroMQ Lab ==="
echo ""

echo -e "${GREEN}1. INTERFACES DE RED${NC}"
ip -4 addr show | grep -E "^[0-9]+:|inet " | sed 's/^/  /'
echo ""

echo -e "${GREEN}2. TABLA DE RUTAS${NC}"
ip route | sed 's/^/  /'
echo ""

ETH0_IP=$(ip -4 addr show eth0 2>/dev/null | grep inet | awk '{print $2}' | cut -d'/' -f1)
ETH1_IP=$(ip -4 addr show eth1 2>/dev/null | grep inet | awk '{print $2}' | cut -d'/' -f1)
ETH2_IP=$(ip -4 addr show eth2 2>/dev/null | grep inet | awk '{print $2}' | cut -d'/' -f1)

echo -e "${GREEN}3. IPs CONFIGURADAS${NC}"
echo "  NAT (eth0):             ${ETH0_IP:-N/A}"
echo "  Private Network (eth1): ${ETH1_IP:-N/A}"
echo "  Bridged Network (eth2): ${ETH2_IP:-N/A}"
echo ""

echo -e "${GREEN}4. CONECTIVIDAD${NC}"
ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1 && echo -e "  Internet: ${GREEN}✓ OK${NC}" || echo -e "  Internet: ${RED}✗ FALLO${NC}"
command -v tcpdump >/dev/null 2>&1 && echo -e "  tcpdump: ${GREEN}✓ Instalado${NC}" || echo -e "  tcpdump: ${YELLOW}○ No instalado${NC}"
echo ""

echo -e "${GREEN}5. KERNEL Y EBPF${NC}"
echo "  Kernel: $(uname -r)"
grep -q CONFIG_BPF=y /boot/config-$(uname -r) 2>/dev/null && echo -e "  eBPF: ${GREEN}✓ Soportado${NC}" || echo -e "  eBPF: ${YELLOW}? Desconocido${NC}"
echo ""

echo "=== Diagnóstico completado ==="
