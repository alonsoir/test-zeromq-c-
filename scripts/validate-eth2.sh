#!/bin/bash
# validate-eth2.sh
# Validación rápida específica para eth2 (interfaz bridged)
# Uso: ./scripts/validate-eth2.sh

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================"
echo "VALIDACIÓN DE eth2 (Interfaz Bridged)"
echo "================================================================"
echo ""

# 1. Verificar que eth2 existe
echo -e "${BLUE}[1/7]${NC} Verificando existencia de eth2..."
if ip link show eth2 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} eth2 detectada"
else
    echo -e "${RED}✗${NC} eth2 NO encontrada"
    echo ""
    echo "DIAGNÓSTICO:"
    echo "  - Verifica que Vagrantfile tiene: config.vm.network \"public_network\""
    echo "  - Ejecuta: vagrant reload"
    echo "  - Verifica interfaces disponibles en el host"
    echo ""
    exit 1
fi
echo ""

# 2. Verificar estado de eth2
echo -e "${BLUE}[2/7]${NC} Verificando estado de eth2..."
STATE=$(ip link show eth2 | grep -oP 'state \K\w+')
if [ "$STATE" == "UP" ]; then
    echo -e "${GREEN}✓${NC} eth2 está UP"
else
    echo -e "${YELLOW}⚠${NC} eth2 está $STATE"
    echo "  Intentando levantar eth2..."
    sudo ip link set eth2 up
    sleep 2
    STATE=$(ip link show eth2 | grep -oP 'state \K\w+')
    if [ "$STATE" == "UP" ]; then
        echo -e "${GREEN}✓${NC} eth2 ahora está UP"
    else
        echo -e "${RED}✗${NC} No se pudo levantar eth2"
        exit 1
    fi
fi
echo ""

# 3. Verificar IP asignada
echo -e "${BLUE}[3/7]${NC} Verificando IP de eth2..."
ETH2_IP=$(ip -4 addr show eth2 | grep inet | awk '{print $2}' | cut -d'/' -f1)
if [ -n "$ETH2_IP" ]; then
    echo -e "${GREEN}✓${NC} eth2 tiene IP asignada: $ETH2_IP"

    # Verificar que la IP es de una red local común
    if [[ $ETH2_IP =~ ^192\.168\. ]] || [[ $ETH2_IP =~ ^10\. ]] || [[ $ETH2_IP =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]]; then
        echo -e "${GREEN}✓${NC} IP en rango de red local válido"
    else
        echo -e "${YELLOW}⚠${NC} IP fuera de rangos típicos de red local"
    fi
else
    echo -e "${RED}✗${NC} eth2 NO tiene IP asignada"
    echo ""
    echo "DIAGNÓSTICO:"
    echo "  - Verifica que tu router DHCP está activo"
    echo "  - Prueba renovar DHCP: sudo dhclient -r eth2 && sudo dhclient eth2"
    echo "  - Verifica que la interfaz del host está conectada a la red"
    exit 1
fi
echo ""

# 4. Verificar gateway
echo -e "${BLUE}[4/7]${NC} Verificando gateway para eth2..."
ETH2_GW=$(ip route | grep "^default.*eth2" | awk '{print $3}')
if [ -n "$ETH2_GW" ]; then
    echo -e "${GREEN}✓${NC} Gateway configurado: $ETH2_GW"
else
    # Intentar obtener el gateway de la red
    ETH2_GW=$(ip route | grep "dev eth2.*scope link" | awk '{print $1}' | grep -oP '\d+\.\d+\.\d+\.1$')
    if [ -n "$ETH2_GW" ]; then
        echo -e "${YELLOW}⚠${NC} Gateway inferido: $ETH2_GW (no es default)"
    else
        echo -e "${YELLOW}⚠${NC} No se detectó gateway para eth2"
        echo "  (Esto puede ser normal si eth0 es el default gateway)"
    fi
fi
echo ""

# 5. Test de conectividad básica
echo -e "${BLUE}[5/7]${NC} Probando conectividad desde eth2..."

# Ping al gateway si existe
if [ -n "$ETH2_GW" ]; then
    echo -n "  Ping al gateway ($ETH2_GW): "
    if ping -c 2 -W 3 -I eth2 $ETH2_GW > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FALLO${NC}"
    fi
fi

# Ping a DNS público
echo -n "  Ping a 8.8.8.8: "
if ping -c 3 -W 3 -I eth2 8.8.8.8 > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}FALLO${NC} (puede ser bloqueado por firewall)"
fi

# Ping a un host local común
echo -n "  Ping a 192.168.1.1: "
if ping -c 2 -W 2 -I eth2 192.168.1.1 > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}No responde${NC} (puede no ser tu gateway)"
fi

echo ""

# 6. Verificar capacidades de captura
echo -e "${BLUE}[6/7]${NC} Verificando capacidades de captura en eth2..."

# Verificar que tcpdump está instalado
if command -v tcpdump > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} tcpdump instalado"

    # Test rápido de captura (requiere sudo)
    echo -n "  Test de captura (5 segundos): "

    # Generar tráfico en background
    (ping -c 5 -I eth2 8.8.8.8 > /dev/null 2>&1 &)

    # Capturar
    if sudo timeout 5 tcpdump -i eth2 -c 5 -w /tmp/test_eth2.pcap > /dev/null 2>&1; then
        PACKET_COUNT=$(sudo tcpdump -r /tmp/test_eth2.pcap 2>/dev/null | wc -l)
        if [ "$PACKET_COUNT" -gt 0 ]; then
            echo -e "${GREEN}OK${NC} ($PACKET_COUNT paquetes)"
        else
            echo -e "${YELLOW}0 paquetes${NC}"
        fi
        sudo rm -f /tmp/test_eth2.pcap
    else
        echo -e "${RED}FALLO${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} tcpdump no instalado"
    echo "  Instalar: sudo apt-get install tcpdump"
fi
echo ""

# 7. Verificar que el sniffer puede usar eth2
echo -e "${BLUE}[7/7]${NC} Verificando compatibilidad con sniffer eBPF..."

# Verificar que existe el binario del sniffer
if [ -f "/vagrant/sniffer/build/sniffer" ]; then
    echo -e "${GREEN}✓${NC} Binario del sniffer encontrado"

    # Verificar permisos CAP_NET_RAW
    echo -n "  Verificando capacidades de red: "
    if sudo -n /vagrant/sniffer/build/sniffer --help > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}Requiere sudo${NC}"
    fi

    # Test rápido del sniffer (2 segundos)
    echo -n "  Test del sniffer en eth2 (2 seg): "
    if timeout 2 sudo /vagrant/sniffer/build/sniffer --verbose 2>&1 | grep -q "eth2\|Listening"; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}No pudo detectar actividad${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} Sniffer no compilado"
    echo "  Compilar con: cd /vagrant && make sniffer-build-local"
fi
echo ""

# RESUMEN FINAL
echo "================================================================"
echo "RESUMEN DE VALIDACIÓN"
echo "================================================================"
echo ""

echo "Configuración de eth2:"
echo "  Interfaz: eth2"
echo "  Estado: $STATE"
echo "  IP: ${ETH2_IP:-No asignada}"
echo "  Gateway: ${ETH2_GW:-No detectado}"
echo "  MAC: $(ip link show eth2 | grep link/ether | awk '{print $2}')"
echo ""

# Verificar si todo está OK
VALIDATION_OK=true

if [ "$STATE" != "UP" ]; then
    VALIDATION_OK=false
fi

if [ -z "$ETH2_IP" ]; then
    VALIDATION_OK=false
fi

if [ "$VALIDATION_OK" = true ]; then
    echo -e "${GREEN}✓ eth2 ESTÁ LISTA PARA USAR CON EL SNIFFER${NC}"
    echo ""
    echo "Comandos útiles:"
    echo "  # Ver tráfico en tiempo real"
    echo "  sudo tcpdump -i eth2 -n"
    echo ""
    echo "  # Ejecutar sniffer"
    echo "  cd /vagrant"
    echo "  sudo ./sniffer/build/sniffer --verbose"
    echo ""
    echo "  # Capturar tráfico por 60 segundos"
    echo "  ./scripts/capture_zeromq_traffic.sh eth2 60"
    echo ""
    exit 0
else
    echo -e "${RED}✗ eth2 TIENE PROBLEMAS${NC}"
    echo ""
    echo "Pasos para solucionar:"
    echo "  1. Verifica Vagrantfile tiene: config.vm.network \"public_network\""
    echo "  2. Ejecuta: vagrant reload"
    echo "  3. Verifica que tu WiFi/LAN tiene DHCP activo"
    echo "  4. Ejecuta nuevamente: ./scripts/validate-eth2.sh"
    echo ""
    exit 1
fi