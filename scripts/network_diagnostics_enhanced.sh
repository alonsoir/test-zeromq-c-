#!/bin/bash
# network_diagnostics_enhanced.sh
# Diagnóstico completo de red para validar eth2 bridged

set -e

# Colores
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "DIAGNÓSTICO DE RED - VM ZeroMQ Lab"
echo "================================================================"
echo ""

# 1. INFORMACIÓN DE INTERFACES
echo -e "${BLUE}1. INTERFACES DE RED${NC}"
echo "---"
ip -4 addr show | grep -E "^[0-9]|inet " | sed 's/^/  /'
echo ""

# 2. TABLA DE ROUTING
echo -e "${BLUE}2. TABLA DE ROUTING${NC}"
echo "---"
ip route | sed 's/^/  /'
echo ""

# 3. INFORMACIÓN DETALLADA POR INTERFAZ
echo -e "${BLUE}3. DETALLE DE INTERFACES${NC}"
echo "---"

for iface in eth0 eth1 eth2; do
    if ip link show $iface > /dev/null 2>&1; then
        echo -e "${GREEN}► $iface${NC}"

        # Estado
        STATE=$(ip link show $iface | grep -oP 'state \K\w+')
        echo "  Estado: $STATE"

        # IP
        IP=$(ip -4 addr show $iface | grep inet | awk '{print $2}' | cut -d'/' -f1)
        if [ -n "$IP" ]; then
            echo "  IP: $IP"
        else
            echo "  IP: No asignada"
        fi

        # Máscara y red
        CIDR=$(ip -4 addr show $iface | grep inet | awk '{print $2}')
        if [ -n "$CIDR" ]; then
            echo "  CIDR: $CIDR"
        fi

        # Gateway (si está en la ruta por defecto)
        GW=$(ip route | grep "default.*$iface" | awk '{print $3}')
        if [ -n "$GW" ]; then
            echo "  Gateway: $GW"
        fi

        # MAC
        MAC=$(ip link show $iface | grep link/ether | awk '{print $2}')
        if [ -n "$MAC" ]; then
            echo "  MAC: $MAC"
        fi

        # MTU
        MTU=$(ip link show $iface | grep -oP 'mtu \K\d+')
        echo "  MTU: $MTU"

        # Estadísticas de tráfico
        RX_PACKETS=$(cat /sys/class/net/$iface/statistics/rx_packets 2>/dev/null || echo "0")
        TX_PACKETS=$(cat /sys/class/net/$iface/statistics/tx_packets 2>/dev/null || echo "0")
        echo "  RX packets: $RX_PACKETS"
        echo "  TX packets: $TX_PACKETS"

        echo ""
    else
        echo -e "${YELLOW}► $iface: No detectada${NC}"
        echo ""
    fi
done

# 4. TEST DE CONECTIVIDAD
echo -e "${BLUE}4. TEST DE CONECTIVIDAD${NC}"
echo "---"

# Ping al gateway por defecto
DEFAULT_GW=$(ip route | grep '^default' | awk '{print $3}' | head -1)
if [ -n "$DEFAULT_GW" ]; then
    echo -n "  Gateway ($DEFAULT_GW): "
    if ping -c 2 -W 2 $DEFAULT_GW > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}FALLO${NC}"
    fi
fi

# Ping a DNS público
echo -n "  DNS Público (8.8.8.8): "
if ping -c 2 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}FALLO${NC}"
fi

# Ping desde eth2 si existe
if ip link show eth2 > /dev/null 2>&1; then
    ETH2_IP=$(ip -4 addr show eth2 | grep inet | awk '{print $2}' | cut -d'/' -f1)
    if [ -n "$ETH2_IP" ]; then
        echo -n "  Desde eth2 a 8.8.8.8: "
        if ping -c 2 -W 2 -I eth2 8.8.8.8 > /dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${YELLOW}FALLO${NC}"
        fi
    fi
fi

echo ""

# 5. DNS CONFIGURATION
echo -e "${BLUE}5. CONFIGURACIÓN DNS${NC}"
echo "---"
if [ -f /etc/resolv.conf ]; then
    grep "^nameserver" /etc/resolv.conf | sed 's/^/  /'
else
    echo "  /etc/resolv.conf no encontrado"
fi
echo ""

# 6. HOSTS ESPECIALES
echo -e "${BLUE}6. HOSTS ESPECIALES${NC}"
echo "---"
if grep -q "host.docker.internal" /etc/hosts 2>/dev/null; then
    grep "host.docker.internal\|docker.host.internal" /etc/hosts | sed 's/^/  /'
else
    echo "  No hay hosts Docker configurados"
fi
echo ""

# 7. VARIABLES DE ENTORNO DE RED
echo -e "${BLUE}7. VARIABLES DE ENTORNO${NC}"
echo "---"
if [ -f /etc/environment ]; then
    grep -E "IP|ENDPOINT" /etc/environment | sed 's/^/  /'
else
    echo "  No hay variables de red en /etc/environment"
fi
echo ""

# 8. PUERTOS EN ESCUCHA
echo -e "${BLUE}8. PUERTOS EN ESCUCHA (ZeroMQ relevantes)${NC}"
echo "---"
ZEROMQ_PORTS="5555 5556 5557 5558 5559 5571"
for port in $ZEROMQ_PORTS; do
    if ss -tuln | grep -q ":$port "; then
        echo -e "  Puerto $port: ${GREEN}EN USO${NC}"
        ss -tuln | grep ":$port " | sed 's/^/    /'
    else
        echo "  Puerto $port: Libre"
    fi
done
echo ""

# 9. RECOMENDACIONES PARA SNIFFER
echo -e "${BLUE}9. CONFIGURACIÓN PARA SNIFFER${NC}"
echo "---"

if ip link show eth2 > /dev/null 2>&1; then
    ETH2_IP=$(ip -4 addr show eth2 | grep inet | awk '{print $2}' | cut -d'/' -f1)
    if [ -n "$ETH2_IP" ]; then
        echo -e "  ${GREEN}✓${NC} eth2 está configurada correctamente"
        echo "  IP de eth2: $ETH2_IP"
        echo ""
        echo "  Para ejecutar el sniffer en eth2:"
        echo "    cd /vagrant"
        echo "    sudo ./sniffer/build/sniffer --verbose"
        echo ""
        echo "  Para capturar tráfico en eth2:"
        echo "    ./scripts/capture_zeromq_traffic.sh eth2 60"
    else
        echo -e "  ${YELLOW}⚠${NC} eth2 existe pero no tiene IP asignada"
        echo "  Solución: Verificar que el bridged network está activo"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} eth2 no detectada"
    echo "  Verifica la configuración en Vagrantfile:"
    echo "    config.vm.network \"public_network\""
fi

echo ""
echo "================================================================"
echo "FIN DEL DIAGNÓSTICO"
echo "================================================================"