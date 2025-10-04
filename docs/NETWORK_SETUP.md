NETWORK_SETUP.md Simplificado
markdown# Configuración de Red

## Estado Actual

La VM tiene **3 interfaces configuradas y funcionando**:
eth0: 10.0.2.15/24      (NAT - Internet)
eth1: 192.168.56.20/24  (Host-only)
eth2: 192.168.1.134/24  (Bridged - LAN)

**eth2 está capturando tráfico real de la red local.**

## Verificación Rápida
```bash
# Ver interfaces y IPs
ip -4 -br addr

# Test de captura en eth2
sudo tcpdump -i eth2 -c 5 -n

# Ver servicios activos
sudo ss -tulpn | grep -E ':(5555|2379|2380|3000|5571)'

Próximo Paso: eBPF Sniffer en eth2
El sniffer actualmente está configurado para otra interfaz. Para capturar tráfico ZeroMQ en eth2:

Actualizar configuración: sniffer/config/sniffer.json
Iniciar servicios: Generar tráfico ZeroMQ real
Ejecutar sniffer: Debería capturar todo el tráfico del puerto 5555 en eth2

Ver scripts/test_sniffer_eth2.sh para el procedimiento completo.

