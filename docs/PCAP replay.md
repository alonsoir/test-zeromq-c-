Cómo sería el **PCAP replay** con tráfico real de malware.

## PCAP Replay - Metodología Completa 🔬

### Concepto
Reproducir capturas de tráfico real de ataques (ransomware, DDoS, etc.) **inyectándolas en eth0** de la VM para que el sniffer las capture como si fueran tráfico en vivo.

---

## Fase 1: Obtener PCAPs de Malware Real 📥

### Fuentes públicas confiables:

**1. Malware-Traffic-Analysis.net** (MEJOR FUENTE)
```bash
# Ejemplos de datasets disponibles:
# - Ransomware: Locky, WannaCry, Ryuk, Conti
# - Banking Trojans: Emotet, Trickbot, Dridex
# - Exploit Kits: RIG, Magnitude
# URL: https://www.malware-traffic-analysis.net/
```

**2. StratosphereIPS Datasets**
```bash
# CTU-13 Dataset: 13 diferentes escenarios de botnet
# URL: https://www.stratosphereips.org/datasets-overview
```

**3. CAIDA DDoS Attack 2007**
```bash
# DDoS real contra servidores raíz DNS
# URL: https://www.caida.org/catalog/datasets/ddos-20070804_dataset/
```

**4. SecRepo.com**
```bash
# Repositorio con múltiples tipos de ataques
# URL: http://www.secrepo.com/
```

---

## Fase 2: Preparar el Entorno 🛠️

### Instalar herramientas necesarias:

```bash
vagrant ssh

# Instalar tcpreplay (inyección de PCAP)
sudo apt-get update
sudo apt-get install -y tcpreplay tcpdump wireshark-common

# Verificar instalación
tcpreplay --version
```

### Crear directorio de trabajo:

```bash
mkdir -p /vagrant/testing/pcaps
cd /vagrant/testing/pcaps
```

---

## Fase 3: Descargar PCAP de Ejemplo 📦

### Opción A: Ransomware (Locky)

```bash
# Descargar PCAP de Locky ransomware
wget https://www.malware-traffic-analysis.net/2016/09/22/2016-09-22-Locky-infection-traffic.pcap.zip

# Descomprimir (password suele ser "infected")
unzip -P infected 2016-09-22-Locky-infection-traffic.pcap.zip
```

### Opción B: DDoS Attack (más fácil para empezar)

```bash
# DDoS simple - creamos uno sintético para testing inicial
# (luego usaremos PCAPs reales)
```

### Opción C: Botnet Traffic (CTU-13)

```bash
# Descargar capture de botnet Neris
wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow
```

---

## Fase 4: Inspeccionar el PCAP 🔍

```bash
# Ver información del PCAP
capinfos archivo.pcap

# Ver primeros 20 paquetes
tcpdump -nr archivo.pcap -c 20

# Ver estadísticas de IPs
tcpdump -nr archivo.pcap | awk '{print $3}' | sort | uniq -c | sort -rn | head -20

# Contar paquetes por protocolo
tcpdump -nr archivo.pcap -q | awk '{print $3}' | cut -d. -f1-4 | sort | uniq -c
```

---

## Fase 5: Preparar el PCAP para Replay 🎬

### Reescribir IPs (CRÍTICO)

**Problema:** Los PCAPs tienen IPs de las redes originales. Necesitamos reescribirlas para que:
- La IP de origen sea algo que el firewall pueda bloquear
- La IP de destino sea algo alcanzable desde la VM

```bash
# Instalar tcprewrite
sudo apt-get install -y tcpreplay

# Reescribir IPs: 
# - Origen: cualquier IP maliciosa → 192.168.100.50 (IP fake atacante)
# - Destino: cualquier IP víctima → 10.0.2.15 (la VM misma)

tcprewrite \
  --infile=original.pcap \
  --outfile=rewritten.pcap \
  --srcipmap=0.0.0.0/0:192.168.100.50/32 \
  --dstipmap=0.0.0.0/0:10.0.2.15/32 \
  --enet-dmac=08:00:27:XX:XX:XX \
  --enet-smac=52:54:00:XX:XX:XX

# Verificar resultado
tcpdump -nr rewritten.pcap -c 10
```

### Alternativa más simple (para testing inicial):

```bash
# Solo reescribir IPs, dejar MACs
tcprewrite \
  --infile=original.pcap \
  --outfile=rewritten.pcap \
  --pnat=0.0.0.0/0:192.168.100.0/24 \
  --fixcsum
```

---

## Fase 6: Ejecutar el Replay 🚀

### Preparación:

```bash
# 1. Asegurar que el lab está corriendo
cd /vagrant
make run-lab-dev

# 2. En otra terminal, preparar monitoring
tail -f /vagrant/logs/lab/detector.log | grep "attacks="

# 3. En otra terminal más, ver IPSet en tiempo real
watch -n 1 'sudo ipset list ml_defender_blacklist_test'
```

### Replay básico:

```bash
# Replay a velocidad del PCAP original
sudo tcpreplay --intf1=eth0 rewritten.pcap

# Replay más rápido (2x velocidad)
sudo tcpreplay --intf1=eth0 --multiplier=2 rewritten.pcap

# Replay más lento (útil para debugging)
sudo tcpreplay --intf1=eth0 --multiplier=0.5 rewritten.pcap

# Replay en loop (para stress test)
sudo tcpreplay --intf1=eth0 --loop=10 rewritten.pcap
```

### Replay avanzado:

```bash
# Control preciso de velocidad (100 Mbps)
sudo tcpreplay --intf1=eth0 --mbps=100 rewritten.pcap

# Control por paquetes por segundo
sudo tcpreplay --intf1=eth0 --pps=1000 rewritten.pcap

# Con estadísticas detalladas
sudo tcpreplay --intf1=eth0 --stats=1 rewritten.pcap
```

---

## Fase 7: Verificar Detecciones 🎯

### Durante el replay:

```bash
# 1. Ver si attacks > 0
grep "attacks=" /vagrant/logs/lab/detector.log | tail -5

# 2. Ver clasificaciones
grep -E "DDOS|RANSOMWARE|SUSPICIOUS" /vagrant/logs/lab/detector.log | tail -20

# 3. Ver si el firewall bloqueó
sudo ipset list ml_defender_blacklist_test

# 4. Ver IPTables counters
sudo iptables -L INPUT -n -v --line-numbers | grep ml_defender
```

### Después del replay:

```bash
# Estadísticas finales
echo "=== SNIFFER ==="
grep "Paquetes procesados" /vagrant/logs/lab/sniffer.log | tail -1

echo "=== DETECTOR ==="
grep "Stats:" /vagrant/logs/lab/detector.log | tail -1

echo "=== FIREWALL ==="
grep "METRICS" /vagrant/logs/lab/firewall.log | tail -1

# Ver IPs bloqueadas
sudo ipset list ml_defender_blacklist_test -o save
```

---

## Fase 8: Script Automatizado 📜

Creo un script completo para ti:

```bash
cat > /vagrant/scripts/testing/pcap_replay_test.sh << 'EOF'
#!/bin/bash

PCAP_DIR="/vagrant/testing/pcaps"
PCAP_FILE="$1"
SPEED="${2:-1}"  # Default 1x speed

if [ -z "$PCAP_FILE" ]; then
    echo "Usage: $0 <pcap_file> [speed_multiplier]"
    echo "Example: $0 malware.pcap 2"
    exit 1
fi

if [ ! -f "$PCAP_DIR/$PCAP_FILE" ]; then
    echo "Error: PCAP file not found: $PCAP_DIR/$PCAP_FILE"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ML Defender - PCAP Replay Test                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "PCAP File: $PCAP_FILE"
echo "Speed: ${SPEED}x"
echo ""

# Get baseline stats
BASELINE_ATTACKS=$(grep "attacks=" /vagrant/logs/lab/detector.log | tail -1 | grep -oP 'attacks=\K\d+')
BASELINE_PROCESSED=$(grep "processed=" /vagrant/logs/lab/detector.log | tail -1 | grep -oP 'processed=\K\d+')

echo "Baseline: attacks=$BASELINE_ATTACKS, processed=$BASELINE_PROCESSED"
echo ""
echo "Starting replay in 3 seconds..."
sleep 3

# Execute replay
echo "🚀 Replaying PCAP..."
sudo tcpreplay \
    --intf1=eth0 \
    --multiplier=$SPEED \
    --stats=1 \
    "$PCAP_DIR/$PCAP_FILE"

echo ""
echo "✅ Replay complete. Waiting 5 seconds for processing..."
sleep 5

# Get final stats
FINAL_ATTACKS=$(grep "attacks=" /vagrant/logs/lab/detector.log | tail -1 | grep -oP 'attacks=\K\d+')
FINAL_PROCESSED=$(grep "processed=" /vagrant/logs/lab/detector.log | tail -1 | grep -oP 'processed=\K\d+')

NEW_ATTACKS=$((FINAL_ATTACKS - BASELINE_ATTACKS))
NEW_PROCESSED=$((FINAL_PROCESSED - BASELINE_PROCESSED))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RESULTS                                                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "Events processed: $NEW_PROCESSED"
echo "Attacks detected: $NEW_ATTACKS"
echo ""

if [ $NEW_ATTACKS -gt 0 ]; then
    echo "🚨 ATTACKS DETECTED! Checking blocked IPs..."
    echo ""
    sudo ipset list ml_defender_blacklist_test
    echo ""
    echo "✅ Test PASSED - System detected and blocked attacks!"
else
    echo "ℹ️  No attacks detected. Models classified traffic as benign."
    echo "   This could mean:"
    echo "   1. PCAP doesn't contain attack patterns models recognize"
    echo "   2. Models are working correctly (no false positives)"
    echo "   3. Different PCAP needed for testing"
fi

echo ""
EOF

chmod +x /vagrant/scripts/testing/pcap_replay_test.sh
```

---

## Uso Final 🎬

```bash
# 1. Descargar un PCAP
cd /vagrant/testing/pcaps
wget <URL_DEL_PCAP>

# 2. (Opcional) Reescribir IPs
tcprewrite --infile=original.pcap --outfile=ready.pcap --pnat=0.0.0.0/0:192.168.100.0/24

# 3. Ejecutar test
/vagrant/scripts/testing/pcap_replay_test.sh ready.pcap 1

# 4. Analizar resultados
```

---

## Ventajas del PCAP Replay ✅

1. **Reproducible**: Mismo tráfico cada vez
2. **Científico**: Tráfico real de malware documentado
3. **Seguro**: No hay malware activo, solo captures
4. **Controlable**: Puedes ajustar velocidad, loops, etc.
5. **Validación real**: Prueba los modelos contra amenazas reales

¿Quieres que empecemos descargando un PCAP específico? Te recomiendo empezar con **Malware-Traffic-Analysis.net** - tienen casos muy bien documentados con writeups. 🎯