### 1. README.md - 

```markdown
## ✅ Estado Actual del Proyecto

### Componentes Operativos
- ✅ **Pipeline ZeroMQ + Protobuf**: service1 → service2 → service3 funcionando
- ✅ **etcd Service Discovery**: Registro automático de servicios con heartbeat
- ✅ **eBPF Sniffer v3.1**: Captura de paquetes en kernel space con XDP
- ✅ **Compresión LZ4**: Protobuf messages comprimidos en tránsito
- ✅ **Vagrant + Docker**: Entorno reproducible completo
- ✅ **Verbose Feature Logging**: 3 niveles de inspección de ML features (NUEVO)

### Configuración eBPF Optimizada
- ✅ BPF JIT habilitado automáticamente en provisioning
- ✅ BPF filesystem montado persistentemente en `/sys/fs/bpf`
- ✅ Configuración permanente vía `/etc/fstab`
- ✅ Sistema de logging configurable para debugging

### Últimas Mejoras (2025-10-12)
- **Sistema de Verbose Logging**: 3 niveles para inspección de features ML
    - Nivel 1 (-v): Resumen básico por paquete
    - Nivel 2 (-vv): Features agrupadas por categoría
    - Nivel 3 (-vvv): Dump completo de ~193 features
- Logging con colores y formato estructurado
- Zero overhead cuando verbose está desactivado
- Integración completa en paquete Debian

### Mejoras Anteriores (2025-10-03)
- Sincronización de archivos de configuración JSON (`sniffer.json` ↔ `sniffer-proposal.json`)
- Eliminación de comentarios inline en JSON (parser estricto)
- Corrección de pkg-config para libzmq (`libzmq3` → `libzmq`)
- Provisioning automático de capacidades eBPF en Vagrant
- Target `verify-bpf` para validación de configuración kernel

## ✅ Estado Actual del Proyecto

### Componentes Operativos
- ✅ **Pipeline ZeroMQ + Protobuf**: service1 → service2 → service3 funcionando
- ✅ **etcd Service Discovery**: Registro automático de servicios con heartbeat
- ✅ **eBPF Sniffer v3.1**: Captura de paquetes en kernel space con XDP
- ✅ **Compresión LZ4**: Protobuf messages comprimidos en tránsito
- ✅ **Vagrant + Docker**: Entorno reproducible completo

### Configuración eBPF Optimizada
- ✅ BPF JIT habilitado automáticamente en provisioning
- ✅ BPF filesystem montado persistentemente en `/sys/fs/bpf`
- ✅ Configuración permanente vía `/etc/fstab`

### Últimas Mejoras (2025-10-03)
- Sincronización de archivos de configuración JSON (`sniffer.json` ↔ `sniffer-proposal.json`)
- Eliminación de comentarios inline en JSON (parser estricto)
- Corrección de pkg-config para libzmq (`libzmq3` → `libzmq`)
- Provisioning automático de capacidades eBPF en Vagrant
- Target `verify-bpf` para validación de configuración kernel
```

### 2. DECISIONS.md - Nuevas secciones:

```markdown
### 7. Configuración JSON y Parsing Estricto

**Decisión**: Usar JSON puro sin comentarios
- **Contexto**: El parser JSON estricto de jsoncpp no acepta comentarios inline (`//`)
- **Solución**: Mantener dos archivos separados:
  - `sniffer-proposal.json`: Versión documentada con comentarios (desarrollo)
  - `sniffer.json`: Versión limpia para producción
- **Alternativa considerada**: Usar JSONC, descartado por complejidad adicional
- **Aprendizaje**: La documentación se mantendrá en archivos `.md` separados

### 8. Optimización eBPF en Vagrant

**Decisión**: Habilitar BPF JIT y filesystem automáticamente
- **Problema detectado**: `/proc/sys/kernel/bpf_jit_enable` no existía por defecto
- **Solución implementada**:
  ```bash
  # En Vagrantfile provision:
  echo 1 | tee /proc/sys/net/core/bpf_jit_enable
  mount -t bpf none /sys/fs/bpf
  echo "none /sys/fs/bpf bpf defaults 0 0" >> /etc/fstab
  ```
- **Impacto**: Mejora de rendimiento en compilación JIT de programas eBPF
- **Verificación**: Target `make verify-bpf` para validar configuración

### 9. Gestión de Dependencias con pkg-config

**Decisión**: Usar nombres correctos de paquetes pkg-config
- **Problema**: Confusion entre nombre de paquete Debian y archivo `.pc`
    - Paquete Debian: `libzmq3-dev`
    - Archivo pkg-config: `libzmq.pc` (no `libzmq3.pc`)
- **Solución**: Actualizar Makefile para usar `pkg-config --exists libzmq`
- **Lección**: Siempre verificar con `pkg-config --list-all | grep <lib>`

### 10. Sincronización de Configuraciones

**Decisión**: `sniffer.json` como single source of truth en producción
- **Problema inicial**: Discrepancia entre archivos de configuración
    - `main.h` apuntaba a `sniffer-proposal.json` (desarrollo)
    - `run_sniffer_with_iface.sh` apuntaba a `sniffer.json` (producción)
- **Solución**: Copiar `sniffer-proposal.json` → `sniffer.json` tras validación
- **Proceso**:
    1. Desarrollo en `sniffer-proposal.json` (con comentarios)
    2. Validación y testing
    3. Limpieza y copia a `sniffer.json`
    4. Commit de ambos archivos sincronizados
```

### 3. Nuevo archivo: `docs/JSON_CONFIG.md`

```markdown
# JSON Configuration Guide

## Archivos de Configuración

### `sniffer-proposal.json` (Desarrollo)
- Versión documentada con comentarios inline
- Usada durante desarrollo y experimentación
- **NO usar en producción** (comentarios no válidos en JSON estándar)

### `sniffer.json` (Producción)
- Versión limpia sin comentarios
- Usada por el sniffer en runtime
- Sincronizada desde `sniffer-proposal.json` tras validación

## Proceso de Actualización

1. Editar `sniffer-proposal.json` con comentarios
2. Validar configuración: `make sniffer-test`
3. Limpiar comentarios: `sed 's|//.*||g' sniffer-proposal.json > sniffer.json`
4. Verificar JSON válido: `python3 -m json.tool sniffer.json`
5. Commit ambos archivos

## Campos Críticos

### `batch.max_batches_queued`
**Requerido**: Sí  
**Tipo**: Integer  
**Descripción**: Máximo número de batches en cola antes de backpressure

### `compression.algorithm`
**Requerido**: Sí  
**Valores**: `lz4`, `zstd`, `snappy` (próximamente)  
**Producción**: `lz4` (mejor balance rendimiento/compresión)
```

## Comandos para el commit:

```bash
# 1. Actualizar archivos
git add README.md DECISIONS.md docs/JSON_CONFIG.md

# 2. Actualizar Makefile con verify-bpf y corrección libzmq
git add Makefile

# 3. Sincronizar configuraciones
git add sniffer/config/sniffer.json sniffer/config/sniffer-proposal.json

# 4. Commit descriptivo
git commit -m "feat: BPF JIT optimization and JSON config synchronization

- Enable BPF JIT automatically in Vagrant provisioning
- Mount /sys/fs/bpf filesystem persistently
- Add verify-bpf target for validation
- Fix pkg-config libzmq detection (libzmq3 → libzmq)
- Synchronize sniffer.json with sniffer-proposal.json
- Remove inline comments from production JSON
- Add JSON_CONFIG.md documentation
- Update DECISIONS.md with latest learnings"

# 5. Crear tag semántico
git tag -a v3.1.1 -m "Version 3.1.1 - eBPF optimization and config fixes"

# 6. Merge a main
git checkout main
git merge feature/enhanced-sniffer-config

# 7. Push todo
git push origin main
git push origin v3.1.1
```
## Red y Conectividad

Esta VM tiene 3 interfaces configuradas:

- **eth0** (10.0.2.15) - NAT para acceso a Internet
- **eth1** (192.168.56.20) - Red privada host-only (IP fija)
- **eth2** (DHCP) - Red bridged a tu LAN física

### Diagnóstico de Red
```bash
# Dentro de la VM
cd /vagrant
./scripts/network_diagnostics.sh

### Captura de Tráfico

# Capturar en eth2 durante 60 segundos
./scripts/capture_zeromq_traffic.sh eth2 60

# Ver capturas guardadas
ls -lh /tmp/zeromq_captures/

### Verificación del Sniffer en eth2
# Compilar sniffer
make sniffer-build-local

# Verificar que captura en eth2
sudo ./sniffer/build/sniffer --verbose | grep eth2
```
## 🔍 Debugging y Verbose Logging

### Niveles de Verbosity

El sniffer incluye un sistema de logging configurable para inspeccionar las features ML extraídas:

#### Nivel 1: Resumen Básico (`-v`)
```bash
  sudo ./sniffer/build/sniffer -c sniffer/config/sniffer.json -v
```

Output:

[PKT #312954584793_547881216] TCP 192.168.1.1:443 → 224.0.0.1:0 60B
[PKT #332893414690_547881216] UDP 192.168.1.135:53 → 224.0.0.240:63715 86B

### Uso: Monitoreo en tiempo real, verificación de captura
Nivel 2: Features Agrupadas (-vv)
```bash
  sudo ./sniffer/build/sniffer -c sniffer/config/sniffer.json -vv
```

Output:

=== PACKET #409255656473_130 ===
[BASIC INFO]
Timestamp: 2025-10-12 07:03:45.123456789
Source: 192.168.1.1:443
Destination: 224.0.0.1:0
Protocol: TCP (6)
Total Bytes: 60

[TIMING]
Flow duration: 0.000123 s
Flow IAT mean: 45.6 µs

[RATES & RATIOS]
Bytes/sec: 487804.8
Packets/sec: 8130.08
Download/Upload ratio: 0.0

[TCP FLAGS]
SYN: 1  ACK: 0  FIN: 0  RST: 0

[FEATURE ARRAYS]
General Attack Features (RF): 23 features
Internal Traffic: 4 features
Ransomware Detection: 83 features
DDoS Detection: 83 features

### Uso: Debugging de pipeline, validación de features
Nivel 3: Dump Completo (-vvv)

```bash
    sudo ./sniffer/build/sniffer -c sniffer/config/sniffer.json -vvv > features.log 2>&1
```
Output: ~193 features con índice y valor

=== PACKET #543424975012_547881216 - FULL FEATURE DUMP ===
[BASIC IDENTIFICATION]
Event ID: 543424975012_547881216
Node ID: cpp_sniffer_v31_001
Timestamp: 2025-10-12 07:05:12.547881216
Classification: UNCATEGORIZED
Threat Score: 0.00

[NETWORK FEATURES - BASIC]
[src_ip] 192.168.1.1
[dst_ip] 224.0.0.1
[src_port] 443
[dst_port] 0
[protocol_number] 6
[protocol_name] TCP

[PACKET STATISTICS]
[total_forward_packets] 1
[total_backward_packets] 0
[total_forward_bytes] 60
[total_backward_bytes] 0
[minimum_packet_length] 60
[maximum_packet_length] 60
[packet_length_mean] 60.00
[packet_length_std] 0.00

... (todas las features detalladas)

[GENERAL ATTACK FEATURES] (23 features)
[0] feature_0: 0.000000
[1] feature_1: 1.000000
...

[RANSOMWARE DETECTION FEATURES] (83 features)
[0] ransomware_0: 0.333333
[1] ransomware_1: 0.000000
...

Uso: Análisis exhaustivo, training de modelos ML, documentación

PEDTE

Redirección y Filtrado

# Guardar log completo
sudo ./sniffer -c config.json -vvv > features_$(date +%Y%m%d_%H%M%S).log 2>&1

# Solo paquetes TCP
sudo ./sniffer -c config.json -v | grep TCP

# Análisis de un paquete específico
sudo ./sniffer -c config.json -vvv | grep -A 200 "PACKET #123"

# Ver en tiempo real con colores
sudo ./sniffer -c config.json -vv | less -R