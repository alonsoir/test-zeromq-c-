# Troubleshooting Guide

Guía completa de resolución de problemas para ML Detector Tricapa.

---

## 🔍 Índice

- [Build Issues](#build-issues)
- [Runtime Issues](#runtime-issues)
- [eBPF/XDP Issues](#ebpfxdp-issues)
- [ZMQ Communication Issues](#zmq-communication-issues)
- [ML Detector Issues](#ml-detector-issues)
- [Vagrant/VM Issues](#vagrantvm-issues)
- [Debug Tools](#debug-tools)

---

## Build Issues

### ❌ Error: `libbpf-dev` not found

**Síntoma:**
```
E: Package 'libbpf-dev' has no installation candidate
```

**Causa:** Repositorios no actualizados o Debian versión incorrecta.

**Solución:**
```bash
# Dentro de la VM
sudo apt-get update
sudo apt-get install -y libbpf-dev

# Si persiste, verificar sources.list
cat /etc/apt/sources.list
# Debe incluir: deb http://deb.debian.org/debian bookworm main
```

---

### ❌ Error: `jsoncpp` headers missing

**Síntoma:**
```
fatal error: json/json.h: No such file or directory
```

**Solución:**
```bash
sudo apt-get install -y libjsoncpp-dev
# Verificar instalación
ls /usr/include/json/
```

---

### ❌ Error: `linux-headers-$(uname -r)` not found

**Síntoma:**
```
E: Unable to locate package linux-headers-6.1.0-XX-amd64
```

**Solución:** Usar el metapaquete en vez de versión específica:
```bash
sudo apt-get install -y linux-headers-amd64
```

---

### ❌ Error: Protobuf compilation fails

**Síntoma:**
```
protoc: command not found
```

**Solución:** Instalar stack completo de Protobuf:
```bash
sudo apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    libprotobuf32

# Verificar
protoc --version  # Debe mostrar: libprotoc 3.21.12
```

---

### ❌ Error: CMake can't find BPF headers

**Síntoma:**
```
CMake Error: Could not find bpf/libbpf.h
```

**Solución:**
```bash
# Verificar que están instalados
dpkg -L libbpf-dev | grep libbpf.h

# Si no existe, reinstalar
sudo apt-get install --reinstall libbpf-dev

# Añadir al CMakeLists.txt si es necesario
include_directories(/usr/include/bpf)
```

---

## Runtime Issues

### ❌ Error: "Address already in use" (ZMQ)

**Síntoma:**
```
Address already in use (zmq_bind)
```

**Causa:** Múltiples sockets intentando bind al mismo puerto.

**Diagnóstico:**
```bash
# Ver qué está usando el puerto
sudo netstat -tlnp | grep 5571
# O
sudo lsof -i :5571

# Ver procesos del sniffer/detector
ps aux | grep -E 'sniffer|detector'
```

**Solución:**
```bash
# Opción 1: Matar procesos anteriores
make kill-all

# Opción 2: Reducir socket pool a 1 en sniffer.json
{
  "socket_pools": {
    "push_sockets": 1  // ← Solo 1 socket
  }
}

# Opción 3: Cambiar puerto
{
  "output": {
    "zmq_endpoint": "tcp://127.0.0.1:5573"  // Puerto diferente
  }
}
```

---

### ❌ Error: "Assertion failed: check ()" (ZMQ Crash)

**Síntoma:**
```
Assertion failed: check () (src/msg.cpp:414)
Abortado
```

**Causa:** Bug en lifecycle de mensajes ZMQ bajo carga alta.

**Workaround temporal:**
```json
// En sniffer.json
{
  "batch_processing_size": 1,     // ← Reducir a 1
  "zmq_sender_threads": 1,        // ← Solo 1 thread
  "ring_consumer_threads": 1
}
```

**Fix definitivo:** Pendiente para v3.3.0 (revisar zmq_msg_close calls).

---

### ❌ Error: No packets captured

**Síntoma:** Sniffer arranca pero no captura paquetes.

**Diagnóstico:**
```bash
# 1. Verificar que el programa eBPF está cargado
sudo bpftool prog show | grep sniffer

# 2. Verificar que está adjunto a la interfaz
ip link show eth0  # Buscar 'xdp' en la salida

# 3. Ver estadísticas del ring buffer
sudo bpftool map dump name stats_map

# 4. Generar tráfico de prueba
ping -c 10 8.8.8.8
```

**Posibles causas:**

1. **Interfaz incorrecta:**
```json
// En sniffer.json, verificar:
{
  "capture": {
    "interface": "eth0"  // ← Debe existir (ip link show)
  }
}
```

2. **Filtros demasiado restrictivos:**
```bash
# Ver filtros activos
sudo bpftool map dump name excluded_ports
sudo bpftool map dump name included_ports

# Desactivar filtros temporalmente
{
  "filter": {
    "mode": "passthrough"  // ← Todo pasa
  }
}
```

3. **Ring buffer lleno:**
```bash
# Ver tamaño actual
grep buffer_size config/sniffer.json

# Incrementar si es necesario
{
  "buffer_size": 131072  // ← Duplicar
}
```

---

## eBPF/XDP Issues

### ❌ Error: "Failed to load eBPF program"

**Síntoma:**
```
[ERROR] Failed to load eBPF object from sniffer.bpf.o
```

**Diagnóstico:**
```bash
# 1. Verificar que el archivo existe
ls -lh build/sniffer.bpf.o

# 2. Verificar permisos
sudo chmod 644 build/sniffer.bpf.o

# 3. Verificar BTF support en kernel
cat /sys/kernel/btf/vmlinux | head -1

# 4. Intentar cargar manualmente
sudo bpftool prog load build/sniffer.bpf.o /sys/fs/bpf/test
```

**Soluciones:**

1. **Recompilar programa eBPF:**
```bash
cd build
make clean
make bpf_program  # Solo el programa eBPF
```

2. **Verificar kernel version:**
```bash
uname -r  # Debe ser >= 5.10 para XDP
```

3. **Check libbpf version:**
```bash
dpkg -l | grep libbpf  # Debe ser >= 1.0
```

---

### ❌ Error: "No such file or directory" (BPF Maps)

**Síntoma:**
```
❌ Failed to get BPF map: excluded_ports (errno: 2)
```

**Causa:** Intentando acceder a maps por nombre en lugar de FDs.

**Solución:** Actualizar a v3.2.1+ que usa FD-based access:
```bash
git pull origin feature/ml-detector-tricapa
make rebuild
```

**Verificación:**
```bash
# Ver logs del sniffer, debe mostrar:
[INFO] Found excluded_ports map, FD: 6
[INFO] Found included_ports map, FD: 7
[INFO] Found filter_settings map, FD: 8
```

---

### ❌ Error: XDP mode not supported (virtio)

**Síntoma:**
```
[WARNING] XDP native mode not supported, using SKB mode
```

**Causa:** VirtualBox virtio_net no soporta XDP nativo.

**Solución:** Normal en VM, usar SKB mode (ya lo hace automáticamente):
```cpp
// El código ya lo maneja:
if (xdp_native_failed) {
    use_skb_mode();  // ← Fallback automático
}
```

No afecta funcionalidad, solo rendimiento.

---

## ZMQ Communication Issues

### ❌ Detector no recibe mensajes

**Síntoma:** Sniffer envía, detector no recibe nada.

**Diagnóstico:**
```bash
# 1. Verificar conectividad
telnet 127.0.0.1 5571

# 2. Ver si detector está escuchando
netstat -tlnp | grep 5571

# 3. Ver configuración de endpoints
# Sniffer (sniffer.json):
grep zmq_endpoint config/sniffer.json

# Detector (ml_detector_config.json):
grep zmq_input_endpoint config/ml_detector_config.json
```

**Soluciones:**

1. **Endpoints must match:**
```json
// sniffer.json
{
  "output": {
    "zmq_endpoint": "tcp://127.0.0.1:5571"  // PUB/PUSH
  }
}

// ml_detector_config.json
{
  "zmq_input_endpoint": "tcp://127.0.0.1:5571"  // SUB/PULL
}
```

2. **Verificar patrón ZMQ:**
```cpp
// Debe ser PUSH-PULL (no PUB-SUB) para garantizar entrega
// Verificar en el código que usa zmq_socket(ctx, ZMQ_PUSH/PULL)
```

3. **Test con zmq_proxy:**
```bash
# Herramienta de debug ZMQ
sudo apt-get install -y zeromq-utils
# Ver tráfico
zmq_term
```

---

## ML Detector Issues

### ❌ Error: "Failed to load ONNX model"

**Síntoma:**
```
[ERROR] Could not load ONNX model: level1_rf_model.onnx
```

**Solución:**
```bash
# 1. Verificar que el modelo existe
ls -lh ml-detector/models/level1_rf_model.onnx

# 2. Verificar permisos
chmod 644 ml-detector/models/*.onnx

# 3. Verificar que ONNX Runtime está instalada
dpkg -l | grep onnxruntime

# 4. Regenerar modelo si está corrupto
cd ml-training
python scripts/convert_level2_ddos_to_onnx.py
```

---

### ❌ Error: Feature dimension mismatch

**Síntoma:**
```
[ERROR] Feature size mismatch: expected 23, got 193
```

**Causa:** Modelo entrenado con 23 features pero se envían 193.

**Solución:**
```bash
# Verificar feature extraction en ml-detector
grep "EXPECTED_FEATURES" ml-detector/src/inference/

# Debe ser 23 para Level 1 (RF model)
# Si es diferente, reentrenar modelo con features correctas
```

---

## Vagrant/VM Issues

### ❌ Error: VM no arranca

**Síntoma:**
```
VBoxManage: error: Failed to create the host-only adapter
```

**Solución:**
```bash
# macOS: Dar permisos a VirtualBox
# System Preferences → Security & Privacy → Allow Oracle

# Reinstalar VirtualBox extensions
# https://www.virtualbox.org/wiki/Downloads

# Recrear VM
make destroy
make dev-setup
```

---

### ❌ Error: Shared folder not mounted

**Síntoma:**
```
ls /vagrant: No such file or directory
```

**Solución:**
```bash
# Reinstalar Guest Additions
vagrant plugin install vagrant-vbguest
vagrant reload

# Verificar
vagrant ssh -c "ls /vagrant"
```

---

### ❌ Error: Out of memory

**Síntoma:**
```
Virtual memory exhausted: Cannot allocate memory
```

**Solución:**
```ruby
# En Vagrantfile, incrementar RAM:
config.vm.provider "virtualbox" do |vb|
  vb.memory = "8192"  # ← Aumentar a 8GB
end

vagrant reload
```

---

## Debug Tools

### bpftool - BPF Introspection

```bash
# Ver todos los programas cargados
sudo bpftool prog show

# Ver programa específico
sudo bpftool prog show id <PROG_ID>

# Ver todos los maps
sudo bpftool map list

# Dumpear contenido de un map
sudo bpftool map dump id <MAP_ID>
sudo bpftool map dump name <MAP_NAME>

# Ver BTF info
sudo bpftool btf dump file /sys/kernel/btf/vmlinux

# Ver programa en assembly
sudo bpftool prog dump xlated id <PROG_ID>
```

---

### tcpdump - Network Capture

```bash
# Capturar en la interfaz del sniffer
sudo tcpdump -i eth0 -nn -vv

# Solo tráfico en puertos específicos
sudo tcpdump -i eth0 port 8000

# Guardar a archivo para análisis
sudo tcpdump -i eth0 -w capture.pcap
```

---

### netstat/ss - Network Status

```bash
# Ver puertos en uso
sudo netstat -tlnp
sudo ss -tlnp

# Ver conexiones ZMQ
sudo netstat -an | grep 5571

# Ver todo el tráfico local
sudo netstat -an | grep 127.0.0.1
```

---

### strace - System Call Tracing

```bash
# Trace sniffer startup
sudo strace -f ./build/sniffer -c config/sniffer.json 2>&1 | grep -E 'bpf|open|mmap'

# Trace ZMQ calls
sudo strace -e trace=network ./build/sniffer -c config/sniffer.json
```

---

### perf - Performance Analysis

```bash
# Ver estadísticas del programa eBPF
sudo perf stat -e bpf:* ./build/sniffer -c config/sniffer.json

# Profile CPU usage
sudo perf top -p $(pgrep sniffer)
```

---

## Common Patterns & Solutions

### Pattern: "It worked yesterday, now it doesn't"

**Checklist:**
1. ✅ Reboot VM: `vagrant reload`
2. ✅ Kill orphan processes: `make kill-all`
3. ✅ Clean rebuild: `make rebuild`
4. ✅ Check disk space: `df -h`
5. ✅ Check memory: `free -h`

---

### Pattern: High CPU usage

**Diagnóstico:**
```bash
top  # Ver procesos
# Si sniffer usa >90% CPU:
```

**Soluciones:**
1. Reducir batch size en config
2. Reducir threads
3. Activar filtros para reducir tráfico
4. Verificar que no hay loop infinito en código

---

### Pattern: Memory leak

**Diagnóstico:**
```bash
# Ver memoria del proceso
ps aux | grep sniffer

# Monitorear en tiempo real
watch -n 1 'ps aux | grep sniffer'
```

**Soluciones:**
1. Verificar que zmq_msg_close se llama
2. Verificar que buffers se liberan
3. Usar valgrind para detectar leaks:
```bash
valgrind --leak-check=full ./build/sniffer -c config/sniffer.json
```

---

## Emergency Recovery

### Nuclear Option: Reset Everything

```bash
# En host macOS
cd ~/Code/test-zeromq-docker

# Destruir y recrear VM completa
make destroy
make dev-setup

# ~15 minutos, pero garantiza estado limpio
```

---

## Getting Help

Si ninguna solución funciona:

1. **Recopilar información de debug:**
```bash
# Script de diagnóstico completo
cd /vagrant/sniffer
./scripts/collect_debug_info.sh > debug_report.txt
```

2. **Abrir Issue en GitHub:**
- [GitHub Issues](https://github.com/alonsoir/test-zeromq-c-/issues)
- Adjuntar `debug_report.txt`
- Describir pasos para reproducir

3. **Consultar documentación:**
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
- [Architecture Docs](./ARCHITECTURE.md)

---

<div align="center">

**🔧 Troubleshooting Guide - ML Detector Tricapa 🔧**

*Actualizado: Octubre 25, 2025*

</div>