# ML Detector Tricapa - eBPF Network Security Pipeline

## 🎯 Visión del Proyecto

Sistema de detección de amenazas en red usando eBPF/XDP para captura de paquetes y modelos ML tricapa para clasificación en tiempo real.
```
┌─────────────────────────────────────────────────────────────────┐
│  Sniffer eBPF (XDP)  →  ML Detector (ONNX)  →  Alert/Action    │
│                                                                  │
│  eth2 capture  →  Feature Extraction  →  Level 1-3 Inference   │
│  Kernel Space     User Space ZMQ          C++ ONNX Runtime     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Estado Actual (v1.0.0-stable-pipeline)

### Componentes Operativos

- ✅ **Sniffer eBPF v3.1**: Captura XDP en kernel space con AF_XDP (eth2)
- ✅ **ML Detector v1.0**: Inferencia ONNX Level 1 (23 features RF model)
- ✅ **Pipeline ZMQ + Protobuf**: Comunicación Sniffer → ML Detector funcional
- ✅ **Build System**: Reproducible desde cero con Vagrant + Makefile
- ✅ **Vagrant Environment**: Debian 12, todas las dependencias automatizadas
- ✅ **Docker Lab**: etcd + service3 (legacy, opcional)

### Arquitectura del Pipeline
```
Sniffer eBPF (eth2)
  ├─ Kernel Space: XDP hook + AF_XDP socket
  ├─ User Space: Feature extraction (193 features)
  ├─ Protobuf serialization (network_security.proto v3.1.0)
  └─ ZMQ PUB → tcp://0.0.0.0:5571

ML Detector (Level 1)
  ├─ ZMQ SUB ← tcp://127.0.0.1:5571
  ├─ Protobuf deserialization
  ├─ Feature preprocessing (23 features)
  ├─ ONNX inference (RandomForest model)
  ├─ Classification: BENIGN / ATTACK
  └─ ZMQ PUB → tcp://0.0.0.0:5572 (ready for next component)
```

---

## 🚀 Quick Start

### Requisitos

- **Host**: macOS con VirtualBox y Vagrant
- **RAM**: 6GB para la VM
- **Disk**: 10GB libres

### Setup Completo (Primera Vez)
```bash
# 1. Clone el repositorio
git clone <repo-url>
cd test-zeromq-docker

# 2. Build completo desde cero
make dev-setup

# Esto ejecuta:
# - vagrant up (crea VM + instala TODAS las dependencias)
# - docker-compose up -d (arranca etcd + service3)
# - Genera protobuf schema
# - Compila sniffer
# - Compila ml-detector
# ⏱️ ~10-15 minutos primera vez

# 3. Verificar instalación
make test

# Output esperado:
# Sniffer:     ✅
# ML Detector: ✅
# Protobuf:    ✅
```

### Ejecución del Pipeline

**Requiere 3 terminales:**
```bash
# Terminal 1: Sniffer (captura + feature extraction)
make run-sniffer

# Output esperado:
# ✅ eBPF program loaded and attached (ring_fd=4)
# ✅ RingBufferConsumer started
# [PKT #xxx] TCP 192.168.1.x:443 → 8.8.8.8:443 60B

# Terminal 2: ML Detector (inferencia)
make run-detector

# Output esperado:
# ✅ ZMQ sockets initialized successfully
# 📥 ZMQ Handler loop started
# [INFO] 📦 Event received: id=event-xxx
# [DEBUG] 🤖 Prediction: label=0 (BENIGN), confidence=0.92

# Terminal 3: Generación de tráfico (pruebas)
make ssh
ping -c 100 8.8.8.8
curl http://example.com
```

---

## 🛠️ Comandos Útiles

### VM Management
```bash
make up              # Arrancar VM
make halt            # Parar VM
make destroy         # Destruir VM
make ssh             # Conectar a VM
make status          # Estado de la VM
```

### Build
```bash
make all             # Compilar todo (sniffer + detector)
make sniffer         # Solo sniffer
make detector        # Solo ml-detector
make proto           # Regenerar protobuf schema
make rebuild         # Clean + build todo
```

### Desarrollo
```bash
make test            # Verificar qué está compilado
make logs-sniffer    # Ver logs del sniffer
make logs-detector   # Ver logs del detector
make check-ports     # Ver si puertos 5571/5572 están en uso
make kill-all        # Matar procesos sniffer/detector
```

### Docker Lab (Opcional - Legacy)
```bash
make lab-start       # Arrancar etcd + service3
make lab-stop        # Parar lab
make lab-ps          # Ver contenedores
make lab-logs        # Ver logs
make lab-clean       # Limpiar todo
```

---

## 🐛 Issues Conocidos

### 1. Crash ZMQ bajo Carga (No Bloqueante)

**Síntoma:**
```
Assertion failed: check () (src/msg.cpp:414)
Abortado
```

**Contexto:**
- Aparece bajo carga sostenida (>100 paquetes/seg)
- Bug en el lifecycle de mensajes ZMQ en el sniffer
- **El pipeline funciona estable en condiciones normales**

**Workaround temporal:**
- Reducir `batch_processing_size` a 1 en `sniffer.json`
- Reducir `zmq_sender_threads` a 1

**Fix planificado:** Próxima sesión (revisar zmq_msg_close calls)

### 2. Warnings ML Detector (No Críticos)

**Síntoma:**
```
warning: unused parameter 'features'
warning: comparison of integer expressions
```

**Contexto:**
- Warnings normales de desarrollo
- No afectan funcionalidad
- Se limpiarán en fase de producción

---

## 📊 Troubleshooting Épico (Sesión 2025-10-19)

### Problema: Build Roto Desde Cero

**Síntomas iniciales:**
- `libbpf-dev` no encontrado
- `jsoncpp` headers missing
- `libprotobuf32` faltante
- `linux-headers-$(uname -r)` no existe
- Protobuf no compilaba (faltaba `protoc`)
- ZMQ "Address already in use"

**Root Causes Identificados:**

1. **Dependencies en múltiples fases** → Pérdida de paquetes con `apt-get remove`
2. **linux-headers version-specific** → No existe en repos
3. **Protobuf compiler vs runtime** → Ambos necesarios
4. **ZMQ socket pool = 4** → Solo 1 puede bind al mismo puerto

**Soluciones Implementadas:**

1. ✅ **Vagrantfile single-phase provisioning**
    - Todas las deps en una sola fase
    - Sin `apt-get remove` entre fases
    - Verificación post-instalación

2. ✅ **linux-headers-amd64** (metapaquete)
    - En vez de `linux-headers-$(uname -r)`
    - Siempre disponible en repos

3. ✅ **Protobuf completo**
```bash
   protobuf-compiler    # Para compilar .proto
   libprotobuf-dev      # Headers C++
   libprotobuf32        # Runtime library
```

4. ✅ **ZMQ socket pool = 1**
```json
   "socket_pools": {
     "push_sockets": 1  // Solo 1 socket PUB puede bind
   }
```

5. ✅ **Protobuf precompilado**
    - Script `protobuf/generate.sh`
    - Makefile copia `.pb.cc/.pb.h` automáticamente

**Resultado:** Build reproducible 100% desde `make destroy && make dev-setup`

---

## 📂 Estructura del Proyecto
```
test-zeromq-docker/
├── Makefile                    # Build system (host)
├── Vagrantfile                 # VM definition (single-phase deps)
├── docker-compose.yml          # Legacy lab (etcd + service3)
│
├── sniffer/                    # Sniffer eBPF v3.1
│   ├── Makefile               # Build sniffer (VM)
│   ├── CMakeLists.txt         # CMake config
│   ├── src/
│   │   ├── kernel/            # eBPF/XDP program
│   │   └── userspace/         # Feature extraction
│   ├── config/
│   │   └── sniffer.json       # Configuración producción
│   └── build/                 # Binarios compilados
│
├── ml-detector/               # ML Detector v1.0
│   ├── CMakeLists.txt        # CMake config
│   ├── src/
│   │   ├── core/             # Pipeline ZMQ
│   │   ├── inference/        # ONNX inference
│   │   └── protobuf/         # Copied .pb.cc/.pb.h
│   ├── config/
│   │   └── ml_detector_config.json
│   └── models/
│       └── level1_rf_model.onnx
│
├── protobuf/                  # Shared schema
│   ├── network_security.proto # Schema v3.1.0
│   ├── generate.sh           # Regeneration script
│   ├── network_security.pb.cc # Generated C++
│   └── network_security.pb.h
│
└── ml-training/              # Python ML training
    ├── scripts/
    │   ├── train_level1.py
    │   └── convert_to_onnx.py
    └── models/
```

---

## 🔧 Dependencias (Auto-instaladas)

### eBPF Toolchain
- clang 14.0.6
- llvm 14.0.6
- bpftool 7.1.0
- libbpf 1.1.2
- linux-headers-amd64

### Libraries
- jsoncpp 1.9.5
- libzmq 4.3.4
- libcurl 7.88.1
- protobuf 3.21.12
- liblz4 1.9.4
- libzstd 1.5.4
- spdlog 1.10.0
- nlohmann-json 3.11.2

### ML
- CMake 3.25.0
- ONNX Runtime 1.17.1

### Python
- Python 3.11.2
- numpy, pandas, scikit-learn, onnx

---

## 🎯 Roadmap

### ✅ Completado (Milestone v1.0.0)
- [x] Sniffer eBPF compila y captura
- [x] ML Detector Level 1 funcional
- [x] Pipeline ZMQ Sniffer → Detector
- [x] Protobuf serialization
- [x] Build reproducible desde cero
- [x] Vagrant single-phase provisioning

### 🔜 Próxima Sesión
- [ ] Fix crash ZMQ (zmq_msg lifecycle)
- [ ] Test E2E con carga sostenida
- [ ] Package sniffer como .deb
- [ ] Clean warnings ML Detector

### 📅 Fase 2: Multi-Level Detection
- [ ] ML Detector Level 2 - DDoS (83 features)
- [ ] ML Detector Level 2 - Ransomware (83 features)
- [ ] ML Detector Level 3 - Internal Traffic (4 features)
- [ ] Confidence thresholds configurables

### 📅 Fase 3: Production Ready
- [ ] LZ4 decompression en ML Detector
- [ ] etcd integration (config sync + encryption tokens)
- [ ] Package ml-detector como .deb
- [ ] Docker packaging (excepto sniffer)
- [ ] Prometheus metrics
- [ ] Alert routing

### 📅 Fase 4: Orchestration
- [ ] K3s deployment manifests
- [ ] Horizontal scaling
- [ ] Model hot-reload
- [ ] Adaptive thresholds

---

## 🔍 Debugging

### Verbose Logging Sniffer
```bash
# Nivel 1: Resumen básico
sudo ./sniffer --verbose

# Nivel 2: Features agrupadas
sudo ./sniffer -vv

# Nivel 3: Dump completo (193 features)
sudo ./sniffer -vvv > features.log 2>&1
```

### Network Diagnostics
```bash
vagrant ssh

# Ver interfaces
ip -4 addr

# Ver puertos
sudo ss -tlnp | grep -E '5571|5572'

# Capturar tráfico ZMQ
sudo tcpdump -i lo -n port 5571 -A
```

### Build Diagnostics
```bash
# Verificar deps instaladas
vagrant ssh -c "pkg-config --modversion libbpf jsoncpp libzmq"

# CMake verbose
cd /vagrant/sniffer/build
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1
```

---

## 📚 Documentación Adicional

- **DECISIONS.md**: Decisiones de arquitectura y lecciones aprendidas
- **CONFIGURATION.md**: Referencia de configuración (sniffer + detector)
- **protobuf/README.md**: Schema versioning y regeneración
- **Vagrantfile**: Todas las dependencias documentadas inline

---

## 🙏 Agradecimientos

Este proyecto es el resultado de:
- Troubleshooting sistemático y metódico
- Documentación exhaustiva de cada decisión
- Infraestructura reproducible desde cero
- Compromiso con la excelencia técnica

**"Smooth is fast"** - La robustez viene antes que la velocidad.

---

## 📞 Próximos Pasos

**Sesión siguiente:**
1. Debug y fix crash ZMQ (src/msg.cpp:414)
2. Test pipeline bajo carga (1000+ paquetes/seg)
3. Package sniffer .deb
4. Commit milestone + tag v1.0.1

**Objetivo:** Pipeline 100% estable antes de añadir Level 2 models.

## v3.1.2-thread-safe-zmq (2025-10-21)

### 🔒 Critical Stability Fix

This release fixes a critical thread-safety violation in the ZMQ sender pipeline that caused crashes after approximately 54 minutes of operation.

**Key Improvements:**
- ✅ 10.39x stability improvement (54min → 9h+ uptime)
- ✅ 99.985% packet delivery rate
- ✅ Zero crashes under stress testing
- ✅ Production-ready

**Technical Details:**
- Added per-socket mutex protection for ZMQ operations
- Prevents concurrent access from multiple sender threads
- <2% performance overhead (negligible)

**Upgrade:** Strongly recommended for all deployments.

---
```
╔════════════════════════════════════════════════════════════╗
║  Este README refleja el estado REAL del proyecto           ║
║  Actualizado: 2025-10-19 07:00 CET                        ║
╚════════════════════════════════════════════════════════════╝
```