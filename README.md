# ML Detector Tricapa - eBPF Network Security Pipeline

> Sistema de detección de amenazas de red usando eBPF/XDP para captura de paquetes de alto rendimiento y modelos ML tricapa para clasificación en tiempo real.

```
┌─────────────────────────────────────────────────────────────────┐
│  Sniffer eBPF (XDP)  →  ML Detector (ONNX)  →  Alert/Action    │
│                                                                  │
│  eth0/eth2 capture → Feature Extraction → Level 1-3 Inference  │
│  Kernel Space         User Space ZMQ         C++ ONNX Runtime  │
└─────────────────────────────────────────────────────────────────┘
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-v3.2.1-blue)](https://github.com/alonsoir/test-zeromq-c-/releases/tag/v3.2.1-hybrid-filters)

---

## 🎯 Estado Actual

### v3.2.1 - Hybrid eBPF Filtering (Latest)

**✅ Componentes Operativos:**
- **Sniffer eBPF v3.2**: Captura con filtrado híbrido kernel/userspace
    - ✅ FD-based BPF map access (no pinning required)
    - ✅ Dynamic port filtering (excluded/included lists)
    - ✅ Ring buffer communication (kernel → userspace)
    - ✅ XDP/SKB mode support
- **ML Detector v1.0**: Inferencia ONNX Level 1 (23 features)
- **Pipeline ZMQ + Protobuf**: Comunicación funcional
- **Build System**: Reproducible con Vagrant + CMake

**🎉 Último Milestone (Oct 25, 2025):**
- Solucionado acceso a BPF filter maps vía FDs
- Sistema de filtrado híbrido 100% funcional
- Tests end-to-end validados

---

## 🚀 Quick Start

### Requisitos

- **Host**: macOS/Linux con VirtualBox y Vagrant
- **RAM**: 6GB para la VM
- **Disk**: 10GB libres

### Setup (Primera Vez)

```bash
# 1. Clone y setup automático
git clone https://github.com/alonsoir/test-zeromq-c-.git
cd test-zeromq-c-
make dev-setup  # ~10-15 min primera vez

# 2. Verificar instalación
make test
```

### Ejecución

**Terminal 1 - Sniffer:**
```bash
make run-sniffer
# Output: ✅ eBPF program attached, Filter maps loaded
```

**Terminal 2 - ML Detector:**
```bash
make run-detector
# Output: ✅ ZMQ Handler loop started, Predictions active
```

**Terminal 3 - Tests:**
```bash
make ssh
ping -c 10 8.8.8.8  # Genera tráfico
```

---

## 🛠️ Comandos Principales

### VM Management
```bash
make up          # Arrancar VM
make halt        # Parar VM
make ssh         # Conectar a VM
make status      # Estado actual
```

### Build & Development
```bash
make all         # Compilar todo
make rebuild     # Clean + build completo
make test        # Verificar instalación
make logs-sniffer    # Ver logs sniffer
make logs-detector   # Ver logs detector
```

### Troubleshooting
```bash
make check-ports     # Ver puertos 5571/5572
make kill-all        # Matar procesos
vagrant reload       # Reiniciar VM limpia
```

---

## 📂 Estructura del Proyecto

```
test-zeromq-docker/
├── Makefile                    # Build orchestration (host)
├── Vagrantfile                 # VM definition (Debian 12)
│
├── sniffer/                    # eBPF Sniffer v3.2
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── kernel/            # sniffer.bpf.c (XDP program)
│   │   └── userspace/         # Feature extraction + ZMQ
│   ├── include/               # Headers (ebpf_loader, bpf_map_manager)
│   └── config/sniffer.json    # Runtime config
│
├── ml-detector/               # ML Inference v1.0
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── core/             # ZMQ pipeline
│   │   └── inference/        # ONNX runtime
│   ├── config/ml_detector_config.json
│   └── models/level1_rf_model.onnx
│
├── protobuf/                  # Shared schema
│   ├── network_security.proto
│   └── generate.sh
│
└── ml-training/              # Python training scripts
    └── scripts/train_*.py
```

---

## 🔧 Arquitectura del Filtrado Híbrido (v3.2.1)

### Flujo de Decisión

```
┌─────────────────────────────────────────────────────────────┐
│                    KERNEL SPACE                              │
│  sniffer.bpf.o (XDP Hook)                                   │
│    └─> Packet Arrives                                       │
│        ├─> Check excluded_ports map → DROP if matched      │
│        ├─> Check included_ports map → PASS if matched      │
│        └─> Apply default_action (CAPTURE/DROP)             │
│            └─> If CAPTURE: Ring Buffer → USERSPACE         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    USERSPACE                                 │
│  sniffer userspace process                                  │
│    ├─> Ring Buffer Consumer (thread pool)                  │
│    ├─> Feature Extraction (193 features)                   │
│    ├─> Protobuf Serialization                              │
│    └─> ZMQ PUSH → tcp://127.0.0.1:5571                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  ml-detector (ONNX Inference)                               │
│    ├─> ZMQ SUB ← tcp://127.0.0.1:5571                      │
│    ├─> Preprocessing (23 features subset)                  │
│    ├─> RandomForest inference                              │
│    └─> Classification: BENIGN / ATTACK                     │
└─────────────────────────────────────────────────────────────┘
```

### Configuración de Filtros

**Archivo:** `sniffer/config/sniffer.json`

```json
{
  "filter": {
    "mode": "hybrid",
    "excluded_ports": [22, 4444, 8080],    // Drop en kernel
    "included_ports": [8000],              // Force capture
    "default_action": "capture"            // Other traffic
  }
}
```

### Verificar Filtros Activos

```bash
# Ver programa eBPF cargado
sudo bpftool prog show | grep sniffer

# Ver maps y contenido
sudo bpftool map dump name excluded_ports
sudo bpftool map dump name included_ports
sudo bpftool map dump name filter_settings

# Ver estadísticas
sudo bpftool map dump name stats_map
```

---

## 🧪 Testing

### Pruebas de Filtrado

```bash
# Terminal 1: Arrancar sniffer con filtros
cd /vagrant/sniffer
sudo ./build/sniffer -c config/sniffer.json

# Terminal 2: Generar tráfico
nc -vz localhost 22      # Puerto excluido → No captura
nc -vz localhost 8000    # Puerto incluido → Captura
nc -vz localhost 9999    # Default action → Captura
```

**Resultados Esperados:**
| Puerto | Configuración | Comportamiento |
|--------|---------------|----------------|
| 22     | excluded      | ✅ DROP en kernel (no stats) |
| 8000   | included      | ✅ PASS a userspace |
| 9999   | default       | ✅ PASS según default_action |

---

## 📊 Dependencias Principales

### eBPF Toolchain
- clang 14.0.6, llvm, bpftool
- libbpf 1.1.2
- linux-headers-amd64

### C++ Libraries
- jsoncpp 1.9.5 (config parsing)
- libzmq 4.3.4 (IPC)
- protobuf 3.21.12 (serialization)
- onnxruntime 1.17.1 (ML inference)
- spdlog 1.10.0 (logging)

Todas instaladas automáticamente por `make dev-setup`.

---

## 🐛 Known Issues

### 1. ZMQ Crash bajo Carga Alta (No Bloqueante)

**Síntoma:** `Assertion failed: check () (src/msg.cpp:414)`

**Workaround:**
```json
{
  "batch_processing_size": 1,
  "zmq_sender_threads": 1
}
```

**Status:** Fix planificado para próxima release.

### 2. ML Detector Warnings (No Críticos)

Warnings de compilación normales de desarrollo, no afectan funcionalidad.

---

## 📈 Roadmap

### Próximos Pasos

- [ ] **v3.3.0**: Integración ML Levels 2-3 (DDOS, Ransomware)
- [ ] **v3.4.0**: Dynamic filter updates (sin reiniciar sniffer)
- [ ] **v3.5.0**: Dashboard web para monitoreo en tiempo real
- [ ] **v4.0.0**: Production-ready con alta disponibilidad

### Mejoras Técnicas

- [ ] Resolver crash ZMQ bajo carga
- [ ] Implementar caché de FDs en BPFMapManager
- [ ] Suite de tests automatizados
- [ ] Benchmarks de rendimiento
- [ ] Métricas Prometheus/Grafana

---

## 📚 Documentación Adicional

- **[CHANGELOG.md](docs/CHANGELOG.md)** - Historial de cambios detallado
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Guía de resolución de problemas
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Arquitectura técnica profunda
- **[v3.2.1 Release Notes](https://github.com/alonsoir/test-zeromq-c-/releases/tag/v3.2.1-hybrid-filters)**

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una feature branch (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la branch (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

---

## 📝 Changelog Destacado

### v3.2.1 (2025-10-25) - Hybrid Filtering Milestone

**✨ Features:**
- Implementado acceso FD-based a BPF filter maps
- Sistema de filtrado híbrido kernel/userspace completo
- Dynamic port filtering sin necesidad de pinning

**🐛 Fixes:**
- Solucionado "No such file or directory" en BPF map access
- Corregido constructor de EbpfLoader (orden de inicialización)
- Eliminada dependencia de `/sys/fs/bpf/` pinning

**🧪 Testing:**
- End-to-end tests validados con tráfico real
- Verificación de estadísticas en kernel space

### v3.2.0 (2025-10-20) - Enhanced Configuration

- Añadido soporte para filtros híbridos en configuración JSON
- Implementado BPFMapManager para gestión de maps
- Mejoras en logging y diagnóstico

### v3.3.1-rf-features-doc -m "Level 1 RF Features Documentation

Added missing rf_23_features.json:
- 23 features for Level 1 attack detection
- Reconstructed from model metadata + C++ code
- Complete documentation with mappings
- Model accuracy: 99.88%

Fixes technical debt violation of 'single source of truth'.

🔍 What This JSON Contains
Complete Documentation for Each Feature:

{
"id": 1,
"name": "Packet Length Std",
"model_name": " Packet Length Std",  // ← Exact name model expects
"type": "float",
"unit": "bytes",
"description": "Standard deviation of packet lengths...",
"cpp_mapping": "PACKET_LEN_STD (feature_extractor.cpp line 106)",
"extraction_function": "extract_packet_len_std",
"category": "packet_size",
"expected_range": [0.0, 10000.0]
}

9 Feature Categories:

Packet Size (10 features) - Size statistics
TCP Flags (2 features) - ACK, PSH counts
Volume (2 features) - Total bytes
Rate (2 features) - Packets/s, bytes/s
Timing (2 features) - Duration, IAT
Subflow (3 features) - Window aggregations
TCP Window (1 feature) - Initial window
Network (1 feature) - Destination port
Activity (1 feature) - Active packets

Model Performance Metrics:

Accuracy: 99.88%
Precision: 99.44%
Recall: 99.93%
F1-Score: 99.68%
ROC-AUC: 99.99%


⚠️ Critical Notes
1. Feature Order is SACRED
   The order in the JSON MUST match the order the model expects:

# Model expects this exact order:
[" Packet Length Std", " Subflow Fwd Bytes", " Fwd Packet Length Max", ...]

Do NOT reorder features - it will break inference.
2. Leading Spaces in Names
   Feature names have leading spaces (e.g., " Packet Length Std").
   This comes from the original CIC-IDS2017 CSV and must be preserved.
3. Code Extracts 83, Model Uses 23
   The C++ code extracts 83 features total:

Lines 90-113: ORIGINAL 23 (Level 1 uses these)
Lines 115-135: PHASE 1 - 20 more features
Lines 137-152: PHASE 2 - 15 more features
Lines 154-174: PHASE 3 - 20 more features
Lines 176-181: PHASE 4 - 5 final features

Level 1 model only uses the first 23.
4. Feature Name Mapping
   Some feature names in metadata don't match C++ enum names exactly:

Model: " Packet Length Std" → C++: PACKET_LEN_STD
Model: "Fwd Packets/s" → C++: SRATE
Model: " Flow Duration" → C++: DURATION

The JSON documents both names for clarity.

🎯 Verification Checklist
Before pushing:

JSON file exists in correct location
JSON syntax is valid
Feature count = 23
Feature names match model metadata exactly
sniffer.json references the file correctly
System builds without errors
Runtime test passes (no config errors)
Committed with descriptive message
(Optional) Tagged for documentation


📊 Impact
Before Fix:

❌ No formal documentation of Level 1 features
❌ Violated "single source of truth"
❌ Impossible to reproduce training
❌ Hard to validate feature extraction

After Fix:

✅ Complete documentation in JSON
✅ Single source of truth restored
✅ Clear mapping: JSON ↔ C++ ↔ Model
✅ Can validate extraction logic
✅ Can reproduce training

Consider for other models:

ddos_83_features.json - probably also missing
internal_4_features.json - probably also missing
Create these next to complete documentation
💡 Lessons Learned
"Via Appia Quality" means:

Every model needs a corresponding JSON schema
Code is NOT documentation
Metadata is NOT documentation
JSON schema IS the single source of truth

For future models:

Train model → Export ONNX
Generate metadata.json
Create feature_schema.json immediately
Document mapping to C++ code
Only then commit

This prevents accumulating technical debt.

---

## 👥 Créditos

**Desarrollador:** [Alonso](https://github.com/alonsoir)  
**Asistente IA:** Claude (Anthropic)  
**Licencia:** MIT

---

## 📞 Soporte

- **Issues:** [GitHub Issues](https://github.com/alonsoir/test-zeromq-c-/issues)
- **Discussions:** [GitHub Discussions](https://github.com/alonsoir/test-zeromq-c-/discussions)

---

<div align="center">

**🎉 ML Detector Tricapa - Powered by eBPF 🎉**

*Actualizado: Octubre 25, 2025*

</div>