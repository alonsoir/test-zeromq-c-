# Changelog

Todos los cambios notables del proyecto están documentados aquí.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v3.2.1-hybrid-filters] - 2025-10-25

### ✨ Added

- **FD-based BPF Map Access**: Implementado acceso directo a BPF filter maps mediante File Descriptors
    - Nuevos getters en `EbpfLoader`: `get_excluded_ports_fd()`, `get_included_ports_fd()`, `get_filter_settings_fd()`
    - Nuevo método `load_filter_config_with_fds()` en `BPFMapManager`
    - Captura automática de FDs durante `load_program()`

- **Hybrid Filtering System**: Sistema de filtrado completo kernel/userspace
    - Port exclusion list (drop en kernel)
    - Port inclusion list (force capture)
    - Configurable default action (capture/drop)

### 🔧 Fixed

- **BPF Map Accessibility**: Solucionado error "No such file or directory (errno: 2)"
    - Eliminada dependencia de maps pinneados en `/sys/fs/bpf/`
    - Acceso directo vía FDs más eficiente y robusto

- **EbpfLoader Constructor**: Corregido orden de inicialización de miembros
    - Eliminada coma doble que causaba error de compilación
    - Reordenados miembros según declaración en header (warning resuelto)

### 🧪 Testing

- ✅ Validación end-to-end con tráfico real
- ✅ Puerto 22 (excluido): filtrado correctamente en kernel
- ✅ Puerto 8000 (incluido): capturado en userspace
- ✅ Puerto 9999 (default): comportamiento según configuración
- ✅ BPF maps verificados con `bpftool` (FDs: 6, 7, 8)

### 📝 Technical Details

**Modified Files:**
- `include/ebpf_loader.hpp`: +3 FD members, +3 getters
- `src/userspace/ebpf_loader.cpp`: Constructor fix, FD capture, getter implementation
- `include/bpf_map_manager.h`: New `load_filter_config_with_fds()` declaration
- `src/userspace/bpf_map_manager.cpp`: Full implementation of FD-based loading
- `src/userspace/main.cpp`: Updated to use new FD-based method

**Benefits:**
- No pinning required
- More efficient (no filesystem lookups)
- Container-friendly (no `/sys/fs/bpf/` mount needed)
- Better resource management (FDs auto-cleanup)

---

## [v3.2.0] - 2025-10-20

### ✨ Added

- **Enhanced Configuration System**:
    - Soporte completo para filtros híbridos en JSON
    - Validación de configuración de filtros
    - Profiles (lab, production, testing)

- **BPFMapManager Module**:
    - Nueva clase para gestión centralizada de BPF maps
    - Port list validation
    - Batch port operations

- **Config Types**:
    - `FilterConfig` struct con modo híbrido
    - `config_types.cpp` implementation

### 🔧 Changed

- **sniffer.json**: Añadida sección `filter` con configuración híbrida
- **CMakeLists.txt**: Build config para nuevos módulos
- **main.h**: Updated declarations para nuevas funcionalidades

### 📝 Technical Details

**New Files:**
- `include/bpf_map_manager.h`
- `src/userspace/bpf_map_manager.cpp`
- `src/userspace/config_types.cpp`

---

## [v3.1.0] - 2025-10-19

### 🔧 Fixed

- **Build System Overhaul**: Build reproducible 100% desde cero
    - Vagrantfile single-phase provisioning
    - Sin `apt-get remove` entre fases
    - Todas las dependencias en una sola fase

- **Dependencies Resolution**:
    - `linux-headers-amd64` (metapaquete) en vez de version-specific
    - Protobuf completo: compiler + dev + runtime
    - libbpf-dev correctamente instalado
    - jsoncpp headers disponibles

### 🐛 Bug Fixes

- **ZMQ Socket Pool**: Reducido de 4 a 1 (solo 1 socket puede bind al mismo puerto)
- **Protobuf Generation**: Script `generate.sh` automatizado
- **CMake Config**: Rutas de headers corregidas

### 📊 Improvements

- **Documentation**: README completo con troubleshooting
- **Makefile**: Comandos útiles para desarrollo
- **Vagrant**: Provisioning mejorado y verificado

---

## [v1.0.0-stable-pipeline] - 2025-10-15

### ✨ Initial Release

- **Sniffer eBPF v3.1**:
    - XDP program con AF_XDP socket
    - Feature extraction (193 features)
    - ZMQ PUB output
    - Ring buffer communication

- **ML Detector v1.0**:
    - Level 1 inference (RandomForest)
    - ONNX Runtime integration
    - 23 features preprocessing
    - ZMQ SUB input

- **Pipeline**:
    - Protobuf schema v3.1.0
    - ZMQ communication working end-to-end
    - Sniffer → Detector → Classification

### 🔧 Technical Stack

- **eBPF**: clang 14, libbpf 1.1.2, bpftool 7.1.0
- **ML**: ONNX Runtime 1.17.1
- **IPC**: ZeroMQ 4.3.4, Protobuf 3.21.12
- **Build**: CMake 3.25, C++20
- **Infrastructure**: Vagrant (Debian 12), Docker Compose

---

## Known Issues

### Active

- **[Issue #1]** ZMQ crash bajo carga alta (>100 pkt/sec)
    - Workaround: batch_size=1, zmq_threads=1
    - Status: Investigando lifecycle de zmq_msg_t

### Resolved

- ~~**[Issue #2]** BPF map pinning dependency~~ → Fixed in v3.2.1
- ~~**[Issue #3]** Build failures desde cero~~ → Fixed in v3.1.0
- ~~**[Issue #4]** Protobuf generation manual~~ → Fixed in v3.1.0

---

## Migration Guides

### v3.2.0 → v3.2.1

**No breaking changes.** Recompilar sniffer:

```bash
cd /vagrant/sniffer/build
make clean && make -j4
```

**Config changes (optional):** Si deseas usar filtros híbridos, añade a `sniffer.json`:

```json
{
  "filter": {
    "mode": "hybrid",
    "excluded_ports": [22, 4444],
    "included_ports": [8000],
    "default_action": "capture"
  }
}
```

### v3.1.0 → v3.2.0

**New modules required:**
- `bpf_map_manager.cpp` debe estar compilado
- `config_types.cpp` debe estar presente

Ejecutar:
```bash
make rebuild  # Limpia y recompila todo
```
## [3.2.1] - 2025-10-28

### Changed
- **BREAKING**: None - fully backward compatible
- Refactored sniffer.json: eliminated 4 duplicate interface fields
- Refactored ml_detector_config.json: eliminated duplicate base_dir
- Enhanced CMakeLists.txt (sniffer): automatic config copying
- Enhanced CMakeLists.txt (ml-detector): automatic symlink creation
- Fixed Makefile: run-sniffer now includes config path

### Philosophy
- Enforced "single source of truth" across all configs
- Eliminated manual copy requirements
- Made it impossible to misconfigure

### Technical Debt Eliminated
- Config duplication (7 → 3 interface definitions)
- Manual file copying (sniffer config)
- Manual directory copying (ml-detector models)
- Sync issues between source and build

### Validation
- ✅ Full rebuild successful
- ✅ Sniffer operational (eth0 capture)
- ✅ ML-Detector operational (2 models loaded)
- ✅ End-to-end test: 8 events, 0 errors
- ✅ Performance: 1.7ms latency, 93.67% accuracy
---

## [v3.2.2-protocol-signed-map] - 2025-11-01

### ✨ Added

* **Protocol Map Future Extension (Design Stage)**
  Añadido al backlog el diseño de un sistema de **extensión firmada** de la tabla de protocolos (`protocol_numbers.hpp`).
  Aunque no se implementa en esta versión, se define la especificación técnica y filosofía de seguridad que lo guiará:

    * Tabla principal sigue compilada en binario (inmutable, `constexpr`).
    * Soporte futuro para **extensión mediante JSON firmado (ECDSA)**.
    * Validación estricta de integridad:

        * `SHA-256` del archivo debe coincidir con el firmado.
        * Firma validada contra clave pública embebida.
        * Si falla la verificación, se ignora la extensión y se continúa con la tabla estática.
    * Modo *opt-in*, sin impacto en rendimiento por defecto.

# Changelog

All notable changes to the Enhanced Network Sniffer project.

## [3.2.1] - 2025-11-01 - Phase 1E: Live Traffic Validation ✅

### 🎉 Major Milestone: MVP Complete

**Runtime:** 271 seconds (4.5 minutes)  
**Events Processed:** 222  
**Alerts Generated:** 150+  
**Crashes:** 0  
**Average Processing Time:** 229.66 μs/event

### Added
- ✅ **Live Traffic Validation**
    - Tested with real network traffic for 271 seconds
    - Generated 150+ ransomware alerts
    - Zero crashes, zero memory leaks
    - Graceful shutdown working perfectly

### Performance
- ⚡ 229.66 μs average processing time per event
- 📊 0.82 events/second sustained rate
- 🎯 <1ms latency end-to-end
- 💪 Zero dropped events

### Validated
- Two-layer detection system working in production
- FastDetector generating real-time alerts
- FeatureProcessor extracting features every 30s
- Thread-safe architecture under real load
- Clean component shutdown

---

## [3.2.0] - 2025-11-01 - Phase 1D: Two-Layer Detection Integration ✅

### Added
- ✅ **Two-Layer Ransomware Detection System**
    - Layer 1: FastDetector (10s window, heuristics)
    - Layer 2: RansomwareFeatureProcessor (30s aggregation)
    - Integrated in RingBufferConsumer main loop

- ✅ **Protobuf Schema Compliance**
    - `send_fast_alert()` using NetworkSecurityEvent
    - `send_ransomware_features()` using NetworkFeatures.ransomware
    - Correct field mapping: source_ip, destination_ip, protocol_number
    - Threat scoring: overall_threat_score, final_classification

- ✅ **Thread-Local FastDetector**
    - Zero contention between threads
    - Per-thread state isolation
    - Definition: `thread_local FastDetector RingBufferConsumer::fast_detector_`

- ✅ **Statistics Tracking**
    - `stats_.ransomware_fast_alerts` - Layer 1 alerts
    - `stats_.ransomware_feature_extractions` - Layer 2 extractions
    - `stats_.ransomware_confirmed_threats` - High-confidence detections
    - `stats_.ransomware_processing_time_us` - Performance metrics

### Fixed
- Namespace resolution for IPProtocol enum (sniffer::)
- Protobuf field naming (source_ip vs src_ip)
- thread_local static member definition
- CMakeLists.txt: added fast_detector.cpp to SNIFFER_SOURCES
- IP address conversion (htonl + inet_ntop)

### Changed
- feature_logger.cpp: uses sniffer::IPProtocol
- flow_tracker.cpp: uses sniffer::protocol_to_string()
- ransomware_feature_processor.cpp: uses IPProtocol enum

---

## [3.1.0] - 2025-11-01 - Phase 1C: FastDetector Implementation ✅

### Added
- ✅ **FastDetector Class** (`include/fast_detector.hpp`, `src/userspace/fast_detector.cpp`)
    - 10-second sliding window
    - 4 heuristic rules:
        1. External IPs: >10 in 10s
        2. SMB connections: >5 in 10s
        3. Port scanning: >15 unique ports in 10s
        4. RST ratio: >30% in 10s
    - Thread-local storage for zero contention
    - Microsecond-level latency

- ✅ **Comprehensive Testing** (`tests/test_fast_detector.cpp`)
    - 5 test cases covering all heuristics
    - Window expiration logic
    - Threshold validation
    - All tests passing ✅

### Performance
- <1 microsecond per ingest() call
- Zero memory allocations in fast path
- Thread-safe via thread_local

---

## [3.0.0] - 2025-11-01 - Phase 1A: Protocol Numbers Standardization ✅

### Added
- ✅ **Protocol Numbers Header** (`include/protocol_numbers.hpp`)
    - 30+ IANA standard protocol definitions
    - Type-safe enum class `IPProtocol`
    - Helper functions: `protocol_to_string()`, `protocol_to_number()`
    - Complete documentation with RFC references

### Changed
- **Zero Magic Numbers Policy**
    - Replaced all numeric protocol constants (6, 17, 1, etc.)
    - Updated all comparisons to use IPProtocol enum
    - Improved code readability and maintainability

### Benefits
- Type safety at compile time
- Self-documenting code
- Easy to extend with new protocols
- Follows industry standards (IANA)

---

## [2.0.0] - 2025-09-18 - Base System

### Features
- eBPF/XDP packet capture
- Ring buffer processing
- Protobuf serialization
- ZMQ communication
- Basic feature extraction (83+ features)
- LZ4/Zstd compression
- Multi-threaded pipeline

---

## Version Format

`[MAJOR.MINOR.PATCH] - YYYY-MM-DD - Description`

- **MAJOR:** Breaking changes or major milestones
- **MINOR:** New features, backwards compatible
- **PATCH:** Bug fixes, minor improvements

### 🧩 Motivation

Permitir en el futuro la incorporación de **nuevos protocolos registrados por IANA o internos** sin recompilar el sistema, preservando la filosofía de **seguridad determinista y compatibilidad hacia adelante**.

### 🧠 Philosophy

> *“Smooth is fast — the future is modular but signed.”*
> Toda ampliación en el IDS deberá ser verificable criptográficamente, incluso en runtime.

### 📝 Status

* **Implementation:** Deferred to Phase 4 (2026)
* **Assigned:** Architecture Group (Claude + Alonso + GPT)
* **Backlog Reference:** `ISSUE-012`

---


## Upcoming Changes

### v3.3.0 (Planned)

- [X] Integration of ML Level 2 (DDoS detection)
- [ ] Integration of ML Level 3 (Ransomware detection)
- [ ] Multi-level decision pipeline
- [ ] Enhanced logging with structured output

### v3.4.0 (Planned)

- [ ] Dynamic filter updates without restart
- [ ] REST API for runtime configuration
- [ ] Metrics endpoint (Prometheus format)
- [ ] Performance optimizations

### v4.0.0 (Future)

- [ ] Production hardening
- [ ] High availability support
- [ ] Multi-node deployment
- [ ] Complete monitoring stack

---

## Contributors

- **Alonso** (@alonsoir)          - Project Lead & Development
- **Claude** (Anthropic)          - AI Assistant for Architecture & Debugging
- **ChatGPT5** (OpenAI)           - AI Assistant for Architecture
- **Parallels.ai** (Parallels.ai) - AI Assistant for Architecture
---

## License

MIT License - See [LICENSE](LICENSE) file for details

---

<div align="center">

**📝 For detailed technical documentation, see [docs/](docs/) folder**

*Last Updated: October 25, 2025*

</div>