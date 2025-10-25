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

---

## Upcoming Changes

### v3.3.0 (Planned)

- [ ] Integration of ML Level 2 (DDoS detection)
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

- **Alonso** (@alonsoir) - Project Lead & Development
- **Claude** (Anthropic) - AI Assistant for Architecture & Debugging

---

## License

MIT License - See [LICENSE](LICENSE) file for details

---

<div align="center">

**📝 For detailed technical documentation, see [docs/](docs/) folder**

*Last Updated: October 25, 2025*

</div>