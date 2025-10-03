### 1. README.md - 

```markdown
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

