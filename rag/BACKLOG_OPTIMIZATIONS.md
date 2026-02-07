# Firewall-ACL-Agent: Async Queue + Batch ipset restore Optimization

**Fecha**: 7 Febrero 2026 - Day 52  
**Estado**: PLANIFICADO (implementar después de estabilización)  
**Prioridad**: HIGH  
**Impacto esperado**: 10x-100x speedup en operaciones IPSet

---

## 🎯 Objetivo

Reemplazar N × `ipset add` (N syscalls) con 1 × `ipset restore` (1 syscall) usando arquitectura async queue + worker pool + batch processing.

---

## 🏗️ Arquitectura Propuesta
```
NetworkSecurityEvent (ZMQ)
    ↓
Async Queue (thread-safe, lockfree o mutex)
    ↓
Worker Pool (1-N threads)
    ↓
Batch Accumulator (estructura en memoria)
    ↓
Trigger: batch_size >= threshold OR timeout
    ↓
Flush: ipset restore -exist (1 syscall)
    ↓
Reiniciar batch vacío
```

### Componentes

1. **Async Queue**
    - Lockfree queue o std::queue + std::mutex
    - Recibe IPs de ZMQSubscriber
    - Bounded capacity (ej: 100K elementos max)

2. **Worker Pool**
    - 1-4 threads configurables
    - Extraen IPs de la cola
    - Acumulan en batch local

3. **Batch Accumulator**
    - `std::unordered_set<std::string>` para deduplicar en memoria
    - `std::vector<IPEntry>` con metadata (ip, timeout, comment)

4. **Flush Logic**
    - Genera string con formato ipset restore
    - Ejecuta: `echo "$batch" | ipset restore -exist`
    - Limpia batch después de flush exitoso

---

## ⚙️ Configuración
```json
{
  "batch_processor": {
    "batch_size_threshold": 5000,        // Configurable: 100, 1K, 5K, 10K, 50K
    "batch_time_threshold_ms": 1000,     // Dual trigger: tiempo máximo
    "worker_threads": 2,                 // Threads en worker pool
    "max_queue_size": 100000,            // Límite de cola
    "enable_deduplication": true,        // Dedup en memoria (además de -exist)
    "ipset_name": "ml_defender_blacklist", // Desde config (NO hardcoded)
    "default_timeout_sec": 600
  }
}
```

---

## 🔄 Flujo Detallado

### 1. Enqueue
```cpp
void BatchProcessor::add_detection(const Detection& det) {
    async_queue_.push(det.src_ip);
    stats_.queue_depth++;
}
```

### 2. Worker Thread
```cpp
void BatchProcessor::worker_loop() {
    while (running_) {
        std::string ip = async_queue_.pop();  // Blocking
        batch_accumulator_.insert(ip);
        
        if (should_flush()) {
            flush_batch();
        }
    }
}
```

### 3. Flush Trigger (Dual)
```cpp
bool BatchProcessor::should_flush() {
    return batch_accumulator_.size() >= config_.batch_size_threshold
        || (now() - last_flush_time_) >= config_.batch_time_threshold_ms;
}
```

### 4. Flush Execution
```cpp
void BatchProcessor::flush_batch() {
    // Generar string ipset restore
    std::stringstream restore_input;
    for (const auto& ip : batch_accumulator_) {
        restore_input << "add " << config_.ipset_name 
                     << " " << ip 
                     << " timeout " << config_.default_timeout_sec 
                     << "\n";
    }
    
    // Ejecutar (1 syscall para N IPs)
    int result = exec_ipset_restore(restore_input.str());
    
    if (result == 0) {
        stats_.ipset_successes++;
        stats_.ips_blocked += batch_accumulator_.size();
    } else {
        stats_.ipset_failures++;
        log_error("ipset restore failed", result);
    }
    
    // Reiniciar batch vacío
    batch_accumulator_.clear();
    last_flush_time_ = now();
}
```

### 5. ipset restore Format
```bash
# Input string generado por C++:
add ml_defender_blacklist 192.168.1.1 timeout 600
add ml_defender_blacklist 192.168.1.2 timeout 600
add ml_defender_blacklist 192.168.1.3 timeout 600
...
add ml_defender_blacklist 192.168.1.5000 timeout 600

# Ejecución:
echo "$batch_string" | ipset restore -exist
```

**Flag `-exist`**: Ignora duplicados sin error (idempotente)

---

## ✅ Validaciones Pre-Startup
```cpp
void BatchProcessor::validate_prerequisites() {
    // 1. Verificar privilegios root
    if (geteuid() != 0) {
        throw std::runtime_error("BatchProcessor requires root privileges");
    }
    
    // 2. Verificar que ipset existe en el kernel
    std::string cmd = "ipset list " + config_.ipset_name + " -n";
    int result = system(cmd.c_str());
    if (result != 0) {
        throw std::runtime_error("IPSet '" + config_.ipset_name + "' does not exist");
    }
    
    // 3. Verificar comando ipset disponible
    result = system("which ipset > /dev/null 2>&1");
    if (result != 0) {
        throw std::runtime_error("ipset command not found in PATH");
    }
    
    LOG_INFO("Prerequisites validated", 
        "ipset_name", config_.ipset_name,
        "running_as", "root");
}
```

---

## 🐛 Fixes de Bugs Existentes (Prerequisitos)

### Bug 1: IPSet Name Hardcoded
```cpp
// ANTES (INCORRECTO):
std::string ipset_name = "ml_defender_blacklist";  // Hardcoded

// DESPUÉS (CORRECTO):
std::string ipset_name = config_.ipset.set_name;  // Desde JSON
```

### Bug 2: Log Path Hardcoded
```cpp
// ANTES (INCORRECTO):
logger_ = std::make_unique<ObservabilityLogger>(
    "/vagrant/logs/firewall-acl-agent/firewall_detailed.log"  // Hardcoded
);

// DESPUÉS (CORRECTO):
logger_ = std::make_unique<ObservabilityLogger>(
    config_.logging.file  // Desde JSON: /vagrant/logs/lab/firewall-agent.log
);
```

### Bug 3: Logging Config Duplicado
```json
// ELIMINAR duplicados, mantener SOLO:
{
  "logging": {
    "level": "info",          // info, debug, warn, error
    "console": true,
    "file": "/vagrant/logs/lab/firewall-agent.log",
    "max_file_size_mb": 10,
    "backup_count": 5
  }
  
  // ELIMINAR "operation.enable_debug_logging"
}
```

**Precedencia**: Solo existe `logging.level`, no ambigüedad.

---

## 📊 Métricas Esperadas

### Antes (200 eventos)
```
ipset add calls: 200
syscalls: 200
ipset_failures: 200 (cuando ipset no existe)
latencia promedio: ~20ms por IP
```

### Después (200 eventos, batch=100)
```
ipset restore calls: 2
syscalls: 2
ipset_failures: 0 (con validación pre-startup)
latencia promedio: ~1ms por IP
```

### Stress Test (50K eventos, batch=5K)
```
ANTES:
  - 50K syscalls ipset add
  - ~6642 ipset failures bajo carga
  - Latencia total: ~1000 segundos

DESPUÉS:
  - 10 syscalls ipset restore (50K/5K)
  - <10 ipset failures (solo errores reales)
  - Latencia total: ~10-20 segundos
  
Speedup: 50x-100x
```

---

## 🔧 Implementación - Fases

### Fase 1: Estabilización (Day 53-54)
```
✅ Fix hardcoded ipset_name
✅ Fix hardcoded log_path
✅ Consolidar logging config
✅ Grep completo por hardcoded values
✅ Stress tests baseline: 1K, 5K, 10K, 20K
✅ Documentar métricas actuales
```

### Fase 2: Async Queue (Day 55)
```
□ Implementar lockfree queue o mutex queue
□ Integrar con BatchProcessor
□ Tests unitarios de queue
□ Métricas: queue_depth, enqueue_rate, dequeue_rate
```

### Fase 3: Worker Pool (Day 56)
```
□ Implementar worker threads configurables
□ Thread-safe batch accumulator
□ Dual trigger (size + timeout)
□ Tests de concurrencia
```

### Fase 4: ipset restore (Day 57)
```
□ Implementar exec_ipset_restore()
□ Formato string correcto
□ Manejo de errores granular
□ Logging por batch (no por IP)
```

### Fase 5: Validación & Stress (Day 58)
```
□ Pre-startup validations
□ Stress tests: 1K, 5K, 10K, 20K, 50K, 100K
□ Comparar métricas vs baseline
□ Tuning de batch_size_threshold
□ Documentar resultados
```

---

## ⚠️ Consideraciones

### Deduplicación
- **En memoria**: `std::unordered_set` previene duplicados en batch
- **Kernel**: `ipset restore -exist` ignora duplicados
- **Doble seguridad**: Eficiencia + Robustez

### Manejo de Errores
```cpp
// Si ipset restore falla:
1. Loggear error completo (stderr del comando)
2. NO perder el batch (guardar en disco?)
3. Incrementar stats_.flush_errors
4. Reintentar o alertar operador
```

### Latencia vs Throughput
```
Batch pequeño (100):   Baja latencia, menor ganancia
Batch medio (1K-5K):   Balance óptimo
Batch grande (50K+):   Alta latencia, máxima ganancia
```

**Recomendación inicial**: `batch_size=5000, timeout=1000ms`

### Privilegios
```bash
# firewall-acl-agent DEBE ejecutarse como root:
sudo ./firewall-acl-agent -c config.json

# Validar en startup:
if (geteuid() != 0) { error y exit }
```

---

## 📝 Commits Relacionados

- **Day 52**: etcd-client compression fix (prerequisito pipeline estable)
- **Day 53**: Fix hardcoded values (prerequisito config correcta)
- **Day 54**: Baseline stress tests (prerequisito métricas)
- **Day 55+**: Implementar async queue + worker pool + ipset restore

---

## 🎯 Success Criteria
```
✅ 50K eventos procesados sin ipset failures
✅ Latencia < 50ms percentil 99
✅ Throughput > 10K eventos/sec
✅ CPU usage < 20% durante operación normal
✅ Memory footprint < 500MB
✅ 0 hardcoded values en código
✅ Logging configurado correctamente (1 solo sitio)
```

---

**Autor**: Alonso Isidoro Roman + Claude (Anthropic)  
**Guardado en**: `/vagrant/rag/BACKLOG_OPTIMIZATIONS.md`  
**Via Appia Quality**: Built to last decades 🏛️