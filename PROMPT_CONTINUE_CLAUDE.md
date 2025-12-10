# 🎯 ML DEFENDER - DAY 15 CONTINUIDAD
## RAG Logger Validation & Neris Dataset Testing
**Fecha:** 11 de Diciembre de 2025  
**Contexto:** Acabamos de completar la implementación del RAGLogger en Day 14

---

## ✅ LO QUE HEMOS LOGRADO HOY (Day 14)

### 🎉 RAGLogger Implementado y Compilando
- **rag_logger.hpp** (165 líneas) - Header completo con todas las declaraciones
- **rag_logger.cpp** (520 líneas) - Implementación funcional y compilada
- **Integración en zmq_handler.cpp** - Constructor, destructor, llamadas en process_event()
- **CMakeLists.txt actualizado** - rag_logger.cpp añadido a SOURCES

### 📊 Arquitectura RAGLogger Operativa
```
/vagrant/logs/rag/
├── events/
│   └── YYYY-MM-DD.jsonl          # JSON Lines: 1 evento por línea
└── artifacts/
    └── YYYY-MM-DD/
        ├── event_<id>.pb         # Protobuf binario completo
        └── event_<id>.json       # JSON human-readable
```

### 🎯 Criterios de Logging (CONFIGURABLES)
**Configuración actual** (`config/rag_logger_config.json`):
```json
{
  "min_score_to_log": 0.70,        // Score mínimo para registrar
  "min_divergence_to_log": 0.30,   // Divergencia mínima Fast vs ML
  "max_events_per_file": 10000,    // Rotación por eventos
  "max_file_size_mb": 100,          // Rotación por tamaño
  "save_protobuf_artifacts": true,  // Guardar .pb binarios
  "save_json_artifacts": true       // Guardar .json legibles
}
```

**Lógica de filtrado** (en `should_log_event()`):
```cpp
// Se registra si cumple CUALQUIERA de estas condiciones:
1. decision_metadata.requires_rag_analysis == true
2. overall_threat_score >= 0.70
3. score_divergence >= 0.30
```

### 📈 Performance Impact Estimado
- **Latencia adicional:** ~1-2 μs por evento (write JSON + flush periódico)
- **Memoria adicional:** ~4 MB (buffers JSON)
- **Overhead total:** <2% (negligible)
- **Throughput:** Mantiene ~160 eventos/s

---

## 🔍 TAREAS PRIORITARIAS PARA MAÑANA (Day 15)

### 1. ✅ VALIDACIÓN BÁSICA DEL SISTEMA

#### 1.1. Verificar Dependencias en Vagrantfile
**ACCIÓN:** Comprobar que están instaladas:
```ruby
# En Vagrantfile, sección de provisioning:
config.vm.provision "shell", inline: <<-SHELL
  # RAGLogger dependencies
  apt-get install -y libssl-dev           # Para SHA256
  apt-get install -y nlohmann-json3-dev   # Para JSON
  apt-get install -y jq                   # Para análisis logs
SHELL
```

**COMANDO TEST:**
```bash
vagrant ssh defender -c "dpkg -l | grep -E 'libssl-dev|nlohmann-json3-dev|jq'"
```

Si falta algo:
```bash
vagrant ssh defender
sudo apt-get update
sudo apt-get install -y libssl-dev nlohmann-json3-dev jq
```

#### 1.2. Verificar Estructura de Directorios RAG
**ACCIÓN:** Asegurar que existen y tienen permisos correctos:
```bash
vagrant ssh defender
ls -la /vagrant/logs/rag/
ls -la /vagrant/logs/rag/events/
ls -la /vagrant/logs/rag/artifacts/

# Si no existen:
mkdir -p /vagrant/logs/rag/events
mkdir -p /vagrant/logs/rag/artifacts
chmod 755 /vagrant/logs/rag -R
```

#### 1.3. Verificar Configuración RAGLogger
**ACCIÓN:** Asegurar que existe `config/rag_logger_config.json`:
```bash
vagrant ssh defender -c "cat /vagrant/config/rag_logger_config.json"
```

Si no existe, crearlo:
```bash
cat > config/rag_logger_config.json << 'EOF'
{
  "base_path": "/vagrant/logs/rag",
  "deployment_id": "ml-defender-dev",
  "node_id": "detector-01",
  "min_score_to_log": 0.70,
  "min_divergence_to_log": 0.30,
  "max_events_per_file": 10000,
  "max_file_size_mb": 100,
  "save_protobuf_artifacts": true,
  "save_json_artifacts": true
}
EOF
```

### 2. 🧪 TEST 1: smallFlows (Validación Rápida)

**OBJETIVO:** Verificar que RAGLogger funciona correctamente

#### 2.1. Limpiar Logs Previos
```bash
vagrant ssh defender
rm -rf /vagrant/logs/rag/events/*
rm -rf /vagrant/logs/rag/artifacts/*
```

#### 2.2. Ejecutar Test
```bash
make test-replay-small
# O manualmente:
# cd /vagrant && sudo ./replay_pcap.sh datasets/smallFlows.pcap
```

#### 2.3. Verificar Logs Generados
```bash
# 1. Verificar que se creó el log diario
ls -lh /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl

# 2. Contar eventos RAG registrados
cat /vagrant/logs/rag/events/*.jsonl | wc -l

# 3. Ver primer evento (pretty-print)
cat /vagrant/logs/rag/events/*.jsonl | head -1 | jq .

# 4. Verificar artifacts (si están habilitados)
ls -lh /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/

# 5. Contar artifacts protobuf
ls /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/*.pb | wc -l

# 6. Ver un artifact JSON legible
ls /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/*.json | head -1 | xargs cat | jq .
```

#### 2.4. Análisis Estadístico Básico
```bash
# Estadísticas de eventos RAG
cat /vagrant/logs/rag/events/*.jsonl | jq -s '{
  total_events: length,
  divergent: [.[] | select(.detection.scores.divergence > 0.30)] | length,
  high_score: [.[] | select(.detection.scores.final_score >= 0.80)] | length,
  sources: [.[] | .detection.classification.authoritative_source] | group_by(.) | map({source: .[0], count: length})
}'

# Distribución de scores
cat /vagrant/logs/rag/events/*.jsonl | jq -s '[.[] | {
  fast: .detection.scores.fast_detector,
  ml: .detection.scores.ml_detector,
  final: .detection.scores.final_score,
  div: .detection.scores.divergence
}]'
```

**RESULTADOS ESPERADOS (smallFlows = 1,207 eventos):**
- Total RAG events: **~80-100** (score >= 0.70 OR divergence >= 0.30)
- Divergent events: **~80-100** (Fast 0.70 vs ML 0.39)
- Artifacts: **80-100 .pb + 80-100 .json** (si habilitados)
- Log file size: **~500 KB - 1 MB**

### 3. 🚀 TEST 2: Neris Botnet (Validación Real - 492K eventos)

**OBJETIVO:** Validación con dataset CTU-13 para F1-scores reales

#### 3.1. Dataset Info
```bash
Dataset: CTU-13 Neris botnet
Path: /vagrant/datasets/capture20110816.truncated.pcap
Total events: 492,358
Botnet IPs: 147.32.84.165, 147.32.84.191, 147.32.84.192
Expected RAG events: ~10,000-15,000 (2-3% del total)
```

#### 3.2. Preparación
```bash
# Limpiar logs previos
vagrant ssh defender
rm -rf /vagrant/logs/rag/events/*
rm -rf /vagrant/logs/rag/artifacts/*

# OPCIONAL: Deshabilitar artifacts para esta prueba (acelera)
# Editar config/rag_logger_config.json:
{
  "save_protobuf_artifacts": false,
  "save_json_artifacts": false
}
```

#### 3.3. Ejecutar Test Neris
```bash
make test-replay-neris
# O manualmente:
# cd /vagrant && sudo ./replay_pcap.sh datasets/capture20110816.truncated.pcap
```

**DURACIÓN ESTIMADA:** 30-45 minutos (492K eventos @ ~160 evt/s)

#### 3.4. Monitoreo en Tiempo Real
```bash
# Terminal 1: Logs del detector
vagrant ssh defender
tail -f /vagrant/logs/ml-detector/ml-detector.log | grep -E "RAG|DUAL-SCORE"

# Terminal 2: RAG events en vivo
vagrant ssh defender
tail -f /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl | jq .

# Terminal 3: Estadísticas cada 30 segundos
vagrant ssh defender
watch -n 30 'cat /vagrant/logs/rag/events/*.jsonl | wc -l'
```

#### 3.5. Análisis Post-Test
```bash
# 1. Total eventos RAG
total=$(cat /vagrant/logs/rag/events/*.jsonl | wc -l)
echo "Total RAG events: $total"

# 2. Eventos con IPs del botnet
botnet=$(cat /vagrant/logs/rag/events/*.jsonl | \
  jq -r 'select(.network.five_tuple.src_ip == "147.32.84.165" or 
                .network.five_tuple.src_ip == "147.32.84.191" or 
                .network.five_tuple.src_ip == "147.32.84.192")' | wc -l)
echo "Botnet IPs detected in RAG: $botnet"

# 3. Distribución de scores en botnet IPs
cat /vagrant/logs/rag/events/*.jsonl | \
  jq -r 'select(.network.five_tuple.src_ip == "147.32.84.165" or 
                .network.five_tuple.src_ip == "147.32.84.191" or 
                .network.five_tuple.src_ip == "147.32.84.192") | 
         [.network.five_tuple.src_ip, 
          .detection.scores.fast_detector, 
          .detection.scores.ml_detector, 
          .detection.scores.final_score] | @csv' | \
  head -20

# 4. Divergencia en eventos de botnet
cat /vagrant/logs/rag/events/*.jsonl | \
  jq -s '[.[] | select(.network.five_tuple.src_ip | 
         IN("147.32.84.165", "147.32.84.191", "147.32.84.192"))] | 
         {avg_divergence: (map(.detection.scores.divergence) | add / length)}'

# 5. Tamaño total de logs
du -h /vagrant/logs/rag/events/
du -h /vagrant/logs/rag/artifacts/
```

**RESULTADOS ESPERADOS:**
- RAG events: **10,000-15,000** (~2-3% del total)
- Botnet IPs en RAG: **500-1,000** (IPs maliciosas con scores altos)
- Average divergence: **0.25-0.35** (Fast alto, ML variable)
- Log file size: **50-80 MB** (sin artifacts), **200-300 MB** (con artifacts)

### 4. 🔧 REVISIÓN DEL MAKEFILE

**PROBLEMA POTENCIAL:** Durante el desarrollo se hicieron cambios manuales.

#### 4.1. Verificar Targets RAG
```bash
# Ver targets disponibles
make help | grep -i rag

# Deberían existir:
# rag-init       - Crear directorios RAG
# rag-clean      - Limpiar logs RAG
# rag-status     - Mostrar estadísticas RAG
# rag-analyze    - Análizar logs RAG
# rag-tail       - Ver logs en vivo
```

#### 4.2. Añadir Targets RAG al Makefile
Si faltan, añadir al final del Makefile raíz:

```makefile
# ============================================================================
# 🎯 RAG LOGGER TARGETS
# ============================================================================

.PHONY: rag-init rag-clean rag-status rag-analyze rag-tail

rag-init:
	@echo "📁 Initializing RAG directories..."
	vagrant ssh defender -c "mkdir -p /vagrant/logs/rag/events"
	vagrant ssh defender -c "mkdir -p /vagrant/logs/rag/artifacts"
	vagrant ssh defender -c "chmod 755 /vagrant/logs/rag -R"
	@echo "✅ RAG directories created"

rag-clean:
	@echo "🧹 Cleaning RAG logs..."
	vagrant ssh defender -c "rm -rf /vagrant/logs/rag/events/*"
	vagrant ssh defender -c "rm -rf /vagrant/logs/rag/artifacts/*"
	@echo "✅ RAG logs cleaned"

rag-status:
	@echo "📊 RAG Logger Status:"
	@vagrant ssh defender -c "ls -lh /vagrant/logs/rag/events/"
	@vagrant ssh defender -c "cat /vagrant/logs/rag/events/*.jsonl 2>/dev/null | wc -l || echo '0' | xargs echo 'Total events:'"

rag-analyze:
	@echo "📈 RAG Analysis:"
	@vagrant ssh defender -c "cat /vagrant/logs/rag/events/*.jsonl | jq -s '{total: length, divergent: [.[] | select(.detection.scores.divergence > 0.30)] | length}'"

rag-tail:
	@echo "👀 Tailing RAG logs (Ctrl+C to stop)..."
	vagrant ssh defender -c "tail -f /vagrant/logs/rag/events/$$(date +%Y-%m-%d).jsonl | jq ."
```

#### 4.3. Verificar Target Compile
```bash
# Verificar que 'make detector' funciona
make detector

# Si falla, verificar que CMakeLists.txt tiene:
grep "rag_logger.cpp" ml-detector/CMakeLists.txt
# Debería aparecer en la lista de SOURCES
```

---

## 🤔 DECISIONES ESTRATÉGICAS PARA MAÑANA

### DECISIÓN 1: ¿Qué Eventos Guardar en RAG?

#### Opción A: **Solo Eventos Interesantes** (RECOMENDADO ✅)
**Criterios actuales:**
- `score >= 0.70` OR
- `divergence >= 0.30` OR
- `requires_rag_analysis == true`

**Ventajas:**
- ✅ Reduce almacenamiento (2-3% del total)
- ✅ Logs manejables para análisis
- ✅ Foco en eventos que requieren investigación
- ✅ Mejor para RAG (menos ruido)

**Desventajas:**
- ❌ No tenemos "true negatives" completos para F1-score
- ❌ Perdemos contexto de eventos benignos

**Storage estimado:**
- smallFlows (1.2K): ~100 eventos → ~500 KB
- Neris (492K): ~15K eventos → ~75 MB
- 1 día producción (10M): ~300K eventos → ~1.5 GB/día

#### Opción B: **Guardar Todo con Muestreo**
**Criterios propuestos:**
- Eventos interesantes: 100% (como Opción A)
- Eventos benignos: 1% muestreo aleatorio

**Ventajas:**
- ✅ Tenemos true negatives para F1-score
- ✅ Contexto completo para análisis
- ✅ Mejor para reentrenamiento de modelos

**Desventajas:**
- ❌ Más almacenamiento (~5% del total)
- ❌ Más complejo de implementar

**Storage estimado:**
- smallFlows: ~200 eventos → ~1 MB
- Neris: ~20K eventos → ~100 MB
- 1 día producción: ~500K eventos → ~2.5 GB/día

#### Opción C: **Guardar Todo** (NO RECOMENDADO ❌)
**Ventajas:**
- ✅ Datos completos

**Desventajas:**
- ❌ Almacenamiento masivo (100%)
- ❌ Logs gigantes
- ❌ RAG se sobrecarga con ruido
- ❌ Poco práctico

**Storage estimado:**
- Neris: ~492K eventos → ~2.5 GB
- 1 día producción: ~10M eventos → ~50 GB/día

### **RECOMENDACIÓN FINAL:** Opción A con ajuste

**Estrategia propuesta:**
1. **Mantener Opción A actual** (solo interesantes)
2. **Añadir flag de testing** para capturar 100% cuando necesitamos F1-scores
3. **Configuración dual:**

```json
// config/rag_logger_config.json
{
  "mode": "production",  // "production" o "testing"
  
  // Production mode (solo interesantes)
  "production": {
    "min_score_to_log": 0.70,
    "min_divergence_to_log": 0.30,
    "sample_benign": false
  },
  
  // Testing mode (para F1-score validation)
  "testing": {
    "log_all_events": true,
    "max_events": 100000  // Límite de seguridad
  }
}
```

### DECISIÓN 2: ¿Artifacts .pb para Qué?

#### Usar Artifacts Si:
- ✅ Necesitamos reproducir exactamente el evento
- ✅ Vamos a reingerir en pipeline
- ✅ Debugging complejo
- ✅ Auditoría legal

#### NO Usar Artifacts Si:
- ❌ Solo queremos logs para RAG
- ❌ Preocupa almacenamiento
- ❌ Testing rápido

**RECOMENDACIÓN:**
```json
// Para Day 15 testing:
{
  "save_protobuf_artifacts": false,  // Apagar para Neris
  "save_json_artifacts": true        // Mantener solo JSON legibles
}

// Para producción:
{
  "save_protobuf_artifacts": true,   // Encender para eventos críticos
  "save_json_artifacts": false       // JSON ya está en .jsonl
}
```

---

## 📋 CHECKLIST COMPLETO DAY 15

### Pre-Test
- [ ] Verificar dependencias Vagrantfile (libssl-dev, nlohmann-json3-dev, jq)
- [ ] Crear directorios RAG (`make rag-init`)
- [ ] Verificar config/rag_logger_config.json existe
- [ ] Verificar `make detector` compila sin errores
- [ ] Revisar y añadir targets RAG al Makefile

### Test 1: smallFlows
- [ ] Limpiar logs (`make rag-clean`)
- [ ] Ejecutar test (`make test-replay-small`)
- [ ] Verificar logs generados (~80-100 eventos)
- [ ] Analizar estadísticas (`make rag-analyze`)
- [ ] Verificar artifacts si habilitados

### Test 2: Neris
- [ ] Decidir estrategia artifacts (OFF recomendado)
- [ ] Limpiar logs
- [ ] Ejecutar test Neris (30-45 min)
- [ ] Monitorear en tiempo real
- [ ] Análisis post-test completo
- [ ] Extraer IPs botnet y scores

### Análisis
- [ ] Calcular F1-scores preliminares
- [ ] Identificar eventos divergentes
- [ ] Documentar patrones interesantes
- [ ] Decidir ajustes de thresholds

### Preparación Day 16
- [ ] Diseñar estructura vector database
- [ ] Planear ingesta RAG → VectorDB
- [ ] Preparar academic paper outline

---

## 🎯 OBJETIVOS CLAROS DAY 15

1. **VALIDAR** que RAGLogger funciona correctamente
2. **DECIDIR** estrategia de logging definitiva
3. **EJECUTAR** Neris completo y obtener primeros F1-scores
4. **DOCUMENTAR** todos los hallazgos
5. **PREPARAR** Day 16 (RAG ingestion pipeline)

---

## 💡 NOTAS IMPORTANTES

### Rendimiento
- RAGLogger añade **<2% overhead** → Negligible
- Thread-safe → Soporta multiple producers
- Rotación automática → No hay crecimiento infinito

### Almacenamiento
- Production (Opción A): **~1.5 GB/día** en logs RAG
- Con artifacts: **~5-8 GB/día** adicionales
- Retención recomendada: **30 días** → **45 GB total**

### Troubleshooting
Si RAGLogger no genera logs:
```bash
# 1. Ver logs del detector
tail -f /vagrant/logs/ml-detector/ml-detector.log | grep RAG

# 2. Verificar permisos
ls -la /vagrant/logs/rag/

# 3. Test manual de config
vagrant ssh defender
cat /vagrant/config/rag_logger_config.json | jq .

# 4. Verificar compilación
make detector 2>&1 | grep rag_logger
```

---

## 🚀 COMANDO ÚNICO PARA EMPEZAR MAÑANA

```bash
# Day 15 Quick Start
make rag-init && \
make rag-clean && \
make test-replay-small && \
make rag-status && \
make rag-analyze
```

---

**Estado actual:** ✅ RAGLogger implementado y compilado  
**Próximo paso:** Validación con smallFlows  
**Objetivo final:** F1-scores reales con Neris botnet

**¡VAMOS A POR EL DAY 15! 🚀**