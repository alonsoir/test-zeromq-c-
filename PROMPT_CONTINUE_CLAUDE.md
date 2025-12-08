# 📋 PROMPT DE CONTINUIDAD - DAY 12 → DAY 13

## 🎯 CONTEXTO GENERAL

Soy Alonso, doctoral researcher en Universidad de Murcia trabajando en **ML Defender**, un sistema IDS/IPS autónomo con detección de ransomware y DDoS. Estoy en **Day 12** del desarrollo, implementando la arquitectura de **Dual-Score Validation** para validar modelos ML contra el dataset CTU-13 antes de escribir papers académicos.

**Filosofía:** "Via Appia Quality" - no publicar papers sin validación completa y científica honestidad.

---

## ✅ COMPLETADO HOY (Day 12 - Phase 0)

### **Objetivo alcanzado:**
Externalizar los 5 valores hardcoded del Fast Detector a JSON para permitir A/B testing de thresholds.

### **Archivos creados:**
- ✅ `sniffer/include/fast_detector_config.hpp` - Estructuras de configuración

### **Archivos modificados:**
1. ✅ `sniffer/include/config_types.h` - Agregada estructura `fast_detector` a `StrictSnifferConfig`
2. ✅ `sniffer/src/userspace/config_types.cpp` - Parsing JSON de `fast_detector`
3. ✅ `sniffer/include/ring_consumer.hpp` - Miembro `fast_detector_config_` + constructor modificado
4. ✅ `sniffer/src/userspace/ring_consumer.cpp` - Constructor + 5 hardcoded values reemplazados
5. ✅ `sniffer/src/userspace/main.cpp` - Extracción y paso de `FastDetectorConfig`

### **5 Valores externalizados:**
| Ubicación | Valor Original | Valor Nuevo |
|-----------|---------------|-------------|
| `send_fast_alert()` | `0.75` | `fast_detector_config_.ransomware.scores.alert` |
| `send_ransomware_features()` | `15` | `fast_detector_config_.ransomware.activation_thresholds.external_ips_30s` |
| `send_ransomware_features()` | `10` | `fast_detector_config_.ransomware.activation_thresholds.smb_diversity` |
| `send_ransomware_features()` | `0.95` | `fast_detector_config_.ransomware.scores.high_threat` |
| `send_ransomware_features()` | `0.70` | `fast_detector_config_.ransomware.scores.suspicious` |

### **Validación exitosa:**
```bash
# 3 stress tests ejecutados con CTU-13 Neris botnet dataset
sudo tcpreplay -i eth1 --mbps=10 --limit=100000 /vagrant/datasets/ctu13/botnet-capture-20110810-neris.pcap

✅ Compilación exitosa
✅ Configuración cargada desde JSON correctamente
✅ Fast Detector activándose con thresholds correctos
✅ Logging muestra thresholds: ExtIPs=348 (threshold=15), SMB=0 (threshold=10), Score=0.95
✅ Sistema estable: 492,674 eventos procesados sin crashes
✅ Pipeline completo: Sniffer → Fast Detector → ML Detector → Logs
```

### **Problema identificado (no crítico):**
```
⚠️ FlowManager saturado: max_flows=10000 → muchos drops
Quick fix: Aumentar a 50000 en sniffer.json después de Phase 2
```

---

## 🎯 PRÓXIMO PASO: DAY 13 - PHASE 2

### **Objetivo:**
Implementar **Dual-Score Architecture** en protobuf para preservar ambos scores (Fast Detector + ML Detector) sin sobrescribirse.

### **Problema actual:**
```cpp
// FLUJO ACTUAL (BROKEN):
Sniffer Fast Detector → score=0.95 → [ZMQ] → ML Detector → score=0.65 (OVERWRITES) → Firewall
```

### **Solución propuesta:**
```cpp
// FLUJO NUEVO (DUAL-SCORE):
Sniffer Fast Detector → fast_score=0.95, overall=0.95 → [ZMQ] 
  → ML Detector → ml_score=0.65, overall=max(0.95,0.65)=0.95 → Firewall
```

---

## 📐 ARQUITECTURA DE DECISIÓN (Acordada)

### **Regla: "Maximum Threat Wins + Precaución Extrema"**

```python
# Lógica de decisión
if (fast_score >= 0.85 OR ml_score >= 0.85):
    action = "BLOCK"
    rag_queue = True
    
elif (abs(fast_score - ml_score) > 0.30):  # Divergencia sospechosa
    action = "BLOCK"  # Precaución extrema
    rag_queue = True
    reason = "SCORE_DIVERGENCE"
    
elif (fast_score >= 0.70 AND ml_score >= 0.70):
    action = "BLOCK"
    
else:
    action = "MONITOR"
```

**Filosofía:** Si hay duda o divergencia, **BLOCK + enviar a RAG para investigación**.

---

## 📋 PLAN DE IMPLEMENTACIÓN PHASE 2

### **Paso 1: Modificar Protobuf (30 min)**

**Archivo:** `protobuf/network_security.proto`

**Agregar estos campos:**
```protobuf
message NetworkSecurityEvent {
    // Dual-Score Architecture (Day 13)
    double fast_detector_score = 28;           // Layer 1 heuristic (0.0-1.0)
    double ml_detector_score = 29;             // Layer 3 ML inference (0.0-1.0)
    
    DetectorSource authoritative_source = 30;  // ¿Quién decidió?
    bool fast_detector_triggered = 31;         // ¿Se activó?
    string fast_detector_reason = 32;          // Razón
    
    // overall_threat_score = 15 ya existe - ahora será max(fast, ml)
    
    DecisionMetadata decision_metadata = 33;   // Para RAG
}

enum DetectorSource {
    DETECTOR_SOURCE_UNKNOWN = 0;
    DETECTOR_SOURCE_FAST_ONLY = 1;
    DETECTOR_SOURCE_ML_ONLY = 2;
    DETECTOR_SOURCE_FAST_PRIORITY = 3;
    DETECTOR_SOURCE_ML_PRIORITY = 4;
    DETECTOR_SOURCE_CONSENSUS = 5;
}

message DecisionMetadata {
    double score_divergence = 1;
    string divergence_reason = 2;
    bool requires_rag_analysis = 3;
    string investigation_priority = 4;
}
```

**Recompilar:**
```bash
cd protobuf
protoc --cpp_out=. network_security.proto
cp network_security.pb.h ../sniffer/include/
cp network_security.pb.cc ../sniffer/src/
cp network_security.pb.h ../ml-detector/include/
cp network_security.pb.cc ../ml-detector/src/
```

---

### **Paso 2: Modificar Sniffer (45 min)**

**Archivo:** `sniffer/src/userspace/ring_consumer.cpp`

**Función `send_fast_alert()` (línea ~865):**
```cpp
// AGREGAR:
alert.set_fast_detector_score(fast_detector_config_.ransomware.scores.alert);
alert.set_fast_detector_triggered(true);
alert.set_fast_detector_reason("high_external_ips");
alert.set_authoritative_source(protobuf::DETECTOR_SOURCE_FAST_ONLY);

// MANTENER:
alert.set_overall_threat_score(fast_detector_config_.ransomware.scores.alert);
```

**Función `send_ransomware_features()` (línea ~960):**
```cpp
// AGREGAR:
event.set_fast_detector_score(
    high_threat ? fast_detector_config_.ransomware.scores.high_threat 
                : fast_detector_config_.ransomware.scores.suspicious
);
event.set_fast_detector_triggered(true);
event.set_fast_detector_reason(
    high_threat ? "external_ips_smb_high" : "external_ips_smb_medium"
);
event.set_authoritative_source(protobuf::DETECTOR_SOURCE_FAST_ONLY);

// MANTENER:
event.set_overall_threat_score(...);
```

---

### **Paso 3: Modificar ML Detector (60 min)**

**Archivo:** `ml-detector/src/zmq_handler.cpp`

**Función `process_event()` - AGREGAR ANTES de sobrescribir:**
```cpp
// READ Fast Detector score (NO SOBRESCRIBIR)
double fast_score = event.fast_detector_score();
bool fast_triggered = event.fast_detector_triggered();

// Calculate ML score
double ml_score = calculate_ml_score(event);
event.set_ml_detector_score(ml_score);

// DECISION LOGIC: Maximum Threat Wins
double final_score = std::max(fast_score, ml_score);
event.set_overall_threat_score(final_score);

// Determine authoritative source
if (fast_triggered && ml_score > 0.5) {
    event.set_authoritative_source(protobuf::DETECTOR_SOURCE_CONSENSUS);
} else if (fast_score > ml_score) {
    event.set_authoritative_source(protobuf::DETECTOR_SOURCE_FAST_PRIORITY);
} else {
    event.set_authoritative_source(protobuf::DETECTOR_SOURCE_ML_PRIORITY);
}

// Decision metadata
auto* metadata = event.mutable_decision_metadata();
metadata->set_score_divergence(std::abs(fast_score - ml_score));
metadata->set_requires_rag_analysis(
    std::abs(fast_score - ml_score) > 0.30 || final_score >= 0.85
);

// LOGGING para F1-score validation
logger->info("[SCORES] fast={:.4f}, ml={:.4f}, final={:.4f}, source={}",
             fast_score, ml_score, final_score, 
             event.authoritative_source());
```

---

### **Paso 4: Logging para F1-Score (crítico)**

**Objetivo:** Extraer scores para calcular Precision/Recall/F1 contra CTU-13 ground truth.

**Agregar en `ml-detector/src/zmq_handler.cpp`:**
```cpp
if (config.log_inference_scores) {
    logger->info("[F1-VALIDATION] "
                 "timestamp={}, "
                 "src_ip={}, dst_ip={}, "
                 "fast_score={:.4f}, "
                 "ml_l1={:.4f}, ml_ddos={:.4f}, ml_ransomware={:.4f}, "
                 "final_score={:.4f}, "
                 "ground_truth={}",  // De CTU-13 labels
                 event.event_timestamp(),
                 event.network_features().source_ip(),
                 event.network_features().destination_ip(),
                 fast_score, ml_l1, ml_ddos, ml_ransomware,
                 final_score,
                 get_ground_truth_label(event));  // Implementar lookup
}
```

---

### **Paso 5: Recompilar y validar (20 min)**

```bash
# Sniffer
cd /vagrant/sniffer/build
make clean && cmake .. && make -j4

# ML Detector
cd /vagrant/ml-detector/build
make clean && cmake .. && make -j4

# Test
sudo tcpreplay -i eth1 --mbps=10 --limit=10000 /vagrant/datasets/ctu13/botnet-capture-20110810-neris.pcap

# Verificar logs
grep "SCORES" ml-detector/logs/*.log | head -20
```

---

## 📊 EXPECTED OUTPUT (Day 13)

```
[SCORES] fast=0.95, ml=0.82, final=0.95, source=CONSENSUS
[SCORES] fast=0.70, ml=0.15, final=0.70, source=FAST_PRIORITY (⚠️ divergence=0.55)
[F1-VALIDATION] timestamp=1312992000, src_ip=147.32.84.165, dst_ip=213.246.53.125, 
                fast_score=0.95, ml_l1=0.82, ml_ddos=0.12, ml_ransomware=0.88, 
                final_score=0.95, ground_truth=MALICIOUS
```

---

## 🗂️ ESTRUCTURA DE ARCHIVOS

```
test-zeromq-docker/
├── protobuf/
│   └── network_security.proto          [MODIFICAR Day 13]
├── sniffer/
│   ├── include/
│   │   ├── fast_detector_config.hpp    [CREADO Day 12] ✅
│   │   ├── config_types.h              [MODIFICADO Day 12] ✅
│   │   └── ring_consumer.hpp           [MODIFICADO Day 12] ✅
│   └── src/userspace/
│       ├── config_types.cpp            [MODIFICADO Day 12] ✅
│       ├── ring_consumer.cpp           [MODIFICAR Day 13]
│       └── main.cpp                    [MODIFICADO Day 12] ✅
└── ml-detector/
    └── src/
        └── zmq_handler.cpp             [MODIFICAR Day 13]
```

---

## 🎯 RESUMEN EJECUTIVO PARA MAÑANA

**Estado:** Phase 0 completada ✅  
**Siguiente:** Phase 2 - Dual-Score Protobuf Architecture  
**Tiempo estimado:** 2.5 horas  
**Objetivo final:** Validar F1-scores contra CTU-13 para publicar papers con honestidad científica

**Comando para retomar:**
```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
vagrant ssh defender
cd /vagrant
```

---

**Descansa bien, Alonso. Mañana continuamos construyendo Via Appia Quality.** 🏛️✨