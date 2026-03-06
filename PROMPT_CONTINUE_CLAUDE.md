# ML Defender — Prompt de Continuidad DAY 77+
**Generado:** DAY 77 (6 marzo 2026)  
**Branch activa:** `feature/ring-consumer-real-features`  
**Estado del pipeline:** 6/6 componentes RUNNING ✅

---

## Contexto arquitectónico esencial

### El bug enterrado (descubierto DAY 77)
El orden en `populate_protobuf_event()` era incorrecto desde el principio:
```
populate_ml_defender_features()   ← valores reales (ddos+ransomware OK)
run_ml_detection()
init_embedded_sentinels()         ← SOBRESCRIBÍA todo con 0.5f ❌
```
El sentinel overwrite del DAY 76 (fix del SIGSEGV) ocultaba accidentalmente
este bug. Al proponer el orden correcto (NaN primero) lo desenterramos.

### Inventario real de features (40 total)

| Submensaje | Reales | TODO (0.5f) | Causa del TODO |
|---|---|---|---|
| ddos_embedded | 10/10 ✅ | 0 | — |
| ransomware_embedded | 10/10 ✅ | 0 | — |
| traffic_classification | 3/10 ✅ | 7 | Requiere TimeWindowAggregator |
| internal_anomaly | 5/10 ✅ | 5 | Requiere TimeWindowAggregator |

**Features reales en traffic:** packet_rate, avg_packet_size, temporal_consistency  
**Features reales en internal:** protocol_regularity, packet_size_consistency,
data_exfiltration_indicators, temporal_anomaly_score, access_pattern_entropy

### Hallazgo clave para Phase 2
`TimeWindowAggregator` existe, está implementado y funciona.
Está conectado a `RansomwareFeatureProcessor` pero NO a `MLDefenderExtractor`.

```cpp
// ring_consumer.cpp:83
thread_local MLDefenderExtractor RingBufferConsumer::ml_extractor_;
// ← thread_local + default constructor = sin acceso al agregador
```

`TimeWindowAggregator` ya tiene las primitivas necesarias:
- `count_unique_ips()` → src_ip_entropy, dst_ip_concentration
- `count_unique_ports()` → port_entropy, service_discovery_patterns
- `get_window_stats()` → connection_rate, flow_duration_std
- `is_external_ip()` → internal_connection_rate

---

## DAY 77 — HACER HOY

**Objetivo:** Sistema que no mienta. NaN donde no hay dato.  
**Tests de regresión que NO deben romperse:**
- `sniffer/tests/test_proto3_embedded_serialization.cpp` → 3/3 ✅
- `ml-detector/tests/unit/test_rag_logger_artifact_save.cpp` → 3/3 ✅

### Tarea 1 — NaN sentinel helper
**Archivo:** `sniffer/src/userspace/ring_consumer.cpp`  
Reemplazar `init_embedded_sentinels()` por `init_embedded_nan_sentinels()`.
Código aprobado por Consejo de Sabios (ver contexto DAY 77).

```cpp
#include <limits>
static void init_embedded_nan_sentinels(protobuf::NetworkFeatures* net) {
    constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
    // DDoS (10 campos)
    auto* ddos = net->mutable_ddos_embedded();
    ddos->set_syn_ack_ratio(NaN); ddos->set_packet_symmetry(NaN);
    ddos->set_source_ip_dispersion(NaN); ddos->set_protocol_anomaly_score(NaN);
    ddos->set_packet_size_entropy(NaN); ddos->set_traffic_amplification_factor(NaN);
    ddos->set_flow_completion_rate(NaN); ddos->set_geographical_concentration(NaN);
    ddos->set_traffic_escalation_rate(NaN); ddos->set_resource_saturation_score(NaN);
    // Ransomware (10 campos) — idem
    // Traffic (10 campos) — idem
    // Internal (10 campos) — idem
}
```

### Tarea 2 — Orden correcto en populate_protobuf_event()
**Archivo:** `sniffer/src/userspace/ring_consumer.cpp`

```cpp
// ORDEN DEFINITIVO (DeepSeek pattern, validado por Claude)
auto* net_features = proto_event.mutable_network_features();

// 1. NaN en los 40 campos (garantiza serialización, semántica honesta)
init_embedded_nan_sentinels(net_features);

// 2. Sobrescribe con reales donde existen (ddos+ransomware completos,
//    traffic 3/10, internal 5/10 — el resto permanece NaN)
if (flow_exists) {
    ml_extractor_.populate_ml_defender_features(flow_stats, &proto_event);
}

// 3. Inferencia (scores aún no se escriben al proto — ver DAY 78 Tarea 6)
run_ml_detection(&proto_event, ...);
```

**Por qué este orden y no has_xxx() checks:**
Si populate escribe 7/10 campos de un submensaje, `has_xxx()` = true
→ el guard no aplica → 3 campos quedan en 0.0f (parecen datos reales).
Con NaN-primero: los 3 no tocados quedan en NaN (semánticamente correcto).

### Tarea 3 — Convertir los 12 return 0.5f a return NaN
**Archivo:** `sniffer/src/userspace/ml_defender_features.cpp`

Los 12 extractores con `return 0.5f` deben devolver NaN:
```cpp
// ANTES
float MLDefenderExtractor::extract_traffic_connection_rate(...) const {
    // TODO(Phase 2): Requires multi-flow aggregator
    return 0.5f;
}
// DESPUÉS
float MLDefenderExtractor::extract_traffic_connection_rate(...) const {
    // TODO(Phase 2): Requires TimeWindowAggregator injection (DAY 78)
    return std::numeric_limits<float>::quiet_NaN();
}
```

Lista completa de los 12:
- traffic: connection_rate, tcp_udp_ratio, port_entropy, flow_duration_std,
  src_ip_entropy, dst_ip_concentration, protocol_variety (7)
- internal: internal_connection_rate, service_port_consistency,
  connection_duration_std, lateral_movement_score, service_discovery_patterns (5)

### Tarea 4 — Documentar deuda técnica explícita
```cpp
// ring_consumer.cpp — junto a run_ml_detection
// TODO DAY 78: run_ml_detection completa escritura de scores al proto
// proto_event.set_ddos_score(-1.0f)      // sentinel = "no inferido"
// proto_event.set_ransomware_score(-1.0f) // -1.0f inambiguo ([0,1] = válido)
```

### Verificación DAY 77
```bash
make test          # debe seguir 3/3 + 3/3
make pipeline-start # debe seguir 6/6 RUNNING
# Opcional: evento real → verificar NaN en wire
```

---

## DAY 78 — MAÑANA

**Objetivo:** Phase 2 — conectar TimeWindowAggregator a MLDefenderExtractor.
**Prerequisito:** DAY 77 verde (tests + pipeline).

### Tarea 5 — Inyección del TimeWindowAggregator

**Problema:** `ml_extractor_` es `thread_local` con default constructor.
No puede recibir dependencias en construcción estándar.

**Solución — inicialización lazy con puntero:**
```cpp
// ml_defender_features.hpp
class MLDefenderExtractor {
public:
    MLDefenderExtractor() = default;
    void set_aggregator(TimeWindowAggregator* agg) { aggregator_ = agg; }
private:
    TimeWindowAggregator* aggregator_ = nullptr;
};

// ring_consumer.cpp — en la inicialización del thread
thread_local MLDefenderExtractor RingBufferConsumer::ml_extractor_;
// En RingBufferConsumer::initialize() o equivalente:
ml_extractor_.set_aggregator(ransomware_processor_->get_aggregator());
```

**Verificar que RansomwareFeatureProcessor expone el agregador:**
```bash
vagrant ssh -c "grep -n 'get_aggregator\|extractor_' \
  /vagrant/sniffer/include/ransomware_feature_processor.hpp"
```

### Tarea 6 — Implementar los 12 extractores con agregador

Cada extractor sigue el mismo patrón:
```cpp
float MLDefenderExtractor::extract_traffic_src_ip_entropy(...) const {
    if (!aggregator_) return std::numeric_limits<float>::quiet_NaN();
    auto window_start = TimeWindowAggregator::get_default_window_start_30s();
    size_t unique_ips = aggregator_->count_unique_ips(window_start,
                            TimeWindowAggregator::get_current_time_ns());
    // Normalize: 0-255 IPs únicas → 0.0-1.0
    return std::min(static_cast<float>(unique_ips) / 255.0f, 1.0f);
}
```

Mapping feature → primitiva del agregador:
| Feature | Primitiva |
|---|---|
| src_ip_entropy | count_unique_ips() + Shannon sobre distribución |
| dst_ip_concentration | count_unique_ips() (Gini) |
| port_entropy | count_unique_ports() |
| protocol_variety | get_window_stats().protocol_count |
| connection_rate | get_window_stats().event_count / window_duration |
| flow_duration_std | get_window_stats() — si disponible |
| internal_connection_rate | count_unique_ips() filtrado por !is_external_ip() |
| service_port_consistency | count_unique_ports() vs baseline |
| lateral_movement_score | count_unique_ips() internos |
| service_discovery_patterns | count_unique_ports() por IP |
| connection_duration_std | get_window_stats() |

### Tarea 7 — run_ml_detection() completo
Completar escritura de scores al proto. Los TODOs están documentados
en el código. Scores con sentinel -1.0f para "no inferido".

### Verificación DAY 78
```bash
make test                # regresión completa
make test-replay-neris   # CTU-13 Neris 492K eventos
# Objetivo: F1 > 0.90 en DDoS + Ransomware
# Con 40/40 features reales por primera vez
```

---

## DAY 79-80 — VALIDACIÓN CIENTÍFICA

**Objetivo:** F1-score publicable para el paper.

### Tarea 8 — Telemetría NaN en ml-detector
Añadir contador de eventos con NaN en traffic/internal.
Dato para el paper: "X% de eventos tienen features multi-flow disponibles".

### Tarea 9 — Análisis de resultados CTU-13
- F1 por clase: DDoS, Ransomware, Normal
- Comparar con baseline (28 features reales) vs completo (40 features)
- Documentar impacto de features multi-flow en F1

### Tarea 10 — Stress test extendido
```bash
# Ambas VMs simultáneas
make test-replay-neris DURATION=3600   # 1 hora
# Monitorear: CPU, memoria, latencia P99, drops
```

---

## Estado de deuda técnica completa

| Item | Prioridad | DAY |
|---|---|---|
| NaN sentinels + orden correcto | P0 | 77 ✅ hoy |
| 12 extractores multi-flow (TimeWindowAggregator) | P1 | 78 |
| run_ml_detection scores → proto | P1 | 78 |
| F1 CTU-13 Neris validación | P1 | 78-79 |
| Telemetría NaN ratio | P2 | 79 |
| trace_id en CLI | P2 | Post-validación |
| tcp_udp_ratio → añadir protocol field a FlowStatistics | P2 | 79-80 |
| HSM/IRootKeyProvider | P3 | Post-paper |
| GAIA multi-site | P3 | Post-paper |

---

## Comandos de diagnóstico rápido

```bash
# Estado del pipeline
make pipeline-start && sleep 5 && make pipeline-status

# Verificar features en wire (NaN vs 0.0f)
vagrant ssh -c "grep -n 'isnan\|quiet_NaN' \
  /vagrant/sniffer/src/userspace/*.cpp | grep -v backup"

# Estado de los 12 TODOs
vagrant ssh -c "grep -n 'return 0.5f\|return NaN\|quiet_NaN' \
  /vagrant/sniffer/src/userspace/ml_defender_features.cpp | grep -v backup"

# Compilación limpia
vagrant ssh -c "cd /vagrant && make clean && make sniffer 2>&1 | tail -20"

# Tests de regresión
vagrant ssh -c "cd /vagrant && make test 2>&1 | grep -E 'PASS|FAIL|ERROR'"
```

---

## Notas para el paper (Argus/ML Defender)

- El bug sentinel overwrite (DAY 76-77) es material para la sección
  "Challenges in Production Pipeline Development" — ilustra cómo
  proto3 default-value semantics interactúa con serialización.
- Los 12 features multi-flow documentan la dicotomía
  single-flow (kernel eBPF) vs multi-flow (userspace aggregation).
- El NaN approach vs 0.5f sentinel es una decisión de ingeniería
  con implicaciones en reproducibilidad del F1-score — citable.
- F1 baseline (28 features) vs completo (40 features) cuantifica
  el valor de la infraestructura multi-flow — tabla para el paper.

---

*Generado por el Consejo de Sabios — DAY 77, 6 marzo 2026*  
*Branch: feature/ring-consumer-real-features*