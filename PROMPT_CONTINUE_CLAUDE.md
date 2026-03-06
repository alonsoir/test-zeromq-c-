# ML Defender — Prompt de Continuidad DAY 78
**Generado:** Cierre DAY 77 (6 marzo 2026)
**Branch activa:** `feature/ring-consumer-real-features`
**Estado del pipeline:** 6/6 componentes RUNNING ✅

---

## Qué se hizo en DAY 77 — COMPLETADO ✅

**Archivos modificados (git diff --stat HEAD):**
- `sniffer/src/userspace/ring_consumer.cpp` — +67/-58 líneas
- `sniffer/src/userspace/ml_defender_features.cpp` — +24/-24 líneas

**Cambios realizados:**

### T1: NaN sentinel (ring_consumer.cpp líneas 21-72)
Reemplazada `init_embedded_sentinels()` — 40 campos de `0.5f` → `quiet_NaN()`.
`0.5f` era peligroso: coincide con tráfico real (tcp_udp_ratio≈0.5).
NaN es non-default en proto3 → serialización garantizada + semántica honesta.

### T2: Orden correcto en populate_protobuf_event()
```
ANTES (bug desde el principio):
  populate_ml_defender_features()  ← reales
  run_ml_detection()
  init_embedded_sentinels()        ← SOBRESCRIBÍA todo ❌

AHORA (correcto):
  init_embedded_sentinels()        ← NaN primero (línea 738)
  populate_ml_defender_features()  ← sobrescribe con reales (línea 757)
  run_ml_detection()               ← inferencia (línea 776)
```

### T3: 12 extractores TODO → quiet_NaN (ml_defender_features.cpp)
Los 12 extractores con `return 0.5f` Phase 2 devuelven ahora `quiet_NaN()`.
Los otros `return 0.5f` (fallbacks en lógica real) NO fueron tocados.

Lista de los 12 convertidos:
- traffic: connection_rate, tcp_udp_ratio, port_entropy, flow_duration_std,
  src_ip_entropy, dst_ip_concentration, protocol_variety (7)
- internal: internal_connection_rate, service_port_consistency,
  connection_duration_std, lateral_movement_score, service_discovery_patterns (5)

### T4: Documentación deuda run_ml_detection()
Comentario DAY 77 añadido en cabecera de `run_ml_detection()` documentando
que los scores no se escriben al proto y el plan DAY 78.

**Resultado tests:**
- crypto-transport: 3/3 ✅
- etcd-client HMAC: 12/12 ✅
- ml-detector (test_rag_logger_artifact_save incluido): 9/9 ✅
- test_trace_id: 44/46 ✅ (2 fallos PREEXISTENTES del DAY 72, no relacionados)
- Compilación sniffer: limpia ✅

---

## Estado real de features tras DAY 77

| Submensaje | Reales | NaN (Phase 2) |
|---|---|---|
| ddos_embedded | 10/10 | 0 |
| ransomware_embedded | 10/10 | 0 |
| traffic_classification | 3/10 | 7 |
| internal_anomaly | 5/10 | 5 |
| **Total** | **28/40** | **12/40** |

**Features reales en traffic:** packet_rate, avg_packet_size, temporal_consistency
**Features reales en internal:** protocol_regularity, packet_size_consistency,
data_exfiltration_indicators, temporal_anomaly_score, access_pattern_entropy

---

## DAY 78 — HACER MAÑANA

**Objetivo:** Conectar TimeWindowAggregator a MLDefenderExtractor.
Desbloquea los 12 extractores restantes con datos reales.

### Contexto arquitectónico clave

`TimeWindowAggregator` existe, está implementado y funciona.
Conectado a `RansomwareFeatureProcessor` pero NO a `MLDefenderExtractor`.

```cpp
// ring_consumer.cpp:83 — problema de diseño
thread_local MLDefenderExtractor RingBufferConsumer::ml_extractor_;
// thread_local + default constructor = sin acceso al agregador
```

`MLDefenderExtractor` constructor:
```cpp
// ml_defender_features.hpp:36
MLDefenderExtractor() = default;  // sin dependencias
```

### Tarea 5 — Diagnóstico previo (primer comando del día)
```bash
# Ver qué expone RansomwareFeatureProcessor
vagrant ssh -c "grep -n 'get_aggregator\|extractor_\|TimeWindowAggregator' \
  /vagrant/sniffer/include/ransomware_feature_processor.hpp"

# Ver cómo se instancia RingBufferConsumer y dónde vive el RansomwareProcessor
vagrant ssh -c "grep -n 'ransomware_processor\|RansomwareFeatureProcessor' \
  /vagrant/sniffer/src/userspace/ring_consumer.cpp | grep -v backup | head -20"
```

### Tarea 5 — Inyección del TimeWindowAggregator

**Solución — inicialización lazy (compatible con thread_local):**
```cpp
// ml_defender_features.hpp — añadir:
class MLDefenderExtractor {
public:
    MLDefenderExtractor() = default;
    void set_aggregator(TimeWindowAggregator* agg) { aggregator_ = agg; }
    bool has_aggregator() const { return aggregator_ != nullptr; }
private:
    TimeWindowAggregator* aggregator_ = nullptr;
};

// ring_consumer.cpp — en inicialización del thread o constructor:
ml_extractor_.set_aggregator(ransomware_processor_->get_aggregator());
```

**Patrón guard en cada extractor:**
```cpp
float MLDefenderExtractor::extract_traffic_src_ip_entropy(...) const {
    if (!aggregator_) return std::numeric_limits<float>::quiet_NaN();
    // ... implementación real
}
```

### Tarea 6 — Implementar los 12 extractores

Mapping feature → primitiva de TimeWindowAggregator:

| Feature | Primitiva disponible |
|---|---|
| traffic: src_ip_entropy | count_unique_ips() + Shannon |
| traffic: dst_ip_concentration | count_unique_ips() (Gini) |
| traffic: port_entropy | count_unique_ports() |
| traffic: protocol_variety | get_window_stats().protocol_count |
| traffic: connection_rate | get_window_stats().event_count / window_s |
| traffic: flow_duration_std | get_window_stats() si disponible |
| traffic: tcp_udp_ratio | necesita protocol field en FlowStatistics (*) |
| internal: internal_connection_rate | count_unique_ips() filtrado !is_external_ip() |
| internal: service_port_consistency | count_unique_ports() vs baseline |
| internal: lateral_movement_score | count_unique_ips() internos |
| internal: service_discovery_patterns | count_unique_ports() por IP |
| internal: connection_duration_std | get_window_stats() si disponible |

(*) tcp_udp_ratio puede necesitar añadir campo `protocol` a FlowStatistics.
Verificar primero si ya existe: `grep -n 'protocol' /vagrant/sniffer/include/flow_manager.hpp`

**Verificar primitivas disponibles antes de implementar:**
```bash
vagrant ssh -c "grep -n 'count_unique\|get_window_stats\|get_default_window' \
  /vagrant/sniffer/include/time_window_aggregator.hpp"
```

### Tarea 7 — run_ml_detection() scores → proto

Los TODOs ya están documentados en el código (Phase1-Day4).
Implementar al menos la escritura básica de scores:

```cpp
// Sentinel -1.0f = "no inferido" (inambiguo: probabilidades van [0,1])
proto_event.set_ddos_score(-1.0f);          // default
proto_event.set_ransomware_score(-1.0f);    // default

if (ddos_pred.is_ddos(0.7f)) {
    proto_event.set_ddos_score(ddos_pred.ddos_prob);
    proto_event.set_final_classification("DDOS");
    proto_event.set_overall_threat_score(ddos_pred.ddos_prob);
}
if (ransomware_pred.is_ransomware(0.75f)) {
    proto_event.set_ransomware_score(ransomware_pred.ransomware_prob);
    proto_event.set_final_classification("RANSOMWARE");
}
```

Verificar que los campos existen en el proto:
```bash
vagrant ssh -c "grep -n 'ddos_score\|ransomware_score\|final_classification' \
  /vagrant/proto/*.proto"
```

### Verificación DAY 78
```bash
make test                # regresión completa — mismos resultados que DAY 77
make test-replay-neris   # CTU-13 Neris 492K eventos
# Objetivo: F1 > 0.90 en DDoS + Ransomware
```

---

## Deuda técnica conocida (no tocar en DAY 78)

| Item | Prioridad | DAY |
|---|---|---|
| test_trace_id 2 fallos (fallback_applied edge case) | P2 | Post-validación |
| tcp_udp_ratio: añadir protocol field a FlowStatistics | P2 | 79 |
| Thresholds desde JSON (TODO Phase1-Day4-CRITICAL) | P1 | 79 |
| Telemetría: ratio eventos con NaN vs reales | P2 | 79 |
| trace_id en CLI | P2 | Post-validación |
| HSM/IRootKeyProvider | P3 | Post-paper |

---

## Comandos de diagnóstico rápido

```bash
# Verificar estado NaN vs 0.5f
vagrant ssh -c "grep -c 'quiet_NaN' /vagrant/sniffer/src/userspace/ml_defender_features.cpp"
# Debe ser 12

vagrant ssh -c "grep -c 'quiet_NaN' /vagrant/sniffer/src/userspace/ring_consumer.cpp"
# Debe ser 1 (en la función init_embedded_sentinels)

# Verificar orden correcto en populate_protobuf_event
vagrant ssh -c "grep -n 'init_embedded_sentinel\|populate_ml_defender\|run_ml_detection' \
  /vagrant/sniffer/src/userspace/ring_consumer.cpp | grep -v backup"
# Debe mostrar: sentinel(738) < populate(757) < run_ml(776)

# Tests de regresión
vagrant ssh -c "cd /vagrant && make test 2>&1 | grep -E '100%|FAILED|passed'"

# Pipeline
make pipeline-start && sleep 5 && make pipeline-status
```

---

## Notas para el paper

- El bug sentinel overwrite (DAY 76-77) ilustra cómo proto3 default-value
  semantics interactúa inesperadamente con la serialización en pipelines reales.
- La dicotomía single-flow (eBPF kernel) vs multi-flow (userspace aggregation)
  es material para la sección de arquitectura — 28/40 features son single-flow.
- F1 baseline (28 features) vs completo (40 features) cuantificará el valor
  de la infraestructura multi-flow — tabla comparativa para el paper.
- El NaN approach vs 0.5f sentinel es una decisión de ingeniería reproducible
  y citable: afecta directamente la distribución de features vista por el modelo.

---

*Consejo de Sabios — Cierre DAY 77, 6 marzo 2026*
*Branch: feature/ring-consumer-real-features*
*Próximo: DAY 78 — TimeWindowAggregator injection + 12 extractores + scores proto*