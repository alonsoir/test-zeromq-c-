# ML Defender — Prompt de Continuidad DAY 78
**Generado:** Cierre DAY 77 (6 marzo 2026) — versión final con consenso del Consejo
**Branch activa:** `feature/ring-consumer-real-features`
**Estado del pipeline:** 6/6 componentes RUNNING ✅

---

## ⚠️ DECISIÓN PENDIENTE CRÍTICA — Primera tarea del día

**Contexto:** En DAY 77 se introdujeron 12 `quiet_NaN()` como sentinel de
"feature no computada". Post-commit se descubrió que ml-detector trata NaN
como error:

```cpp
// ml-detector/src/feature_extractor.cpp:201-202
if (std::isnan(features[i])) {
    logger_->error("Feature[{}] {} is NaN", i, LEVEL1_FEATURE_NAMES[i]);
```

**Consenso del Consejo (DAY 77 segunda ronda):**

NaN es semánticamente correcto pero incompatible con el consumidor actual.
La solución correcta NO es 0.0f, -1.0f, ni -100.0f arbitrarios — es un
valor escalar **matemáticamente inalcanzable** por cualquier split de los
modelos ensemble embebidos (traducidos directamente a C++20).

**Para elegir ese valor científicamente, hay que inspeccionar los modelos
antes de decidir.** Esos son los primeros comandos del día.

---

## PRIMEROS COMANDOS DAY 78 — Inspección de modelos ensemble

```bash
# 1. Localizar los modelos ensemble embebidos
vagrant ssh -c "ls -la /vagrant/sniffer/models/ 2>/dev/null || \
  find /vagrant/sniffer -name '*.cpp' | xargs grep -l \
  'threshold\|tree\|forest\|ensemble' | grep -v backup"

# 2. Ver rangos de features en los splits de los árboles (C++20)
vagrant ssh -c "grep -rn 'feature\[0\]\|feature\[1\]\|threshold\|<= \|>= ' \
  /vagrant/sniffer/src/userspace/ml_defender_models*.cpp \
  /vagrant/sniffer/src/userspace/*detector*.cpp 2>/dev/null | head -40"

# 3. Ver cómo está definido predict() en los modelos
vagrant ssh -c "grep -rn 'class.*Detector\|predict\|DecisionTree\|RandomForest' \
  /vagrant/sniffer/include/*.hpp | grep -v backup | head -30"

# 4. Diagnóstico complementario: rango mínimo/máximo de thresholds en los árboles
vagrant ssh -c "grep -rn '<= \|>= \|< \|> ' \
  /vagrant/sniffer/src/userspace/*detector*.cpp \
  /vagrant/sniffer/src/userspace/*model*.cpp 2>/dev/null | \
  grep -oE '[-]?[0-9]+\.[0-9]+' | sort -n | head -5 && echo '---' && \
  grep -rn '<= \|>= \|< \|> ' \
  /vagrant/sniffer/src/userspace/*detector*.cpp \
  /vagrant/sniffer/src/userspace/*model*.cpp 2>/dev/null | \
  grep -oE '[-]?[0-9]+\.[0-9]+' | sort -n | tail -5"
```

**Con estos outputs el Consejo decide el valor sentinel correcto.**

Criterio de selección:
- Debe ser escalar (no NaN, no Inf)
- Fuera del rango [min_threshold, max_threshold] de todos los árboles
- Típicamente: si todos los thresholds son [0.0, 1.0] → sentinel = -9999.0f
- Si hay thresholds negativos: sentinel = valor por debajo del mínimo absoluto
- El modelo lo enviará siempre a la misma rama (left/right default) de forma
  determinista → comportamiento predecible, no aleatorio como con NaN

---

## Qué se hizo en DAY 77 — COMPLETADO ✅

**Archivos modificados:**
- `sniffer/src/userspace/ring_consumer.cpp` — +67/-58 líneas
- `sniffer/src/userspace/ml_defender_features.cpp` — +24/-24 líneas

### T1: NaN sentinel (ring_consumer.cpp líneas 21-72)
40 campos `0.5f` → `quiet_NaN()` en `init_embedded_sentinels()`.

### T2: Orden correcto en populate_protobuf_event()
```
ANTES (bug desde día 1):            AHORA (correcto):
  populate_ml_defender_features()     init_embedded_sentinels() NaN (738)
  run_ml_detection()                  populate_ml_defender_features() (757)
  init_embedded_sentinels() ❌        run_ml_detection() (776)
```

### T3: 12 extractores TODO → quiet_NaN
Lista de los 12:
- traffic: connection_rate, tcp_udp_ratio, port_entropy, flow_duration_std,
  src_ip_entropy, dst_ip_concentration, protocol_variety (7)
- internal: internal_connection_rate, service_port_consistency,
  connection_duration_std, lateral_movement_score, service_discovery_patterns (5)

### T4: Documentación deuda run_ml_detection()

**3 call sites de init_embedded_sentinels:**
- Línea 738: ruta principal ✅
- Línea 1172: send_fast_alert (sin FlowStatistics) ✅
- Línea 1257: ruta ransomware (sin FlowStatistics) ✅

**Tests al cierre DAY 77:**
- crypto-transport: 3/3 ✅
- etcd-client HMAC: 12/12 ✅
- ml-detector: 9/9 ✅
- test_trace_id: 44/46 — 2 fallos PREEXISTENTES DAY 72

---

## Estado real de features

| Submensaje | Reales | NaN→sentinel (Phase 2) |
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

## Plan DAY 78 — orden definitivo

### PASO 0 — Inspeccionar modelos → elegir sentinel (30 min)
Ejecutar los 4 comandos de arriba. El Consejo decide el valor.
Implementar: reemplazar `quiet_NaN()` → `MISSING_FEATURE_SENTINEL` en los 12
extractores Y en `init_embedded_sentinels()` (los 40 campos).

```cpp
// Constante a definir tras inspección — valor a determinar
constexpr float MISSING_FEATURE_SENTINEL = ???;  // DAY 78 Paso 0
```

Verificar que ml-detector ya no logguea errores:
```bash
make pipeline-start && sleep 10 && \
vagrant ssh -c "grep -i 'NaN\|isnan' /var/log/ml-defender.log 2>/dev/null || \
  journalctl -u ml-detector --since '1 min ago' | grep -i nan"
```

### PASO 1 — Diagnóstico TimeWindowAggregator (10 min)
```bash
vagrant ssh -c "grep -A 5 -n 'get_aggregator\|aggregator_' \
  /vagrant/sniffer/include/ransomware_feature_processor.hpp"

vagrant ssh -c "grep -n 'ransomware_processor_\|RansomwareFeatureProcessor' \
  /vagrant/sniffer/src/userspace/ring_consumer.cpp | grep -v backup | head -20"

vagrant ssh -c "grep -n 'count_unique\|get_window_stats\|get_default_window' \
  /vagrant/sniffer/include/time_window_aggregator.hpp"

vagrant ssh -c "grep -n 'protocol\|ip_proto' \
  /vagrant/sniffer/include/flow_manager.hpp | head -10"
```

### PASO 2 — Inyección TimeWindowAggregator en MLDefenderExtractor (20 min)
```cpp
// ml_defender_features.hpp — añadir
void set_aggregator(TimeWindowAggregator* agg) {
    if (!aggregator_) aggregator_ = agg;  // proteger contra reinit thread_local
}
bool has_aggregator() const { return aggregator_ != nullptr; }
private:
    TimeWindowAggregator* aggregator_ = nullptr;

// ring_consumer.cpp — inyección lazy en populate_protobuf_event o consume()
if (!ml_extractor_.has_aggregator() && ransomware_processor_) {
    ml_extractor_.set_aggregator(ransomware_processor_->get_aggregator());
}
```

### PASO 3 — Implementar 12 extractores (60 min)
Prioridad por facilidad/impacto:

| Feature | Primitiva | Dificultad |
|---|---|---|
| traffic: connection_rate | event_count / window_s | fácil |
| traffic: src_ip_entropy | count_unique_ips() + Shannon | media |
| traffic: dst_ip_concentration | count_unique_ips() Gini | media |
| traffic: port_entropy | count_unique_ports() + Shannon | media |
| traffic: protocol_variety | unique protocols en ventana | media |
| traffic: flow_duration_std | get_window_stats() | media |
| traffic: tcp_udp_ratio | protocol field (*) | difícil |
| internal: internal_connection_rate | !is_external_ip() count | media |
| internal: service_port_consistency | unique_ports vs baseline | media |
| internal: lateral_movement_score | unique internal IPs | media |
| internal: service_discovery_patterns | unique_ports por IP | difícil |
| internal: connection_duration_std | get_window_stats() | media |

(*) Si `protocol` no existe en FlowStatistics → dejar MISSING_FEATURE_SENTINEL
con TODO DAY 79. Publicar paper con 39/40. Honesto.

Patrón guard en cada extractor:
```cpp
float MLDefenderExtractor::extract_traffic_src_ip_entropy(...) const {
    if (!aggregator_) return MISSING_FEATURE_SENTINEL;
    // implementación real con TimeWindowAggregator
}
```

### PASO 4 — run_ml_detection() scores → proto (30 min)
```bash
# Verificar campos en proto
vagrant ssh -c "grep -n 'ddos_score\|ransomware_score\|final_classification' \
  /vagrant/proto/*.proto"
```

```cpp
// Sentinel -1.0f para scores (probabilidades en [0,1] → -1.0f inambiguo)
proto_event.set_ddos_score(-1.0f);
proto_event.set_ransomware_score(-1.0f);
if (ddos_pred.is_ddos(0.7f)) {
    proto_event.set_ddos_score(ddos_pred.ddos_prob);
    proto_event.set_final_classification("DDOS");
    proto_event.set_overall_threat_score(ddos_pred.ddos_prob);
}
```

### PASO 5 — Validación (30 min)
```bash
make test                # mismos resultados DAY 77
make test-replay-neris   # CTU-13 Neris 492K — objetivo F1 > 0.90
```

---

## Deuda técnica conocida

| Item | Prioridad | DAY |
|---|---|---|
| Elegir MISSING_FEATURE_SENTINEL tras inspección modelos | **P0 — primero** | 78 |
| TimeWindowAggregator → MLDefenderExtractor | P1 | 78 |
| 12 extractores multi-flow | P1 | 78 |
| run_ml_detection scores → proto | P1 | 78 |
| tcp_udp_ratio: protocol field en FlowStatistics | P2 | 79 |
| Thresholds desde JSON (TODO Phase1-Day4-CRITICAL) | P1 | 79 |
| Telemetría: ratio eventos sentinel vs reales | P2 | 79 |
| test_trace_id 2 fallos preexistentes | P2 | post-validación |
| trace_id en CLI | P2 | post-validación |
| HSM/IRootKeyProvider | P3 | post-paper |

---

## Notas para el paper

- "Semantic corruption via proto3 default-value elision + sentinel misuse +
  incorrect pipeline ordering → silent feature corruption" — sección
  lessons learned.
- Contrato explícito entre capas del pipeline sobre missing data:
  extractor puede generar sentinel, ML inference layer nunca recibe NaN.
- 70% features single-flow (eBPF) vs 30% multi-flow (TimeWindowAggregator).
- F1 baseline (28 features) vs completo (39-40) — tabla comparativa.
- El sentinel matemáticamente inalcanzable es citable como decisión de
  ingeniería rigurosa vs valores arbitrarios (-1.0f, 0.5f, etc.).

---

## Sanity check al arrancar

```bash
vagrant ssh -c "grep -c 'quiet_NaN' \
  /vagrant/sniffer/src/userspace/ml_defender_features.cpp"
# Debe ser 12 (los cambiaremos a MISSING_FEATURE_SENTINEL en Paso 0)

vagrant ssh -c "grep -n 'init_embedded_sentinels(' \
  /vagrant/sniffer/src/userspace/ring_consumer.cpp | grep -v backup"
# Debe mostrar líneas 738, 1172, 1257

vagrant ssh -c "grep -n 'init_embedded_sentinel\|populate_ml_defender\|run_ml_detection' \
  /vagrant/sniffer/src/userspace/ring_consumer.cpp | grep -v backup"
# Orden: sentinel(738) < populate(757) < run_ml(776)
```

---

*Consejo de Sabios — Cierre DAY 77 versión final, 6 marzo 2026*
*DAY 78 arranca con: inspección modelos ensemble → sentinel científico →
inyección agregador → 12 extractores → scores proto → F1 CTU-13*