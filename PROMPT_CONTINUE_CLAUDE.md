# ML Defender — Prompt de Continuidad DAY 79
**Generado:** Cierre DAY 78 (7 marzo 2026)
**Branch activa:** `feature/ring-consumer-real-features`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 44/46 ✅

---

## Estado de features DAY 78

| Submensaje | Reales | SENTINEL (-9999) | 0.5f hardcoded |
|---|---|---|---|
| ddos_embedded | 10/10 | 0 | 0 |
| ransomware_embedded | 6/10 | 0 | 4 (*) |
| traffic_classification | 6/10 | 4 | 0 |
| internal_anomaly | 7/10 | 1 | 0 |
| **Total** | **29/40** | **5** | **4** |

(*) ransomware: io_intensity, resource_usage, file_operations,
process_anomaly — devuelven 0.5f (dentro del rango de splits →
puede activar splits incorrectos, peor que SENTINEL)

**Impacto de SENTINEL en clasificadores:**
- `-9999.0f <= threshold` siempre TRUE → siempre `left_child`
- Determinista pero biased — el ensemble vota con esa rama fija
- **Peor:** los 4 que devuelven `0.5f` están DENTRO del rango [0,5.1]
  y pueden activar splits incorrectamente de forma no predecible

---

## ORDEN DAY 79 — deuda técnica primero, luego F1

**Razón:** medir F1 con features mejoradas da más información que
el baseline actual. Cada feature real añadida es una mejora medible
en la tabla del paper.

### TAREA 0 — Inspección FlowStatistics (10 min)
Antes de implementar, verificar qué campos tiene FlowStatistics
que podamos aprovechar:
```bash
cat sniffer/include/flow_manager.hpp | grep -A 60 'struct FlowStatistics'
```

Features potencialmente atacables si FlowStatistics tiene el campo:
- `tcp_udp_ratio`: ¿hay conteo de paquetes TCP vs UDP?
- `protocol_variety`: ¿hay set de protocolos vistos?
- `flow_duration_std`: imposible con un solo flow — confirmar SENTINEL

### TAREA 1 — Corregir 4 features con 0.5f → MISSING_FEATURE_SENTINEL (15 min)

Las 4 features de ransomware que devuelven `0.5f` son peores que
SENTINEL porque están dentro del rango de splits. Cambiarlas:
```cpp
// ransomware: io_intensity, resource_usage, file_operations, process_anomaly
// ANTES: return 0.5f;
// DESPUÉS: return MISSING_FEATURE_SENTINEL;
```

También verificar si `ddos_geographical_concentration` devuelve `0.5f`:
```bash
grep -n '0\.5f' sniffer/src/userspace/ml_defender_features.cpp
```

### TAREA 2 — tcp_udp_ratio (si viable tras TAREA 0)

Si FlowStatistics tiene `protocol` o conteos TCP/UDP:
```cpp
float MLDefenderExtractor::extract_traffic_tcp_udp_ratio(
    const FlowStatistics& flow) const {
    // tcp_packets / (tcp_packets + udp_packets)
    // o via protocol field si existe
}
```

Si no existe el campo → dejar SENTINEL con TODO DAY 80. Honesto.

### TAREA 3 — thresholds desde JSON (Phase1-Day4-CRITICAL) (45 min)

El TODO más antiguo del proyecto. "JSON is the law."
```bash
# Ver si existe ModelConfig o similar
find ml-detector -name '*.hpp' | xargs grep -l 'threshold\|ModelConfig' | head -5

# Ver estructura JSON de los modelos embedded
ls ml-detector/models/production/
cat ml-detector/config/ml_detector_config.json | grep -A3 'thresholds'
```

Los thresholds hardcodeados en `ring_consumer.cpp`:
- DDoS: `0.7f` → debe venir de config
- Ransomware: `0.75f` → debe venir de config
- Traffic: `0.7f` → debe venir de config
- Internal: `0.00000000065f` → debe venir de config

Patrón a implementar: leer en `initialize()` de `RingBufferConsumer`
y almacenar en `struct MLThresholds` como miembro.

### TAREA 4 — test-replay-neris: F1 baseline (30 min)
```bash
make pipeline-start && sleep 15 && make test-replay-neris
```

**Objetivo:** F1 > 0.90 con 29+ features reales.
**Documentar:** F1 baseline (29 features) vs F1 completo (post-deuda).
**Para el paper:** tabla comparativa de ambos valores.

---

## Deuda técnica completa actualizada

| Item | Prioridad | DAY |
|---|---|---|
| 4 features 0.5f → MISSING_FEATURE_SENTINEL | **P0** | 79 |
| Inspección FlowStatistics → tcp_udp_ratio | P1 | 79 |
| Thresholds desde JSON (Phase1-Day4-CRITICAL) | P1 | 79 |
| test-replay-neris F1 CTU-13 Neris 492K | P1 | 79 |
| flow_duration_std / connection_duration_std | P2 | 79-80 |
| protocol_variety | P2 | 79-80 |
| is_forward dirección flow (ransomware_processor) | P2 | 79-80 |
| DNS payload parsing real (vs pseudo-domain) | P2 | 80 |
| Telemetría: ratio eventos sentinel vs reales | P2 | 80 |
| test_trace_id 2 fallos preexistentes DAY 72 | P2 | post-validación |
| trace_id en CLI | P2 | post-validación |
| tcp_udp_ratio si protocol en FlowStatistics | P2 | 79-80 |
| HSM/IRootKeyProvider | P3 | post-paper |
| io_intensity/resource_usage/file_operations/process_anomaly | P3 | post-paper (requiere eBPF tracepoints) |
| geographical_concentration | SKIP | decisión arquitectural deliberada |

---

## Notas para el paper (acumuladas)

- Sentinel matemáticamente inalcanzable: rango splits [0.0, 5.1],
  sentinel = -9999.0f → 3 órdenes de magnitud fuera del dominio.
  Routing determinista: siempre left_child. Citable como decisión
  de ingeniería rigurosa vs valores arbitrarios.
- 0.5f dentro del rango de splits es PEOR que sentinel — lección
  aprendida documentada en lessons learned.
- 70% features single-flow (eBPF) vs 30% multi-flow (TimeWindowAggregator).
- F1 baseline (29 features) vs completo (31-32) — tabla comparativa.
- "Semantic corruption via proto3 default-value elision + sentinel
  misuse + incorrect pipeline ordering → silent feature corruption"
- Contrato explícito entre capas: extractor puede generar sentinel,
  ML inference layer nunca recibe NaN.

---

## Sanity check al arrancar DAY 79
```bash
# Confirmar estado de features
grep -c 'MISSING_FEATURE_SENTINEL' \
  sniffer/src/userspace/ml_defender_features.cpp
# Esperado: ≥17 (12 originales + ddos_source_ip_dispersion ya no,
# más los que cambiaremos de 0.5f)

grep -c '0\.5f' sniffer/src/userspace/ml_defender_features.cpp
# Esperado: 4 (los que hay que cambiar en TAREA 1)

# Confirmar orden correcto en ring_consumer
grep -n 'init_embedded_sentinel\|populate_ml_defender\|run_ml_detection' \
  sniffer/src/userspace/ring_consumer.cpp | grep -v backup
# Orden: sentinel < populate < run_ml

# Pipeline arriba
make pipeline-status
```
## Referencias académicas para features pendientes

### Implementables mañana (infraestructura lista):
- `flow_duration_std` / `connection_duration_std`: algoritmo trivial,
  dato disponible en TimeWindowAggregator. Ref: CIC-IDS2017 dataset
  feature documentation, Universidad de New Brunswick.
- `protocol_variety` / `tcp_udp_ratio`: estándar desde 2004.
  Ref: Moore & Papagiannaki, "Toward the Accurate Identification of
  Network Applications", IMC 2004.

### Investigación activa (secreto industrial o frontera abierta):
- `io_intensity` / `resource_usage` desde tráfico de red puro:
  Ref: Mirsky et al., "Kitsune: An Ensemble of Autoencoders for
  Online Network Intrusion Detection", NDSS 2018.
  Línea: network-based host profiling sin agente endpoint.
  Sistemas comerciales (CrowdStrike, SentinelOne): agente endpoint,
  secreto industrial. Frontera conocida, documentada, no resuelta
  públicamente con la precisión requerida.

### Decisión DAY 79:
1. Implementar los 4 atacables → 33/40 features reales
2. Documentar los 2 restantes citando Kitsune + literatura EDR
3. Medir F1 con 33/40 y documentar delta vs 29/40

---

*Consejo de Sabios — Cierre DAY 78, 7 marzo 2026*
*DAY 79 arranca con: 0.5f→SENTINEL → inspección FlowStatistics →
thresholds JSON → F1 CTU-13 baseline*