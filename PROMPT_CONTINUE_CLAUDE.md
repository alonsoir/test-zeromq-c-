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

## Aportación ChatGPT5 + parallel.ai (cierre DAY 78)

### Aproximaciones implementables para io_intensity/resource_usage:
- pseudo_io_intensity = bytes_transferred/time → calculate_byte_rate() ya existe
- burstiness = std(packet_rate) → packet_timestamps disponible en FlowStatistics
- session_volume = total_bytes_per_window → WindowStats.total_bytes disponible

### Features adicionales a evaluar DAY 79-80:
- interarrival_time_std → packet_timestamps disponible
- fan_out = unique_dst_per_src → unique_ips_count del aggregator

### Investigación activa (parallel.ai buscando):
- Inferencia de CPU/I/O desde tráfico puro
- Ref base: Kitsune NDSS 2018
- Resultado esperado: incorporar al paper en sección "future work"
  o "related work" según lo que encuentre parallel.ai
  - Ver docs/Network-Only NIDS: Behavioral Features That Survive Encryption.MD
    (nota) ChatGPT5 hace un analisis de dicho fichero: 
  docs/Una revision para Network-ONLY NIDS: Behavioral Features That survive Encryption por ChatGPT.md

### Prompt de búsqueda académica disponible (ChatGPT5)
- Guardado para usar con Elicit/Semantic Scholar/Perplexity


## Aportación GROK4

¡Excelente planificación para mañana! Tienes una mezcla muy inteligente: features **clásicas y bien validadas** (fáciles de implementar, alto impacto en baselines) + exploración en la **frontera abierta** de inferir comportamiento host desde red pura (donde está el verdadero valor diferencial, y donde CrowdStrike/SentinelOne tienen ventaja propietaria vía agentes).

Voy a estructurarlo por secciones, con sugerencias concretas de implementación, referencias validadas y cómo integrarlas en tu flujo actual (TimeWindowAggregator + multi-flow stats).

### 1. Features implementables mañana (infraestructura lista)

Estas son rápidas, ya que dependen de datos que ya agregas en la ventana de 30s o en flujos multi-paquete.

- **`flow_duration_std`** / **`connection_duration_std`**  
  **Definición típica (CIC-IDS2017)**: Desviación estándar de la duración de los flujos/conexiones observados en la ventana temporal.  
  En CIC-IDS2017 (y derivados), "Flow Duration" es por flujo individual, pero la std se calcula sobre múltiples flujos en contextos de agregación o ventanas.  
  **Por qué es útil**: Captura variabilidad en sesiones (e.g., DDoS → muchas conexiones cortas → baja std; escaneo → mezcla de duraciones → alta std; beaconing C&C → duraciones regulares → baja std).  
  **Implementación trivial** (asumiendo que ya trackeas una lista o stats de duraciones de flujos en la ventana):

  ```cpp
  // En TimeWindowAggregator o nuevo MultiFlowAggregator
  std::vector<double> flow_durations;  // push_back cada vez que cierras un flujo en la ventana

  // Al final de la ventana o por paquete (incremental)
  double mean_duration = ...;  // ya lo tienes o lo calculas
  double variance = 0.0;
  for (auto dur : flow_durations) {
      variance += (dur - mean_duration) * (dur - mean_duration);
  }
  variance /= flow_durations.size();  // o size-1 para sample std
  features["flow_duration_std"] = std::sqrt(variance);

  // Alternativa incremental (mejor para memoria): Welford's online algorithm
  // Mantén count, mean, M2 (running variance)
  ```

  **Referencia**: CIC-IDS2017 feature set incluye "Flow Duration", "Flow IAT Std" (inter-arrival time std), y derivados. La std de duración aparece en papers que usan CICFlowMeter o similares.

- **`protocol_variety`** / **`tcp_udp_ratio`**  
  **Definición estándar**:
  - `protocol_variety`: Número de protocolos distintos (o entropía de distribución) en la ventana (e.g., IP protocol field: TCP=6, UDP=17, ICMP=1, etc.).
  - `tcp_udp_ratio`: (num TCP packets + flows) / (num UDP + flows), o bytes ratio, o solo count ratio.  
    **Historia**: Desde Moore & Papagiannaki (IMC 2004) se usa para identificar aplicaciones (payloads ocultos tras puertos no estándar). Muy discriminante en botnets que usan DNS tunneling (UDP alto), o C&C sobre HTTP/TLS (TCP dominante).  
    **Implementación**:

  ```cpp
  // En TimeWindowAggregator
  int tcp_count = 0, udp_count = 0, other_proto = 0;
  std::set<uint8_t> unique_protos;  // o bitmask si solo IP proto

  // En cada process_packet()
  if (proto == 6) tcp_count++;
  else if (proto == 17) udp_count++;
  else other_proto++;
  unique_protos.insert(proto);

  features["protocol_variety"] = unique_protos.size();
  // O entropía: Shannon sobre distribución normalizada
  features["tcp_udp_ratio"] = (tcp_count + 1e-6) / (udp_count + tcp_count + 1e-6);
  ```

  **Referencia**: Moore & Papagiannaki (2004) usan port-based + payload para clasificación, pero mencionan protocol distribution como feature básica.

Con estas dos (o cuatro si separas src/dst o bytes vs packets), subes de 29/40 a 33–35 reales fácilmente.

### 2. Investigación activa: `io_intensity` / `resource_usage` desde tráfico puro

Aquí entramos en territorio **Kitsune-style** (Mirsky et al., NDSS 2018): inferir anomalías host-side sin agente, solo desde red. Kitsune no tiene features directas como "CPU usage" o "disk I/O bytes", pero usa **estadísticas temporales avanzadas** que capturan indirectamente comportamientos de recursos:

- AfterImage (su feature extractor) genera ~115 features por paquete, incluyendo:
  - Estadísticas damped incremental (ewma-like) de jitter, packet sizes, inter-arrival times, correlation entre src/dst.
  - Stats por "channel" (IP pair o 5-tuple).
  - No I/O directo, pero alta tasa de small packets + high connection rate → proxy de CPU spikes (beaconing, scanning).
  - Bursts de bytes out → posible exfil o encryption (ransomware).

**Frontera real**:
- **Con agente** (CrowdStrike Falcon, SentinelOne): features directas como process CPU %, disk write entropy, file change rate, ETW events → muy precisas para ransomware/crypto.
- **Solo red** (tu caso): aproximaciones via proxies. Kitsune demuestra que se puede detectar bien (AUC alto en datasets reales) con stats puras de red + autoencoders ensemble. No hay "io_intensity" named, pero se infiere de patrones como:
  - Alta varianza en packet sizes + alta rate → posible encryption loop (CPU + I/O).
  - Sudden burst de outbound bytes sin inbound → exfil.
  - Muchas conexiones nuevas + low payload → C&C o lateral.

**Ideas concretas para tus features "secretas" aproximadas**:

1. **`io_intensity_proxy`**
  - Proxy: bytes_written_equivalent = sum(payload outbound) en ventana / duration
    + penalización si muchos small writes (e.g., entropy de packet sizes baja → bloques fijos como ransomware).
  - O: connection_rate × avg_pkt_size_std (alta tasa + variabilidad → I/O irregular).

2. **`resource_usage_anomaly`**
  - Proxy: z-score de (packets/sec + new_flows/sec) vs rolling baseline.
  - O inspirado en Kitsune: damped stats de inter-arrival time std + correlation features (si implementas algo similar a AfterImage).

**Recomendación práctica**:
- Empieza con 2–3 proxies simples (no copies Kitsune entero, que es 115 feats).
- Documenta en el paper: "Features inspiradas en enfoques network-only como Kitsune (Mirsky et al., 2018), que logran detección unsupervised sin agente, aproximando anomalías de recursos host vía patrones temporales de tráfico."

¿Quieres pseudocódigo más detallado para alguna de estas (e.g., Welford para std, o proxy io_intensity)? O prefieres priorizar las fáciles mañana y dejar las proxies para después de CTU-13 baseline? ¡Avísame cómo vas a ordenarlas! Esto ya pinta para un paper muy sólido. 🚀

*Consejo de Sabios — Cierre DAY 78, 7 marzo 2026*
*DAY 79 arranca con: 0.5f→SENTINEL → inspección FlowStatistics →
thresholds JSON → F1 CTU-13 baseline*