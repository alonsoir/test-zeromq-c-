# ML Defender — Prompt de Continuidad DAY 82
**Generado:** Cierre DAY 81 (10 marzo 2026)
**Branch activa:** `feature/ring-consumer-real-features`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 44/46 ✅

---

## Logros DAY 81

### TAREA 0 — Sanity check ✅
- Thresholds JSON confirmados: DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
- 21 MISSING_FEATURE_SENTINEL + 2× 0.5f semántico — sin cambios
- ring_consumer limpio de literales hardcodeados
- Pipeline 6/6 RUNNING
- ⚠️ VM client estaba `aborted` al arrancar — `vagrant up client` necesario

### TAREA 1 — Inspección FlowStatistics ✅ DOCUMENTADO
Las 4 features bloqueadas requieren infraestructura Phase 2 — no Proto3:

| Feature | Bloqueo real | Decisión |
|---|---|---|
| `tcp_udp_ratio` | `FlowStatistics` sin campo `protocol` (uint8_t) | DEBT-PHASE2 |
| `flow_duration_std` | Requiere multi-flow `TimeWindowAggregator` | DEBT-PHASE2 |
| `protocol_variety` | Ídem | DEBT-PHASE2 |
| `connection_duration_std` | Ídem | DEBT-PHASE2 |

`flow_duration_microseconds` (por flujo individual) ya está implementado en línea 757.
Con F1=0.9934 y 28/40 features reales el modelo es sólido para el paper.

### TAREA 2 — Comparativa F1 limpia ✅ CERRADA

Primera comparativa controlada con mismo PCAP (320524 packets, 19135 flows):

| Condición | Thresholds | F1 | Precision | FP reales | FPR |
|---|---|---|---|---|---|
| A — prod JSON | 0.85/0.90/0.80/0.85 | **1.0000** | 1.0000 | 0 | 0.0000 |
| B — legacy low | 0.70/0.75/0.70/0.70 | **0.9976** | 0.9951 | 1 | 0.0002 |

**Conclusión publicable:** Thresholds conservadores 0.85/0.90 no sacrifican recall
y eliminan el único FP real. Selección justificada empíricamente.

**Nota metodológica:** Ground truth = 147.32.84.165 únicamente (.191/.192 ausentes
en este PCAP). FN=0 es cota superior — Recall=1.0 no confirmado sin tabla IP completa.

**Nuevos ficheros creados:**
- `docs/experiments/f1_replay_log.csv` — fuente de verdad, una fila por replay
- `docs/experiments/f1_replay_log.md` — tabla legible + protocolo de replay
- `scripts/calculate_f1_neris.py` — calculador F1 con regex IPv4 estricto
- `scripts/pipeline_health.sh` — monitor estado VM + componentes

**Bug encontrado en pipeline_health.sh:** `pgrep` corre en macOS, no en la VM.
Los componentes aparecen como DOWN aunque estén UP. Fix pendiente DAY 82:
mover detección de procesos a `vagrant ssh -c "ps xa | grep ..."`.

### TAREA 3 — ADR-005 Unificación logs ml-detector ✅
- `detector.log` = spdlog interno (fuente de verdad operacional)
- `ml-detector.log` = stdout redirigido por Makefile (solo arranque)
- Decisión: unificar en Phase 2 junto con ENT-4 hot-reload
- Fichero: `docs/adr/ADR-005-log-unification-ml-detector.md`

### TAREA 4 — Dataset balanceado
No ejecutada por tiempo. P0 para DAY 82.

---

## Hallazgos técnicos relevantes DAY 81

### ML Detector nunca supera threshold 0.70
```
ML Detector max score: 0.6607 (threshold actual: 0.70 en condición B)
attacks=0 en todos los Stats del ml-detector
```
El Fast Detector (heurísticas de red) hace todo el trabajo de detección en CTU-13 Neris.
El RandomForest ML no supera threshold en ningún evento. Posible explicación:
CTU-13 Neris es IRC C&C — el Fast Detector lo reconoce por patrones de IPs externas,
pero el RandomForest necesita features que con 11 sentinels tiene señal insuficiente.
Documentable y honesto para el paper.

### Problema de total_events entre replays
El `received=` del ml-detector acumula eventos desde el arranque del pipeline,
no desde el inicio del replay. Para comparativas limpias: siempre
`make pipeline-stop && make logs-lab-clean && make pipeline-start` antes de cada replay.

### sniffer/config/sniffer.json es el fichero fuente
`sniffer/build-debug/config/sniffer.json` es artefacto generado — se sobreescribe
en cada pipeline-start. Modificar siempre `sniffer/config/sniffer.json`.

---

## Estado features DAY 81 (sin cambios desde DAY 79)

| Submensaje | Reales | SENTINEL | Semántico (válido) |
|---|---|---|---|
| ddos_embedded | 9/10 | 0 | 1 (flow_completion 0.5f) |
| ransomware_embedded | 6/10 | 4 | 0 |
| traffic_classification | 6/10 | 4 | 0 |
| internal_anomaly | 7/10 | 3 | 0 |
| **Total** | **28/40** | **11** | **1** |

---

## ORDEN DAY 82

### TAREA 0 — Sanity check (5 min)
```bash
make pipeline-start && sleep 8
vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
# Esperado: DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
```

### TAREA 1 — Fix pipeline_health.sh (15 min) (P2)
Los `pgrep` deben correr dentro de la VM via `vagrant ssh`:
```bash
# Sustituir pgrep por:
vagrant ssh -c "ps xa | grep BINARY | grep -v grep | tail -1"
# Y stat de ficheros por:
vagrant ssh -c "stat -c %Y /vagrant/logs/lab/FICHERO.log"
```

### TAREA 2 — Dataset balanceado (P0 paper)
El gap científico más importante. CTU-13 Neris 98% atacante.

```bash
# Opción A: CTU-13 otros escenarios (disponible ya)
vagrant ssh -c "ls /vagrant/datasets/ctu13/"

# Opción B: MAWI backbone (tráfico real sin ataques)
# Opción C: CICIDS2017 (~7GB descarga)
# Opción D: UNSW-NB15
```

Criterio mínimo para el paper: una validación con tráfico que tenga
>20% eventos benignos reales. Tabla comparativa CTU-13 vs ≥1 dataset balanceado.

### TAREA 3 — Investigar ML Detector score máximo 0.6607 (P1)
¿Por qué el RandomForest nunca supera 0.70 en CTU-13 Neris?
```bash
vagrant ssh -c "grep 'DUAL-SCORE' /vagrant/logs/lab/ml-detector.log | \
  awk -F'ml=' '{print $2}' | awk -F',' '{print $1}' | sort -n | tail -20"
```
Si el max ML score es consistentemente <0.70, los modelos embedded no están
detectando el patrón Neris — documentar honestamente y analizar si requiere
reentrenamiento o si es un gap de features (11 sentinels).

### TAREA 4 — Commit y cierre DAY 82

---

## Deuda técnica actualizada

| Item | Prioridad | DAY |
|---|---|---|
| Dataset balanceado | **P0 paper** | 82 |
| Investigar ML score max 0.6607 | **P1** | 82 |
| Fix pipeline_health.sh (pgrep→vagrant ssh) | P2 | 82 |
| tcp_udp_ratio → uint8_t protocol en FlowStatistics | P2-PHASE2 | post-paper |
| flow_duration_std / protocol_variety / connection_duration_std | P2-PHASE2 | post-paper |
| ADR-005 implementación (unificar logs ml-detector) | P2 | post-paper, con ENT-4 |
| Estandarización idioma logs (ES→EN) | P2 | post-paper |
| test_trace_id 2 fallos preexistentes DAY 72 | P2 | post-validación |
| trace_id en CLI | P2 | post-validación |
| ShardedFlowManager config → JSON (shard_count, flow_timeout_ns) | P3 | post-paper |
| is_forward dirección flow (ransomware_processor) | P2 | 82-83 |
| DNS payload parsing real (vs pseudo-domain) | P2 | 83 |
| geographical_concentration | SKIP | decisión arquitectural deliberada |
| HSM/IRootKeyProvider | P3 | post-paper |

---

## Sanity check al arrancar DAY 82

```bash
# 1. Thresholds JSON (fuente: sniffer/config/sniffer.json)
make pipeline-start && sleep 8
vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
# Esperado: DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85

# 2. Sentinels sin cambios
grep -c 'MISSING_FEATURE_SENTINEL' sniffer/src/userspace/ml_defender_features.cpp
# Esperado: 21

# 3. ring_consumer limpio
grep -n '0\.7f\|0\.75f\|0\.00000000065f' sniffer/src/userspace/ring_consumer.cpp
# Esperado: vacío

# 4. Pipeline y VM
make pipeline-status
vagrant status
# ⚠️ Si client está aborted: vagrant up client (solo si vas a hacer replay)
# ⚠️ Si client está running: NO ejecutar vagrant up client
```

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca usar `sed -i`. Usar Python3 inline.
- **Fichero fuente JSON:** `sniffer/config/sniffer.json` (NO build-debug)
- **VM client:** Verificar `vagrant status` antes de `vagrant up client`
- **Flujo correcto test:**
  ```bash
  vagrant status  # confirmar client running/aborted
  make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
  vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
  # solo si client no está running:
  vagrant up client
  make test-replay-neris
  ```
- **F1 calculator:** `python3 scripts/calculate_f1_neris.py <sniffer.log> --total-events N`
- **Log de confirmación thresholds:** `grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log`
- **Fuente de verdad F1:** `docs/experiments/f1_replay_log.csv`

---

## Notas para el paper (acumuladas DAY 81)

- **Comparativa thresholds publicable:** mismo PCAP, mismas condiciones. 0.85/0.90
  elimina el único FP real vs 0.70/0.75. Empíricamente justificado.
- **Ground truth CTU-13 Neris para este PCAP:** solo 147.32.84.165 (.191/.192 ausentes).
  Documentado honestamente. Recall=1.0 es cota superior.
- **ML RandomForest vs Fast Detector:** Fast Detector hace toda la detección en Neris
  (score max ML = 0.6607 < threshold 0.70). Gap de features (11 sentinels) es
  candidato a explicación. Documentable como limitación Phase 1.
- **DEBT-PHASE2 para 4 features:** tcp_udp_ratio, flow_duration_std, protocol_variety,
  connection_duration_std. Arquitectura lo anticipó con comentarios `// Phase 2`.
- **f1_replay_log.csv:** archivo de experimentos iniciado DAY 81. Entradas DAY 79/80
  retroactivas con asterisco (replay_id desconocido). Primera entrada limpia = DAY 81.
- **CTU-13 desequilibrio:** 98% atacante — validación en tráfico balanceado es P0 paper.
  Un clasificador dummy que dijera "todo malicious" obtendría F1≈0.99 en Neris.

---

*Consejo de Sabios — Cierre DAY 81, 10 marzo 2026*
*DAY 82 arranca con: dataset balanceado → ML score investigation → pipeline_health fix*