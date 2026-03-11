# ML Defender — Prompt de Continuidad DAY 83
**Generado:** Cierre DAY 82 (11 marzo 2026)
**Branch activa:** `feature/ring-consumer-real-features`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 44/46 ✅

---

## Logros DAY 82

### TAREA 0 — Sanity check ✅
- Thresholds JSON confirmados: DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
- 21 MISSING_FEATURE_SENTINEL — sin cambios
- ring_consumer limpio de literales hardcodeados
- Pipeline 6/6 RUNNING
- VM client estaba `aborted` al arrancar — `vagrant up client` necesario (comportamiento normal)

### TAREA 1 — Dataset balanceado: smallFlows.pcap ✅ EJECUTADO Y DOCUMENTADO

**Replay DAY82-001:** CTU-13 smallFlows.pcap (9.1MB, 14261 packets, 1209 flows)

| Métrica | Valor |
|---|---|
| ML attacks | 0 ✅ |
| ML score máximo | 0.3818 |
| Fast Detector alertas | 3,741 ⚠️ |
| Ground truth | IP botnet 147.32.84.165 AUSENTE en este PCAP |
| Conclusión | **Todos FPs** — Microsoft CDN, Google, Windows Update |

**Hallazgo crítico — DEBT-FD-001:**
Fast Detector Path A (`is_suspicious()`) usa constantes hardcodeadas en `fast_detector.hpp`,
ignorando completamente la configuración JSON. Viola "JSON is the law".

| Constante | Valor hardcodeado | JSON equivalente |
|---|---|---|
| `THRESHOLD_EXTERNAL_IPS` | **10** | `external_ips_30s: 15` (Path B) |
| `THRESHOLD_SMB_CONNS` | **3** | `smb_diversity: 10` (Path B) |
| `THRESHOLD_PORT_SCAN` | **10** | — sin equivalente |
| `THRESHOLD_RST_RATIO` | **0.2** | — sin equivalente |
| `WINDOW_NS` | **10s** | — sin equivalente |

Construido DAY 13, antes del sistema de configuración JSON. Fix PHASE2.
Documentado en `docs/adr/ADR-006-fast-detector-hardcoded-thresholds.md`.

### TAREA 2 — Dataset balanceado: bigFlows.pcap ✅ EJECUTADO Y DOCUMENTADO

**Replay DAY82-002:** CTU-13 bigFlows.pcap (352MB, 791615 packets, 40467 flows)

| Métrica | Valor |
|---|---|
| ML label=1 log (🚨 ATTACK) | **7** |
| ML attacks_detected (stats, conf≥0.65) | **2** |
| ML score máximo | **0.6897** |
| Fast Detector alertas | 31,065 |
| IPs top atacantes | 172.16.133.x (red privada) |
| Ground truth | ❌ 172.16.133.x no está en binetflow Neris |

**Hallazgo: Tres contadores con semántica distinta (no es bug, es arquitectura):**

| Contador | Condición | Valor |
|---|---|---|
| log `🚨 ATTACK` | `label_l1 == 1` (voto binario RF) | 7 |
| `stats_.attacks_detected` | `label_l1==1 AND conf>=level1_attack(0.65)` | 2 |
| `final_classification=MALICIOUS` | `final_score>=malicious_threshold` | pendiente |

`level1_attack: 0.65` está en `ml-detector/config/ml_detector_config.json` (no en sniffer.json).

**Nota:** `ml-detector/src/ml_detector.cpp` es un fichero vacío (0 bytes). La lógica real
está en `ml-detector/src/zmq_handler.cpp` (927 líneas). Anotado para limpieza.

### TAREA 3 — ADR-006 ✅ DOCUMENTADO
`docs/adr/ADR-006-fast-detector-hardcoded-thresholds.md` — diagnóstico completo
de arquitectura dual-threshold y plan de migración a JSON en PHASE2.

### TAREA 4 — f1_replay_log actualizado ✅
Dos nuevas entradas limpias: DAY82-001 (smallFlows) y DAY82-002 (bigFlows).

---

## Hallazgos técnicos DAY 82

### Arquitectura dual-threshold Fast Detector (bug histórico DAY 13)

Hay DOS paths de alerta completamente independientes en `ring_consumer.cpp`:

```
Path A (línea 538): fast_detector_.ingest(event) → is_suspicious()
  - Evalúa en CADA PAQUETE
  - Usa constantes compiladas: THRESHOLD_EXTERNAL_IPS=10, WINDOW_NS=10s
  - Imprime snapshot.external_ips_10s (ventana corta, informativo)
  - Genera [FAST ALERT] en sniffer.log

Path B (línea 1270): send_ransomware_features()
  - Evalúa sobre agregados temporales del RansomwareProcessor
  - Usa JSON: external_ips_30s=15, smb_diversity=10
  - Opera sobre new_external_ips_30s() (ventana larga)
  - No genera [FAST ALERT] directamente
```

Path A es el responsable de los FPs en tráfico Windows. El threshold real que dispara
es 10 (hardcodeado), no 15 (JSON). Fix PHASE2: pasar config al constructor de FastDetector.

### ML Level1 score sube con más volumen de tráfico
- Neris (19135 flows): max score 0.6607
- smallFlows (1209 flows): max score 0.3818
- bigFlows (40467 flows): max score **0.6897**

El RandomForest necesita diversidad de tráfico para activar patrones. Con más flows,
más probabilidad de encontrar combinaciones de features que superen el umbral.
7 eventos label=1 en bigFlows (2 con conf≥0.65). Progreso documentable para el paper.

### ground truth bigFlows desconocido
Las IPs 172.16.133.x son red privada distinta al escenario Neris. No aparecen en
`capture20110810.binetflow`. bigFlows es de una red diferente — necesita su propio
ground truth para calcular F1. Para el paper de especificidad necesitamos confirmar
si bigFlows tiene o no tráfico botnet.

---

## Estado features (sin cambios desde DAY 79)

| Submensaje | Reales | SENTINEL | Semántico (válido) |
|---|---|---|---|
| ddos_embedded | 9/10 | 0 | 1 (flow_completion 0.5f) |
| ransomware_embedded | 6/10 | 4 | 0 |
| traffic_classification | 6/10 | 4 | 0 |
| internal_anomaly | 7/10 | 3 | 0 |
| **Total** | **28/40** | **11** | **1** |

---

## ORDEN DAY 83

### TAREA 0 — Sanity check (5 min)
```bash
vagrant status  # confirmar client running o aborted
make pipeline-start && sleep 8
vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
# Esperado: DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
grep -c 'MISSING_FEATURE_SENTINEL' sniffer/src/userspace/ml_defender_features.cpp
# Esperado: 21
```

### TAREA 1 — Confirmar ground truth bigFlows (P0 paper) (20 min)
¿bigFlows.pcap tiene tráfico botnet o es benigno puro?

```bash
# Ver qué escenarios CTU-13 existen — bigFlows puede ser escenario distinto
vagrant ssh -c "cat /vagrant/datasets/ctu13/index.html | grep -i 'scenario\|botnet\|big\|small'"

# ¿Las IPs 172.16.133.x aparecen como From-Botnet en algún binetflow?
# Si hay más binetflows disponibles:
vagrant ssh -c "ls /vagrant/datasets/"

# Buscar documentación CTU-13 sobre bigFlows/smallFlows
# bigFlows y smallFlows son típicamente subsets del escenario completo
# pero de una red diferente a 147.32.84.x
```

Si bigFlows tiene botnet → los 7 label=1 pueden ser TPs → necesitamos ground truth.
Si bigFlows es benigno → FPR del ML también es 0 (attacks=0 con conf≥0.65 en tráfico no-Neris).

### TAREA 2 — Investigar ML score máximo (P1) (30 min)
Con bigFlows ya tenemos el dato: max=0.6897 con 40K flows. El umbral level1_attack=0.65
se supera en 2 casos. La pregunta es si son TPs o FPs.

```bash
# ¿Qué IPs generaron los 2 attacks_detected?
vagrant ssh -c "grep 'DUAL-SCORE' /vagrant/logs/lab/ml-detector.log | \
  grep -E 'ml=0\.(6[5-9]|[7-9])' | head -10"

# Extraer event_id de los 2 attacks y cruzar con sniffer.log para obtener IPs
```

### TAREA 3 — Fix pipeline_health.sh (P2) (15 min)
```bash
# Sustituir pgrep por vagrant ssh + ps xa
# En pipeline_health.sh, reemplazar:
#   pgrep -f BINARY
# Por:
#   vagrant ssh -c "ps xa | grep BINARY | grep -v grep | wc -l"
```

### TAREA 4 — CSV Pipeline E2E validation (P2)
Primer paso hacia merge a main.
```bash
make test-replay-neris
vagrant ssh -c "ls -lh /vagrant/logs/lab/*.csv 2>/dev/null || echo 'No CSV'"
```

### TAREA 5 — Commit y cierre DAY 83

---

## Deuda técnica actualizada

| Item | Prioridad | DAY |
|---|---|---|
| Confirmar ground truth bigFlows | **P0 paper** | 83 |
| Investigar IPs de los 2 attacks_detected bigFlows | **P1** | 83 |
| Fix pipeline_health.sh (pgrep→vagrant ssh) | P2 | 83 |
| DEBT-FD-001: FastDetector Path A hardcoded thresholds → JSON | P1-PHASE2 | post-paper |
| ml_detector.cpp vacío (0 bytes) — limpiar o documentar | P3 | post-paper |
| Normalizar esquema f1_replay_log.csv (columnas DAY82-001 distintas) | P2 | post-paper |
| tcp_udp_ratio → uint8_t protocol en FlowStatistics | P2-PHASE2 | post-paper |
| flow_duration_std / protocol_variety / connection_duration_std | P2-PHASE2 | post-paper |
| ADR-005 implementación (unificar logs ml-detector) | P2 | post-paper, con ENT-4 |
| test_trace_id 2 fallos preexistentes DAY 72 | P2 | post-validación |
| CSV Pipeline E2E con tráfico real | P2 | 83 |
| is_forward dirección flow (ransomware_processor) | P2 | 83-84 |
| DNS payload parsing real (vs pseudo-domain) | P2 | 84 |
| ShardedFlowManager config → JSON (shard_count, flow_timeout_ns) | P3 | post-paper |
| geographical_concentration | SKIP | decisión arquitectural deliberada |
| HSM/IRootKeyProvider | P3 | post-paper |
| ADR-007: Consenso AND para bloqueo firewall (max→AND lógico) | P1-PHASE2 | zmq_handler.cpp + JSON scoring config |

---

## Criterio de merge a main (sin cambios)

- [ ] Al menos 1 dataset balanceado validado (>20% benigno)
- [ ] ML score investigation documentada ← **parcialmente completada DAY 82**
- [ ] F1 comparativa limpia documentada ✅ DAY 81
- [ ] Pipeline 6/6 RUNNING post-merge
- [ ] F1 ≥ 0.99 reproducible con `make test-replay-neris`

---

## Notas para el paper (acumuladas DAY 82)

- **DEBT-FD-001 publicable:** Fast Detector Path A hardcodeado desde DAY 13 — deuda
  técnica honesta en sistema de 82 días. Fix PHASE2 identificado y documentado (ADR-006).
- **Arquitectura dual-score demuestra valor:** ML no confirma FPs del Fast Detector.
  El sistema no bloquea tráfico legítimo — solo alerta. Diseño defensivo correcto.
- **ML score progresa con volumen:** 0.3818 (1209 flows) → 0.6897 (40467 flows).
  Consistente con hipótesis de señal insuficiente con pocos flows.
- **Tres contadores con semántica distinta:** documentados y explicados. No es bug.
- **bigFlows ground truth desconocido:** necesita resolución antes de publicar
  métricas de especificidad con este dataset.
- **f1_replay_log.csv esquema inconsistente:** DAY82-001/002 tienen columnas distintas
  a DAY81. Normalizar en PHASE2.

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca usar `sed -i`. Usar Python3 inline.
- **Fichero fuente JSON sniffer:** `sniffer/config/sniffer.json` (NO build-debug)
- **Fichero fuente JSON ml-detector:** `ml-detector/config/ml_detector_config.json`
- **level1_attack threshold:** 0.65 (en ml_detector_config.json, NO en sniffer.json)
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
- **Fuente de verdad F1:** `docs/experiments/f1_replay_log.csv`
- **ADRs disponibles:** ADR-001 (cifrado), ADR-002 (provenance), ADR-005 (logs),
  ADR-006 (fast detector thresholds)

---

*Consejo de Sabios — Cierre DAY 82, 11 marzo 2026*
*DAY 83 arranca con: ground truth bigFlows → ML score investigation → CSV E2E → commit*