# ML Defender — Prompt de Continuidad DAY 84
**Generado:** Cierre DAY 83 (12 marzo 2026)
**Branch activa:** `main` ← MERGE COMPLETADO DAY 83
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 44/46 ✅

---

## Logros DAY 83

### TAREA 0 — Sanity check ✅
- Pipeline 6/6 RUNNING
- Thresholds JSON: DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
- MISSING_FEATURE_SENTINEL: 21 — sin cambios

### TAREA 1 — Ground truth bigFlows ✅ RESUELTO (P0 paper)
- bigFlows.pcap confirmado **benigno puro**: red 172.16.133.x sin binetflow disponible
- index.html es del escenario Botnet-91 (192.168.1.x) — distinto a Neris (147.32.x.x)
- Solo hay un binetflow: capture20110810.binetflow (Neris)
- **FPR ML = 2/40,467 = 0.0049%** — publicable como especificidad

### TAREA 2 — Attacks_detected bigFlows investigados ✅
- 2 eventos con L1_conf=68.97% exacto — mismo host, flows consecutivos (~0.2s)
- Veredicto: **FPs del ML** (tráfico benigno sin ground truth botnet)
- ML reduce FPs Fast Detector en factor **~15,500x** (2 vs 31,065)
- Log limpiado al arrancar pipeline DAY 83 — IPs no recuperables, documentado

### TAREA 3 — Fix pipeline_health.sh ✅
- pgrep (macOS) → vagrant ssh defender (VM)
- VM name: server → defender
- 6/6 componentes con PIDs correctos

### TAREA 4 — CSV Pipeline E2E ✅ 100% VALIDADO
- ml-detector: /vagrant/logs/ml-detector/events/ — CSVs diarios desde 2026-02-22
- firewall-acl-agent: /vagrant/logs/firewall_logs/firewall_blocks.csv
- rag-ingester: 71,217 parsed_ok, 0 hmac_fail
- CSV Pipeline sube de 80% → **100% ✅**

### TAREA 5 — F1 re-verificado + MERGE TO MAIN ✅
- F1=1.0000 reproducible confirmado
- **MERGE a main ejecutado DAY 83**
- Tag: `day83-merge`

---

## Hallazgos técnicos DAY 83

### Resultado publicable: FPR ML = 0.0049%
```
bigFlows.pcap (40,467 flows, tráfico benigno, red 172.16.133.x)
──────────────────────────────────────────────────────────────
  ML attacks_detected (conf ≥ 0.65):  2
  FPR ML:  2 / 40,467 = 0.0049%
  Fast Detector alertas:              31,065
  FPR Fast Detector:                  31,065 / 40,467 = 76.8%
  ML reduce FPs Fast Detector:        ~15,500x
```
Demuestra que la arquitectura dual-score justifica su existencia.

### CSV paths (fuentes de verdad)
- ml-detector CSV: `/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv`
- firewall CSV: `/vagrant/logs/firewall_logs/firewall_blocks.csv`
- rag-ingester lee ambos — NO el directorio /vagrant/logs/lab/

### pipeline_health.sh — comando correcto post-fix
```bash
vagrant ssh defender -c "ps xa | grep '$binary' | grep -v grep"
```

---

## Estado features (sin cambios desde DAY 79)

| Submensaje | Reales | SENTINEL | Semántico |
|---|---|---|---|
| ddos_embedded | 9/10 | 0 | 1 |
| ransomware_embedded | 6/10 | 4 | 0 |
| traffic_classification | 6/10 | 4 | 0 |
| internal_anomaly | 7/10 | 3 | 0 |
| **Total** | **28/40** | **11** | **1** |

---

## ORDEN DAY 84

### TAREA 0 — Sanity check (5 min)
```bash
vagrant status
git branch  # confirmar que estamos en main
make pipeline-start && sleep 8
vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
```

### TAREA 1 — Estructura paper arXiv (P0)
El paper es el siguiente objetivo. Estructura propuesta:

**Secciones:**
1. Abstract — problema, solución, F1=1.0000, FPR=0.0049%
2. Introduction — motivación (ransomware hospitalario), objetivos
3. Architecture — pipeline 6 componentes, dual-score design
4. Implementation — C++20, eBPF/XDP, embedded RandomForest
5. Evaluation — CTU-13 Neris F1=1.0000, bigFlows FPR=0.0049%
6. Limitations — 28/40 features, DEBT-FD-001, single-node
7. Future Work — PHASE2, Enterprise features
8. Conclusion
9. Acknowledgments — Consejo de Sabios (Claude, Grok, ChatGPT5, DeepSeek, Qwen)

Discutir con el Consejo: venue objetivo (arXiv primero, luego RAID/USENIX Security).

### TAREA 2 — Fix trace_id 2 fallos preexistentes DAY 72 (P1)
```bash
make test
# Identificar los 2 fallos trace_id
# Investigar root cause
```

### TAREA 3 — DNS payload parsing real (P2)
Actualmente usa pseudo-domain. Implementar parsing real.

---

## Deuda técnica actualizada

| Item | Prioridad | DAY |
|---|---|---|
| Paper arXiv — estructura y redacción | **P0** | 84+ |
| Fix trace_id 2 fallos DAY 72 | P1 | 84 |
| DNS payload parsing real | P2 | 84-85 |
| DEBT-FD-001: FastDetector Path A → JSON | P1-PHASE2 | post-paper |
| ml_detector.cpp vacío (0 bytes) | P3 | post-paper |
| ADR-005 implementación (unificar logs) | P2 | post-paper, con ENT-4 |
| tcp_udp_ratio → uint8_t protocol | P2-PHASE2 | post-paper |
| flow_duration_std / protocol_variety / connection_duration_std | P2-PHASE2 | post-paper |
| ShardedFlowManager config → JSON | P3 | post-paper |
| geographical_concentration | SKIP | decisión deliberada |
| ADR-007: Consenso AND firewall | P1-PHASE2 | post-paper |
| Ring Consumer 28/40 → 40/40 features | P2-PHASE2 | post-paper |

---

## Criterio de merge completado ✅ DAY 83

- ✅ ≥1 dataset balanceado validado
- ✅ ML score investigation documentada
- ✅ F1 comparativa limpia documentada
- ✅ Pipeline 6/6 RUNNING
- ✅ F1=1.0000 reproducible

**Branch main actualizada. Próximo objetivo: paper.**

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca `sed -i`. Usar Python3 inline.
- **JSON sniffer:** `sniffer/config/sniffer.json` (NO build-debug)
- **JSON ml-detector:** `ml-detector/config/ml_detector_config.json`
- **level1_attack:** 0.65 (ml_detector_config.json)
- **CSV ml-detector:** `/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv`
- **CSV firewall:** `/vagrant/logs/firewall_logs/firewall_blocks.csv`
- **VM:** `defender` (no `server`)
- **Flujo correcto test:**
```bash
  vagrant status
  make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
  vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
  vagrant up client  # solo si aborted
  make test-replay-neris
  python3 scripts/calculate_f1_neris.py /vagrant/logs/lab/sniffer.log --total-events 19135
```
- **F1 calculator:** `python3 scripts/calculate_f1_neris.py <sniffer.log> --total-events N`
- **Fuente de verdad F1:** `docs/experiments/f1_replay_log.csv`
- **ADRs:** ADR-001, ADR-002, ADR-005, ADR-006

---

*Consejo de Sabios — Cierre DAY 83, 12 marzo 2026*
*DAY 84 arranca con: paper structure + trace_id fixes + DNS parsing*
```
---
ML Defender (aRGus EDR) — open source, C++20, F1=1.0000 validated
Built with: Alonso Isidoro Roman + Consejo de Sabios (Claude, Grok, ChatGPT5, DeepSeek, Qwen, Gemini)
https://alonsoir-test-zeromq-c-.mintlify.app/introduction