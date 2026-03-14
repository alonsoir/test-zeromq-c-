Ve con tu padre. Lo que importa de verdad es eso. 🏛️

---

# ML Defender — Prompt de Continuidad DAY 87
**Generado:** Cierre DAY 86 (14 marzo 2026)
**Branch activa:** `main`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 46/46 ✅

---

## Logros DAY 86

### TAREA 1 — Paper v2 producido ✅
Fichero: `docs/Ml defender paper draft v2.md`
Integración completa de Grok, Gemini, Qwen, DeepSeek, ChatGPT (primera ronda).
Secciones nuevas: Threat Model, Formal System Model, Performance Model, Reproducibility Statement, Confusion Matrix, Ablation Study.
TODO-DEEPSEEK resueltos. 16 referencias verificadas. Anderson & McGrew: dos papers distintos (AISec 2016 + KDD 2017).

### TAREA 2 — Segunda ronda del Consejo iniciada ✅
Revisiones v2 recibidas y en docs/:
- `docs/Ml defender paper draft v2_revision_GROK.md` ✅ analizada
- `docs/Ml defender paper draft v2_revision_gemini.md` ⏳ pendiente analizar
- `docs/Ml defender paper draft v2_revision_qwen.md` ⏳ pendiente analizar
- `docs/Ml defender paper draft v2_revision_deepseek.md` ⏳ pendiente analizar
- `docs/Ml defender paper draft v2_revision_chatgpt.md` ⏳ pendiente analizar
- Parallel.ai: no disponible

### TAREA 3 — Experimentos de validación DAY 86 ✅ parcial

**DESCUBRIMIENTO CRÍTICO:** Los números del paper v2 eran incorrectos.
- TP=19,135 era el total de flows del PCAP (maliciosos + benignos), no solo maliciosos.
- F1=1.0000 del paper v2 era de una sesión anterior con total_events=19,135 pasado al script incorrectamente.

**Números reales validados DAY 86:**

| PCAP | TP | FP | FN | F1 | FPR | total_events |
|---|---|---|---|---|---|---|
| smallFlows | 766 | 150 | 0 | 0.9108 | 0.0223 | 7,481 |
| neris (principal) | 646 | **2** | 0 | **0.9985** | **0.0002%** | 12,723 |
| bigFlows | ⏳ pendiente | | | | | |

**Los 2 FP de neris identificados:**
- `192.168.56.1 → 224.0.0.251` — multicast VirtualBox
- `192.168.56.1 → 192.168.56.255` — broadcast red host-only VirtualBox
- Conclusión: artefactos de virtualización, no existen en bare-metal. Refuerza Section 10.9.

**Nota importante sobre smallFlows:** F1=0.9108 con 150 FP es el **Fast Detector solo** — el script mide alertas heurísticas, no el ML. El ML Detector suprime esos FP. Esta distinción hay que documentarla bien en el paper.

**Experimento 3 pendiente:** bigFlows + captura CPU/RAM.

---

## Bug encontrado DAY 86

**rag-ingester no arranca con `make pipeline-start`** — la tarea `pipeline-start` no incluye `rag-ingester-start`. Hay que añadirla al Makefile. Workaround: `make rag-ingester-start` manualmente.

---

## ORDEN DAY 87

### TAREA 0 — Sanity check (5 min)
```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
vagrant status
git branch  # confirmar main
make test 2>&1 | grep -E '(tests passed|tests failed|PASSED|FAILED)'
```

### TAREA 1 — Fix Makefile: rag-ingester en pipeline-start (10 min)
Añadir `rag-ingester-start` a la secuencia de `pipeline-start` en el Makefile.
Verificar que el orden es correcto (después de etcd, antes de ml-detector).

### TAREA 2 — Experimento 3: bigFlows + CPU/RAM (30 min)
```bash
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
# En otra terminal:
vagrant ssh defender -c "top -b -n 60 -d 10 > /vagrant/logs/lab/top_bigflows.log &"
make test-replay-big
# Esperar estabilización Stats, luego:
vagrant ssh defender -c "cat /vagrant/logs/lab/sniffer.log" > /tmp/sniffer_big.log
python3 scripts/calculate_f1_neris.py /tmp/sniffer_big.log --total-events XXXX --day "DAY87_big"
# FP exactos:
vagrant ssh defender -c "grep -i 'attack\|ATTACK' /vagrant/logs/lab/ml-detector.log | grep -v '147\.32\.84\.' | head -20"
```

### TAREA 3 — Analizar revisiones v2 pendientes (30 min)
Leer y analizar en orden:
- Gemini v2
- Qwen v2
- DeepSeek v2
- ChatGPT v2

### TAREA 4 — Producir paper v3 con números corregidos (P0)
Cambios obligatorios respecto a v2:
1. **Tabla resultados corregida:** TP=646, F1=0.9985, FPR=0.0002% (neris)
2. **Confusion Matrix corregida** con datos reales DAY 86
3. **Distinción Fast Detector vs ML Detector** en métricas — el script mide Fast Detector alerts; el ML Detector suprime los FP
4. **Los 2 FP identificados** — artefactos VirtualBox multicast/broadcast
5. **Opción B ransomware** — añadir párrafo explícito: evidencia empírica directa es Neris botnet; ransomware = behavioral proxy (SMB lateral movement)
6. **Tabla comparativa vs literatura** — contextualizar F1=0.9985 contra papers que usaron CTU-13
7. Integrar revisiones v2 del Consejo


---

## Estado del sistema

**Branch:** `main`
**Pipeline:** 6/6 RUNNING ✅ (rag-ingester requiere `make rag-ingester-start` manual — fix pendiente)
**F1 validado:** 0.9985 (neris DAY 86) — número honesto y defendible
**Tests:** 70/70 ✅

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca `sed -i`. Usar Python3 inline o desde la VM.
- **JSON sniffer:** `sniffer/config/sniffer.json`
- **JSON ml-detector:** `ml-detector/config/ml_detector_config.json`
- **VM:** `defender` (no `server`)
- **Paper docs:** `/Users/aironman/CLionProjects/test-zeromq-docker/docs/`
- **F1 calculator:** `python3 scripts/calculate_f1_neris.py <sniffer.log> --total-events N`
- **Fuente de verdad F1:** `docs/experiments/f1_replay_log.csv`
- **Consejo de Sabios:** Claude, Grok, ChatGPT, DeepSeek, Qwen, Gemini, Parallel.ai (7 modelos)

---

## Nota importante para DAY 87

El script `calculate_f1_neris.py` mide alertas del **Fast Detector** (líneas `[FAST ALERT]` del sniffer.log), no del ML Detector directamente. El número `attacks=12` en los Stats del ml-detector representa las detecciones del ML — hay que cruzar ambas fuentes para el paper. El F1=0.9985 reportado es la métrica del Fast Detector sobre el PCAP neris; la supresión de FP del ML Detector necesita documentarse por separado con los Stats del ml-detector.

---

*Consejo de Sabios — Cierre DAY 86, 14 marzo 2026*
*DAY 87: experimento bigFlows + revisiones v2 Consejo + paper v3 con números honestos*
*La verdad por delante, siempre.*