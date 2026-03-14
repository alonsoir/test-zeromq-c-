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
Secciones nuevas: Threat Model, Formal System Model, Performance Model,
Reproducibility Statement, Confusion Matrix, Ablation Study.
TODO-DEEPSEEK resueltos. 16 referencias verificadas.
Anderson & McGrew: DOS papers distintos confirmados:
- AISec 2016: "Identifying Encrypted Malware Traffic with Contextual Flow Data"
- KDD 2017: "Machine Learning for Encrypted Malware Traffic Classification"

### TAREA 2 — Segunda ronda del Consejo iniciada ✅
Revisiones v2 recibidas en docs/:
- `docs/Ml defender paper draft v2_revision_GROK.md` ✅ analizada DAY 86
- `docs/Ml defender paper draft v2_revision_gemini.md` ⏳ pendiente analizar
- `docs/Ml defender paper draft v2_revision_qwen.md` ⏳ pendiente analizar
- `docs/Ml defender paper draft v2_revision_deepseek.md` ⏳ pendiente analizar
- `docs/Ml defender paper draft v2_revision_chatgpt.md` ⏳ pendiente analizar
- Parallel.ai: no disponible

**Hallazgo clave de Grok v2:** "F1=1.0000 es demasiado perfecto para no levantar
sospechas en un reviewer serio" → desencadenó la re-validación experimental.

### TAREA 3 — Re-validación experimental completa DAY 86 ✅

**CORRECCIONES CRÍTICAS al paper v2:**

Los números del paper v2 eran incorrectos. Causa: 19,135 era el total de flows
del PCAP (maliciosos + benignos), no solo maliciosos. El script calculate_f1_neris.py
mide alertas del Fast Detector, no del ML Detector directamente.

**Resultados validados DAY 86 — DEFINITIVOS:**

| PCAP | TP | FP | FN | TN | F1 | FPR | total_events |
|---|---|---|---|---|---|---|---|
| smallFlows | 766 | 150 | 0 | 6,565 | 0.9108 | 2.23% | 7,481 |
| neris (principal) | 646 | 2 | 0 | 12,075+ | **0.9985** | **0.0002%** | 12,723 |
| bigFlows (benigno) | 0 | 2,517 | 0 | 35,547 | N/A | **6.61%** | 38,064 |

**Notas críticas:**
- smallFlows F1=0.9108 = Fast Detector solo (150 FP heurísticos)
- neris F1=0.9985 = métrica principal del paper ✅ ESTABLE
- Los 2 FP de neris identificados: 192.168.56.1→224.0.0.251 (multicast VirtualBox)
  y 192.168.56.1→192.168.56.255 (broadcast host-only VirtualBox)
  → artefactos de virtualización, no existen en bare-metal
- bigFlows FPR=6.61% del Fast Detector (NO 76.8% como decía v2)
  → DEBT-FD-001 se cerró en DAY 80, los thresholds JSON son mucho más precisos

**ML Detector sobre bigFlows (puro benigno): 5 attacks detectados**
Confianzas: 68.97%, 68.97%, 60.04%, 59.04%, 52.52%
Todos por debajo de umbrales de producción (ransomware=0.85, DDoS=0.90)
→ El firewall NO los habría bloqueado. FP reales del ML = 0 en producción.

**Reducción FPs real (corregida):**
Fast Detector bigFlows: 2,517 FP
ML Detector bigFlows: 5 detecciones (0 bloqueos reales)
Reducción: ~500× (no 15,500× como decía v2)

**top_bigflows.log NO se generó** — el `&` backgroundeó en macOS, no en la VM.
Para medir CPU/RAM usar:
```bash
vagrant ssh defender -c "nohup top -b -n 30 -d 5 > /vagrant/logs/lab/top_bigflows.log 2>&1 &"
```

### Bug encontrado DAY 86
**rag-ingester no arranca con `make pipeline-start`** — workaround: `make rag-ingester-start`
Fix pendiente en Makefile.

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
Añadir `$(MAKE) rag-ingester-start` a la secuencia de `pipeline-start`,
después de `rag-start` y antes de `ml-detector-start`.

### TAREA 2 — Captura CPU/RAM (15 min)
```bash
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
vagrant ssh defender -c "nohup top -b -n 30 -d 5 > /vagrant/logs/lab/top_bigflows.log 2>&1 &"
make test-replay-neris
# Esperar estabilización, luego:
vagrant ssh defender -c "cat /vagrant/logs/lab/top_bigflows.log" > /tmp/top_bigflows.log
# Extraer CPU y RAM máximo/medio del pipeline durante el replay
```

### TAREA 3 — Analizar revisiones v2 pendientes (30 min)
Leer y analizar en orden:
1. Gemini v2
2. Qwen v2
3. DeepSeek v2
4. ChatGPT v2

### TAREA 4 — Producir paper v3 con números corregidos (P0)

**Cambios OBLIGATORIOS respecto a v2:**

1. **Tabla resultados principal corregida:**
    - TP=646 (no 19,135)
    - F1=0.9985 (no 1.0000)
    - FPR ML = 0.0002% en neris

2. **FPR Fast Detector corregido:**
    - bigFlows: 6.61% (no 76.8%)
    - Razón: DEBT-FD-001 cerrado DAY 80, thresholds JSON más precisos

3. **Reducción FPs corregida:**
    - ~500× (Fast 2,517 FP vs ML 5 detecciones/0 bloqueos reales en bigFlows)
    - No 15,500×

4. **Confusion Matrix corregida** con datos reales DAY 86

5. **Los 2 FP identificados** — artefactos VirtualBox, no existen en bare-metal

6. **ML Detector sobre bigFlows** — 5 detecciones por debajo de umbral de producción
   → FP reales en producción = 0

7. **Claim ransomware (Opción B de Grok):**
   Añadir párrafo explícito: "ransomware detection validated at behavioral proxy
   level (SMB lateral movement patterns); direct evaluation against modern
   ransomware captures is Future Work 11.1"

8. **Tabla comparativa vs literatura CTU-13** — contextualizar F1=0.9985

9. **Ablation corregido** con datos reales de los 3 experimentos

10. **CPU/RAM** — añadir si se captura en TAREA 2

### TAREA 5 — Commit DAY 87
```bash
git add docs/
git add Makefile
git add docs/experiments/f1_replay_log.csv
git commit -m "docs: paper arXiv v3 + experimentos validados DAY 86

CORRECCIONES CRÍTICAS (números del paper v2 eran incorrectos):
- F1: 1.0000 → 0.9985 (neris, validado DAY 86)
- TP: 19,135 → 646 flows maliciosos reales
- FPR Fast Detector: 76.8% → 6.61% (DEBT-FD-001 cerrado DAY 80)
- FP reduction: 15,500x → ~500x (Fast 2517 vs ML 0 bloqueos reales)
- 2 FP identificados: artefactos VirtualBox multicast/broadcast

Fix Makefile: rag-ingester-start en pipeline-start

Co-authored-by: Claude (Anthropic) <claude@anthropic.com>
Co-authored-by: ChatGPT (OpenAI) <chatgpt@openai.com>
Co-authored-by: DeepSeek <deepseek@deepseek.com>
Co-authored-by: Grok (xAI) <grok@xai.com>
Co-authored-by: Qwen (Alibaba) <qwen@alibaba.com>
Co-authored-by: Gemini (Google) <gemini@google.com>"

git push origin main
```

---

## Tabla resumen experimental DAY 86 — fuente de verdad

| PCAP | Flows tcpreplay | TP | FP Fast | FP ML | FN | F1 Fast | FPR Fast |
|---|---|---|---|---|---|---|---|
| smallFlows | 1,209 | 766 | 150 | — | 0 | 0.9108 | 2.23% |
| neris | 19,135 | 646 | 2 | 0* | 0 | **0.9985** | **0.0002%** |
| bigFlows | 40,467 | 0 (benigno) | 2,517 | 5† | 0 | N/A | 6.61% |

*Los 2 FP de neris son artefactos VirtualBox (multicast + broadcast)
†Los 5 del ML en bigFlows están por debajo del umbral de producción → 0 bloqueos reales

---

## Estado del sistema

**Branch:** `main`
**Pipeline:** 6/6 RUNNING ✅ (rag-ingester requiere `make rag-ingester-start` — fix DAY 87)
**F1 validado:** 0.9985 (neris DAY 86) — número honesto y estable
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
- **Consejo de Sabios:** Claude, Grok, ChatGPT, DeepSeek, Qwen, Gemini, Parallel.ai

---

## Nota metodológica importante para DAY 87

El script `calculate_f1_neris.py` mide alertas del **Fast Detector** (líneas
`[FAST ALERT]` del sniffer.log), no del ML Detector directamente.
Los `attacks=N` en los Stats del ml-detector son las detecciones reales del ML.
Para el paper necesitamos distinguir claramente ambas métricas:

- **Fast Detector F1** = lo que mide el script (sobre sniffer.log)
- **ML Detector attacks** = `attacks=12` en neris (Stats ml-detector)
- **ML Detector FP en bigFlows** = 5 detecciones, 0 bloqueos reales (bajo umbral)

El paper v3 debe presentar ambas métricas con esta distinción clara.

---

*Consejo de Sabios — Cierre DAY 86, 14 marzo 2026*
*La verdad por delante, siempre.*
*F1=0.9985 honesto > F1=1.0000 sospechoso*

Con eso tenemos: núcleos asignados, RAM total/usada, modelo CPU del host virtualizado, y consumo durante el replay. 
Todo lo que necesita la Section 8.1 del paper para que sea reproducible.

# Configuración Vagrant (núcleos + RAM asignados)
vagrant ssh defender -c "nproc && free -h && cat /proc/cpuinfo | grep 'model name' | head -1"

# Y durante el replay (en paralelo):
vagrant ssh defender -c "nohup top -b -n 30 -d 5 > /vagrant/logs/lab/top_neris.log 2>&1 &"
make test-replay-neris

## El plan de stress test para DAY 87:

# 10 Mbps — baseline (ya tenemos datos)
# 25 Mbps
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
vagrant ssh defender -c "nohup top -b -n 60 -d 5 > /vagrant/logs/lab/top_25mbps.log 2>&1 &"
vagrant ssh client -c "sudo tcpreplay -i eth1 --mbps=25 --stats=5 /vagrant/datasets/ctu13/botnet-capture-20110810-neris.pcap"

# 50 Mbps
# 100 Mbps
# 200 Mbps — probablemente aquí empieza a romperse en VirtualBox
# seguir hasta que Failed packets > 10% o F1 degrada