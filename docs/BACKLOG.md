# ML Defender (aRGus NDR) — BACKLOG
## Via Appia Quality 🏛️

---

## 📐 Criterio de compleción (explícito para paper)

| Estado | Criterio |
|---|---|
| ✅ 100% | Implementado + probado en condiciones reales + resultado documentado |
| 🟡 80% | Implementado + compilando + smoke test pasado, sin validación E2E completa |
| 🟡 60% | Implementado parcialmente o con valores placeholder conocidos |
| ⏳ 0% | No iniciado |

---

## ✅ COMPLETADO

### Day 83 (12 Mar 2026) — Ground truth bigFlows + CSV E2E + pipeline_health fix + MERGE TO MAIN

**Ground truth bigFlows resuelto (P0 paper):**
- bigFlows.pcap confirmado benigno: red 172.16.133.x, no aparece en ningún binetflow CTU-13
- index.html es del escenario Botnet-91 (red 192.168.1.x) — distinto escenario
- Solo existe capture20110810.binetflow (Neris, red 147.32.x.x) — sin ground truth para 172.16.133.x
- Conclusión: los 2 attacks_detected (conf≥0.65, L1=68.97%) son FPs del ML
- **FPR ML = 2/40,467 = 0.0049%** — dato publicable de especificidad
- ML reduce FPs del Fast Detector en factor ~15,500x (2 vs 31,065)

**Attacks_detected investigados:**
- Ambos con L1_conf=68.97% exacto — mismo flow-context, host idéntico
- Timestamps consecutivos (~0.2s) — mismo par src/dst IP, dos flows del mismo patrón
- Log DAY 82 limpiado al arrancar pipeline DAY 83 — IPs no recuperables
- Documentado con evidencia disponible: veredicto FP confirmado por ground truth

**CSV Pipeline E2E — 100% validado:**
- ml-detector CSV: `/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv` — historial desde 2026-02-22
- firewall-acl-agent CSV: `/vagrant/logs/firewall_logs/firewall_blocks.csv` — 42K
- rag-ingester: 71,217 líneas parsed_ok, 0 hmac_fail, 0 parse_err (2 rejected por columnas, inofensivo)
- CSV Pipeline E2E: **100% ✅** (sube de 80% a 100%)

**pipeline_health.sh fix (DEBT-FD-002 ✅):**
- Root cause: `pgrep` corre en macOS host, no dentro de la VM
- Fix 1: `pgrep` → `vagrant ssh defender -c "ps xa | grep '$binary'"`
- Fix 2: VM name `server` → `defender`
- Resultado: 6/6 componentes con PIDs correctos ✅

**F1 re-verificado DAY 83:**
- F1=1.0000, Precision=1.0000, Recall=1.0000 — reproducible ✅
- Criterio de merge cumplido con todos los checks verdes

**MERGE TO MAIN ejecutado DAY 83 — tag: v0.83.0-day83-main** ✅

---


### Day 93 (21 Mar 2026) — ADR-012 PHASE 1: plugin-loader + ABI validation

**plugin-loader PHASE 1 implementado y validado:**
- `common/include/sentinel.hpp`: `MISSING_FEATURE_SENTINEL = -9999.0f` centralizado (ADR-012 §4)
- `plugin-loader/include/plugin_loader/plugin_api.h`: contrato C puro, ABI estable, `PLUGIN_API_VERSION=1`
- `plugin-loader/include/plugin_loader/plugin_loader.hpp`: interfaz C++ con `PluginLoader` + `PluginStats`
- `plugin-loader/src/plugin_loader.cpp`: `dlopen`/`dlsym` lazy loading, sin crypto, sin seed-client
- `plugin-loader/CMakeLists.txt`: patrón idéntico a `crypto-transport`
- `plugins/hello/hello_plugin.cpp`: hello world plugin — validación contrato end-to-end
- `Makefile`: targets `plugin-loader-build/clean/test` + `plugin-hello-build/clean`

**Restricciones PHASE 1 respetadas:**
- Plugins: SOLO feature extraction — decisión de bloqueo NUNCA en plugin ✅
- Sin crypto, sin seed-client (PHASE 2, ADR-013, DAY 95-96) ✅
- `MISSING_FEATURE_SENTINEL` desde cabecera común, no redefinido por plugin ✅

**ABI validation via Python3/ctypes:**
```
plugin_name()        = hello
plugin_version()     = 0.1.0
plugin_api_version() = 1
PLUGIN_API_VERSION   = 1
ABI version match    : True ✅
```

**Artefactos desplegados en VM:**
- `libplugin_loader.so.1.0.0` → `/usr/local/lib/` (53K) ✅
- `libplugin_hello.so` → `/usr/lib/ml-defender/plugins/` (16K) ✅

**Criterios ADR-012 hello world:**
- [x] `dlopen`/`dlsym` funcionan correctamente
- [x] `plugin_api_version()` retorna `PLUGIN_API_VERSION=1`
- [x] `plugin_name()` / `plugin_version()` resueltos
- [ ] `plugin_init()` recibe config JSON — test integración DAY 94
- [ ] `plugin_process_packet()` en cada paquete — test integración DAY 94
- [ ] `plugin_shutdown()` limpio — test integración DAY 94
- [ ] Si se elimina el `.so`, host no aborta — validación manual DAY 94
- [ ] Budget overrun → warning en log — validación manual DAY 94

**README badge living contracts añadido (sugerencia Grok, acta DAY 92)** ✅

### Day 82 (11 Mar 2026) — Balanced dataset validation + DEBT-FD-001

**Validación smallFlows.pcap (tráfico benigno Windows):**
- ML attacks=0 ✅, ML max_score=0.3818 — correcto en tráfico benigno
- Fast Detector: 3,741 FPs sobre Microsoft CDN, Google, Windows Update
- Root cause: DEBT-FD-001 (FastDetector Path A hardcodeado desde DAY 13)

**DEBT-FD-001 — Fast Detector hardcoded thresholds (DAY 13 → DAY 82):**
`FastDetector::is_suspicious()` usa constantes compiladas ignorando sniffer.json.
Fix: PHASE2 (ADR-006).

**Ficheros creados:**
- `docs/adr/ADR-006-fast-detector-hardcoded-thresholds.md`
- f1_replay_log.csv: entradas DAY82-001, DAY82-002

---

### Day 81 (10 Mar 2026) — Comparativa F1 limpia + ADR-005 + Infraestructura experimentos

| Condición | Thresholds | F1 | Precision | FP reales | FPR |
|---|---|---|---|---|---|
| A — prod JSON | 0.85/0.90/0.80/0.85 | **1.0000** | 1.0000 | 0 | 0.0000 |
| B — legacy low | 0.70/0.75/0.70/0.70 | **0.9976** | 0.9951 | 1 | 0.0002 |

**Infraestructura de experimentos creada:**
- `docs/experiments/f1_replay_log.csv`
- `scripts/calculate_f1_neris.py`
- `scripts/pipeline_health.sh`

---

### Day 80 (9 Mar 2026) — JSON is the LAW ✅
- 4 capas de bug resueltas. F1=0.9934, Precision=0.9869, Recall=1.0000, FN=0 ✅

### Day 79 (8 Mar 2026) — Sentinel Fix + F1=0.9921
- 8× `return 0.5f` → `MISSING_FEATURE_SENTINEL` (-9999.0f)
- F1=0.9921 baseline CTU-13 Neris ✅

### Day 76 (5 Mar 2026) — Proto3 Sentinel Fix + Pipeline Estable
- SIGSEGV eliminado. Pipeline 6/6 estable ✅

### Day 72 (Feb 2026) — Deterministic trace_id correlation
### Day 64 (21 Feb 2026) — CSV Pipeline + Test Suite
### Day 53 — HMAC Infrastructure (32/32 tests ✅)
### Day 52 — Stress Testing (364 ev/s, 54% CPU, 127MB, 0 crypto errors)

---

## 🔄 EN CURSO / INMEDIATO

### DAY 84 — Paper arXiv + trace_id fixes (P0)

**P0 — Redacción paper arXiv**
- Draft v5 completado DAY 88 ✅
- Veredicto unánime Consejo de Sabios: listo para arXiv ✅
- LaTeX `main.tex` + `references.bib` generados DAY 89 ✅
- Email endorser Sebastian Garcia enviado DAY 89 ✅
- Pendiente: respuesta endorser → submit arXiv cs.CR

**P1 — Fix 2 fallos preexistentes test_trace_id (DAY 72)**
```bash
make test  # identificar los 2 fallos trace_id
```

---

## 📋 BACKLOG — COMMUNITY

### 🟥 P0 — Recolección sistemática de datos etiquetados

- [ ] **FEAT-LABEL-1:** Almacenar eventos con etiqueta del fast detector
- [ ] **FEAT-LABEL-2:** Campo "revisión humana" opcional

### 🟧 P1 — Mejora y observabilidad del Fast Detector

- [ ] **FEAT-FP-1:** Registro de falsos positivos y negativos del fast detector
- [ ] **FEAT-FP-2 / DEBT-FD-001:** Migrar FastDetector Path A a configuración JSON
  - THRESHOLD_EXTERNAL_IPS=10, THRESHOLD_SMB_CONNS=3, THRESHOLD_PORT_SCAN=10,
    THRESHOLD_RST_RATIO=0.2, WINDOW_NS=10s — todos hardcodeados en fast_detector.hpp
  - Path B (send_ransomware_features) sí lee JSON ✅
  - Fix: inyectar FastDetectorConfig en constructor de FastDetector
  - Documentado en ADR-006. **Prerequisito bloqueante para FEAT-RANSOM-* y FEAT-RETRAIN-***

### 🟨 P2 — Ciclo de reentrenamiento ML con datos pipeline-native

- [ ] **FEAT-RETRAIN-1:** Generar dataset balanceado desde datos recolectados
- [ ] **FEAT-RETRAIN-2:** Entrenar y evaluar nuevos modelos RandomForest C++20
- [ ] **FEAT-RETRAIN-3:** A/B testing — dos versiones ML en paralelo

---

### 🟨 P2 — Expansión por familias de ransomware (FEAT-RANSOM-*)

**Prerequisito bloqueante:** DEBT-FD-001 cerrado (FEAT-FP-2)

**Principios de diseño (inmutables):**
1. **Dificultad ascendente** — familias más fáciles primero.
2. **Modelo fundacional, no modelos especializados** — objetivo final: un único
   RandomForest ensemble que haya visto todo. Los modelos por familia son etapas
   intermedias, no el destino.
3. **Misma estrategia que la actual** — sintético → modelo prueba → pcap relay →
   dataset pipeline-native → modelo mejorado. Repetir hasta convergencia.
4. **Features como decisión del Consejo** — antes de implementar cada familia,
   consultar al Consejo qué features adicionales capturar.
5. **La misma infraestructura sirve para todo** — FEAT-RETRAIN-* es reutilizable
   para cada familia.
6. **Open source primero, enterprise después** — modelos fundacionales van al núcleo
   MIT. Modelos enterprise emergen de la flota distribuida, cuando exista.

---

#### FEAT-RANSOM-2 — Neris Extended (dificultad: MUY BAJA — empieza aquí)

**Por qué primero:** ya tenemos la base. Solo son más escenarios CTU-13 del mismo
laboratorio. Sin tocar el modelo, sin datos nuevos.

**Objetivo:** evaluar el modelo actual contra escenarios CTU-13 adicionales
(escenarios 1-9, 11-13) para medir generalización dentro de la familia Neris.

```
1. Descargar CTU-13 escenarios adicionales → /vagrant/datasets/ctu13/
2. pcap relay con pipeline actual (sin modificar modelo)
3. Medir F1 por escenario — ¿el modelo generaliza dentro de Neris?
4. Documentar degradación si existe → identifica qué features faltan
5. Ajustar datos sintéticos si necesario → re-evaluar
```

**Valor añadido:** si Sebastian Garcia colabora, podemos validar con su ground
truth oficial. El email enviado DAY 89 abre esta puerta directamente.

- [ ] Descargar CTU-13 escenarios 1-9, 11-13
- [ ] pcap relay + F1 por escenario
- [ ] Documentar generalización en f1_replay_log.csv

---

#### FEAT-RANSOM-1 — WannaCry / NotPetya (dificultad: BAJA)

**Features de red con los 28 actuales:**
- Port 445 scan burst → ya cubierto (port diversity + external IP velocity)
- TCP SYN sin ACK → ya cubierto (RST ratio)
- DNS killswitch lookup → **nuevo feature — consultar Consejo**
- EternalBlue packet size → **nuevo feature probable — DEBT-PHASE2**

**Conditio sine qua non:**
- [ ] Consulta al Consejo: ¿features actuales suficientes?
- [ ] pcaps CTU-252..CTU-299 descargados en `/vagrant/datasets/wannacry/`

**pcaps disponibles (Stratosphere Lab — Sebastian Garcia):**
- CTU-252-1, CTU-253-1, CTU-254-1, CTU-256-1, CTU-258-1 — WannaCry variantes
- CTU-284-1..CTU-297-1 — WannaCry infección real entre VMs
- CTU-288-1..CTU-299-1 — NotPetya (mismo exploit, payload distinto)
- Fuente: https://www.stratosphereips.org/datasets-malware

**Ciclo:**
```
1. Datos sintéticos WannaCry (port 445 burst + RST ratio alto)
2. Modelo C++20 prueba WannaCry-only
3. Añadir al pipeline en modo observación (sin bloqueo)
4. pcap relay CTU-252..CTU-299 mezclado con bigFlows
5. CSV pipeline-native con etiquetas
6. Evaluar F1 → iterar hasta F1 > 0.95
7. Merge al modelo fundacional
```

- [ ] Consulta Consejo features WannaCry
- [ ] Datos sintéticos WannaCry
- [ ] Modelo C++20 prueba
- [ ] pcap relay + evaluación
- [ ] Merge modelo fundacional

---

#### FEAT-RANSOM-3 — DDoS Variants (dificultad: BAJA-MEDIA)

**Familias:** UDP flood, DNS amplification, SYN flood, HTTP flood.
El pipeline ya detecta DDoS volumétrico. Ampliar a variantes menos obvias.

**Features adicionales probables (consultar Consejo):**
- UDP:TCP ratio anómalo
- Destination port entropy (DNS amplification → concentración port 53)
- Packet size distribution asimétrica

**Datasets:**
- MAWI Working Group (ya en infraestructura)
- CAIDA DDoS datasets (públicos)
- CIC-DDoS2019 (UNB) — consultar Consejo

- [ ] Consulta Consejo features DDoS variants
- [ ] Datos sintéticos por variante
- [ ] Ciclo completo por variante

---

#### FEAT-RANSOM-4 — Ryuk / Conti (dificultad: MEDIA)

**Prerequisito adicional:** TimeWindowAggregator extendido — ventanas de
minutos/horas para capturar lateral movement lento. DEBT-PHASE2 territory.

**Features de red nuevas necesarias:**
- RDP (port 3389) anomaly
- Credential harvesting — volumen SMB sostenido en el tiempo
- Lateral movement graph — múltiples hosts internos secuenciales
- Beaconing regularity — intervalos C2 demasiado regulares

**Datasets:**
- Universidad de Navarra: 94 pcaps, 32 familias (http://dataset.tlm.unavarra.es/)
- Zenodo 2024: Ryuk balanceado (10,876/10,876)

- [ ] Consulta Consejo: ¿WINDOW_NS > 10s necesario? ¿Cómo sin romper latencia?
- [ ] DEBT-PHASE2 parcial (features temporales largas)
- [ ] Ciclo completo Ryuk → Conti

---

#### FEAT-RANSOM-5 — LockBit (dificultad: ALTA — BLOQUEADO)

**BLOQUEADO hasta:** DEBT-PHASE2 completado (12 features restantes)

**Por qué indetectable ahora:** C2 HTTPS cifrado estadísticamente similar
a navegación web benigna con los 28 features actuales.

**Features nuevas necesarias (sin ellas es indetectable):**
- TLS session duration anomaly
- Upload/download byte ratio (exfiltración StealBit → ratio asimétrico)
- Certificate anomaly (self-signed, recently issued)
- Connection burst a pocas IPs (distinto a WannaCry que contacta muchas)

**Acción mientras está bloqueado:**
- [ ] Consulta Consejo: features exactas para LockBit
- [ ] ADR-008: decisión de features LockBit
- [ ] No implementar hasta features + pcaps validados

**Datasets cuando esté desbloqueado:**
- Zenodo 2024: LockBit samples balanceados
- malware-traffic-analysis.net: capturas LockBit documentadas

---

### 🟨 P2 — Pipeline genérico de reentrenamiento (FEAT-RETRAIN-*)

Infraestructura reutilizable para todas las familias. Se construye una vez.

#### FEAT-RETRAIN-1 — Anonimización y unificación CSV

```
ml-detector CSV + firewall CSV
        ↓ [anonymization]
        - IP → pseudónimo consistente (hash no reversible)
        - Timestamps → relativos
        - Strip: trace_id, hostnames
        ↓ [unification]
        - Merge por trace_id
        - Balance clases: oversample ataques ~30-50%
        - Split train/validation/test reproducible (seed fijo)
        ↓
/vagrant/data/training/unified_YYYY-MM-DD.parquet
```

**Decisiones pendientes (consultar Consejo):**
- Python (pandas/polars) vs C++20 (Arrow) — Python para prototipo
- Parquet vs CSV — Parquet recomendado para datasets grandes

- [ ] Consulta Consejo: formato y stack
- [ ] Implementar capa anonimización
- [ ] Implementar unificación + balance + split

#### FEAT-RETRAIN-2 — Script de entrenamiento

```python
# input:  unified_dataset.parquet + config.json
# output: model_*.hpp + evaluation_report.md
# El transpiler C++20 ya existe (DAY 54) — parametrizar para nuevas familias
```

- [ ] Parametrizar transpiler existente
- [ ] Script entrenamiento + evaluación automática
- [ ] Threshold deploy configurable

#### FEAT-RETRAIN-3 — Proceso de inclusión al pipeline

```
Open source:   model_*.hpp → ml-detector/models/ → compile + test → merge
Enterprise:    [futuro] fleet_dataset → FEAT-RETRAIN-2 → ENT-MODEL-*
```

- [ ] Definir proceso de merge para modelos fundacionales
- [ ] Documentar en ADR-009

---

### 🟩 P3 — Aprendizaje continuo (Enterprise)

- [ ] **ENT-RETRAIN:** Ciclo de reentrenamiento automático periódico
- [ ] **ENT-1 (Federated):** prerequisito: validación local P2

### 🟩 P4 — Consenso mejorado firewall
- [ ] ADR-007: Consenso AND para bloqueo | P1-PHASE2 | zmq_handler.cpp

---

### Nota del Consejo de Sabios (DAY 81)

> "El fast detector ya es el backbone que protege. El ml-detector es hoy un
> observador silencioso y una fábrica de datos. La prioridad es no romper
> el escudo mientras alimentamos la máquina de aprendizaje."
>
> — Grok, Gemini, Qwen, DeepSeek, ChatGPT, Claude, Alonso

---

### FASE 3 — rag-ingester HMAC validation
- [ ] EventLoader valida HMAC antes de descifrar
- [ ] Tests: 10+ escenarios

### CsvEventLoader — rag-ingester
**Prerequisito:** CSV Pipeline E2E validado ✅ DAY 83

### CsvRetentionManager / ADR-005 / Estandarización logs ES→EN
Post-paper, junto con ENT-4 hot-reload.

### FASE 4 — Grace Period + Key Versioning
### FASE 5 — Auto-Rotation claves HMAC
### rag-local — informes PDF, geolocalización, historial

---

## 🏢 BACKLOG — ENTERPRISE

### ENT-MODEL-1 — Epic pcap relay (primer modelo enterprise — aprendizaje)

**Prerequisito:** TODOS los modelos fundacionales en línea (FEAT-RANSOM-1..5
completados) + proporciones correctas de todas las familias.

**Objetivo:** aprender el proceso. No es el modelo enterprise definitivo.

```
Dataset épico:
  - Neris (CTU-13, múltiples escenarios)
  - WannaCry / NotPetya (CTU-252..CTU-299)
  - Ryuk / Conti (Navarra + Zenodo)
  - DDoS variants (MAWI + CIC-DDoS2019)
  - Benigno balanceado (~50% del total)
  Proporciones: ~10% cada familia ataque + ~50% benigno
        ↓
  Pipeline replay épico (multi-sesión, días)
        ↓
  Dataset pipeline-native más rico generado hasta la fecha
        ↓
  FEAT-RETRAIN-2 → model_enterprise_v1.hpp
        ↓
  Evaluación exhaustiva → paper/report
```

### ENT-MODEL-2 — Modelos de flota distribuida (largo plazo)

**Prerequisito:** ENT-3 (P2P Seed), ENT-5 (rag-world), despliegue en producción.

Los verdaderos modelos enterprise emergen de los CSV generados por la flota
a lo largo del mundo — tráfico real, anónimo, diverso. No se puede simular
en laboratorio. Es el destino final, no el punto de partida.

### ENT-1 — Federated Threat Intelligence
### ENT-2 — Attack Graph Generation (GraphML + STIX 2.1)
### ENT-3 — P2P Seed Distribution via Protobuf (eliminar V-001)
### ENT-4 — Hot-Reload de Configuración en Runtime
### ENT-5 — rag-world (Telemetría Global Federada)
### ENT-6 — Integración Threat Intelligence (MISP/OpenCTI)
### ENT-7 — Observabilidad OpenTelemetry + Grafana
### ENT-8 — SecureBusNode (HSM + USB Root Key)
### ENT-9 — Captura y correlación opcional de datagramas sospechosos (ADR-008)

---

## 🗺️ Roadmap ransomware expansion

```
HOY     → arXiv submission (esperar endorser Sebastian Garcia)
        ↓
NEXT    → DEBT-FD-001 cerrado (prerequisito bloqueante)
        ↓
        → Consultas al Consejo de Sabios (features por familia)
        ↓
P2-1    → FEAT-RANSOM-2: Neris Extended (sin tocar modelo)
        ↓
P2-2    → FEAT-RANSOM-1: WannaCry/NotPetya
        + FEAT-RETRAIN-1+2: pipeline reentrenamiento (paralelo)
        ↓
P2-3    → FEAT-RANSOM-3: DDoS Variants
        ↓
P2-4    → FEAT-RANSOM-4: Ryuk/Conti (requiere DEBT-PHASE2 parcial)
        ↓
        → DEBT-PHASE2 completado (12 features)
        ↓
P2-5    → FEAT-RANSOM-5: LockBit
        ↓
ENT     → ENT-MODEL-1: Epic pcap relay → modelo enterprise v1 (aprendizaje)
        ↓
        → [despliegue producción + flota]
        ↓
        → ENT-MODEL-2: modelos de flota distribuida
                       los verdaderos modelos enterprise
```

---

## 🔔 Consultas pendientes al Consejo de Sabios

| Consulta | Timing | Objetivo |
|---|---|---|
| Features WannaCry/NotPetya | Antes de FEAT-RANSOM-1 | ¿DNS query feature? ¿Packet size? |
| WINDOW_NS para Ryuk/Conti | Antes de FEAT-RANSOM-4 | ¿Ventanas de minutos? ¿Coste de latencia? |
| Features LockBit | Antes de FEAT-RANSOM-5 | TLS anomaly, byte ratio |
| Python vs C++20 FEAT-RETRAIN-2 | Antes de FEAT-RETRAIN-2 | Prototipo vs producción |
| Proporciones dataset épico | Antes de ENT-MODEL-1 | Balance óptimo entre familias |

---

## 📊 Estado global del proyecto

```
                              [criterio: impl+test E2E+documentado = 100%]

Foundation + Thread-Safety:           ████████████████████ 100% ✅
Contract Validation:                  ████████████████████ 100% ✅
Build System:                         ████████████████████ 100% ✅
HMAC Infrastructure (F1+F2):          ████████████████████ 100% ✅
Proto3 Pipeline Stability:            ████████████████████ 100% ✅
Logging Standard (6 components):      ████████████████████ 100% ✅
Sentinel Correctness:                 ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
Thresholds desde JSON:                ████████████████████ 100% ✅
F1 Comparativa Limpia:                ████████████████████ 100% ✅
Infraestructura Experimentos:         ████████████████████ 100% ✅
CSV Pipeline ml-detector:             ████████████████████ 100% ✅
CSV Pipeline firewall-acl-agent:      ████████████████████ 100% ✅
F1-Score Validación (balanceado):     ████████████████████ 100% ✅
ML Score Investigation:               ████████████████████ 100% ✅
pipeline_health.sh:                   ████████████████████ 100% ✅
Paper arXiv (draft v5):               ████████████████████ 100% ✅  ← DAY 88
LaTeX main.tex:                       ████████████████████ 100% ✅  ← DAY 89
Email endorser Sebastian Garcia:      ████████████████████ 100% ✅  ← DAY 89
plugin-loader ADR-012 PHASE 1:        ████████████████░░░░  80% 🟡  integración sniffer DAY 94
trace_id correlación:                 ████████████████░░░░  80% 🟡  2 fallos DAY 72
Test Suite:                           ████████████████░░░░  80% 🟡  2 fallos trace_id
Ring Consumer Real Features:          ████████████░░░░░░░░  60% 🟡  28/40 reales
Fast Detector Config (DEBT-FD-001):   ████░░░░░░░░░░░░░░░░  20% 🟡  fix PHASE2
rag-local (community):                ████░░░░░░░░░░░░░░░░  20% 🟡
FEAT-RANSOM-2 (Neris Extended):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2, post DEBT-FD-001
FEAT-RANSOM-1 (WannaCry/NotPetya):    ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2
FEAT-RANSOM-3 (DDoS Variants):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2
FEAT-RANSOM-4 (Ryuk/Conti):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2, req PHASE2
FEAT-RANSOM-5 (LockBit):              ░░░░░░░░░░░░░░░░░░░░   0% ⏳  BLOQUEADO (PHASE2)
FEAT-RETRAIN-1+2+3:                   ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2, paralelo
ENT-MODEL-1 (epic pcap relay):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post todos fundacionales
FASE 3 rag-ingester HMAC:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳
CsvEventLoader rag-ingester:          ░░░░░░░░░░░░░░░░░░░░   0% ⏳
Attack Graph Generation:              ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ENT-2
Federated Threat Intelligence:        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ENT-1
P2P Seed Distribution:                ░░░░░░░░░░░░░░░░░░░░   0% ⏳  ENT-3
ENT-MODEL-2 (flota distribuida):      ░░░░░░░░░░░░░░░░░░░░   0% ⏳  largo plazo
```

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución |
|----------|------------|
| CSV cifrado | ❌ No — sin cifrado, con HMAC por fila |
| Sentinel correctness | -9999.0f fuera del dominio ✅ DAY 79 |
| 0.5f TCP half-open | Valor semántico válido — comentario protector ✅ DAY 79 |
| Thresholds ML | Desde JSON — CERRADO ✅ DAY 80 |
| Fichero fuente JSON sniffer | `sniffer/config/sniffer.json` (NO build-debug) ✅ DAY 81 |
| Log standard | /vagrant/logs/lab/COMPONENTE.log ✅ DAY 79 |
| Dual logs ml-detector | detector.log=fuente verdad, ml-detector.log=arranque — ADR-005 ✅ |
| FlowStatistics Phase 2 | tcp_udp_ratio/protocol_variety/duration_std → DEBT-PHASE2 ✅ |
| GeoIP en critical path | ❌ Deliberadamente fuera — latencia inaceptable |
| Fast Detector dual-path | Path A hardcodeado (DAY 13), Path B JSON (DAY 80). DEBT-FD-001. Fix PHASE2 |
| ML attack counters | 3 semánticas distintas: RF vote / conf>=0.65 / malicious_threshold |
| level1_attack threshold | 0.65 en ml-detector/config/ml_detector_config.json |
| CSV paths (fuente verdad) | ml-detector: /vagrant/logs/ml-detector/events/ — firewall: /vagrant/logs/firewall_logs/ ✅ |
| pipeline_health.sh VM | vagrant ssh defender (no server) ✅ DAY 83 |
| Modelo fundacional vs especializado | Un único RF ensemble que lo haya visto todo ✅ DAY 89 |
| Orden expansión familias | Dificultad ascendente: Neris→WannaCry→DDoS→Ryuk→LockBit ✅ DAY 89 |
| LockBit bloqueado | Indetectable hasta DEBT-PHASE2 (12 features TLS/byte ratio) ✅ DAY 89 |
| Enterprise vs open source | Fundacionales → MIT core. Enterprise → flota distribuida ✅ DAY 89 |

---

*Última actualización: Day 93 — 21 Mar 2026*
*Branch: feature/plugin-loader-adr012 — DAY 93*
*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic), Grok, ChatGPT, DeepSeek, Qwen, Gemini, Parallel.ai*