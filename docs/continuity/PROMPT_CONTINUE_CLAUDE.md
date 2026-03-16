# ML Defender — Prompt de Continuidad DAY 89
**Generado:** Cierre DAY 88 (16 marzo 2026)
**Branch activa:** `main`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 46/46 ✅

---

## Logros DAY 88

### TAREA 1 — Paper draft v5 ✅
- Fichero: `docs/Ml defender paper draft v5.md`
- Feedback del Consejo de Sabios (ronda v5) integrado — 5/5 modelos: ✅
- **Rename definitivo: aRGus EDR → aRGus NDR** — scope NDR actual, EDR = roadmap FEAT-EDR-1
- Cambios integrados: GEM1, DSV1-DSV4, CGV2-CGV6, G3, G5, Q1, Q3, CGP1
- Veredicto unánime del Consejo: "Listo para endorsers / publicable en arXiv"
- Commit: `bd94dcfb`

### TAREA 2 — README.md actualizado ✅
- Métricas corregidas: F1=0.9985, TP=646, FPR=0.0002%, ~500× FP reduction
- Stress test results integrados (DAY 87)
- FEAT-ENTRY-* roadmap añadido
- Rename NDR, Gemini añadido al Consejo
- Sección Funding/Collaboration (ENISA, INCIBE, Horizon Europe)

### TAREA 3 — Limpieza repo ✅
- Rama `cleanup/repo-presentation` creada y mergeada a main
- Eliminados: Docker (Dockerfile.service1/2/3, docker-compose.yaml), debian/, paper_academic_trap/, service1/2/3/, cppcheck reports, etcd_client_*.cpp dispersos, backups, logs en raíz
- Movidos: docs/history/, docs/continuity/, docs/gateway/, scripts/tsan/, scripts/stress/
- Renombrado: claude.md → CLAUDE.md
- .gitignore actualizado: backups, macOS artifacts, logs en raíz
- Vagrantfile: Docker eliminado del provisioning (ya no instalará Docker en vagrant provision)

---

## Estado del repo post-DAY 88

### Raíz limpia
```
CLAUDE.md          ← memoria del proyecto (desactualizado — ver TAREA 1 DAY 89)
LICENSE.txt
Makefile
README.md
Vagrantfile
```

### Carpetas activas del pipeline
```
sniffer/           ml-detector/       etcd-server/
firewall-acl-agent/ rag-ingester/     rag/
common/            common-rag-ingester/ protobuf/
etcd-client/       crypto-transport/  tools/
scripts/           docs/              third_party/
```

### Carpetas pendientes de revisión futura (no urgente)
```
models/            722 MB local, solo README.md trackeado — correcto
ml-training/       historia de experimentos de entrenamiento — se queda
pcap_testing/      pcaps pequeños, no trackeados
testing/           no trackeado
mawi/              datasets MAWI
datasets/          datasets locales, no trackeados (gitignore)
contract-validation/ DAY 52 experiment
contrib/           revisar en sesión futura
```

---

## Tareas DAY 89 — en orden de prioridad

### P0 — CLAUDE.md actualización
El CLAUDE.md actual (22KB, febrero 2026) está desactualizado — no refleja:
- Rename aRGus NDR
- F1=0.9985, TP=646, métricas correctas
- Stress test DAY 87
- Paper draft v5
- Test suite 70/70
- FEAT-ENTRY-* roadmap
- Arquitectura final (6 componentes estables)

Un CLAUDE.md bien escrito = cualquier sesión futura arranca con contexto completo.
Estructura sugerida: arquitectura actual | métricas validadas | decisiones clave (ADRs) | deuda técnica | estado del paper | próximos pasos.

### P0 — Email a Sebastian Garcia (endorser arXiv)
```
Para: sebastian.garcia@agents.fel.cvut.cz
Asunto: arXiv endorsement request — ML NIDS/NDR using CTU-13 dataset
```
Contenido: 3 párrafos — quién eres, qué es ML Defender (aRGus NDR), por qué CTU-13 conecta con su trabajo.
Adjunto: PDF del preprint (generar via pandoc o LaTeX preliminar).
Link: repositorio GitHub.

### P0 — Dataset setup en README y §13 del preprint
Añadir nota de rutas esperadas:
```markdown
**Dataset setup:**
- CTU-13: https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/
  (mirror: https://www.stratosphereips.org/datasets-ctu13)
- Place files in `/vagrant/datasets/ctu13/`
  - neris.pcap (CTU-13 scenario 10)
  - bigFlows.pcap (CTU-13 scenario bigFlows)
```

### P1 — Lista de endorsers completa
Tier 1 (contactar primero):
1. Sebastian Garcia — sebastian.garcia@agents.fel.cvut.cz (CTU Prague, autor CTU-13)
2. Yisroel Mirsky — Ben-Gurion University (autor Kitsune, citado en §2 y Tabla 4)
3. Ariel Shabtai — Ben-Gurion University (co-autor Kitsune)

Tier 2 (si Tier 1 no responde en 2 semanas):
4. Ali Habibi Lashkari — York University (autor CIC-IDS2017)
5. Iman Sharafaldin — UNB (co-autor CIC-IDS2017, citado en §8.2)
6. Battista Biggio — Universidad de Cagliari (adversarial ML para seguridad)

### P1 — LaTeX conversion
Plantilla: arXiv standard (`article` class)
- Estructura `main.tex` con secciones del draft v5
- Fórmulas ya en LaTeX en el Markdown — migración directa
- Tablas: conversión de Markdown a `tabular`
- Fichero `.bib` con todas las referencias
- Diagrama pipeline §4.1 (opcional pero recomendado)
- UTF-8 / caracteres especiales verificados

---

## Pendiente técnico post-paper (no DAY 89)

| Item | Prioridad | Estado |
|---|---|---|
| Bare-metal throughput stress test | P1 | post-paper |
| DEBT-FD-001: FastDetector Path A → JSON | P1-PHASE2 | post-paper |
| ADR-007: AND-consensus firewall | P1-PHASE2 | post-paper |
| DNS payload parsing real | P2 | post-paper |
| FEAT-NET-1: DNS anomaly / DGA detection | P1 | roadmap |
| FEAT-NET-2: Threat intelligence feeds | P1 | roadmap |
| FEAT-AUTH-1: Auth log ingestion | P2 | roadmap |
| FEAT-EDR-1: Lightweight endpoint agent | P3 | roadmap |

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca `sed -i`. Usar Python3 inline **desde dentro de la VM**.
- **JSON sniffer:** `sniffer/config/sniffer.json` (NO build-debug/)
- **JSON ml-detector:** `ml-detector/config/ml_detector_config.json`
- **VM:** `defender` (no `server`)
- **CSV ml-detector:** `/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv`
- **CSV firewall:** `/vagrant/logs/firewall_logs/firewall_blocks.csv`
- **F1 calculator:** `python3 scripts/calculate_f1_neris.py <sniffer.log> --total-events 19135`
- **Dataset paths:** `/vagrant/datasets/ctu13/neris.pcap` | `/vagrant/datasets/ctu13/bigFlows.pcap`

---

## Consejo de Sabios
Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI), DeepSeek, Qwen (Alibaba), Gemini (Google), Parallel.ai — 7 modelos.
Veredicto unánime v5: listo para arXiv.

---

## Métricas definitivas (paper v5)
- F1=0.9985, Precision=0.9969, Recall=1.0000 (CTU-13 Neris, DAY 86, 4 runs estables)
- TP=646 (flows C2 activos), TN=12,075, FP=2 (artefactos VirtualBox), FN=0
- FPR ML=0.0002% | FPR Fast Detector=6.61% bigFlows | FP reduction ~500×
- Latencia: 0.24 μs (DDoS) a 1.06 μs (Ransomware)
- Stress test: ~33–38 Mbps techo VirtualBox NIC, 2.37M packets, 0 drops, 0 errors, RAM ~1.28 GB estable

---

## Secuencia DAY 89

1. **CLAUDE.md** — actualizar con estado completo DAY 88
2. **Dataset URLs** — añadir en README.md y §13 del preprint
3. **Email Sebastian Garcia** — redactar y enviar
4. **LaTeX** — empezar conversión main.tex

*DAY 89 arranca con: CLAUDE.md + dataset URLs + email endorser + LaTeX*
*Test suite: 100% ✅ | Stress test: completado ✅ | Paper: draft v5 ✅ | Repo: limpio ✅*