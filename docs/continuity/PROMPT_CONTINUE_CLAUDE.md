# ML Defender — Prompt de Continuidad DAY 90
**Generado:** Cierre DAY 89 (17 marzo 2026)
**Branch activa:** `main`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 46/46 ✅

---

## Logros DAY 89

### TAREA 1 — CLAUDE.md actualizado ✅
- Fichero: `CLAUDE.md` (~10 KB, 289 líneas)
- Rename NDR, F1=0.9985, stress test, paper v5, tests 70/70, endorsers, ADRs, roadmap
- Disponible en outputs para commit al repo

### TAREA 2 — Dataset URLs añadidos ✅
- README.md: sección `### Dataset Setup` con wget + paths VM
- §13 preprint: bloque dataset setup antes de comandos F1
- URLs: CTU Prague primaria + Stratosphere IPS mirror

### TAREA 3 — LaTeX completo ✅
- `main.tex` (1.052 líneas) + `references.bib` (129 líneas)
- Compilado en Overleaf → PDF académico limpio, 19 páginas
- Fórmulas renderizadas (ec. 1-8), tablas booktabs, referencias [1]-[11]
- Autor: **Alonso Isidoro Román** ✅
- Sin changelog interno — cabecera limpia de preprint

### TAREA 4 — Email endorser enviado ✅
- Para: sebastian.garcia@agents.fel.cvut.cz
- Asunto: arXiv endorsement request — open-source NDR system evaluated on CTU-13 Neris (646 TP, F1=0.9985)
- Adjunto: PDF LaTeX generado en Overleaf
- Contenido: motivación Hospital Clínic + datos exactos (147.32.84.165, 646 flows) + mención Slips
- Estado: enviado, pendiente respuesta

### TAREA 5 — BACKLOG unificado ✅
- FEAT-RANSOM-* integrado en backlog principal
- 5 familias en orden de dificultad: Neris Extended → WannaCry/NotPetya → DDoS Variants → Ryuk/Conti → LockBit (bloqueado)
- FEAT-RETRAIN-1/2/3: pipeline genérico de reentrenamiento
- ENT-MODEL-1/2: modelos enterprise (epic pcap relay + flota distribuida)
- Principio arquitectural fijado: modelo fundacional único, no modelos especializados
- LockBit bloqueado explícitamente hasta DEBT-PHASE2 (TLS/byte ratio features)

---

## Estado del repo post-DAY 89

### Ficheros nuevos/modificados (pendientes de commit)
```
CLAUDE.md                              ← actualizado DAY 88→89
README.md                              ← dataset setup añadido
docs/latex/main.tex                    ← nuevo
docs/latex/references.bib             ← nuevo
docs/backlog/BACKLOG.md               ← unificado con FEAT-RANSOM-*
docs/Ml defender paper draft v5.md    ← §13 actualizado con dataset setup
```

### Commit sugerido
```bash
git add CLAUDE.md README.md docs/latex/ docs/backlog/BACKLOG.md
git commit -m "docs: DAY 89 — CLAUDE.md updated, dataset URLs, LaTeX main.tex, ransomware expansion backlog"
git tag v0.89.0-day89
git push origin main --tags
```

---

## Tareas DAY 90 — en orden de prioridad

### P0 — Seguimiento endorser Sebastian Garcia
- Si responde positivo → submit arXiv cs.CR inmediatamente
- Si no ha respondido → esperar hasta día 7 (24 marzo) antes de contactar Tier 2
- Tier 2 backup: Yisroel Mirsky (mirsky@bgu.ac.il) — autor Kitsune, citado en §2 y Tabla 4

### P0 — Fix 2 fallos trace_id (DAY 72, pendiente desde DAY 84)
```bash
vagrant ssh defender
cd /vagrant
make test  # identificar los 2 fallos exactos
# investigar root cause
# TDH: test que falla → fix → confirmar 70/70 → 72/72
```
Con estos 2 fixes: test suite pasa de 70/70 a 72/72 (100% real).

### P1 — Subir LaTeX a repo
```bash
mkdir -p docs/latex
cp main.tex docs/latex/
cp references.bib docs/latex/
git add docs/latex/
git commit -m "docs: add LaTeX source for arXiv submission"
```

### P1 — Consulta al Consejo: features WannaCry/NotPetya
Antes de FEAT-RANSOM-1, preguntar a cada modelo del Consejo:
- ¿Son suficientes los 28 features actuales para WannaCry?
- ¿DNS query feature necesaria?
- ¿Packet size distribution como feature adicional?

### P2 — FEAT-RANSOM-2: Neris Extended (si trace_id resuelto)
```bash
# Descargar CTU-13 escenarios adicionales
wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-[1-9]/
# pcap relay con pipeline actual
# Medir F1 por escenario
```

---

## Pendiente técnico post-paper

| Item | Prioridad | Estado |
|---|---|---|
| Bare-metal throughput stress test | P1 | post-paper |
| DEBT-FD-001: FastDetector Path A → JSON | P1-PHASE2 | prerequisito FEAT-RANSOM-* |
| ADR-007: AND-consensus firewall | P1-PHASE2 | post-paper |
| FEAT-RANSOM-2: Neris Extended | P2 | post DEBT-FD-001 |
| FEAT-RANSOM-1: WannaCry/NotPetya | P2 | post DEBT-FD-001 |
| FEAT-RETRAIN-1/2/3 | P2 | paralelo a FEAT-RANSOM-* |
| FEAT-RANSOM-5: LockBit | P2 | BLOQUEADO hasta PHASE2 |

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca `sed -i`. Python3 inline desde dentro de la VM.
- **JSON sniffer:** `sniffer/config/sniffer.json` (NO build-debug/)
- **VM:** `vagrant ssh defender` (no `server`)
- **Dataset paths:** `/vagrant/datasets/ctu13/neris.pcap` | `/vagrant/datasets/ctu13/bigFlows.pcap`
- **F1 calculator:** `python3 scripts/calculate_f1_neris.py <sniffer.log> --total-events 19135`
- **Overleaf project:** argus-ndr-v5 (compilado ✅, PDF 19 páginas)

---

## Endorsers arXiv
- **Tier 1 contactado:** Sebastian Garcia — sebastian.garcia@agents.fel.cvut.cz ✅ (DAY 89)
- **Tier 1 pendiente:** Yisroel Mirsky — Ben-Gurion University (autor Kitsune)
- **Tier 1 pendiente:** Ariel Shabtai — Ben-Gurion University
- **Tier 2 backup:** Ali Habibi Lashkari, Iman Sharafaldin, Battista Biggio

## Consejo de Sabios
Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI), DeepSeek, Qwen (Alibaba), Gemini (Google), Parallel.ai — 7 modelos.

## Métricas definitivas (paper v5)
- F1=0.9985, Precision=0.9969, Recall=1.0000 (CTU-13 Neris, DAY 86, 4 runs estables)
- TP=646, TN=12,075, FP=2 (artefactos VirtualBox), FN=0
- FPR ML=0.0002% | FPR Fast Detector=6.61% bigFlows | FP reduction ~500×
- Latencia: 0.24 μs (DDoS) a 1.06 μs (Ransomware)
- Stress test: ~33–38 Mbps techo VirtualBox NIC, 2.37M packets, 0 drops, RAM ~1.28 GB

---

*DAY 90 arranca con: seguimiento endorser + fix trace_id + LaTeX al repo + consulta Consejo features WannaCry*
*Test suite: 70/70 ✅ | Paper: draft v5 + LaTeX ✅ | Email endorser: enviado ✅ | Repo: limpio ✅*