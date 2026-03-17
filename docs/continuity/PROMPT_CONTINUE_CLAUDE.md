# ML Defender — Planificación DAY 90–103
## "Fase de documentación y diseño" (sin hardware bare-metal)
**Período:** 18 marzo – 31 marzo 2026
**Constraint:** sin PC bare-metal Linux disponible
**Trigger de cambio de fase:** respuesta de Sebastian Garcia (endorser arXiv)

---

## Principio organizador

Estas dos semanas tienen un objetivo claro: cuando llegue el hardware
(o la respuesta de Sebastian), el proyecto debe estar tan bien documentado
que cualquier colaborador externo pueda entender el sistema, y cualquier
experimento planificado pueda ejecutarse en un día limpio sin ambigüedad.

Tres flujos en paralelo:
- **Flujo A — Código:** lo que se puede hacer en la VM existente
- **Flujo B — Documentación y diseño:** trabajo intelectual puro
- **Flujo C — Consejo de Sabios:** consultas técnicas a los 7 modelos

---

## SEMANA 1 (DAY 90–96) — Base sólida

### DAY 90 (18 mar) — Fix trace_id + ADR-007
**Flujo A:**
- [ ] Fix 2 fallos trace_id (DAY 72) — TDH: test que falla → fix → 72/72
- [ ] Confirmar test suite 72/72 ✅

**Flujo B:**
- [ ] ADR-007: AND-consensus firewall
    - Decisión ya tomada (OR alertas / AND bloqueos)
    - Documento formal: contexto, decisión, consecuencias, implementación futura
    - Path: `docs/adr/ADR-007-and-consensus-firewall.md`

**Flujo C:**
- [ ] Consulta Consejo #1: features WannaCry/NotPetya
    - Pregunta: ¿son suficientes los 28 features actuales para separar WannaCry
      de tráfico benigno? ¿Qué añadirías?
    - Esperar respuestas divergentes → arbitrar → documentar decisión

---

### DAY 91 (19 mar) — ARCHITECTURE.md + diseño WannaCry sintético
**Flujo B:**
- [ ] `ARCHITECTURE.md` — descripción técnica completa del pipeline
    - Para nuevos colaboradores y potenciales deployment partners
    - Secciones: componentes, flujo de datos, decisiones clave, deployment
    - Incluir diagrama ASCII del pipeline (ya existe en README, expandir)
    - Path: `ARCHITECTURE.md` (raíz del repo)

- [ ] Especificación datos sintéticos WannaCry/NotPetya
    - Distribuciones estadísticas basadas en literatura + CTU-13
    - Port 445 burst: distribución, rate, duración
    - RST ratio esperado: rango, varianza
    - DNS killswitch lookup: frecuencia, TTL
    - Path: `docs/design/synthetic_data_wannacry_spec.md`

**Flujo C:**
- [ ] Integrar respuestas Consejo #1 → decisión features WannaCry documentada

---

### DAY 92 (20 mar) — ADR-008 LockBit + CONTRIBUTING.md
**Flujo B:**
- [ ] ADR-008: features necesarias para LockBit
    - TLS session duration anomaly — especificación técnica
    - Upload/download byte ratio — cómo extraerlo del pipeline actual
    - Certificate anomaly — viable sin DPI?
    - Conclusión: qué hay que implementar en DEBT-PHASE2 primero
    - Path: `docs/adr/ADR-008-lockbit-features.md`

- [ ] `CONTRIBUTING.md` — guía para colaboradores externos
    - Cómo levantar el entorno (Vagrant)
    - Cómo ejecutar tests
    - Cómo añadir un nuevo modelo al pipeline
    - Código de conducta (Via Appia Quality)
    - Path: `CONTRIBUTING.md` (raíz del repo)

**Flujo C:**
- [ ] Consulta Consejo #2: WINDOW_NS para Ryuk/Conti
    - Pregunta: ¿cómo extender ventanas temporales a minutos/horas
      sin comprometer la latencia de detección en tiempo real?
    - Posibles aproximaciones: ventanas deslizantes multi-escala,
      buffer secundario, agregación asíncrona

---

### DAY 93 (21 mar) — Protocolo bare-metal + diseño FEAT-RETRAIN
**Flujo B:**
- [ ] Protocolo bare-metal stress test (escrito antes de tener hardware)
    - Metodología exacta: tcpreplay rates, métricas a capturar, duración
    - Hardware objetivo: specs mínimas, NIC recomendada
    - Criterios de éxito: qué números necesitamos ver
    - Path: `docs/experiments/bare_metal_stress_test_protocol.md`

- [ ] Especificación FEAT-RETRAIN-1 (anonimización CSV)
    - Schema de entrada (127 columnas CSV ml-detector)
    - Schema de salida (dataset unificado anonimizado)
    - Reglas de anonimización: IP hashing, timestamp relativos
    - Balance de clases: algoritmo de oversampling
    - Path: `docs/design/feat_retrain_1_spec.md`

**Flujo C:**
- [ ] Integrar respuestas Consejo #2 → decisión WINDOW_NS documentada
- [ ] Consulta Consejo #3: Python vs C++20 para FEAT-RETRAIN-2
    - Pregunta: prototipo Python (pandas + scikit-learn) primero,
      luego C++20 si necesario — ¿o directo C++20?

---

### DAY 94 (22 mar) — ADR-009 + diseño Ryuk/Conti sintético
**Flujo B:**
- [ ] ADR-009: proceso de inclusión de modelos nuevos al pipeline
    - Open source: modelo → tests → merge → tag
    - Enterprise: flujo desde fleet data → modelo → distribución
    - Criterios de aceptación: F1 mínimo, FPR máximo, latencia máxima
    - Path: `docs/adr/ADR-009-model-inclusion-process.md`

- [ ] Especificación datos sintéticos Ryuk/Conti
    - Diferencias con WannaCry: lateral movement lento, RDP, credenciales
    - Features temporales necesarias: ¿cómo capturarlas con WINDOW_NS actual?
    - Path: `docs/design/synthetic_data_ryuk_spec.md`

**Seguimiento arXiv:**
- [ ] Si Sebastian no ha respondido (día 7) → email Yisroel Mirsky (Tier 2)

---

### DAY 95 (23 mar) — FEAT-RANSOM-2 metodología + DDoS variants diseño
**Flujo B:**
- [ ] Protocolo FEAT-RANSOM-2: Neris Extended
    - Lista exacta de escenarios CTU-13 a descargar (1-9, 11-13)
    - Metodología de evaluación: F1 por escenario, métricas de generalización
    - Criterios: ¿qué F1 mínimo necesitamos para considerar que generaliza?
    - Path: `docs/experiments/feat_ransom_2_neris_extended_protocol.md`

- [ ] Especificación DDoS Variants (FEAT-RANSOM-3)
    - UDP flood vs DNS amplification vs SYN flood: diferencias de features
    - Datasets disponibles: MAWI, CAIDA, CIC-DDoS2019
    - Path: `docs/design/feat_ransom_3_ddos_spec.md`

**Flujo C:**
- [ ] Integrar respuestas Consejo #3 → decisión stack reentrenamiento
- [ ] Consulta Consejo #4: proporciones dataset épico (ENT-MODEL-1)
    - Pregunta: ¿qué proporción benigno/ataque por familia?
      ¿50% benigno + 10% por familia ataque? ¿Otra distribución?

---

### DAY 96 (24 mar) — Revisión semana 1 + commit
**Flujo B:**
- [ ] Revisión y coherencia de todos los documentos semana 1
- [ ] Actualizar BACKLOG.md con decisiones tomadas
- [ ] Actualizar CLAUDE.md con estado DAY 96

**Commit semana 1:**
```bash
git add docs/adr/ADR-007* docs/adr/ADR-008* docs/adr/ADR-009*
git add ARCHITECTURE.md CONTRIBUTING.md
git add docs/design/ docs/experiments/
git commit -m "docs: DAY 90-96 — ADR-007/008/009, ARCHITECTURE, CONTRIBUTING, design specs"
git tag v0.96.0-day96
git push origin main --tags
```

---

## SEMANA 2 (DAY 97–103) — Profundidad y preparación

### DAY 97 (25 mar) — DEBT-FD-001 diseño técnico
**Flujo A:**
- [ ] Diseño técnico detallado de DEBT-FD-001
    - Análisis del código actual: `FastDetector::is_suspicious()` en fast_detector.hpp
    - Plan de migración: qué constexpr → qué campo JSON
    - Tests necesarios: casos de prueba antes de implementar
    - Path: `docs/design/debt_fd_001_migration_plan.md`

**Flujo B:**
- [ ] Guía de deployment para hospitales/escuelas
    - Hardware mínimo recomendado
    - Proceso de instalación paso a paso
    - Configuración inicial (sniffer.json)
    - Consideraciones de seguridad operacional
    - Path: `docs/deployment/hospital_deployment_guide.md`

---

### DAY 98 (26 mar) — Paper v2 preparación
**Flujo B:**
- [ ] Identificar mejoras para paper v2 (en base a feedback Consejo + Gepeto)
    - ¿Qué limitaciones ampliar en §10?
    - ¿Tabla comparativa mejorada con dataset column ya añadida?
    - ¿Diagrama pipeline (figura real en LaTeX)?
    - Checklist de cambios v1 → v2
    - Path: `docs/paper/v2_improvement_checklist.md`

- [ ] Si Sebastian ha respondido positivo → **submit arXiv cs.CR**

**Flujo C:**
- [ ] Integrar respuestas Consejo #4 → proporciones dataset épico documentadas
- [ ] Consulta Consejo #5: ¿qué haría el sistema más valioso para un hospital real?
    - Pregunta abierta: fuera del radar técnico habitual

---

### DAY 99 (27 mar) — Especificación FEAT-RETRAIN-2 completa
**Flujo B:**
- [ ] Especificación completa script de entrenamiento
    - Input schema, output schema
    - Hiperparámetros RandomForest: cuáles fijar, cuáles buscar
    - Criterios de deploy: F1 > X, FPR < Y, latencia < Z μs
    - Proceso de transpilación a C++20 (parametrizar transpiler existente DAY 54)
    - Path: `docs/design/feat_retrain_2_training_spec.md`

- [ ] Especificación FEAT-LABEL-1 (recolección datos etiquetados)
    - Schema del dataset pipeline-native
    - Rotación diaria, retención configurable
    - Path: `docs/design/feat_label_1_spec.md`

---

### DAY 100 (28 mar) — Hito: documentación enterprise
**Flujo B:**
- [ ] Documento ENT-MODEL-1: epic pcap relay plan
    - Lista exacta de datasets a combinar
    - Proporciones por familia (resultado Consejo #4)
    - Duración estimada del replay
    - Hardware necesario (esto sí requiere bare-metal)
    - Criterios de éxito del modelo enterprise v1
    - Path: `docs/enterprise/ent_model_1_epic_replay_plan.md`

- [ ] Documento ENT-MODEL-2: visión flota distribuida
    - Arquitectura conceptual de la flota
    - Flujo de datos: nodo → anonimización → agregación → modelo
    - Privacidad: qué datos viajan, qué datos se quedan locales
    - Path: `docs/enterprise/ent_model_2_fleet_vision.md`

**LinkedIn DAY 100** — hito simbólico, post especial

---

### DAY 101 (29 mar) — Mintlify docs + FAQ
**Flujo B:**
- [ ] Actualizar documentación Mintlify con estado actual
    - Nuevas secciones: FEAT-RANSOM-*, FEAT-RETRAIN-*
    - Resultados stress test
    - Estado arXiv submission
    - URL: https://alonsoir-test-zeromq-c-.mintlify.app/introduction

- [ ] FAQ técnico para potenciales colaboradores
    - ¿Por qué C++20 y no Python?
    - ¿Por qué RandomForest y no deep learning?
    - ¿Por qué TinyLlama local y no GPT-4?
    - ¿Por qué ChaCha20 y no AES?
    - Path: `docs/faq.md`

---

### DAY 102 (30 mar) — Revisión general + coherencia
**Flujo B:**
- [ ] Auditoría de coherencia entre todos los documentos nuevos
    - ¿Contradicen algún ADR existente?
    - ¿Están alineados con el paper v5?
    - ¿Están alineados entre sí?
- [ ] Actualizar BACKLOG.md con todas las decisiones de diseño tomadas
- [ ] Actualizar CLAUDE.md con estado DAY 102

---

### DAY 103 (31 mar) — Commit final + planificación siguiente fase
**Commit semana 2:**
```bash
git add docs/
git commit -m "docs: DAY 97-103 — deployment guide, paper v2 checklist, enterprise specs, Mintlify, FAQ"
git tag v0.103.0-day103
git push origin main --tags
```

**Planificación siguiente fase:**
- Si Sebastian (o Mirsky) ha respondido → arXiv submitted → fase validación
- Si no → Tier 3 + empezar FEAT-RANSOM-2 en VM (Neris Extended, sin hardware nuevo)
- Si hay hardware disponible → bare-metal stress test según protocolo DAY 93

---

## Resumen visual

```
DAY 90  ─── Fix trace_id (72/72) + ADR-007 + Consejo #1 (WannaCry features)
DAY 91  ─── ARCHITECTURE.md + spec sintético WannaCry
DAY 92  ─── ADR-008 (LockBit) + CONTRIBUTING.md + Consejo #2 (WINDOW_NS Ryuk)
DAY 93  ─── Protocolo bare-metal + spec FEAT-RETRAIN-1 + Consejo #3 (Python vs C++)
DAY 94  ─── ADR-009 (modelos) + spec Ryuk/Conti + seguimiento Mirsky si necesario
DAY 95  ─── Protocolo Neris Extended + spec DDoS + Consejo #4 (proporciones)
DAY 96  ─── Revisión + commit semana 1 ──────────────────────────────────────────
DAY 97  ─── Diseño técnico DEBT-FD-001 + guía deployment hospitales
DAY 98  ─── Paper v2 checklist + Consejo #5 (valor hospital real)
DAY 99  ─── Spec FEAT-RETRAIN-2 completa + FEAT-LABEL-1
DAY 100 ─── ENT-MODEL-1 plan + ENT-MODEL-2 visión + LinkedIn hito DAY 100
DAY 101 ─── Mintlify actualización + FAQ técnico
DAY 102 ─── Auditoría coherencia + BACKLOG + CLAUDE.md
DAY 103 ─── Commit semana 2 + planificación siguiente fase ──────────────────────
```

---

## Trigger de cambio de fase

| Evento | Acción inmediata |
|---|---|
| Sebastian responde positivo | Submit arXiv cs.CR ese mismo día |
| Sebastian no responde en 7 días (DAY 96) | Email Yisroel Mirsky (Tier 2) |
| Mirsky no responde en 7 días (DAY 103) | Tier 3 + arXiv sin endorser (cs.CR permite) |
| Hardware bare-metal disponible | Ejecutar protocolo DAY 93 inmediatamente |
| Colaborador externo interesado | CONTRIBUTING.md + ARCHITECTURE.md ya listos |

---

*Planificación generada: DAY 89 — 17 marzo 2026*
*Revisión prevista: DAY 96 (24 marzo) y DAY 103 (31 marzo)*
*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic) and the rest of the crew*