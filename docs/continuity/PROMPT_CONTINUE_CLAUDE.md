# ML Defender — Prompt de Continuidad DAY 92
## 20 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING (etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall)
**Test suite:** 31/31 ✅ (crypto 3/3, etcd-hmac 12/12, ml-detector 9/9, rag-ingester 7/7)
**Rama activa:** main
**Último tag:** DAY 91

---

## Lo que se hizo en DAY 91

### Flujo B — Documentación (todo el día)

**ARCHITECTURE.md v6.0.0** — documento técnico completo para colaboradores externos.
- Path: raíz del repo (`ARCHITECTURE.md`)
- 14 secciones: visión, 6 componentes, flujo de datos, diagramas ASCII, stack tecnológico, 12 ADRs, deployment Vagrant + bare-metal, seguridad, ML subsystem, performance, limitaciones, roadmap
- Incluye tabla de generalización (WannaCry recall 0.70–0.85 sin reentrenamiento)
- Incluye GAIA vision como long-term context

**`docs/design/synthetic_data_wannacry_spec.md`** — spec completa para generación de datos sintéticos WannaCry/NotPetya.
- 11 secciones con perfiles cuantitativos WannaCry y NotPetya
- Feature targets con rangos precisos para `rst_ratio`, `syn_ack_ratio`, `flow_duration_min`
- Control negativo: 6 tipos de tráfico Windows legítimo con regla dura de exclusión
- Dataset objetivo: 20,000 flows, split 40% maligno / 60% benigno
- Backlog SYN-1 a SYN-12 con critical path

**Acta Consejo #1 — CERRADA.** 7/7 modelos han respondido.

### Consejo de Sabios — Consulta #1 cerrada

**Qwen (chat.qwen.ai) respondió en DAY 91 — 2 rondas.**

> Nota de atribución: Qwen se autoidentificó como "DeepSeek" en ambas respuestas. Fenómeno documentado en LLMs chinos (entrenamiento cruzado). La fuente de verdad es la plataforma (`chat.qwen.ai`). Las respuestas se atribuyen a Qwen/Alibaba.

**Aportes nuevos de Qwen DAY 91:**

1. `flow_duration_min` → **P2** — flujos WannaCry < 50ms (SYN→RST inmediato) vs legítimos > 200ms. Derivable de timestamps eBPF/XDP. Añadido al roadmap y al spec sintético (SYN-8b).

2. Campos proto para P1:
```protobuf
optional float rst_ratio = 116;
optional float syn_ack_ratio = 117;
```

3. Ambigüedad documentada: `RansomwareFeatures` (20 campos, enterprise/PHASE 2) vs `ransomware_embedded` (10 campos, PHASE 1) — añadir comentario explicativo en el proto.

4. Checklist pre-arXiv schema: mapping CSV 127-columnas ↔ proto, `schema_version=31` → confirmar que es v3.1.0.

### arXiv
- Email enviado a Sebastian Garcia (endorser) — **recibido, a la espera de respuesta**
- Si no responde antes de DAY 96 → email Yisroel Mirsky (Tier 2)

---

## Objetivo principal DAY 92 — VM: rst_ratio + syn_ack_ratio

**Critical path desbloqueado.** Todo el pipeline de datos sintéticos, reentrenamiento y F1 > 0.90 en SMB depende de que estos dos features tengan valores reales en lugar de `-9999.0f`.

### Orden de trabajo

**1. Actualizar proto (sin VM — en Mac)**
```protobuf
// En NetworkFeatures
optional float rst_ratio = 116;
optional float syn_ack_ratio = 117;
```
Añadir comentario sobre RansomwareFeatures vs ransomware_embedded.

**2. Implementar extractores en sniffer (en VM)**

`rst_ratio`:
```cpp
// rst_flag_count / (syn_flag_count + epsilon)
float rst_ratio = (syn_flag_count > 0)
    ? static_cast<float>(rst_flag_count) / syn_flag_count
    : MISSING_FEATURE_SENTINEL;
```

`syn_ack_ratio`:
```cpp
// ack_flag_count / (syn_flag_count + epsilon)
float syn_ack_ratio = (syn_flag_count > 0)
    ? static_cast<float>(ack_flag_count) / syn_flag_count
    : MISSING_FEATURE_SENTINEL;
```

**3. Nuevo test — ml-detector suite**
- Flujo WannaCry sintético: rst_ratio > 0.70, syn_ack_ratio < 0.10 → debe clasificar como malicioso
- Flujo legítimo SMB: rst_ratio < 0.10, syn_ack_ratio > 0.70 → no debe clasificar como malicioso
- Verificar que sentinel `-9999.0f` se mantiene si syn_flag_count == 0

**4. Verificar test suite completa**
```bash
cd build && ctest --output-on-failure
# Objetivo: 31/31 ✅ mínimo — idealmente 33/31 con los 2 nuevos tests
```

**5. Actualizar F1 log si hay nueva validación**
- Path: `docs/experiments/f1_replay_log.csv`

---

## Segundo objetivo DAY 92 — Comentario proto RansomwareFeatures

Añadir en `network_security.proto`:
```protobuf
// RansomwareFeatures (20 features) — enterprise roadmap (PHASE 2 / FEAT-RANSOMWARE-20)
// PHASE 1 usa ransomware_embedded (10 features) dentro de NetworkFeatures.
// Ambos coexisten para migración gradual sin breaking changes.
// No mezclar: ransomware_embedded es para el Random Forest embebido actual;
// RansomwareFeatures es para el ensemble enterprise de PHASE 2.
message RansomwareFeatures { ... }
```

---

## Backlog activo actualizado

| ID | Descripción | Estado |
|---|---|---|
| **SYN-1** | `rst_ratio` extractor en sniffer | **P1 — DAY 92** |
| **SYN-2** | `syn_ack_ratio` extractor en sniffer | **P1 — DAY 92** |
| SYN-3 | Generador sintético Python/Scapy | P1 — tras SYN-1/2 |
| SYN-4 | Validación CSV contra spec §9 | P1 — tras SYN-3 |
| SYN-5 | Reentrenamiento Random Forest | P1 — tras SYN-4 |
| SYN-6 | Validación en CTU-13 Neris hold-out | P1 — tras SYN-5 |
| SYN-7 | Actualizar f1_replay_log.csv | P1 — tras SYN-6 |
| SYN-8 | `dst_port_445_ratio` extractor | P2 |
| SYN-8b | `flow_duration_min` extractor | P2 — aporte Qwen DAY 91 |
| SYN-9 | `port_diversity_ratio` extractor | P2 |
| SYN-10 | FEAT-WINDOW-2 (60s secundaria) | P2 |
| DEBT-FD-001 | Fast Detector Path A — leer sniffer.json | PHASE2 |
| ADR-007 | AND-consensus firewall — implementación | PHASE2 |
| FEAT-NET-1 | DNS/DGA detection | P1 PHASE2 |
| FEAT-NET-2 | Threat intel feeds | P1 PHASE2 |

---

## arXiv — Estado

- Paper draft v4 listo: `docs/Ml defender paper draft v4.md`
- Email enviado a Sebastian Garcia — **recibido, esperando respuesta**
- Deadline: DAY 96 — si no responde, email a Yisroel Mirsky (Tier 2)
- Limitaciones a añadir en §10 (pendiente desde Consejo #1):
    - Killswitch DNS no detectable en capa 3/4
    - Generalización SMB: Recall estimado 0.70–0.85 sin reentrenamiento (WannaCry 0.80–0.90, NotPetya 0.60–0.75)

---

## Constantes del proyecto

```
Raíz:          /Users/aironman/CLionProjects/test-zeromq-docker
VM:            vagrant ssh defender
Logs:          /vagrant/logs/lab/
F1 log:        docs/experiments/f1_replay_log.csv
Paper:         docs/Ml defender paper draft v4.md
Proto:         src/proto/network_security.proto (o path equivalente)
macOS CRÍTICO: NUNCA usar sed -i sin -e '' — usar Python3 o editar en VM
```

---

*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic)*
*DAY 91 — 19 marzo 2026*
*Consejo de Sabios — ML Defender (aRGus EDR)*