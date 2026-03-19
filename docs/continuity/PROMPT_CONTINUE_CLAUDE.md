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

## Decisión de arquitectura DAY 91 — ADR-013

**ADR-013 redactado y aceptado:** Seed Distribution and Component Authentication.
Path: `docs/adr/ADR-013-seed-distribution-component-authentication.md`
**ADR-013 tiene precedencia** sobre cualquier documento anterior que describa
distribución de seeds o autenticación entre componentes.

### Resumen ejecutivo ADR-013

El seed ChaCha20 deja de ser responsabilidad de etcd-server. Pasa a generarse
**una única vez en el script de instalación bash** (`scripts/provision.sh`).

| Mecanismo | Responsable | Estado |
|---|---|---|
| ZMQ CURVE (canal cifrado) | ya en stack | sin cambios |
| ChaCha20 seed | `provision.sh` bash | nuevo — DAY 95-96 |
| HMAC-SHA256 CSVs | crypto-transport | sin cambios |
| etcd-server | solo ciclo de vida + hot-reload | refactor DAY 97+ |

**Nueva shared library:** `libs/seed-client` — lee y descifra `seed.enc` del
filesystem. Al estilo `libs/crypto-transport`. Sin red. Sin generación de seeds.

**rag-ingester y rag-security** fuera del seed — solo HMAC.

**Backlog CONGELADO** — no se añaden nuevos ítems salvo valor extraordinario.
A partir de DAY 92: implementar, no diseñar.

---

## Objetivo principal DAY 92 — rama nueva + proto + sniffer

### Paso 0 — Rama nueva
```bash
git checkout -b feature/smb-detection-features
```

### Paso 1 — Proto (sin VM, en Mac)

Verificar si `rst_ratio` y `syn_ack_ratio` ya tienen estructura propia en
`network_security.proto`. Si no la tienen, crearla:

```protobuf
// SMBScanFeatures — features de escaneo SMB (WannaCry/NotPetya)
message SMBScanFeatures {
    optional float rst_ratio            = 1;  // RST/SYN
    optional float syn_ack_ratio        = 2;  // ACK/SYN
    optional float flow_duration_min_ms = 3;  // P2 — flujos WannaCry < 50ms
}
```

Si no cabe estructura propia, añadir como opcionales en NetworkFeatures
(campos 116/117) y documentar por qué en comentario.

Añadir también el comentario de ambigüedad RansomwareFeatures:
```protobuf
// RansomwareFeatures (20 features) — enterprise roadmap (PHASE 2)
// PHASE 1 usa ransomware_embedded (10 features) dentro de NetworkFeatures.
// Ambos coexisten para migración gradual sin breaking changes.
```

### Paso 2 — Extractores en sniffer (VM)

```cpp
float rst_ratio = (syn_flag_count > 0)
    ? static_cast<float>(rst_flag_count) / syn_flag_count
    : MISSING_FEATURE_SENTINEL;

float syn_ack_ratio = (syn_flag_count > 0)
    ? static_cast<float>(ack_flag_count) / syn_flag_count
    : MISSING_FEATURE_SENTINEL;
```

### Paso 3 — Tests nuevos
- Flujo WannaCry sintético: `rst_ratio > 0.70`, `syn_ack_ratio < 0.10` → malicioso
- Flujo legítimo SMB: `rst_ratio < 0.10`, `syn_ack_ratio > 0.70` → benigno
- `syn_flag_count == 0` → sentinel `-9999.0f` en ambos

### Paso 4 — Test suite completa
```bash
cd build && ctest --output-on-failure
# Mínimo: 31/31 ✅ — objetivo: 33/31 con los 2 nuevos tests
```

---

## Secuencia de implementación acordada

```
DAY 92  — rama nueva + proto SMBScanFeatures + rst_ratio + syn_ack_ratio
DAY 93-94 — plugin-loader minimalista (ADR-012, SIN seed-client todavía)
            documentado explícitamente como "sin autenticación hasta seed-client"
DAY 95-96 — scripts/provision.sh bash + libs/seed-client
DAY 97+   — refactor etcd-server/etcd-client + datos sintéticos WannaCry
```

---

## Backlog activo (CONGELADO)

| ID | Descripción | Estado |
|---|---|---|
| **SYN-1** | `rst_ratio` extractor en sniffer | **P1 — DAY 92** |
| **SYN-2** | `syn_ack_ratio` extractor en sniffer | **P1 — DAY 92** |
| ADR-012 | plugin-loader minimalista (sin seed-client) | DAY 93-94 |
| provision.sh | Script bash generación keypairs + seeds | DAY 95-96 |
| seed-client | Mini-componente libs/seed-client | DAY 95-96 |
| etcd refactor | Eliminar responsabilidades criptográficas | DAY 97+ |
| SYN-3 | Generador sintético Python/Scapy | DAY 97+ |
| SYN-4 | Validación CSV contra spec §9 | tras SYN-3 |
| SYN-5 | Reentrenamiento Random Forest | tras SYN-4 |
| SYN-6 | Validación en CTU-13 Neris hold-out | tras SYN-5 |
| SYN-7 | Actualizar f1_replay_log.csv | tras SYN-6 |
| SYN-8 | `dst_port_445_ratio` extractor | P2 |
| SYN-8b | `flow_duration_min` extractor | P2 |
| SYN-9 | `port_diversity_ratio` extractor | P2 |
| SYN-10 | FEAT-WINDOW-2 (60s secundaria) | P2 |
| DEBT-FD-001 | Fast Detector Path A — leer sniffer.json | PHASE2 |
| ADR-007 | AND-consensus firewall — implementación | PHASE2 |

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