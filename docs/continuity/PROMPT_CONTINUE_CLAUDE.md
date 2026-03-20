# ML Defender — Prompt de Continuidad DAY 93
## 21 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING (etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall)
**Test suite:** 33/31 ✅ (crypto 3/3, etcd-hmac 12/12, ml-detector 9/9, rag-ingester 7/7, sniffer 1/1 NEW)
**Rama activa:** `feature/smb-detection-features`
**Último tag:** DAY 92

---

## Lo que se hizo en DAY 92

### Flujo A — Documentación (mañana)

**Rama `feature/documentation-contracts`** — mergeada a main.

Tres documentos vivos en `docs/contracts/`:
- `protobuf-contract.md` — `network_security.proto` documentado completo (13 secciones, todas las tablas de campos, dual-score, ADR-002, ambigüedad RansomwareFeatures, spec SMB pendiente)
- `json-contracts.md` — contratos JSON de los 5 componentes con thresholds, sockets ZMQ, paths de logs y diagrama ASCII del pipeline
- `rag-security-commands.md` — whitelist completo de comandos, patrones regex, claves restringidas, referencia de cada comando con ejemplos JSON

### Flujo B — SMB Detection Features

**Rama `feature/smb-detection-features`** — activa.

**Proto (`protobuf/network_security.proto`):**
- Nuevo mensaje `SMBScanFeatures` (líneas 87-97):
  - `rst_ratio` (campo 1) — SYN-1, P1
  - `syn_ack_ratio` (campo 2) — SYN-2, P1
  - `flow_duration_min_ms` (campo 3) — SYN-8b, P2
- `NetworkFeatures.smb_scan = field 116`
- Comentario de ambigüedad `RansomwareFeatures` vs `RansomwareEmbeddedFeatures` (línea 663)

**Extractor (`sniffer/src/userspace/ml_defender_features.cpp`, líneas 939-950):**
```cpp
auto* smb = net_features->mutable_smb_scan();
const float syn_count = static_cast<float>(flow.syn_count);
if (syn_count > 0.0f) {
    smb->set_rst_ratio(rst_count / syn_count);
    smb->set_syn_ack_ratio(ack_count / syn_count);
} else {
    smb->set_rst_ratio(MISSING_FEATURE_SENTINEL);    // -9999.0f
    smb->set_syn_ack_ratio(MISSING_FEATURE_SENTINEL);
}
```

**Tests (`sniffer/tests/test_smb_scan_features.cpp`):**
- Test 1: WannaCry sintético — `rst_ratio > 0.70`, `syn_ack_ratio < 0.10` → MALICIOUS ✅
- Test 2: SMB legítimo — `rst_ratio < 0.10`, `syn_ack_ratio > 0.70` → BENIGN ✅
- Test 3: `syn_flag_count == 0` → sentinel `-9999.0f` en ambos campos ✅

**`sniffer/CMakeLists.txt`:** `enable_testing()` añadido + `test_smb_scan_features` registrado.

**Suite completa:** 33/31 ✅ — zero regresiones.

---

## Backlog activo — estado actualizado

| ID | Descripción | Estado |
|---|---|---|
| **SYN-1** | `rst_ratio` extractor en sniffer | ✅ DONE — DAY 92 |
| **SYN-2** | `syn_ack_ratio` extractor en sniffer | ✅ DONE — DAY 92 |
| **ADR-012** | plugin-loader minimalista (sin seed-client) | **P1 — DAY 93-94** |
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

## Objetivo principal DAY 93 — ADR-012 plugin-loader minimalista

### Contexto ADR-012

Plugin-loader minimalista **sin seed-client todavía** — documentado explícitamente como "sin autenticación hasta seed-client (DAY 95-96)".

El objetivo es un mecanismo de carga dinámica de componentes/plugins ligero, que no introduzca complejidad criptográfica antes de que `libs/seed-client` esté listo.

### Punto de partida

Antes de implementar, revisar el ADR-012 existente:
```bash
cat docs/adr/ADR-012-*.md
```

Si no existe aún como fichero:
```bash
ls docs/adr/
```

### Secuencia acordada

```
DAY 93-94 — plugin-loader minimalista (ADR-012, SIN seed-client todavía)
            documentado explícitamente como "sin autenticación hasta seed-client"
DAY 95-96 — scripts/provision.sh bash + libs/seed-client
DAY 97+   — refactor etcd-server/etcd-client + datos sintéticos WannaCry
```

---

## arXiv — Estado

- Paper draft v4 listo: `docs/argus_ndr_v5.pdf`
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
F1 log:        /vagrant/docs/experiments/f1_replay_log.csv
Paper:         /vagrant/docs/Ml defender paper draft v4.md
Proto:         /vagrant/protobuf/network_security.proto
macOS CRÍTICO: NUNCA usar sed -i sin -e '' — usar Python3 o editar en VM
```

---

*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic)*
*DAY 92 — 20 marzo 2026*
*Consejo de Sabios — ML Defender (aRGus EDR)*