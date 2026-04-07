# ML Defender (aRGus NDR) — BACKLOG
## Via Appia Quality 🏛️

---

## 📐 Criterio de compleción

| Estado | Criterio |
|---|---|
| ✅ 100% | Implementado + probado en condiciones reales + resultado documentado |
| 🟡 80% | Implementado + compilando + smoke test pasado, sin validación E2E completa |
| 🟡 60% | Implementado parcialmente o con valores placeholder conocidos |
| ⏳ 0% | No iniciado |

---

## ✅ COMPLETADO

### Day 110 (7 Apr 2026) — PluginMode + PHASE 2c + Paper v13

**PluginMode + mode field (Q1 Consejo DAY 109) ✅**
- `plugin_api.h`: enum `PluginMode` (NORMAL=0, READONLY=1), `annotation[64]` restaurado,
  `mode uint8_t` consume 1 byte de `reserved[60]` → `reserved[59]`
- `plugin_loader.cpp`: D8-pre coherence check — READONLY+payload!=nullptr → std::terminate()
  `snap_mode` añadido a snapshot D8 y al invariant check
- `rag-ingester/src/main.cpp`: PHASE 2b reconstruida — PluginLoader init + ctx_readonly
  con `mode=PLUGIN_MODE_READONLY` antes de `embed_chronos()`, early return si result_code!=0
- `test_integ_4b.cpp`: TEST-INTEG-4b PASSED (Caso A: accept + Caso B: mode propagation)

**PHASE 2c — sniffer + plugin_process_message() (Q2 Consejo DAY 109) ✅**
- `ring_consumer.hpp`: `set_plugin_loader()` setter + `plugin_loader_` member
- `ring_consumer.cpp`: `invoke_all(ctx_msg)` en `process_raw_event()` con payload real,
  `mode=PLUGIN_MODE_NORMAL`, D8-v2 CRC32 activo, result_code!=0 → events_dropped++ + return
- `sniffer/src/userspace/main.cpp`: `set_plugin_loader(&plugin_loader_)` tras set_stats_interval()
- `sniffer.json`: hello plugin active=false (no exporta plugin_process_message — D1 OK)
- Pipeline: 6/6 RUNNING con binarios actualizados, sniffer log confirma D1 Graceful Degradation
- TEST-INTEG-4c: pendiente DAY 111

**Paper Draft v13 (Q3 Consejo DAY 109) ✅**
- §4 Integration Philosophy: 4 argumentos como enumerate LaTeX:
    1. Latencia determinista (<10ms, raw TCP vs HTTP/Kafka jitter)
    2. Superficie de ataque (parsers HTTP = CVEs; raw TCP >90% reducción)
    3. Sin broker = sin SPOF (Kafka/Redis incompatibles con host único 150-200 USD)
    4. Footprint mínimo (sin librdkafka, sin libcurl, sin boost.asio)
- Compilación limpia Overleaf confirmada
- Pendiente: Replace en arXiv cuando submit/7438768 sea anunciado

**Commits DAY 110:**
- Commit 1: feat(plugin-api): PluginMode + mode + PHASE 2b reconstruida + TEST-INTEG-4b
- Commit 2: feat(sniffer): PHASE 2c — plugin_process_message con payload real
- Push: ebc1d0e7..360faf8b feature/plugin-crypto

---

### Day 109 (6 Apr 2026) — FIX-A/B + PHASE 2b + D8-light READ-ONLY + Paper v12

**FIX-A — MLD_ALLOW_UNCRYPTED escape hatch ✅**
**FIX-B — provision.sh rag-security config dir ✅**
**PHASE 2b — rag-ingester plugin_process_message() ✅**
**D8-light READ-ONLY fix ✅**
**TEST-INTEG-4b — PASSED ✅**
**Paper Draft v12 ✅**

---

### Day 107–108 · Day 106 · Day 105 · Day 104 · Day 103–62
*(ver historial completo en git log)*

---

## 🔜 PRÓXIMO — TEST-INTEG-4c + PHASE 2d

### TEST-INTEG-4c — Gate PHASE 2c (sniffer)

Escribir `plugins/test-message/test_integ_4c.cpp`:
- Caso A: mode=NORMAL, payload real → PLUGIN_OK, errors==0
- Caso B: mode=NORMAL, plugin modifica campo read-only → D8 VIOLATION
- Caso C: mode=NORMAL, result_code=-1 → error registrado, no crash

Añadir al Makefile `plugin-integ-test` (tras 4b).
Gate: `make plugin-integ-test` verde (4a + 4b + 4c).

### ADR-023 PHASE 2d — ml-detector

Archivos: `ml-detector/src/main.cpp`, `ml-detector/CMakeLists.txt`
Contrato: payload post-inferencia, mode=PLUGIN_MODE_NORMAL, D8-v2 CRC32 activo.
Gate: TEST-INTEG-4d.

### ADR-023 PHASE 2e — rag-security

Requiere `g_plugin_loader` global para signal handler.
Gate: TEST-INTEG-4e.

---

## 📋 BACKLOG ACTIVO

### P1 — Deuda inmediata

| ID | Tarea | Origen |
|----|-------|--------|
| TEST-INTEG-4c | Gate PHASE 2c: sniffer + plugin con payload real | ADR-023 DAY 110 |
| TEST-INTEG-4d | Gate PHASE 2d: ml-detector | ADR-023 |
| TEST-INTEG-4e | Gate PHASE 2e: rag-security | ADR-023 |
| ADR-028 | RAG Ingestion Trust Model — escribir antes de primer plugin write-capable | ChatGPT5 DAY 108 |
| TEST-PROVISION-1 | Gate CI: vagrant destroy → up → 6/6 RUNNING | ChatGPT5 DAY 108 |
| DEBT-SNIFFER-SEED | Unificar sniffer bajo SeedClient (eliminar get_encryption_seed manual) | DAY 107 |
| arXiv Replace v13 | Subir main_v13.tex cuando submit/7438768 sea anunciado | DAY 110 |

### P1 — feature/plugin-crypto (branch activa)

| ID | Tarea |
|----|-------|
| ADR-025 | Plugin Integrity Verification (Ed25519 + TOCTOU-safe dlopen) — post PHASE 2 |
| ADR-024 impl | Noise_IKpsk3 dynamic key agreement — FASE 3 post-arXiv |

### P2 — Post-PHASE 2

| ID | Tarea | Origen |
|----|-------|--------|
| BARE-METAL-IMAGE | Imagen Debian Bookworm hardened — exportable a USB | P2 |
| BARE-METAL stress | tcpreplay 100/250/500/1000 Mbps en NIC físico | P2 |
| DEBT-FD-001 | Fast Detector Path A → JSON thresholds | DAY 80 |
| DEBT-CRYPTO-003a | mlock() en seed_client.cpp | ADR-022 |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie | P2 |
| FEAT-ROTATION-1 | provision.sh rotate-all + SEED_ROTATION_DAYS | P2 |
| DOCS-APPARMOR | 6 perfiles AppArmor por componente | P2 |

---

## 📊 Estado global del proyecto
Foundation + Thread-Safety:           ████████████████████ 100% ✅
HMAC Infrastructure:                  ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
CryptoTransport (HKDF+nonce+AEAD):    ████████████████████ 100% ✅  DAY 97
contexts.hpp (HKDF simétricos):       ████████████████████ 100% ✅  DAY 99
TEST-INTEG-1/2/3 (gate arXiv):        ████████████████████ 100% ✅  DAY 99
set_terminate() 6/6 main():           ████████████████████ 100% ✅  DAY 100
plugin-loader ADR-012 PHASE 1b 6/6:   ████████████████████ 100% ✅  DAY 101-102
ADR-023 PHASE 2a (firewall):          ████████████████████ 100% ✅  DAY 105-106 — TEST-INTEG-4a
ADR-023 PHASE 2b (rag-ingester):      ████████████████████ 100% ✅  DAY 109-110 — TEST-INTEG-4b
ADR-023 PHASE 2c (sniffer código):    ████████████████████ 100% ✅  DAY 110 — TEST-INTEG-4c ⏳
PluginMode + mode field (Q1-109):     ████████████████████ 100% ✅  DAY 110
D8-pre coherence check:               ████████████████████ 100% ✅  DAY 110
Paper v13 (§4 4 argumentos):          ████████████████████ 100% ✅  DAY 110
ADR-028 (RAG Ingestion Trust Model):  ████████████████████ 100% ✅  DAY 109 — APROBADO Consejo 5/5
D8-light READ-ONLY contract:          ████████████████████ 100% ✅  DAY 109
MLD_ALLOW_UNCRYPTED escape hatch:     ████████████████████ 100% ✅  DAY 109
provision.sh reproducible (destroy):  ████████████████████ 100% ✅  DAY 108
arXiv submitted:                      ████████████████████ 100% ✅  DAY 106 — submit/7438768
TEST-INTEG-4c (gate PHASE 2c):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 111
ADR-023 PHASE 2d (ml-detector):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-023 PHASE 2e (rag-security):      ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-025 impl. (plugin signing):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-PHASE 2 completa
ADR-028 escrito:                      ░░░░░░░░░░░░░░░░░░░░   0% ⏳  antes de write-capable plugins
ADR-024 impl. (Noise IK):             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  FASE 3 post-arXiv
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado por hardware
DEBT-FD-001 (JSON thresholds):        ████░░░░░░░░░░░░░░░░  20% 🟡  PHASE 2
TEST-PROVISION-1 (CI gate):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post PHASE 2

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| PluginMode field | flag explícito uint8_t, D8-pre enforce, Consejo 4/5 | 110 |
| PHASE 2c contrato | payload real, mode=NORMAL, D8-v2 CRC32 — Consejo 5/5 | 110 |
| Integration Philosophy | 4 argumentos formales: latencia, superficie, SPOF, footprint | 110 |
| ADR-028 timing | diferir hasta primer write-capable plugin — Grok+DeepSeek | 109 |
| Plugin rag-ingester trust | read-only, payload=nullptr — Consejo 4/5 | 108 |
| MLD_ALLOW_UNCRYPTED | escape hatch explícito dev, terminate prod | 109 |
| D8-light READ-ONLY | nullptr+0 es contrato legítimo, no violación | 109 |
| Integration protocol | raw TCP + protobuf + ChaCha20, sin HTTP/Kafka/WS | 109 |
| Threat model scope | NDR = vigilancia de red; USB/físico fuera por diseño | 109 |

---

### Notas del Consejo de Sabios

> DAY 110 — PluginMode + PHASE 2c + Paper v13:
> "Implementación pura. No se convocó Consejo.
> Todas las decisiones DAY 109 cerradas: Q1 (mode field) ✅, Q2 (payload real) ✅,
> Q3 (4 argumentos §4) ✅, Q4 (ADR-028 diferido) ✅.
> Incidente: 3 ficheros vacíos, resuelto con backups. PHASE 2b reconstruida correctamente.
> TEST-INTEG-4c pendiente DAY 111."
> — Claude (Anthropic) · Implementación DAY 110

---

*Última actualización: DAY 110 — 7 Apr 2026*
*Branch: feature/plugin-crypto*
*Tests: 25/25 + TEST-INTEG-4a 3/3 + TEST-INTEG-4b PASSED*
*Paper: Draft v13 ✅ · arXiv: submit/7438768 SUBMITTED ✅ (pendiente moderación)*
*Pipeline: 6/6 RUNNING ✅*
*PHASE 2a: ✅ · PHASE 2b: ✅ · PHASE 2c: ✅ (código) / ⏳ (TEST-INTEG-4c) · PHASE 2d/2e: ⏳*