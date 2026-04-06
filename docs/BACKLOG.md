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

### Day 109 (6 Apr 2026) — FIX-A/B + PHASE 2b + D8-light READ-ONLY + Paper v12

**FIX-A — MLD_ALLOW_UNCRYPTED escape hatch ✅**
- 3 adaptadores `etcd_client.cpp` (ml-detector, sniffer, firewall-acl-agent)
- En dev (`MLD_ALLOW_UNCRYPTED=1`): `cerr FATAL[DEV]` + `return` (constructor void)
- En producción: `std::terminate()` garantizado
- 6/6 RUNNING verificado tras rebuild

**FIX-B — provision.sh rag-security config dir ✅**
- `mkdir -p /vagrant/rag-security/config` en sección rag-security
- Symlinks JSON automáticos si el directorio se crea en el momento
- Eliminado warning "Config dir no existe aún"
- Verificado: `provision.sh full` → "Config dir creado" sin errores

**PHASE 2b — rag-ingester plugin_process_message() ✅**
- `rag-ingester/src/main.cpp`: invocación antes de `embed_chronos()`
- Contrato READ-ONLY: `ctx_readonly.payload = nullptr`, `ctx_readonly.payload_len = 0`
- Early return si `result_code != 0` — evento no se ingesta en FAISS
- Plugin decide accept/reject sin acceso al payload (decisión Consejo 4/5 DAY 108)

**D8-light READ-ONLY fix — plugin_loader.cpp ✅**
- `is_readonly = (payload == nullptr && payload_len == 0)` — excepción legítima
- D8-light y CRC32 saltan si `is_readonly` — sin `std::terminate()` falso positivo
- Comentario de documentación del contrato en código

**TEST-INTEG-4b — PASSED ✅**
- `plugins/test-message/test_integ_4b.cpp` — variantes A (accept) y B (reject)
- Variante A: `result_code=0`, `payload_intact=YES` → PASS
- Variante B: D8 VIOLATION detectada, `payload_intact=YES` → PASS
- Integrado en `make plugin-integ-test` (ejecuta 4a + 4b)
- force-add necesario por `**/test_*` en `.gitignore`

**Paper Draft v12 ✅**
- `§3.4 Out-of-Scope`: vector físico/USB consciente · USB=responsabilidad IT · Wazuh complementario
- `§4 Integration Philosophy: Composability over Monolithism` (nueva subsección)
- FEAT-INT-1: raw TCP + protobuf + ChaCha20, graph quality motivation, ADR-024/028 gates
- `§11.16`: referencia corta a §4
- Compilación limpia Overleaf confirmada
- Pendiente: Replace en arXiv cuando submit/7438768 sea anunciado

**Commits DAY 109:**
- `da7355d8` — FIX-A + FIX-B
- `81ab2101` — Makefile plugin-integ-test (4a+4b)
- `d13b35d1` — PHASE 2b + D8-light READ-ONLY + test_integ_4b.cpp
- `[hash]`   — DAY 109 cierre (BACKLOG + README + prompts)

---

### Day 107–108 (4–5 Apr 2026) — MAC failure root cause + provision.sh + ADR-026/027

**DAY 107 — Root cause MAC verification failed ✅**
**DAY 108 — provision.sh reproducible + ADR-026/027 + gate PASO 4 verde ✅**
- PASO 1: Swap CTX_ETCD_TX/RX verificado empíricamente — necesario (ADR-027)
- PASO 2: Invariant fail-fast en 3 adaptadores
- PASO 3: provision.sh 9 fixes
- PASO 4: `vagrant destroy && vagrant up` → 6/6 RUNNING sin intervención ✅

**Consejo de Sabios DAY 108 — 5/5 respondieron ✅**

---

### Day 106 (3 Apr 2026) — PHASE 2a CERRADA + TEST-INTEG-4a-PLUGIN + arXiv submitted
### Day 105 (2 Apr 2026) — PHASE 2a + Paper v10 + arXiv cuenta
### Day 104 (1 Apr 2026) — Paper v9 + ADR-023 + ADR-024 + Consejo 2 rondas
### Day 103–102–101–100–99–98–97–96–95–93–83–76–82–63–75–1–62
*(ver historial completo en git log)*

---

## 🔜 PRÓXIMO — PHASE 2c (sniffer)

### ADR-023 PHASE 2c — sniffer + plugin_process_message()

**Pregunta abierta (pendiente Consejo DAY 109):** ¿contrato READ-ONLY o payload real?
El sniffer tiene acceso al payload del paquete en el punto de invocación.

**Archivos a tocar:**
- `sniffer/src/userspace/main.cpp`
- `sniffer/CMakeLists.txt`
- `sniffer/config/sniffer.json`

**Gate:** TEST-INTEG-4c

**Secuencia completa PHASE 2:**
- PHASE 2a — firewall-acl-agent ✅ (DAY 105-106)
- PHASE 2b — rag-ingester ✅ (DAY 109) — READ-ONLY
- PHASE 2c — sniffer ⏳ — contrato TBD (Consejo DAY 109)
- PHASE 2d — ml-detector ⏳
- PHASE 2e — rag-security ⏳ (`g_plugin_loader` global para signal handler)

---

## 📋 BACKLOG ACTIVO

### P1 — Consejo DAY 109 (pendiente respuestas)

| ID | Pregunta | Estado |
|----|----------|--------|
| Q1-109 | D8-light READ-ONLY: flag explícito vs inferencia por nullptr | Pendiente |
| Q2-109 | PHASE 2c sniffer: READ-ONLY vs payload real | Pendiente |
| Q3-109 | Integration Philosophy §4: justificación raw TCP suficiente | Pendiente |
| Q4-109 | ADR-028: ¿bloqueante antes de PHASE 2c o sólo antes de write-capable plugins? | Pendiente |

### P1 — Deuda inmediata

| ID | Tarea | Origen |
|----|-------|--------|
| TEST-INTEG-4c | Gate PHASE 2c: sniffer + plugin | ADR-023 |
| TEST-INTEG-4d | Gate PHASE 2d: ml-detector | ADR-023 |
| TEST-INTEG-4e | Gate PHASE 2e: rag-security | ADR-023 |
| ADR-028 | RAG Ingestion Trust Model — antes de write-capable plugins | ChatGPT5 DAY 108 |
| TEST-PROVISION-1 | Gate CI: vagrant destroy → up → 6/6 RUNNING | ChatGPT5 DAY 108 |
| DEBT-SNIFFER-SEED | Unificar sniffer bajo SeedClient (eliminar get_encryption_seed manual) | DAY 107 |
| arXiv Replace v12 | Subir main_v12.tex cuando submit/7438768 sea anunciado | DAY 109 |

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
| DEBT-CRYPTO-003a | `mlock()` en seed_client.cpp | ADR-022 |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie | P2 |
| FEAT-ROTATION-1 | `provision.sh rotate-all` + SEED_ROTATION_DAYS | P2 |
| DOCS-APPARMOR | 6 perfiles AppArmor por componente | P2 |

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:           ████████████████████ 100% ✅
HMAC Infrastructure:                  ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
CryptoTransport (HKDF+nonce+AEAD):    ████████████████████ 100% ✅  DAY 97
contexts.hpp (HKDF simétricos):       ████████████████████ 100% ✅  DAY 99
TEST-INTEG-1/2/3 (gate arXiv):        ████████████████████ 100% ✅  DAY 99
set_terminate() 6/6 main():           ████████████████████ 100% ✅  DAY 100
plugin-loader ADR-012 PHASE 1b 6/6:   ████████████████████ 100% ✅  DAY 101-102
ADR-023 PHASE 2a (firewall):          ████████████████████ 100% ✅  DAY 105 — TEST-INTEG-4a
ADR-023 PHASE 2b (rag-ingester):      ████████████████████ 100% ✅  DAY 109 — TEST-INTEG-4b
D8-light READ-ONLY contract:          ████████████████████ 100% ✅  DAY 109
MLD_ALLOW_UNCRYPTED escape hatch:     ████████████████████ 100% ✅  DAY 109
provision.sh rag-security config:     ████████████████████ 100% ✅  DAY 109
Paper v12 (threat model + int. phil): ████████████████████ 100% ✅  DAY 109
provision.sh reproducible (destroy):  ████████████████████ 100% ✅  DAY 108
arXiv submitted:                      ████████████████████ 100% ✅  DAY 106 — submit/7438768
ADR-023 PHASE 2c (sniffer):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  SIGUIENTE
ADR-023 PHASE 2d (ml-detector):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-023 PHASE 2e (rag-security):      ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-025 impl. (plugin signing):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-PHASE 2 completa
ADR-028 (RAG Ingestion Trust Model):  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  antes de write-capable plugins
ADR-024 impl. (Noise IK):             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  FASE 3 post-arXiv
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado por hardware
DEBT-FD-001 (JSON thresholds):        ████░░░░░░░░░░░░░░░░  20% 🟡  PHASE 2
TEST-PROVISION-1 (CI gate):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post PHASE 2b
```

---

## 🔑 Decisiones de diseño consolidadas

*(todas las anteriores más:)*

| Decisión | Resolución | DAY |
|---|---|---|
| Plugin rag-ingester trust | read-only, payload=nullptr — Consejo 4/5 ✅ | 108 |
| MLD_ALLOW_UNCRYPTED | escape hatch explícito dev, terminate prod ✅ | 109 |
| D8-light READ-ONLY | nullptr+0 es contrato legítimo, no violación ✅ | 109 |
| Integration protocol | raw TCP + protobuf + ChaCha20, sin HTTP/Kafka/WS ✅ | 109 |
| Threat model scope | NDR = vigilancia de red; USB/físico fuera por diseño ✅ | 109 |

---

### Notas del Consejo de Sabios

> DAY 109 — FIX-A/B + PHASE 2b + Paper v12:
> "Q1: D8-light READ-ONLY flag vs nullptr inference. Q2: sniffer PHASE 2c contrato.
> Q3: Integration Philosophy raw TCP justificación académica. Q4: ADR-028 timing.
> TEST-INTEG-4b PASSED. Paper v12 compilación limpia."
> — Pendiente respuestas Consejo

> DAY 108 — provision.sh formalizado + ADR-026/027:
> "Q1: terminate prod + MLD_ALLOW_UNCRYPTED dev — aprobado. Q2: rebuild limpio unánime.
> Q3: rag-ingester plugin read-only (ctx_readonly.payload=nullptr) — 4/5.
> Q4: rag-security/config crear en provision.sh — unánime.
> Nuevos: TEST-PROVISION-1 (ChatGPT5), ADR-028 RAG Ingestion Trust Model (ChatGPT5)"
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (auto-identifica DeepSeek)

---

*Última actualización: DAY 109 — 6 Apr 2026*
*Branch: feature/plugin-crypto*
*Tests: 25/25 + TEST-INTEG-4a 3/3 + TEST-INTEG-4b PASSED*
*Paper: Draft v12 ✅ · arXiv: submit/7438768 SUBMITTED ✅ (pendiente moderación)*
*Pipeline: 6/6 RUNNING reproducible desde cero ✅*
*PHASE 2a: ✅ · PHASE 2b: ✅ · PHASE 2c/2d/2e: ⏳*