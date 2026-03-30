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

### Day 102 (30 Mar 2026) — ADR-012 PHASE 1b COMPLETA + TEST-PLUGIN-INVOKE-1

**TEST-PLUGIN-INVOKE-1 ✅**

Test unitario en `plugin-loader/tests/test_invoke_all.cpp`:
- `PacketContext` sintético → `invoke_all()` → `invocations=1, errors=0, overruns=0`
- Valida hot path completo: load → init → invoke → shutdown
- CTest: 100% passed, 0 failed out of 1
- Tests totales: **25/25 ✅** (nuevo récord)

**ADR-012 PHASE 1b — firewall-acl-agent ✅**

Plugin-loader integrado con guard `#ifdef PLUGIN_LOADER_ENABLED`:
- `firewall-acl-agent/CMakeLists.txt`: find_library + link + define
- `firewall-acl-agent/src/main.cpp`: unique_ptr<PluginLoader> + load/shutdown
- `firewall-acl-agent/config/firewall.json`: sección `plugins` con hello plugin
- Smoke test: `[plugin:hello] init OK` + `shutdown OK` — invocations=0

**ADR-012 PHASE 1b — rag-ingester ✅**

Plugin-loader integrado:
- `rag-ingester/CMakeLists.txt`: find_library + link + define
- `rag-ingester/src/main.cpp`: unique_ptr<PluginLoader> + load/shutdown
- `rag-ingester/config/rag-ingester.json`: sección `plugins` con hello plugin
- Smoke test: 1 plugin cargado OK

**ADR-012 PHASE 1b — rag-security ✅**

Plugin-loader integrado (variante: g_plugin_loader global por signal handler):
- `rag/CMakeLists.txt`: find_library + link + define
- `rag/src/main.cpp`: g_plugin_loader global + load/shutdown (signal handler + graceful)
- `rag/config/rag-config.json`: sección `plugins` con hello plugin
- Smoke test: 1 plugin cargado OK

**arXiv endorser — Andrés Caro Lindo confirmado ✅**

Prof. Andrés Caro Lindo (Cátedra INCIBE-UEx-EPCC) respondió positivamente.
Ofrece endorsement + posible colaboración futura (revistas, congresos).
Llamada telefónica acordada: jueves 2 abril. Tel: 657 33 10 10.

---

### Day 101 (29 Mar 2026) — ADR-012 PHASE 1b bug fix + ml-detector plugin-loader

**fix(plugin-loader): extract_enabled_objects ✅**
**ADR-012 PHASE 1b — sniffer ✅**
**ADR-012 PHASE 1b — ml-detector ✅**
**Tests: 24/24 suites ✅**

---

### Day 100 (28 Mar 2026) — ADR-021 + ADR-022 + set_terminate() + CI honesto

**set_terminate() en los 6 main() — ADR-022 fail-closed ✅**
**ADR-021 — deployment.yml SSOT + seed families ✅**
**ADR-022 — Threat model formal + Opción 2 descartada ✅**
**CI reescrito — honesto y funcional ✅**

---

### Day 99 (27 Mar 2026) — contexts.hpp + TEST-INTEG + fail-closed

**ADR-013 PHASE 2 completado — contextos HKDF simétricos ✅**
**TEST-INTEG-1/2/3 — gate arXiv ✅**
**Fail-closed EventLoader + RAGLogger ✅**

---

### Day 98 — CryptoTransport migración 6/6
### Day 97 — CryptoTransport HKDF + libsodium 1.0.19
### Day 96 — seed-client + Makefile dep order
### Day 95 — Cryptographic Provisioning Infrastructure
### Day 93 — ADR-012 PHASE 1: plugin-loader + ABI validation
### Day 83 — Ground truth bigFlows + CSV E2E
### Days 76–82 — Proto3 · Sentinel · F1=0.9985 · DEBT-FD-001
### Days 63–75 — Pipeline 6/6 · ChaCha20 · FAISS · HMAC · trace_id
### Days 1–62 — Foundation: eBPF/XDP · protobuf · ZMQ · RandomForest C++20

---

## 🔄 PRÓXIMO MILESTONE — arXiv submission

### P1 — Antes de enviar el paper

| ID | Tarea | Estado |
|----|-------|--------|
| MAKEFILE-RAG | Alinear rag-build al patrón estándar (cmake + PROFILE) | ⏳ DAY 103 |
| MAKEFILE-RAG | Añadir rag-attach (tmux attach -t rag-security) | ⏳ DAY 103 |
| MAKEFILE-RAG | Añadir rag tests a test-components y test-all | ⏳ DAY 103 |
| MAKEFILE-RAG | Añadir rag-build a build-unified y pipeline-* | ⏳ DAY 103 |
| PAPER-ADR022 | §6 subsección HKDF Context Symmetry case study | ⏳ DAY 103 |
| BARE-METAL | Stress test sin VirtualBox — validar ≥100 Mbps | ⏳ |
| PAPER-FINAL | Actualizar métricas DAY 102 (25/25 tests, PHASE 1b completa) | ⏳ |
| DOCS-APPARMOR | 6 perfiles AppArmor por componente | ⏳ |

### P2 — Post-arXiv, pre-FASE 3

| ID | Tarea | Origen |
|----|-------|--------|
| DEBT-CRYPTO-003a | `mlock()` seed_client.cpp | ADR-022 threat model |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie (libsodium 1.0.19 en apt) | P2 |
| DEBT-INFRA-002 | Sustituir `haveged` por `rng-tools5` + hardware RNG | P2 |
| FEAT-ROTATION-1 | `provision.sh rotate-all` + política SEED_ROTATION_DAYS | P2 |
| DEBT-NAMING-001 | `libseed_client` → `libseedclient` (sin underscore) | P3 |

### FASE 3 — Post-arXiv (requiere revisión Consejo)

| ID | Tarea |
|----|-------|
| ADR-021 impl. | `deployment.yml` SSOT + families en `provision.sh` |
| MULTI-VM | Vagrantfile multi-VM con topología distribuida real |
| CI-FULL | Self-hosted runner "argus-debian-bookworm" |
| ANSIBLE | Receta Ansible + Jinja2 (patrón Ericsson) |

---

## 🔐 BACKLOG CRIPTOGRÁFICO ACTIVO

### DEBT-CRYPTO-003a — mlock() sobre buffer del seed (🟡 P2)

```cpp
mlock(seed_.data(), seed_.size());
```
**Decisión consolidada (Consejo DAY 97):** WARNING + log instructivo, no error fatal.

### FEAT-ROTATION-1 — Política de rotación de seeds (🟡 P2 — DAY 105+)

- `SEED_ROTATION_DAYS=30` — SSOT en provision.sh
- Hot-reload descartado (split-brain risk) — ENT-4 para PHASE 3+.

### FEAT-CRYPTO-2 — Handshake efímero Noise (P3 — PHASE 2)
### FEAT-CRYPTO-3 — TPM 2.0 / HSM enterprise (P3 — ENT-8)

---

## 📋 DOCUMENTACIÓN PENDIENTE

### DOCS-2 — Perfiles AppArmor por componente (6 perfiles) ⏳

---

## 📋 BACKLOG — COMMUNITY & FEATURES

### 🟥 P0 — Paper arXiv
- Draft v6 ✅ · LaTeX ✅
- Sebastian Garcia (CTU Prague) ✅ — respondió, recibió PDF
- Yisroel Mirsky (BGU) ⏳ — enviado DAY 96, sin respuesta
- Andrés Caro Lindo (UEx) ✅ — endorsement confirmado, llamada jueves 2 abril
- Pendiente: §6 HKDF case study + métricas DAY 102

### 🟧 P1 — Fast Detector Config (DEBT-FD-001)
### 🟨 P2 — Expansión ransomware (prerequisito: DEBT-FD-001)
### 🟨 P2 — Pipeline reentrenamiento
### 🟩 P3 — Enterprise (ENT-1..8)

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:           ████████████████████ 100% ✅
HMAC Infrastructure:                  ████████████████████ 100% ✅
Proto3 Pipeline Stability:            ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
CSV Pipeline:                         ████████████████████ 100% ✅
Paper arXiv (draft v6 + LaTeX):       ████████████████████ 100% ✅
Cryptographic Provisioning PHASE 1:   ████████████████████ 100% ✅  DAY 95
seed-client (libseed_client):         ████████████████████ 100% ✅  DAY 96
libsodium 1.0.19 + provision.sh:      ████████████████████ 100% ✅  DAY 97
CryptoTransport (HKDF+nonce+AEAD):    ████████████████████ 100% ✅  DAY 97
contexts.hpp (HKDF simétricos):       ████████████████████ 100% ✅  DAY 99
TEST-INTEG-1/2/3 (gate arXiv):        ████████████████████ 100% ✅  DAY 99
set_terminate() 6/6 main():           ████████████████████ 100% ✅  DAY 100
ADR-021 (topology SSOT + families):   ████████████████████ 100% ✅  DAY 100
ADR-022 (threat model + Opción 2):    ████████████████████ 100% ✅  DAY 100
CI honesto (ubuntu-latest):           ████████████████████ 100% ✅  DAY 100
plugin-loader ADR-012 PHASE 1b 5/5:   ████████████████████ 100% ✅  DAY 101-102
TEST-PLUGIN-INVOKE-1:                 ████████████████████ 100% ✅  DAY 102
MAKEFILE-RAG alignment:               ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 103
PAPER-ADR022 §6 (HKDF case study):    ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 103
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 pre-arXiv
DEBT-CRYPTO-003a (mlock seed):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2
DEBT-INFRA-001 (Debian Trixie):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 105+
DOCS-2 (AppArmor profiles):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 105+
Fast Detector Config (DEBT-FD-001):   ████░░░░░░░░░░░░░░░░  20% 🟡  PHASE 2
ENT-*:                                ░░░░░░░░░░░░░░░░░░░░   0% ⏳  largo plazo
```

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| Sentinel correctness | -9999.0f fuera del dominio ✅ | 79 |
| Algoritmo cifrado pipeline | ChaCha20-Poly1305 IETF unificado ✅ | 95 |
| HKDF context format | Contexto = CANAL, no componente ✅ | 99 |
| Nonce policy | Contador monotónico 96-bit atómico ✅ | 96-97 |
| Error handling | `throw` en todo el pipeline ✅ | 96 |
| Cifrado obligatorio | SIEMPRE. Sin flag `enabled`. CryptoTransport ✅ | 96-97 |
| Compresión obligatoria | SIEMPRE. Sin flag `enabled`. LZ4 ✅ | 96 |
| Orden operaciones | LZ4 → ChaCha20 (comprimir antes de cifrar) ✅ | 96 |
| libsodium versión | 1.0.19 desde fuente, SHA-256 verificado ✅ | 97 |
| CryptoManager | DEPRECADO — CryptoTransport lo sustituye ✅ | 97-98 |
| Opción 2 multi-instancia | DESCARTADA — reproduce bug asimetría por diseño ✅ | 100 |
| set_terminate() | fail-closed en los 6 main(). abort() ante excepción no capturada ✅ | 100 |
| CI GitHub Actions | Solo validación estática (ubuntu-latest). Full build = self-hosted ✅ | 100 |
| Plugin-loader scope | global si signal handler necesita acceso, local si no ✅ | 102 |

---

### Notas del Consejo de Sabios

> DAY 102 (Consejo — feedback pendiente DAY 103):
> Preguntas abiertas: Q1 Makefile rag alignment · Q2 estructura §6 paper
> Q3 orden prioridades P1 pre-arXiv

> DAY 101 (Consejo — decisiones consolidadas):
> "Orden plugin-loader: firewall → rag-ingester → rag-security (unanimidad 5/5)"
> "TEST-PLUGIN-INVOKE-1 necesario antes de seguir con firewall (unanimidad 5/5)"
> "HKDF Context Symmetry → §6 subsección independiente (árbitro: Alonso)"
> — Grok · ChatGPT5 · DeepSeek · Qwen · Gemini

> DAY 100: "set_terminate() es defensa en profundidad correcta."
> DAY 97: "La rotación real de seeds es el mecanismo correcto de forward secrecy."
> DAY 96: "El diseño sigue RFC 5869 (HKDF), Signal Protocol y TLS 1.3."
> DAY 95: "El seed no es una clave — es material base. HKDF primero."

---

*Última actualización: DAY 102 — 30 Mar 2026*
*Branch: feature/bare-metal-arxiv*
*Tests: 25/25 suites ✅*
*ADR-012 PHASE 1b: 5/5 COMPLETA ✅*
*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic), Grok, ChatGPT, DeepSeek, Qwen, Gemini, Parallel.ai*

---

## 🔌 FEAT-PLUGIN-CRYPTO-1 — Plugin de CryptoTransport (⏳ PHASE 2 — post-arXiv)

**Objetivo:** Migrar el cifrado ChaCha20-Poly1305/HKDF del core de cada componente
a un plugin genérico `libplugin_crypto_transport.so`, configurado por identidad JSON.

### Diseño conceptual

```
config.json → component_id → SeedClient → HKDF(seed, CTX_canal) → ChaCha20-Poly1305
```

El plugin lee su identidad de `PluginConfig.config_json` (campo `component_id`),
selecciona el contexto HKDF correcto (`CTX_SNIFFER_TO_ML`, etc.) y gestiona
el cifrado/descifrado del mensaje de transporte.

### Problema arquitectónico identificado (DAY 102)

El hook actual `plugin_process_packet(PacketContext*)` opera en **capa de red**
(raw bytes, IPs, puertos). El cifrado opera en **capa de transporte ZMQ**
(payload protobuf + LZ4 serializado). Son capas distintas — se necesita un
nuevo hook:

```c
// Propuesta PHASE 2 — requiere PLUGIN_API_VERSION bump
PluginResult plugin_process_message(MessageContext* ctx);
// MessageContext: uint8_t* payload, size_t len, direction tx/rx
```

**Alternativa:** Ampliar `PacketContext` con campo `serialized_payload`.
Decisión pendiente — llevar al Consejo DAY 105+.

### Estrategia de migración (dual-mechanism obligatorio)

```
PHASE 2a: CryptoTransport directo (actual) + CryptoPlugin en paralelo
          Gate: TEST-INTEG-4 valida equivalencia cifrado/descifrado

PHASE 2b: CryptoTransport directo desactivado
          Gate: 72h sin regresiones en bare-metal

PHASE 2c: CryptoTransport eliminado del main.cpp de cada componente
          Código limpio — plugin es el único mecanismo
```

### Pregunta para el Consejo (DAY 105+)

> ¿Extendemos `plugin_api.h` con `MessageContext` para transporte (API limpia,
> breaking change), o ampliamos `PacketContext` con `serialized_payload`
> (no breaking, pero semánticamente impuro)?

### Estado

| Item | Estado |
|------|--------|
| plugin_api.h extensión (MessageContext) | ⏳ diseño pendiente |
| libplugin_crypto_transport.so | ⏳ no iniciado |
| Dual-mechanism en 5 componentes | ⏳ no iniciado |
| TEST-INTEG-4 equivalencia | ⏳ no iniciado |
| Consejo Q1 DAY 105+ | ⏳ pendiente |

**Prerequisitos:** arXiv submission ✅ · PHASE 2 plugin-loader auth (ADR-013)
**Estimación:** 5-7 días de desarrollo + 2-3 días de validación E2E
**Prioridad:** P2 PHASE 2 — post-arXiv