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

### Day 107–108 (4–5 Apr 2026) — MAC failure root cause + provision.sh formalizado + ADR-026/027

**DAY 107 — Root cause MAC verification failed ✅**
- Root cause: `component_config_path` no seteado en `etcd_client.cpp` de ml-detector,
  sniffer y firewall → `tx_` null → datos en claro → MAC failure garantizado
- Fix: `component_config_path` seteado en los 3 adaptadores
- Fix: `get_encryption_seed()` reescrito en sniffer para leer seed.bin local
- Fix confirmado: swap `CTX_ETCD_TX/RX` en `etcd-server/src/component_registry.cpp`
  necesario y correcto (verificado DAY 108 PASO 1)
- Pipeline 6/6 RUNNING al cierre de sesión

**DAY 108 — provision.sh reproducible + ADR-026/027 + gate PASO 4 verde ✅**

- **PASO 1**: Swap CTX_ETCD_TX/RX verificado empíricamente — necesario (ADR-027)
- **PASO 2**: Invariant fail-fast en 3 adaptadores `etcd_client.cpp`:
  `if (config_.encryption_enabled && config.component_config_path.empty()) std::terminate()`
- **PASO 3**: provision.sh formalizado — 9 fixes:
    - `create_component_dir`: chmod 755 + chown root:vagrant (era 700 root:root)
    - `generate_seed`: chmod 640 + chown root:vagrant (era 600 root:root)
    - Seed maestro: etcd-server/seed.bin distribuido a los 5 componentes
    - Symlinks JSON automáticos: /etc/ml-defender/*/ → /vagrant/*/config/*.json
    - libsodium compat: ln -sf libsodium.so.26 libsodium.so.23 + ldconfig
    - `install_shared_libs()`: build + install seed-client, crypto-transport,
      plugin-loader, etcd-client, libsnappy
    - `check_dependencies()`: tmux añadido con auto-install
    - libcrypto_transport: rebuild automático si fecha < hoy
- **PASO 4**: `vagrant destroy && vagrant up` → 6/6 RUNNING sin intervención manual ✅
- **ADR-026**: Arquitectura P2P Fleet Federated — formaliza Consejo DAY 104
- **ADR-027**: CTX_ETCD_TX/RX swap — principio mirror cliente/servidor documentado

**Consejo de Sabios DAY 108 — 5 revisores (Qwen auto-identifica DeepSeek, patrón DAY 103-108) ✅**
- Q1: `std::terminate()` prod + `MLD_ALLOW_UNCRYPTED` dev (no `MLD_DEV_MODE`) — aprobado
- Q2: rebuild limpio siempre en `install_shared_libs()` — unánime
- Q3: plugin rag-ingester read-only (`ctx_readonly.payload = nullptr`) — 4/5
- Q4: `rag-security/config` crear en provision.sh — unánime
- Nuevos ítems: TEST-PROVISION-1 (ChatGPT5), ADR-028 RAG Ingestion Trust Model (ChatGPT5)

### Day 106 (3 Apr 2026) — PHASE 2a CERRADA + TEST-INTEG-4a-PLUGIN + arXiv submitted

**PHASE 2a — 4 condiciones Consejo DAY 105: TODAS CERRADAS ✅**

**1c — nonce/tag NULL contract en `plugin_api.h` ✅**
- Production guarantee: `nonce != NULL && tag != NULL`
- Test/config mode (`--test-config`, `MLD_DEV_MODE`): MAY be NULL
- Plugins MUST check for NULL before dereferencing

**1d — Makefile deps: `plugin-loader-build` en 4 componentes restantes ✅**
- `sniffer`, `ml-detector`, `rag-ingester`, `rag-build` ahora dependen de `plugin-loader-build`
- `firewall-acl-agent` ya lo tenía desde DAY 105
- Los 5 componentes con plugins correctamente declarados

**1a — D8-v2: CRC32 payload integrity en `plugin_loader.cpp` ✅**
- `crc32_fast()` implementado como función estática (CRC-32/ISO-HDLC, 0xEDB88320)
- Guarded por `#ifdef MLD_ALLOW_DEV_MODE` (debug builds únicamente)
- Snapshot antes de invocación, comparación después
- CRC mismatch → `SECURITY D8-v2` log + `stats_.errors++`
- Cierra el gap del D8 existente (pointer identity → content integrity)

**1b — TEST-INTEG-4a-PLUGIN: 3/3 variantes PASSED ✅**
- `plugins/test-message/plugin_test_message.cpp` — variantes A/B/C via `MLD_TEST_VARIANT`
- `plugins/test-message/CMakeLists.txt` — builds `libplugin_test_message.so`
- `plugins/test-message/test_variants.cpp` — runner standalone
- `plugins/test-message/test_config.json` — config aislado (no toca producción)
- `make plugin-integ-test` integrado en suite `test-libs`

Resultados:
```
Variant A: errors=0, result_code=0       → PASS
Variant B: D8 VIOLATION detectada        → PASS
Variant C: PLUGIN_ERROR, sin crash       → PASS
TEST-INTEG-4a-PLUGIN: PASSED (0 failures)
```

**arXiv submission ✅**
- Draft v11: UEx eliminada, 3 figuras TikZ, spec hardware corregida (Intel i9 2.5GHz 32GB DDR4)
- TDH repo link añadido: https://github.com/alonsoir/test-driven-hardening
- Submission ID: `7438768` — STATUS: submitted (pendiente moderación)
- Endorsers: Sebastian Garcia (CTU Prague) ✅, Andrés Caro Lindo (UEx/INCIBE) ✅

---

### Day 105 (2 Apr 2026) — PHASE 2a + Paper v10 + arXiv cuenta creada

**ADR-023 PHASE 2a — firewall-acl-agent + MessageContext ✅**

- `MessageContext` struct añadido a `plugin_api.h` (D2, D3, D7, D9, D11)
- `invoke_all(MessageContext&)` implementado en `plugin_loader.hpp/.cpp`
- Graceful Degradation D1: skip silencioso si plugin no exporta símbolo
- Post-invocation snapshot D8: pointer comparison (debug + prod)
- `firewall-acl-agent/src/main.cpp` integrado con TEST-INTEG-4a smoke test
- Makefile: `firewall` ahora depende de `plugin-loader-build` (orden instalación)

**Gate TEST-INTEG-4a: PASSED ✅**
- `plugin_process_message()` invocado sobre `MessageContext` real
- Graceful Degradation D1 aplicada (hello plugin sin símbolo — correcto)
- `result_code=0` confirmado
- Sin `D8 VIOLATION`
- `CryptoTransport` decryption path sin modificar (diff clean)

**Consejo de Sabios DAY 105 — 4/5 revisores (ChatGPT5 ausente) ✅**

Veredicto: ACCEPTED CON CONDICIONES
- D8-v2: añadir CRC32 del payload en debug builds (3/4 mayoría)
- TEST-INTEG-4a-PLUGIN: gate obligatorio antes de PHASE 2b (4/4 unanimidad)
- nonce/tag NULL: documentar en plugin_api.h (4/4 unanimidad)
- Makefile deps: propagar a los 4 componentes restantes (4/4 unanimidad)
- reserved[60]: suficiente, layout a documentar en ADR-024 (4/4 unanimidad)
- Qwen autoidentificado como DeepSeek (4ª vez consecutiva, patrón consolidado)
- ChatGPT5 ausente — primera vez en el proyecto (registrado)

**Paper v10 ✅**
- §5.6 Plugin Architecture (ADR-023, ADR-024) añadido
- §5.8 gate TEST-INTEG-4a mencionado
- §6.5 rama actualizada a `feature/plugin-crypto`
- Compilación limpia confirmada

**arXiv — cuenta y endorsement ✅**
- Cuenta `alonsoir` creada y verificada (alonsoir@gmail.com)
- Código endorsement `AFKRBO` generado para cs.CR
- Email reenviado a Andrés Caro Lindo con enlace
- Submission pausada esperando endorsement

---

### Day 104 (1 Apr 2026) — Paper v9 + ADR-023 + ADR-024 + Consejo 2 rondas

**Paper v9 — revisión Gepeto (P1–P6) ✅**
**Paper v9 — corrección integridad científica (FP bare-metal) ✅**
**ADR-023 — Multi-Layer Plugin Architecture ✅ ACCEPTED**
**ADR-024 — Dynamic Group Key Agreement ✅ DISEÑO APROBADO**
**Consejo de Sabios — Ronda 1 + Ronda 2 ADR-023/024 ✅**
**Rama feature/plugin-crypto ✅**

---

### Day 103 (31 Mar 2026) — Makefile rag alignment + PAPER-ADR022 §6

**MAKEFILE-RAG alignment ✅**
**PAPER-ADR022 §6 — HKDF Context Symmetry ✅**
Paper: Draft v8 — 21 páginas, compilación limpia
**Merge feature/bare-metal-arxiv → main ✅**
**Consejo ADR-023 + ADR-024 sesión inicial (5 revisores) ✅**

---

### Day 102 (30 Mar 2026) — ADR-012 PHASE 1b COMPLETA + TEST-PLUGIN-INVOKE-1

**TEST-PLUGIN-INVOKE-1 ✅** · **ADR-012 PHASE 1b firewall-acl-agent ✅**
**ADR-012 PHASE 1b rag-ingester ✅** · **ADR-012 PHASE 1b rag-security ✅**
**arXiv endorser — Andrés Caro Lindo confirmado ✅**
Tests totales: **25/25 ✅**

---

### Day 101 (29 Mar 2026) — ADR-012 PHASE 1b bug fix + ml-detector plugin-loader
### Day 100 (28 Mar 2026) — ADR-021 + ADR-022 + set_terminate() + CI honesto
### Day 99 (27 Mar 2026) — contexts.hpp + TEST-INTEG + fail-closed
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

## 🔜 PRÓXIMO — PHASE 2b (DESBLOQUEADA)

### ADR-023 PHASE 2b — rag-ingester + MessageContext

**Archivos a tocar:**
- `rag-ingester/src/main.cpp` — añadir `PluginLoader`, `invoke_all(MessageContext&)`
- `rag-ingester/CMakeLists.txt` — linkear `libplugin_loader.so`
- `rag-ingester/config/rag-ingester.json` — añadir sección `plugins`

**Gate:** TEST-INTEG-4b — patrón idéntico a TEST-INTEG-4a

**Secuencia completa PHASE 2:**
- PHASE 2a — firewall-acl-agent ✅ (DAY 105-106)
- PHASE 2b — rag-ingester ⏳
- PHASE 2c — sniffer ⏳
- PHASE 2d — ml-detector ⏳
- PHASE 2e — rag-security (usa `g_plugin_loader` global para signal handler) ⏳

### P2 — Post-PHASE 2

| ID | Tarea | Origen |
|----|-------|--------|
| TEST-INTEG-4b | Gate PHASE 2b: rag-ingester + MessageContext | ADR-023 |
| TEST-FUZZ-1 | MessageContext fuzzing — pre-requisito PHASE 2c | ADR-023 R3 |
| TEST-INTEG-4c | Gate PHASE 2c: rag-security | ADR-023 |
| DEBT-CRYPTO-003a | `mlock()` seed_client.cpp | ADR-022 |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie | P2 |
| DEBT-INFRA-002 | Sustituir `haveged` por `rng-tools5` | P2 |
| FEAT-ROTATION-1 | `provision.sh rotate-all` + SEED_ROTATION_DAYS | P2 |
| BARE-METAL-IMAGE | Imagen Debian Bookworm hardened — exportable a USB | P2 |
| BARE-METAL-VAGRANT | Vagrantfile nuevo con imagen BARE-METAL-IMAGE | P2 |
| MLD_ALLOW_UNCRYPTED | Flag explícito para desactivar fail-fast en dev | Consejo DAY 108 |
| TEST-PROVISION-1 | Gate CI: vagrant destroy → up → 6/6 RUNNING | ChatGPT5 DAY 108 |
| ADR-028 | RAG Ingestion Trust Model — antes de write-capable plugins | ChatGPT5 DAY 108 |
| UX install_shared_libs | Mensaje "~2 min intencional" en provision.sh | Consejo DAY 108 |

### FASE 3 — Post-arXiv (ADR-024 implementation)

| ID | Tarea |
|----|-------|
| reserved-layout | Documentar layout reserved[60] en ADR-024 |
| ADR-024 OQ-5 | Mecanismo de revocación de claves estáticas X25519 |
| ADR-024 OQ-6 | Política de rotación en reprovisionamiento |
| ADR-024 OQ-7 | Replay first message — documentar en threat model |
| ADR-024 OQ-8 | Performance ARMv8 + comparación Noise_IKpsk3 vs Noise_KK |
| ADR-024 R4 | Evaluar noise-c (<200 KB, <50 ms) vs libsodium puro |
| ADR-024 impl. | provision.sh X25519 keypairs + deployment.yml schema |
| ADR-024 impl. | CryptoTransport::install_session_keys() |
| TEST-INTEG-5 | Noise_IKpsk3 handshake E2E (sniffer ↔ ml-detector) |
| TEST-INTEG-6 | PSK mismatch → fail-closed verificado |
| TEST-INTEG-7 | install_session_keys() llamado dos veces → std::terminate() |
| MULTI-VM | Vagrantfile multi-VM con topología distribuida real |
| CI-FULL | Self-hosted runner "argus-debian-bookworm" |
| ANSIBLE | Receta Ansible + Jinja2 |

---

## 🔐 BACKLOG CRIPTOGRÁFICO ACTIVO

### DEBT-CRYPTO-003a — mlock() sobre buffer del seed (🟡 P2)

```cpp
mlock(seed_.data(), seed_.size());
```
**Decisión consolidada (Consejo DAY 97):** WARNING + log instructivo, no error fatal.

### FEAT-ROTATION-1 — Política de rotación de seeds (🟡 P2)

- `SEED_ROTATION_DAYS=30` — SSOT en provision.sh
- Hot-reload descartado (split-brain risk) — ENT-4 para PHASE 3+.

### ADR-024 — Noise_IKpsk3 dynamic key agreement (FASE 3 — post-arXiv)

Diseño aprobado DAY 104. Implementación condicionada a OQ-5..OQ-8.
Layout reserved[60] a documentar al iniciar implementación.

### ADR-025 — Plugin Integrity Verification (Ed25519 + TOCTOU-safe dlopen)

Estado: **APPROVED** (DAY 102). Implementación post-PHASE 2 completa.

- PLUGIN-SIGN-1: Verificación Ed25519 en plugin_loader.cpp (O_NOFOLLOW + fstat + fd discipline)
- PLUGIN-SIGN-2: provision.sh --reset (confirmación, timestamp, fingerprint)
- PLUGIN-SIGN-3: Inyección pubkey en CMakeLists.txt (MLD_PLUGIN_PUBKEY_HEX)
- PLUGIN-SIGN-4: JSON config schemas — require_signature, allowed_key_id
- PLUGIN-SIGN-5: systemd units (Restart=always + unset LD_PRELOAD)
- TEST-INTEG-SIGN-1 → TEST-INTEG-SIGN-7

### FEAT-CRYPTO-3 — TPM 2.0 / HSM enterprise (P3 — ENT-8)

---

## 📋 DOCUMENTACIÓN PENDIENTE

### DOCS-2 — Perfiles AppArmor por componente (6 perfiles) ⏳

---

## 📋 BACKLOG — COMMUNITY & FEATURES

### 🟧 P1 — Fast Detector Config (DEBT-FD-001)
### 🟨 P2 — Expansión ransomware (prerequisito: DEBT-FD-001)
### 🟨 P2 — Pipeline reentrenamiento
### 🟩 P3 — Enterprise (ENT-1..8)

**ENT-1 — Federated Threat Intelligence**
Arquitectura planificada (Fig 3 del paper v11): telemetría anónima → agregación central →
classifiers reentrenados distribuidos como plugins firmados (ADR-025).
Estado: diseño documentado en paper, implementación post-pipeline-estabilización.

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:           ████████████████████ 100% ✅
HMAC Infrastructure:                  ████████████████████ 100% ✅
Proto3 Pipeline Stability:            ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
CSV Pipeline:                         ████████████████████ 100% ✅
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
MAKEFILE-RAG alignment:               ████████████████████ 100% ✅  DAY 103
PAPER-ADR022 §5.5 (HKDF case study):  ████████████████████ 100% ✅  DAY 103
Paper v9 (Gepeto P1–P6 + FP fix):     ████████████████████ 100% ✅  DAY 104
ADR-023 (Plugin Architecture):        ████████████████████ 100% ✅  DAY 104 — ACCEPTED
ADR-024 (Noise IK — diseño):          ████████████████████ 100% ✅  DAY 104 — DISEÑO APROBADO
PHASE 2a firewall (MessageContext):   ████████████████████ 100% ✅  DAY 105 — TEST-INTEG-4a PASSED
Paper v10 (§5.6 Plugin Arch.):        ████████████████████ 100% ✅  DAY 105
arXiv cuenta + endorsement:           ████████████████████ 100% ✅  DAY 105-106
D8-v2 (CRC32 payload debug):          ████████████████████ 100% ✅  DAY 106
TEST-INTEG-4a-PLUGIN (3 variantes):   ████████████████████ 100% ✅  DAY 106 — make plugin-integ-test
nullptr-doc (nonce/tag en API):       ████████████████████ 100% ✅  DAY 106
MAKEFILE-DEPS (5 componentes):        ████████████████████ 100% ✅  DAY 106
Paper v11 (3 figuras TikZ + UEx):     ████████████████████ 100% ✅  DAY 106
arXiv submitted:                      ████████████████████ 100% ✅  DAY 106 — submit/7438768
PHASE 2b rag-ingester:                ░░░░░░░░░░░░░░░░░░░░   0% ⏳  SIGUIENTE
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado por hardware
ADR-024 impl. (Noise IK):             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  FASE 3 post-arXiv
ADR-025 impl. (plugin signing):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-PHASE 2 completa
DEBT-CRYPTO-003a (mlock seed):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2
DEBT-INFRA-001 (Debian Trixie):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2
DOCS-2 (AppArmor profiles):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳
Fast Detector Config (DEBT-FD-001):   ████░░░░░░░░░░░░░░░░  20% 🟡  PHASE 2
ENT-*:                                ░░░░░░░░░░░░░░░░░░░░   0% ⏳  largo plazo
provision.sh reproducible (destroy→6/6): ████████████████████ 100% ✅  DAY 108
ADR-026 (P2P Fleet Federated):           ████████████████████ 100% ✅  DAY 108 — BORRADOR
ADR-027 (CTX swap etcd):                 ████████████████████ 100% ✅  DAY 108 — ACEPTADO
MLD_ALLOW_UNCRYPTED invariant:           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 109
rag-security/config en provision.sh:     ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 109
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
| Cifrado obligatorio | SIEMPRE. Sin flag. CryptoTransport ✅ | 96-97 |
| Compresión obligatoria | SIEMPRE. Sin flag. LZ4 ✅ | 96 |
| Orden operaciones | LZ4 → ChaCha20 ✅ | 96 |
| libsodium versión | 1.0.19 desde fuente, SHA-256 verificado ✅ | 97 |
| CryptoManager | DEPRECADO — CryptoTransport lo sustituye ✅ | 97-98 |
| Opción 2 multi-instancia | DESCARTADA — reproduce bug asimetría ✅ | 100 |
| set_terminate() | fail-closed en los 6 main() ✅ | 100 |
| CI GitHub Actions | Solo validación estática (ubuntu-latest) ✅ | 100 |
| Plugin-loader scope | global si signal handler, local si no ✅ | 102 |
| Schema protobuf | Abierto intencionado — no cerrar hasta decisión grafos ✅ | 102 |
| FEAT-PLUGIN-CRYPTO-1 API | MessageContext — unanimidad Consejo 5/0 ✅ | 102 |
| ADR-023 plugin trust model | trusted-but-buggy, not tamper-proof ✅ | 104 |
| ADR-023 degradation policy | fail-closed producción; DEV_MODE solo Debug + compile flag ✅ | 104 |
| ADR-023 TCB declaration | plugins operan sobre plaintext → parte del TCB ✅ | 104 |
| ADR-024 handshake pattern | Noise_IKpsk3 confirmado ✅ | 104 |
| ADR-024 PSK info string | "ml-defender:noise-ikpsk3:v1" — domain separation ✅ | 104 |
| ADR-024 key installation | install_session_keys() atómica + gate etcd READY ✅ | 104 |
| D8 snapshot mode | pointer comparison + CRC32 payload (debug) — Consejo 3/4 ✅ | 105-106 |
| nonce/tag nullable | NULL permitido en test-config mode, non-NULL en producción ✅ | 106 |
| Makefile dep order | plugin-loader-build explícito en todos los componentes ✅ | 106 |
| reserved[60] layout | suficiente; layout formal a documentar en ADR-024 ✅ | 105 |
| test_config.json | Config aislado para tests de plugin — no tocar producción ✅ | 106 |
| CTX_ETCD swap (server mirror) | Servidor invierte TX/RX respecto al cliente — ADR-027 ✅ | 107–108 |
| provision.sh reproducibilidad | vagrant destroy → up → 6/6 sin intervención manual ✅ | 108 |
| plugin rag-ingester trust | read-only, payload=nullptr, Consejo 4/5 ✅ | 108 |

---

### Notas del Consejo de Sabios
> DAY 108 — provision.sh formalizado + ADR-026/027 + PASO 4 verde:
> "Q1: terminate prod + MLD_ALLOW_UNCRYPTED dev. Q2: rebuild limpio unánime.
> Q3: rag-ingester plugin read-only (ctx_readonly.payload=nullptr) — 4/5.
> Q4: rag-security/config en provision.sh — unánime.
> Nuevos: TEST-PROVISION-1 como gate CI, ADR-028 RAG Ingestion Trust Model.
> Nota: Qwen auto-identifica como DeepSeek — patrón consolidado DAY 103-108,
> comportamiento de entrenamiento, no identidad real (acceso verificado chat.qwen.ai)."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (auto-identifica DeepSeek)

> DAY 106 — PHASE 2a cierre (paper + arXiv):
> "PHASE 2a completamente cerrada. Todas las condiciones del Consejo DAY 105 satisfechas.
> TEST-INTEG-4a-PLUGIN 3/3 variantes PASSED. arXiv submit/7438768 submitted.
> PHASE 2b desbloqueada."
> — Claude (Anthropic)

> DAY 105 — PHASE 2a (4/5 revisores, ChatGPT5 ausente por primera vez):
> "ACCEPTED CON CONDICIONES. D8 pointer-only insuficiente — añadir CRC32 (3/4).
> TEST-INTEG-4a no ejerce plugin_process_message real — crear plugin de test (4/4).
> nonce/tag NULL documentar en contrato API (4/4). Makefile deps explícitas (4/4).
> reserved[60] suficiente, layout a especificar en ADR-024 (4/4)."
> — DeepSeek · Gemini · Grok · Qwen (→DeepSeek, patrón DAY 103-105 consolidado)
> ChatGPT5: AUSENTE — primera ausencia del proyecto.

> DAY 104 — ADR-023 + ADR-024 (unanimidad ronda 2, 5/5):
> "ADR-023 ACCEPTED. ADR-024 DISEÑO APROBADO. Noise_IKpsk3 confirmado."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (→DeepSeek)

> DAY 103 — ADR-023 + ADR-024 (ronda 1, 5/5 con condiciones)
> DAY 102 — FEAT-PLUGIN-CRYPTO-1 (unanimidad 5/0)
> DAY 101: "Orden plugin-loader: firewall → rag-ingester → rag-security (5/5)"
> DAY 100: "set_terminate() es defensa en profundidad correcta."
> DAY 97: "La rotación real de seeds es el mecanismo correcto de forward secrecy."

---

*Última actualización: DAY 108 — 5 Apr 2026*
*Branch: feature/plugin-crypto*
*Tests: 25/25 suites ✅ + TEST-INTEG-4a-PLUGIN 3/3 ✅*
*Paper: Draft v11 ✅ · arXiv: submit/7438768 SUBMITTED ✅*
*Pipeline: 6/6 RUNNING reproducible desde cero (vagrant destroy) ✅*
*PHASE 2a: COMPLETAMENTE CERRADA · PHASE 2b: DESBLOQUEADA*