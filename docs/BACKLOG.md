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

## 🔄 PRÓXIMO MILESTONE — arXiv submission + PHASE 2a cierre

### P0 — arXiv (bloqueado por endorsement)

| ID | Tarea | Estado |
|----|-------|--------|
| ENDORSER-ANDRES | Endorsement arXiv cs.CR — código AFKRBO enviado | 📧 esperando respuesta |
| ARXIV-SUBMIT | Continuar submission en arxiv.org/user | ⏳ post-endorsement |
| BARE-METAL | Stress test sin VirtualBox — validar ≥100 Mbps | 🔴 BLOQUEADO — sin hardware físico |

### P1 — PHASE 2a cierre (DAY 106 — rama feature/plugin-crypto)

**BLOQUEANTE antes de avanzar a PHASE 2b:**

| ID | Tarea | Origen | Estado |
|----|-------|--------|--------|
| D8-v2 | CRC32 payload snapshot en debug builds | Consejo DAY 105 (3/4) | 🔴 DAY 106 |
| TEST-INTEG-4a-PLUGIN | Plugin de test con símbolo exportado — 3 variantes | Consejo DAY 105 (4/4) | 🔴 DAY 106 |
| nullptr-doc | nonce/tag NULL documentado en plugin_api.h | Consejo DAY 105 (4/4) | 🔴 DAY 106 |
| MAKEFILE-DEPS | Dependencia plugin-loader-build en sniffer, ml-detector, rag-ingester, rag-security | Consejo DAY 105 (4/4) | 🔴 DAY 106 |

**TEST-INTEG-4a-PLUGIN — variantes requeridas:**
- Variante A: exporta símbolo, result_code=0, no modifica nada → debe pasar
- Variante B: intenta modificar campo read-only (`direction`) → D8 debe detectarlo
- Variante C: devuelve result_code=-1 → host registra error (no std::terminate())

### P2 — PHASE 2b y siguientes (post-cierre PHASE 2a)

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

### FEAT-CRYPTO-3 — TPM 2.0 / HSM enterprise (P3 — ENT-8)

---

## 📋 DOCUMENTACIÓN PENDIENTE

### DOCS-2 — Perfiles AppArmor por componente (6 perfiles) ⏳

---

## 📋 BACKLOG — COMMUNITY & FEATURES

### 🟥 P0 — Paper arXiv

- Draft v10 ✅ · LaTeX + ZIP ✅
- Cuenta arXiv `alonsoir` creada ✅
- Código endorsement AFKRBO generado ✅
- Pendiente: endorsement Andrés → submission

**Revisores / endorsers:**

| Persona | Perfil | Estado |
|---------|--------|--------|
| Sebastian Garcia (CTU Prague) | Autor CTU-13, ML seguridad | ✅ respondió, recibió PDF |
| Yisroel Mirsky (BGU) | Investigador ML/seguridad | ⏳ enviado DAY 96, sin respuesta |
| Andrés Caro Lindo (UEx/INCIBE) | Director Cátedra INCIBE-UEx | 📧 código AFKRBO enviado DAY 105 |
| Jorge Coronado (QuantiKa14) | DFIR, forense, OSINT | ⏳ email enviado — revisión paper + repo |

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
arXiv cuenta + endorsement AFKRBO:    ████████████████████ 100% ✅  DAY 105
D8-v2 (CRC32 payload debug):          ░░░░░░░░░░░░░░░░░░░░   0% 🔴  DAY 106
TEST-INTEG-4a-PLUGIN (3 variantes):   ░░░░░░░░░░░░░░░░░░░░   0% 🔴  DAY 106 — bloquea 4b
nullptr-doc (nonce/tag en API):       ░░░░░░░░░░░░░░░░░░░░   0% 🔴  DAY 106
MAKEFILE-DEPS (4 componentes):        ░░░░░░░░░░░░░░░░░░░░   0% 🔴  DAY 106
ENDORSER-ANDRES (código AFKRBO):      ████████░░░░░░░░░░░░  40% 📧  esperando respuesta
ARXIV-SUBMIT:                         ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-endorsement
PHASE 2b rag-ingester:                ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-TEST-INTEG-4a-PLUGIN
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado por hardware
ADR-024 impl. (Noise IK):             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  FASE 3 post-arXiv
DEBT-CRYPTO-003a (mlock seed):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2
DEBT-INFRA-001 (Debian Trixie):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2
DOCS-2 (AppArmor profiles):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳
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
| D8 snapshot mode | pointer comparison + CRC32 payload (debug) — Consejo 3/4 ✅ | 105 |
| nonce/tag nullable | NULL permitido en test-config mode, non-NULL en producción ✅ | 105 |
| Makefile dep order | plugin-loader-build explícito en todos los componentes ✅ | 105 |
| reserved[60] layout | suficiente; layout formal a documentar en ADR-024 ✅ | 105 |

---

## 🔌 FEAT-PLUGIN-CRYPTO-1 — Plugin de CryptoTransport

**Prerequisitos completados:** ADR-023 ✅ · ADR-024 (diseño) ✅

**PHASE 2a:** `firewall-acl-agent` ✅ DAY 105
- Gate TEST-INTEG-4a: PASSED (D1 validado)
- Gate TEST-INTEG-4a-PLUGIN: 🔴 PENDIENTE DAY 106 (D8 a validar)

**PHASE 2b:** `rag-ingester` — BLOQUEADA hasta TEST-INTEG-4a-PLUGIN
**Pre-requisito PHASE 2c:** TEST-FUZZ-1 (MessageContext fuzzing)
**PHASE 2c:** `rag-security`

---

### Notas del Consejo de Sabios

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

*Última actualización: DAY 105 — 2 Apr 2026*
*Branch: feature/plugin-crypto*
*Tests: 25/25 suites ✅*
*Paper: Draft v10 ✅*
*PHASE 2a: COMPLETA (TEST-INTEG-4a PASSED) · TEST-INTEG-4a-PLUGIN: 🔴 DAY 106*
*arXiv: cuenta alonsoir ✅ · código AFKRBO enviado a Andrés ✅*
*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic), Grok, DeepSeek, Qwen, Gemini, Parallel.ai*