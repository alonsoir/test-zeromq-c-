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

### Day 104 (1 Apr 2026) — Paper v9 + ADR-023 + ADR-024 + Consejo 2 rondas

**Paper v9 — revisión Gepeto (P1–P6) ✅**

6 mejoras de ChatGPT5 (revisor cs.CR externo) aplicadas:
- P1: §5.5 — cadena causal explícita (contexto incorrecto → claves distintas → MAC failure)
- P2a: §5.4 — TDH propuesto formalmente como metodología
- P2b: Abstract — TDH mencionado explícitamente
- P3: §5.2 — Consejo redefinido como adversarial validation, no ensemble
- P4: §10.11 — nueva subsección Threats to Validity (seed compromise, single-host, no TPM)
- P5: §8 — claim técnico explícito: "throughput limited by virtualized NIC, not pipeline"
- P6: §5.5 — frase conectando contexts.hpp con la clase de bug HKDF

**Paper v9 — corrección integridad científica (FP bare-metal) ✅**

Tres instancias corregidas (Abstract, §7, Conclusión):
- Anterior: "do not occur in bare-metal deployments" (afirmación no verificada)
- Corregido: FP identificados con detalle (mDNS multicast + broadcast, VirtualBox host-only NIC);
  ausencia en bare-metal es expectativa razonable pero no verificada empíricamente;
  referencia a §11.11 (Future Work)

**ADR-023 — Multi-Layer Plugin Architecture ✅ ACCEPTED**

Dos rondas del Consejo de Sabios (DAY 103 + DAY 104) — unanimidad en ronda 2.
Decisiones críticas incorporadas: D1–D11.

Highlights:
- `MessageContext` struct con contrato completo de ownership/lifetime
- Plugin trust model explícito: trusted-but-buggy, not tamper-proof
- TCB declaration: plugins operan sobre plaintext → forman parte del TCB
- Security invariants + post-invocation validation con código C concreto
- Graceful degradation policy: fail-closed en producción; DEV_MODE solo en
  builds Debug + `MLD_ALLOW_DEV_MODE` compile flag
- Forward-compatibility con ADR-024 declarada explícitamente
- Minorías de Gemini, Grok y ChatGPT registradas formalmente

**ADR-024 — Dynamic Group Key Agreement ✅ DISEÑO APROBADO**

Dos rondas del Consejo de Sabios — unanimidad en ronda 2.
Status: DISEÑO APROBADO — IMPLEMENTACIÓN POST-ARXIV.

Highlights:
- Noise_IKpsk3 confirmado como patrón de handshake
- Domain separation: `"ml-defender:noise-ikpsk3:v1"` como info string canónico
- Tabla de info strings prohibidos
- `install_session_keys()` con contrato completo + transición atómica + gate etcd READY
- OQ-5 (revocación), OQ-6 (rotación), OQ-7 (replay), OQ-8 (perf ARMv8 + Noise_KK)
- Métricas de aceptación noise-c: <200 KB, <50 ms
- Nota de implicación de compromiso de seed_family
- Minorías registradas (ChatGPT: Noise_XX; Grok: Noise_KK; Qwen: libsodium puro)

**Consejo de Sabios — Ronda 1 + Ronda 2 ADR-023/024 ✅**

- Ronda 1: 5 revisores (ChatGPT, DeepSeek, Gemini, Grok, Qwen) — 5/5 ACCEPTED CON CONDICIONES
- Ronda 2: validación consolidada — 5/5 unanimidad
- Qwen autoidentificado como DeepSeek en ambas rondas — patrón registrado
- Sesiones consolidadas producidas y archivadas

**Rama feature/plugin-crypto ✅**

```bash
git checkout -b feature/plugin-crypto
git push -u origin feature/plugin-crypto
```

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

## 🔄 PRÓXIMO MILESTONE — arXiv submission

### P1 — Antes de enviar el paper

| ID | Tarea | Estado |
|----|-------|--------|
| MAKEFILE-RAG | Alinear rag-build al patrón estándar | ✅ DAY 103 |
| PAPER-ADR022 | §5.5 subsección HKDF Context Symmetry case study | ✅ DAY 103 |
| PAPER-V9 | Revisión Gepeto P1–P6 aplicada | ✅ DAY 104 |
| PAPER-V9 | Corrección integridad científica FP bare-metal (3 instancias) | ✅ DAY 104 |
| ADR-023 | Multi-Layer Plugin Architecture — ACCEPTED | ✅ DAY 104 |
| ADR-024 | Dynamic Group Key Agreement — DISEÑO APROBADO | ✅ DAY 104 |
| ENDORSER-ANDRES | Llamada Andrés Caro Lindo — endorsement arXiv | 📞 mañana jueves 3 abril |
| BARE-METAL | Stress test sin VirtualBox — validar ≥100 Mbps | 🔴 BLOQUEADO — sin hardware físico |
| PAPER-FINAL | Actualizar métricas DAY 104 (paper v9, ADR-023/024, 25/25 tests) | ⏳ |
| DOCS-APPARMOR | 6 perfiles AppArmor por componente | ⏳ |

### P2 — Post-arXiv, pre-FASE 3 (rama feature/plugin-crypto activa)

| ID | Tarea | Origen |
|----|-------|--------|
| FEAT-PLUGIN-CRYPTO-1 | Plugin crypto transport — implementación PHASE 2a | ADR-023 |
| ADR-023 D1–D11 impl. | Fail-closed, invariants, TCB, DEV_MODE build flag | DAY 104 |
| TEST-INTEG-4a | Gate PHASE 2a → 2b: firewall-acl-agent + MessageContext | ADR-023 |
| TEST-INTEG-4b | Gate PHASE 2b → 2c: rag-ingester | ADR-023 |
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
Ver ADR-024 para contrato completo.

### FEAT-CRYPTO-3 — TPM 2.0 / HSM enterprise (P3 — ENT-8)

---

## 📋 DOCUMENTACIÓN PENDIENTE

### DOCS-2 — Perfiles AppArmor por componente (6 perfiles) ⏳

---

## 📋 BACKLOG — COMMUNITY & FEATURES

### 🟥 P0 — Paper arXiv

- Draft v9 ✅ · LaTeX ✅
- Correcciones Gepeto P1–P6 aplicadas ✅
- Corrección integridad científica FP bare-metal ✅
- Pendiente: endorsement Andrés (jueves) + PAPER-FINAL métricas + bare-metal stress test

**Revisores / endorsers:**

| Persona | Perfil | Estado |
|---------|--------|--------|
| Sebastian Garcia (CTU Prague) | Autor CTU-13, ML seguridad | ✅ respondió, recibió PDF |
| Yisroel Mirsky (BGU) | Investigador ML/seguridad | ⏳ enviado DAY 96, sin respuesta |
| Andrés Caro Lindo (UEx/INCIBE) | Director Cátedra INCIBE-UEx | ✅ endorsement confirmado — llamada jueves 3 abril |
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
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado por hardware
ENDORSER-ANDRES llamada:              ░░░░░░░░░░░░░░░░░░░░   0% 📞  jueves 3 abril
PAPER-FINAL métricas DAY 104:         ░░░░░░░░░░░░░░░░░░░░   0% ⏳
FEAT-PLUGIN-CRYPTO-1 PHASE 2a:        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-arXiv
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

---

## 🔌 FEAT-PLUGIN-CRYPTO-1 — Plugin de CryptoTransport (⏳ PHASE 2 — post-arXiv)

**Prerequisitos completados:** ADR-023 ✅ · ADR-024 (diseño) ✅

**Implementación PHASE 2a:** `firewall-acl-agent` (gate: TEST-INTEG-4a)
**Implementación PHASE 2b:** `rag-ingester` (gate: TEST-INTEG-4b)
**Pre-requisito PHASE 2c:** TEST-FUZZ-1 (MessageContext fuzzing)
**Implementación PHASE 2c:** `rag-security` (gate: TEST-INTEG-4c)

Ver ADR-023 para contrato completo de MessageContext, trust model,
security invariants y graceful degradation policy.

---

## 🖥️ BARE-METAL — Imagen hardened + stress test real (⏳ P2)

Bloqueo: hardware físico disponible. No bloquea arXiv ni PHASE 2.
Ver sección completa en versión anterior del backlog.

---

## 📋 DEBT-PROTO-001 — Revisión contrato protobuf (⏳ P3 — FASE 3)

Schema abierto intencionado. No cerrar hasta decisión sobre grafos.
No bloquea arXiv.

---

### Notas del Consejo de Sabios

> DAY 104 — ADR-023 + ADR-024 (unanimidad ronda 2, 5/5):
> "ADR-023 ACCEPTED. Plugin trust model explícito. TCB declaration. Post-invocation
> invariant validation con código concreto. MLD_DEV_MODE solo en builds Debug."
> "ADR-024 DISEÑO APROBADO. Noise_IKpsk3 confirmado. Domain separation en info string.
> install_session_keys() atómica. OQ-5..OQ-8 formalizadas."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (→DeepSeek, patrón registrado)

> DAY 103 — ADR-023 + ADR-024 (ronda 1, 5/5 con condiciones):
> "ADR-023 ACCEPTED CON CONDICIONES. ADR-024 DISEÑO CON RESERVAS."
> "fail-closed en producción. domain separation HKDF. install_session_keys() atómica."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen

> DAY 102 — FEAT-PLUGIN-CRYPTO-1 (unanimidad 5/0):
> "Opción A (MessageContext). Símbolo opcional PHASE 2a → obligatorio PHASE 2b."
> "TEST-INTEG-4a/4b/4c como gates. Core read-only durante PHASE 2a."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen

> DAY 101: "Orden plugin-loader: firewall → rag-ingester → rag-security (5/5)"
> DAY 100: "set_terminate() es defensa en profundidad correcta."
> DAY 97: "La rotación real de seeds es el mecanismo correcto de forward secrecy."

---

*Última actualización: DAY 104 — 1 Apr 2026*
*Branch: feature/plugin-crypto*
*Tests: 25/25 suites ✅*
*Paper: Draft v9 ✅*
*ADR-023: ACCEPTED ✅ · ADR-024: DISEÑO APROBADO ✅*
*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic), Grok, ChatGPT, DeepSeek, Qwen, Gemini, Parallel.ai*