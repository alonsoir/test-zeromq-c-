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

### Day 97 (25 Mar 2026) — CryptoTransport HKDF + libsodium 1.0.19

**CryptoTransport — ADR-013 PHASE 2 completado ✅**

Eslabón central de la cadena de confianza: HKDF-SHA256 nativo via libsodium 1.0.19
+ ChaCha20-Poly1305 IETF + nonce 96-bit monotónico atómico.

```
crypto-transport/include/crypto_transport/transport.hpp
crypto-transport/src/transport.cpp
crypto-transport/tests/test_crypto_transport.cpp  ← 10 tests TC-CT-001..010
```

**Cadena de confianza completa:**
```
provision.sh → seed.bin → SeedClient → CryptoTransport(HKDF) → ChaCha20-Poly1305
```

**libsodium 1.0.19 desde fuente:**
- SHA-256 verificado: `018d79fe0a045cca07331d37bd0cb57b2e838c51bc48fd837a1472e50068bbea`
- `provision.sh`: `install_libsodium_1019()` + `check_entropy()` (DEBT-CRYPTO-003b ✅)
- Reinstala ZeroMQ post-remove (apt arrastra libzmq5 al quitar libsodium23)
- CMake: `NO_DEFAULT_PATH` para priorizar `/usr/local` sobre sistema

**Decisiones técnicas resueltas:**
- API real libsodium 1.0.19: `extract_final(&st, prk[])` → `expand(okm, len, ctx, prk[])`
- HKDF salt: 32 bytes cero (RFC 5869 default)
- Nonce layout: `[0x00000000 || uint64_LE_counter]`
- `CryptoManager` DEPRECADO — sustituido por `CryptoTransport` en DAY 98

**ADR-020 — Cifrado + compresión siempre obligatorios ✅ (documentado)**
Flags `enabled` eliminados del contrato. Migración JSONs pendiente DAY 98.

**etcd-server CMakeLists:** `add_test()` añadido — tests ahora registrados en ctest.

**Tests: 22/22 suites ✅**
- crypto-transport: 4/4 · seed-client: 1/1 · etcd-server: 1/1
- rag-ingester: 7/7 · ml-detector: 9/9

---

### Day 96 (24 Mar 2026) — seed-client + Makefile dependency order

**libs/seed-client — PHASE 1 completado ✅**

Nueva librería `libseed_client.so` (con underscore — DEBT-NAMING-001 P3 pendiente).
Tests: 6/6 ✅ · Instalado en: `/usr/local/lib/libseed_client.so.1.0.0`

**Makefile — nueva cadena de dependencias:**
```
libseed_client → crypto-transport → etcd-client → componentes
```

**arXiv outreach DAY 96:** Email enviado a Prof. Yisroel Mirsky (BGU) ✅

---

### Day 95 (23 Mar 2026) — Cryptographic Provisioning Infrastructure

- `tools/provision.sh`: 4 modos full/status/verify/reprovision
- Ed25519 keypairs + ChaCha20 seeds (32B) para 6 componentes
- Paths AppArmor-compatible: `/etc/ml-defender/{component}/` chmod 600
- Bloque `identity` en los 6 JSONs de componentes (ADR-013)
- `pipeline-start` depende de `provision-check` (fail-closed security)
- Consejo DAY 95 (6/7): unanimidad.

---

### Day 93 — ADR-012 PHASE 1: plugin-loader + ABI validation
### Day 83 — Ground truth bigFlows + CSV E2E · tag: v0.83.0-day83-main ✅
### Days 76–82 — Proto3 · Sentinel · F1=0.9985 · DEBT-FD-001
### Days 63–75 — Pipeline 6/6 · ChaCha20 · FAISS · HMAC · trace_id
### Days 1–62 — Foundation: eBPF/XDP · protobuf · ZMQ · RandomForest C++20

---

## 🔄 EN CURSO / INMEDIATO

### DAY 98 — Integrar CryptoTransport en los 6 componentes + ADR-020 JSONs

---

## 🔐 BACKLOG CRIPTOGRÁFICO

### DEBT-CRYPTO-004 — Migrar CryptoManager → CryptoTransport (🔴 P1 — DAY 98)

`CryptoManager` usa el seed directamente como clave sin HKDF — "USO INCORRECTO"
documentado en `seed_client.hpp`. Debe sustituirse en los 6 componentes.

**Patrón de migración:**
```cpp
// ANTES (CryptoManager — DEPRECADO)
CryptoManager crypto(seed_string);
auto encrypted = crypto.encrypt(plaintext);

// DESPUÉS (CryptoTransport — ADR-013 PHASE 2)
SeedClient sc("/etc/ml-defender/sniffer/sniffer.json");
sc.load();
CryptoTransport tx(sc, "ml-defender:sniffer:v1:tx");
CryptoTransport rx(sc, "ml-defender:sniffer:v1:rx");
auto encrypted = tx.encrypt(plaintext_bytes);
```

**Orden de migración:**
1. `etcd-server/src/crypto_manager.cpp`
2. `sniffer`
3. `ml-detector`
4. `firewall-acl-agent`
5. `rag-ingester`
6. `rag-security`

### DEBT-ETCD-001 — etcd-client: integrar CryptoTransport (🔴 P1 — DAY 98)

`etcd-client` usa `CryptoManager`. Migrar a `CryptoTransport` con contexto
`"ml-defender:etcd-client:v1:tx"`.

### DEBT-CRYPTO-003a — mlock() sobre buffer del seed (🟡 P2 — DAY 98)

```cpp
// En seed_client.cpp, tras leer seed.bin:
mlock(seed_.data(), seed_.size());
```
Evita que el material criptográfico llegue al swap.
**Decisión pendiente:** ¿fallo fatal o advertencia si mlock() falla con ENOMEM?

### FEAT-CRYPTO-1 — Rotación de claves sin downtime (P2 — DAY 98+)

⚠️ Rotación con `make provision-reprovision` requiere reinicio ordenado del pipeline.
Documentar en SECURITY_MODEL.md.

### DEBT-NAMING-001 — libseed_client → libseedclient (P3 — DAY tranquilo)

CMakeLists genera `libseed_client.so` (underscore). Consejo acordó sin underscore.
No bloquea nada.

### DEBT-INFRA-001 — Migrar a Debian Trixie (13) (🟡 P2 — DAY 105+)

Debian Bookworm solo distribuye libsodium 1.0.18. La compilación desde fuente
en `provision.sh` es un parche funcional, no una solución de producción:
- La librería en `/usr/local/lib/` no está gestionada por apt
- No recibe actualizaciones automáticas de seguridad
- Entornos con políticas estrictas (hospitales, AAPP) requieren paquetes `.deb` firmados

**Solución:** migrar la box Vagrant de `debian/bookworm64` a `debian/trixie64`.
Trixie tiene libsodium 1.0.19 en repos oficiales — instalable con `apt-get install libsodium-dev`.
Requiere validar todas las dependencias del pipeline (etcd, eBPF headers, ZeroMQ) en Trixie.

### DEBT-INFRA-002 — Sustituir haveged por rng-tools5 (🟡 P2 — DAY 105+)

`haveged` no está certificado por NIST, FIPS 140-2/3 ni Common Criteria.
Es aceptable para desarrollo, no para producción en infraestructura crítica.

**Solución:** `rng-tools5` con detección automática de fuente de hardware:
- Bare-metal Intel/AMD: RDRAND/RDSEED via `/dev/hwrng`
- VMs KVM/QEMU: `virtio-rng` device
- Sin hardware RNG: `jitterentropy-rngd` (certificado BSI, NIST SP 800-90B)

Ver `docs/SECURITY_MODEL.md` §4.2 para detalle completo.

### FEAT-CRYPTO-2 — Handshake efímero Noise (P3 — PHASE 2)
### FEAT-CRYPTO-3 — TPM 2.0 / HSM enterprise (P3 — ENT-8)

---

## 🔴 ADR-020 — Eliminar flags enabled de los 6 JSONs (P1 — DAY 98)

ADR documentado ✅ (`docs/adr/ADR-020-crypto-mandatory.md`).
Implementación pendiente — eliminar de:

| Componente | JSON |
|---|---|
| sniffer | `sniffer/config/sniffer.json` |
| ml-detector | `ml-detector/config/ml_detector_config.json` |
| firewall-acl-agent | `firewall-acl-agent/config/firewall.json` |
| rag-ingester | `rag-ingester/config/rag-ingester.json` |
| rag-security | `rag/config/rag-config.json` |
| etcd-server | `etcd-server/config/etcd-server.json` |

---

## 🧪 TESTS DE INTEGRACIÓN E2E (P1 — DAY 98)

### TEST-INTEG-1 — Pipeline completo: provision → seed → crypto → mensaje
### TEST-INTEG-2 — Upload JSON a etcd-server: LZ4 → ChaCha20 → round-trip

---

## 📋 DOCUMENTACIÓN PENDIENTE

### DOCS-1 — SECURITY_MODEL.md
- Límites PHASE 1 · rotación requiere reinicio · roadmap PHASE 2

### DOCS-1 — SECURITY_MODEL.md ✅ DAY 97

Creado en `docs/SECURITY_MODEL.md`:
- Cadena de confianza completa
- Límites PHASE 1 (seed en claro, sin forward secrecy por sesión, mlock() no garantizado)
- Rotación de seeds — advertencia split-brain
- Entropía: haveged vs rng-tools5 + hardware RNG
- CRIME/BREACH — análisis del threat model
- Threat model alcance PHASE 1

### DOCS-2 — Perfiles AppArmor por componente (6 perfiles)

---

## 📋 BACKLOG — COMMUNITY & FEATURES

### 🟥 P0 — Paper arXiv
- Draft v5 ✅ · LaTeX ✅ · Email Sebastian Garcia ✅ (sin respuesta)
- Email Yisroel Mirsky ✅ DAY 96 (esperando)
- Si sin respuesta 48h → Tier 3: Martin Grill o Battista Biggio

### 🟧 P1 — Fast Detector Config (DEBT-FD-001)
### 🟨 P2 — Expansión ransomware (prerequisito: DEBT-FD-001)
### 🟨 P2 — Pipeline reentrenamiento
### 🟩 P3 — Enterprise (ENT-1..8)
### 🟩 P3 — ADR-012 PHASE 1b: plugin-loader integrado en sniffer

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:           ████████████████████ 100% ✅
HMAC Infrastructure:                  ████████████████████ 100% ✅
Proto3 Pipeline Stability:            ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
CSV Pipeline:                         ████████████████████ 100% ✅
Paper arXiv (draft v5 + LaTeX):       ████████████████████ 100% ✅
Cryptographic Provisioning PHASE 1:   ████████████████████ 100% ✅  DAY 95
seed-client (libseed_client):         ████████████████████ 100% ✅  DAY 96
Makefile dep order:                   ████████████████████ 100% ✅  DAY 96
libsodium 1.0.19 + provision.sh:      ████████████████████ 100% ✅  DAY 97
CryptoTransport (HKDF+nonce+AEAD):    ████████████████████ 100% ✅  DAY 97
DEBT-CRYPTO-001 (nonce 96-bit):       ████████████████████ 100% ✅  DAY 97
DEBT-CRYPTO-002 (HKDF libsodium):     ████████████████████ 100% ✅  DAY 97
DEBT-CRYPTO-003b (entropy check):     ████████████████████ 100% ✅  DAY 97
ADR-020 (documentado):                ████████████████████ 100% ✅  DAY 97
plugin-loader ADR-012 PHASE 1:        ████████████████░░░░  80% 🟡  integ. sniffer P3
DEBT-CRYPTO-004 (migrar CryptoMgr):  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 DAY 98
DEBT-ETCD-001 (etcd-client migrar):  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 DAY 98
ADR-020 JSONs (eliminar flags):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 DAY 98
DEBT-CRYPTO-003a (mlock seed):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 98
TEST-INTEG-1 (pipeline E2E):          ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 98
TEST-INTEG-2 (etcd JSON round-trip):  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 98
SECURITY_MODEL.md:                    ████████████████████ 100% ✅  DAY 97
DEBT-INFRA-001 (Debian Trixie):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 105+
DEBT-INFRA-002 (rng-tools5):          ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 105+
DOCS-2 (AppArmor profiles):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 98+
Fast Detector Config (DEBT-FD-001):   ████░░░░░░░░░░░░░░░░  20% 🟡  PHASE 2
FEAT-RANSOM-*:                        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post DEBT-FD-001
ENT-*:                                ░░░░░░░░░░░░░░░░░░░░   0% ⏳  largo plazo
```

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| Sentinel correctness | -9999.0f fuera del dominio ✅ | 79 |
| Thresholds ML | Desde JSON ✅ | 80 |
| Fast Detector dual-path | Path A hardcodeado (DEBT-FD-001) ✅ | 82 |
| Algoritmo cifrado pipeline | ChaCha20-Poly1305 IETF unificado ✅ | 95 |
| Keys AppArmor-compatible | `/etc/ml-defender/{component}/` chmod 600 ✅ | 95 |
| Fail-closed security | pipeline-start depende de provision-check ✅ | 95 |
| Seed como material base | seed → HKDF → session_key. Nunca directo ✅ | 95 |
| Dep order libs | libseed_client → crypto-transport → etcd-client ✅ | 96 |
| HKDF implementation | libsodium 1.0.19 nativo ✅ | 96-97 |
| HKDF context format | `"ml-defender:{component}:{version}:{tx\|rx}"` ✅ | 96 |
| Nonce policy | Contador monotónico 96-bit atómico ✅ | 96-97 |
| C++20 permanente | Migración C++23 solo si kernel/eBPF lo exige ✅ | 96 |
| Error handling | `throw` en todo el pipeline ✅ | 96 |
| Cifrado obligatorio | SIEMPRE. Sin flag `enabled`. CryptoTransport ✅ | 96-97 |
| Compresión obligatoria | SIEMPRE. Sin flag `enabled`. LZ4 ✅ | 96 |
| Orden operaciones | LZ4 → ChaCha20 (comprimir antes de cifrar) ✅ | 96 |
| Rotación seeds | Requiere reinicio ordenado ✅ | 96 |
| libsodium versión | 1.0.19 desde fuente, SHA-256 verificado ✅ | 97 |
| CryptoManager | DEPRECADO — CryptoTransport lo sustituye ✅ | 97 |
| CMake sodium path | NO_DEFAULT_PATH → priorizar /usr/local ✅ | 97 |
| P1 HKDF context | Contexto estático. Forward secrecy = rotación de seeds ✅ | 97 |
| P2 Migración CryptoMgr | Big-bang controlado. git tag antes del merge ✅ | 97 |
| P3 mlock() | WARNING + log instructivo, no error fatal ✅ | 97 |
| P4 test-integ | make test-integ separado, no en ctest normal ✅ | 97 |
| haveged | Aceptable desarrollo. Producción: rng-tools5 + hardware RNG ✅ | 97 |
| libsodium build | Parche PHASE 1. DEBT-INFRA-001: migrar a Trixie (13) ✅ | 97 |
| etcd-client rol final | Transporte puro de blobs opacos ✅ | 96 |

---

### Notas del Consejo de Sabios

> DAY 97 (Consejo — unanimidad 5/5):
> "La rotación real de seeds es el mecanismo correcto de forward secrecy.
> Big-bang migration DAY 98. mlock() como warning, no fatal.
> make test-integ separado del ctest normal.
> haveged aceptable en desarrollo — rng-tools5 + hardware RNG en producción."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen

> DAY 96: "El diseño sigue RFC 5869 (HKDF), Signal Protocol y TLS 1.3.
> El contrato en el .hpp es defensa en profundidad contra mal uso."
> — Grok · DeepSeek · Gemini · ChatGPT5 (unanimidad 4/4)

> DAY 95: "El seed no es una clave — es material base. HKDF primero."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (unanimidad 5/5)

---

*Última actualización: Day 97 (post-Consejo) — 25 Mar 2026*
*Branch: feature/plugin-loader-adr012*
*Tests: 22/22 suites ✅*
*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic), Grok, ChatGPT5, DeepSeek, Qwen, Gemini, Parallel.ai*