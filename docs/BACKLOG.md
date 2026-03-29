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

### Day 101 (29 Mar 2026) — ADR-012 PHASE 1b bug fix + ml-detector plugin-loader

**fix(plugin-loader): extract_enabled_objects ✅**

`extract_enabled_list()` reemplazada por `extract_enabled_objects()`:
- Parser corregido para array de objetos `{name, path, active, comment}` en lugar de array de strings
- Filtra `active:false` antes de cargar
- Ruta explícita desde JSON (`path`), no reconstruida desde `directory`
- Smoke test: cero WARNINGs de `name/path/active/comment`

**ADR-012 PHASE 1b — ml-detector ✅**

Plugin-loader integrado en ml-detector con guard `#ifdef PLUGIN_LOADER_ENABLED`:
- `ml-detector/CMakeLists.txt`: find_library + link + define
- `ml-detector/src/main.cpp`: PluginLoader instanciado + load/shutdown
- `ml-detector/config/ml_detector_config.json`: sección `plugins` con hello plugin
- Smoke test: `[plugin:hello] init OK` + `shutdown OK` — invocations=0 overruns=0 errors=0

**arXiv endorser — email enviado ✅**

Email enviado a `andresc@unex.es` (Prof. Andrés Caro Lindo, Cátedra INCIBE-UEx-EPCC).
PDF v6 adjunto. Esperando respuesta.

**Tests: 24/24 suites ✅*
*ADR-012 PHASE 1b: sniffer ✅ + ml-detector ✅**

---

### Day 100 (28 Mar 2026) — ADR-021 + ADR-022 + set_terminate() + CI honesto

**set_terminate() en los 6 main() — ADR-022 fail-closed ✅**

`std::set_terminate()` insertado en todos los componentes:
- sniffer, ml-detector, firewall-acl-agent, rag-ingester, rag, etcd-server
- Comportamiento: log `[FATAL]` + `std::abort()` ante excepción no capturada
- `<exception>` añadido donde faltaba
- 24/24 tests verdes tras el parche

**ADR-021 — deployment.yml SSOT + seed families (FASE 3 documentada) ✅**

Arquitectura de familias de canal documentada:
- `family_A`: sniffer → ml-detector
- `family_B`: ml-detector → firewall
- `family_C`: ml-detector + firewall → rag-ingester
- `deployment.yml` como única fuente de verdad de topología distribuida
- Política de versioning de contextos HKDF (`:v1` → bump solo en cambio semántico)
- Pre-requisito: arXiv submission → milestone gate

**ADR-022 — Threat model formal + Opción 2 descartada ✅**

- Threat model completo documentado (activos, vectores, mitigaciones, límites)
- Opción 2 (`instance_id` en contexto HKDF) descartada formalmente: reproduce el bug de asimetría por diseño
- Caso pedagógico para paper arXiv: "el contexto HKDF identifica el canal, no el emisor"
- TDH validado: TEST-INTEG-1/2/3 detectaron el bug antes de producción

**CI reescrito — honesto y funcional ✅**

- `debian-bookworm` runner eliminado (nunca existió en GitHub Actions)
- Nuevo workflow `ubuntu-latest`: 5 validaciones estáticas que SÍ corren
  - JSON configs válidos · CMakeLists presentes · ADRs presentes
  - `contexts.hpp` simétricos · `set_terminate` en los 6 main()
- Self-hosted runner para full build+test: backlog post-arXiv

**DEBT-CRYPTO-004b — tools/ migración CTX_* ✅ (cerrado — no aplicable)**

`grep -rn "ml-defender:" tools/ --include="*.cpp"` → sin resultados.
`tools/` solo contiene scripts shell. No hay CTX_* que migrar. Deuda cerrada.

---

### Day 99 (27 Mar 2026) — contexts.hpp + TEST-INTEG + fail-closed

**ADR-013 PHASE 2 completado — contextos HKDF simétricos ✅**

Bug crítico corregido: contextos asimétricos entre emisor y receptor.
`crypto-transport/include/crypto_transport/contexts.hpp` — nueva fuente de verdad:

```cpp
// Contexto = canal, no componente
constexpr const char* CTX_SNIFFER_TO_ML  = "ml-defender:sniffer-to-ml-detector:v1";
constexpr const char* CTX_ML_TO_FIREWALL = "ml-defender:ml-detector-to-firewall:v1";
// ... 6 canales, todos simétricos
```

**TEST-INTEG-1/2/3 — gate arXiv ✅**
- TEST-INTEG-1: ping cifrado end-to-end (simétrico → pasa)
- TEST-INTEG-2: round-trip JSON etcd con LZ4+ChaCha20
- TEST-INTEG-3: regresión — contextos asimétricos → MAC failure (caso pedagógico)

**Fail-closed EventLoader + RAGLogger ✅**

`EventLoader` y `RAGLogger` lanzan excepción si `CryptoTransport` falla al inicializar.
Ningún componente arranca con cifrado degradado.

**test_hmac_integration habilitado ✅**

Comentado desde DAY 53. Corregido namespace (`etcd` → `etcd_server`) y macro pollution
(`#undef manager`). Re-enabled y verde.

**Tests: 24/24 suites ✅*
*ADR-012 PHASE 1b: sniffer ✅ + ml-detector ✅** (era 22/22 DAY 98)

---

### Day 98 (26 Mar 2026) — CryptoTransport migración 6/6 componentes

**DEBT-CRYPTO-004 — CryptoManager → CryptoTransport en 6 componentes ✅**

Todos los componentes usan ahora `CryptoTransport` con HKDF-SHA256.
`CryptoManager` deprecado y eliminado de los paths activos.

**DEBT-ETCD-001 — etcd-client usa CryptoTransport ✅**

**ADR-020 JSONs — flags `enabled` eliminados de los 6 configs ✅**

**Tests: 22/22 suites ✅**

---

### Day 97 (25 Mar 2026) — CryptoTransport HKDF + libsodium 1.0.19

**CryptoTransport — HKDF-SHA256 + ChaCha20-Poly1305 + nonce 96-bit ✅**

```
provision.sh → seed.bin → SeedClient → CryptoTransport(HKDF) → ChaCha20-Poly1305
```

- libsodium 1.0.19 desde fuente (SHA-256: `018d79fe...`)
- `provision.sh`: modos full/status/verify/reprovision
- CMake: `NO_DEFAULT_PATH` → priorizar `/usr/local`
- `CryptoManager` DEPRECADO

**ADR-020 documentado ✅ · SECURITY_MODEL.md ✅**

---

### Day 96 (24 Mar 2026) — seed-client + Makefile dep order
### Day 95 (23 Mar 2026) — Cryptographic Provisioning Infrastructure
### Day 93 — ADR-012 PHASE 1: plugin-loader + ABI validation
### Day 83 — Ground truth bigFlows + CSV E2E · tag: v0.83.0-day83-main ✅
### Days 76–82 — Proto3 · Sentinel · F1=0.9985 · DEBT-FD-001
### Days 63–75 — Pipeline 6/6 · ChaCha20 · FAISS · HMAC · trace_id
### Days 1–62 — Foundation: eBPF/XDP · protobuf · ZMQ · RandomForest C++20

---

## 🔄 PRÓXIMO MILESTONE — arXiv submission

### P1 — Antes de enviar el paper

| ID | Tarea | Estado |
|----|-------|--------|
| BARE-METAL | Stress test sin VirtualBox — validar ≥100 Mbps | ⏳ |
| PAPER-FINAL | Incorporar caso pedagógico ADR-022 al paper | ⏳ |
| PAPER-FINAL | Actualizar métricas DAY 100 (24/24 tests, contexts.hpp) | ⏳ |
| DOCS-APPARMOR | 6 perfiles AppArmor por componente | ⏳ |

### P2 — Post-arXiv, pre-FASE 3

| ID | Tarea | Origen |
|----|-------|--------|
| PLUGIN-LOADER-FW | plugin-loader en firewall-acl-agent | ADR-012 PHASE 1b |
| PLUGIN-LOADER-RAG | plugin-loader en rag-ingester | ADR-012 PHASE 1b |
| DEBT-CRYPTO-003a | `mlock()` seed_client.cpp | ADR-022 threat model |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie (libsodium 1.0.19 en apt) | P2 |
| DEBT-INFRA-002 | Sustituir `haveged` por `rng-tools5` + hardware RNG | P2 |
| FEAT-ROTATION-1 | `provision.sh rotate-all` + política SEED_ROTATION_DAYS | P2 |
| DEBT-NAMING-001 | `libseed_client` → `libseedclient` (sin underscore) | P3 |
| ADR-020 | Borrar flags `enabled` de JSONs | DAY tranquilo |

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
// En seed_client.cpp, tras leer seed.bin:
mlock(seed_.data(), seed_.size());
```
**Decisión consolidada (Consejo DAY 97):** WARNING + log instructivo, no error fatal.

### FEAT-ROTATION-1 — Política de rotación de seeds (🟡 P2 — DAY 105+)

```bash
sudo bash tools/provision.sh rotate-all
make pipeline-stop && make pipeline-start
```
- `SEED_ROTATION_DAYS=30` — SSOT en provision.sh
- RAG lee metadata etcd → alerta admin. Nunca ejecuta provision.sh.
- Hot-reload descartado (split-brain risk) — ENT-4 para PHASE 3+.

### FEAT-CRYPTO-2 — Handshake efímero Noise (P3 — PHASE 2)
### FEAT-CRYPTO-3 — TPM 2.0 / HSM enterprise (P3 — ENT-8)

---

## 📋 DOCUMENTACIÓN PENDIENTE

### DOCS-2 — Perfiles AppArmor por componente (6 perfiles) ⏳

---

## 📋 BACKLOG — COMMUNITY & FEATURES

### 🟥 P0 — Paper arXiv
- Draft v5 ✅ · LaTeX ✅
- Sebastian Garcia (CTU Prague) ✅ — respondió, recibió PDF
- Yisroel Mirsky (BGU) ✅ — enviado DAY 96, esperando
- Pendiente: incorporar ADR-022 caso pedagógico + métricas DAY 100

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
SECURITY_MODEL.md:                    ████████████████████ 100% ✅  DAY 97
DEBT-CRYPTO-004 (migrar CryptoMgr):   ████████████████████ 100% ✅  DAY 98
DEBT-ETCD-001 (etcd-client migrar):   ████████████████████ 100% ✅  DAY 98
ADR-020 JSONs (eliminar flags):       ████████████████████ 100% ✅  DAY 98
contexts.hpp (HKDF simétricos):       ████████████████████ 100% ✅  DAY 99
TEST-INTEG-1/2/3 (gate arXiv):        ████████████████████ 100% ✅  DAY 99
fail-closed EventLoader+RAGLogger:    ████████████████████ 100% ✅  DAY 99
test_hmac_integration:                ████████████████████ 100% ✅  DAY 99
set_terminate() 6/6 main():           ████████████████████ 100% ✅  DAY 100
ADR-021 (topology SSOT + families):   ████████████████████ 100% ✅  DAY 100
ADR-022 (threat model + Opción 2):    ████████████████████ 100% ✅  DAY 100
CI honesto (ubuntu-latest):           ████████████████████ 100% ✅  DAY 100
DEBT-CRYPTO-004b (tools/ CTX_*):      ████████████████████ 100% ✅  DAY 100 (N/A)
plugin-loader ADR-012 PHASE 1b:       ████████████████████ 100% ✅  sniffer+ml-detector DAY 101
DEBT-CRYPTO-003a (mlock seed):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 pre-arXiv
DEBT-INFRA-001 (Debian Trixie):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 105+
DEBT-INFRA-002 (rng-tools5):          ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 105+
FEAT-ROTATION-1 (política rotación):  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 105+
DOCS-2 (AppArmor profiles):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 105+
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
| HKDF context format | Contexto = CANAL, no componente ✅ | 99 |
| Nonce policy | Contador monotónico 96-bit atómico ✅ | 96-97 |
| C++20 permanente | Migración C++23 solo si kernel/eBPF lo exige ✅ | 96 |
| Error handling | `throw` en todo el pipeline ✅ | 96 |
| Cifrado obligatorio | SIEMPRE. Sin flag `enabled`. CryptoTransport ✅ | 96-97 |
| Compresión obligatoria | SIEMPRE. Sin flag `enabled`. LZ4 ✅ | 96 |
| Orden operaciones | LZ4 → ChaCha20 (comprimir antes de cifrar) ✅ | 96 |
| Rotación seeds | Requiere reinicio ordenado ✅ | 96 |
| libsodium versión | 1.0.19 desde fuente, SHA-256 verificado ✅ | 97 |
| CryptoManager | DEPRECADO — CryptoTransport lo sustituye ✅ | 97-98 |
| CMake sodium path | NO_DEFAULT_PATH → priorizar /usr/local ✅ | 97 |
| P1 HKDF context | Contexto estático. Forward secrecy = rotación de seeds ✅ | 97 |
| mlock() | WARNING + log instructivo, no error fatal ✅ | 97 |
| haveged | Aceptable desarrollo. Producción: rng-tools5 + hardware RNG ✅ | 97 |
| libsodium build | Parche PHASE 1. DEBT-INFRA-001: migrar a Trixie ✅ | 97 |
| Política rotación seeds | provision.sh único SSOT. rotate-all + pipeline restart ✅ | 97 |
| Hot-reload seeds | NO en PHASE 1 — split-brain risk. ENT-4 para PHASE 3+ ✅ | 97 |
| RAG + rotación | RAG lee metadata etcd, alerta admin. Nunca ejecuta provision.sh ✅ | 97 |
| etcd-client rol final | Transporte puro de blobs opacos ✅ | 96 |
| Opción 2 multi-instancia | DESCARTADA — reproduce bug asimetría por diseño ✅ | 100 |
| FASE 3 seed families | Canal = familia. Un componente puede tener N seeds ✅ | 100 |
| deployment.yml SSOT | Topología declarativa. Pre-req: arXiv ✅ | 100 |
| set_terminate() | fail-closed en los 6 main(). abort() ante excepción no capturada ✅ | 100 |
| CI GitHub Actions | Solo validación estática (ubuntu-latest). Full build = self-hosted ✅ | 100 |

---

### Notas del Consejo de Sabios

> DAY 100 (Consejo — revisión cierre DAY 99):
> "set_terminate() es defensa en profundidad correcta.
> ADR-022 documenta honestamente los límites del modelo de instancia única.
> El caso pedagógico del bug de asimetría pertenece al paper arXiv."
> — Grok · ChatGPT · DeepSeek (origen de las acciones DAY 100)

> DAY 97 (Consejo — unanimidad 5/5):
> "La rotación real de seeds es el mecanismo correcto de forward secrecy.
> mlock() como warning, no fatal. make test-integ separado del ctest normal."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen

> DAY 96: "El diseño sigue RFC 5869 (HKDF), Signal Protocol y TLS 1.3."
> — Grok · DeepSeek · Gemini · ChatGPT5 (unanimidad 4/4)

> DAY 95: "El seed no es una clave — es material base. HKDF primero."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (unanimidad 5/5)

---

*Última actualización: DAY 101 — 29 Mar 2026*
*Branch: feature/bare-metal-arxiv*
*Tests: 24/24 suites ✅*
*ADR-012 PHASE 1b: sniffer ✅ + ml-detector ✅*
*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic), Grok, ChatGPT, DeepSeek, Qwen, Gemini, Parallel.ai*