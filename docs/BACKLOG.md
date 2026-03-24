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

### Day 96 (24 Mar 2026) — seed-client + Makefile dependency order

**libs/seed-client — PHASE 1 completado ✅**

Nueva librería `libseedclient.so` (sin underscore — convención ecosistema Linux:
libssl, libcrypto, libprotobuf). Capa de identidad criptográfica base.
Primer eslabón de la cadena de confianza peer-to-peer para plugin-loader (ADR-012).

```
libs/seed-client/
  CMakeLists.txt
  include/seed_client/seed_client.hpp   ← contrato documentado con caja ║
  src/seed_client.cpp                   ← explicit_bzero destructor + buffer
  tests/test_seed_client.cpp            ← 6 tests CTest
```

**Contrato invariante:**
```
CORRECTO:   seed() → HKDF(seed, context) → session_key → ChaCha20
INCORRECTO: seed() → ChaCha20 directamente  ← sin forward secrecy
```

**Tests: 6/6 ✅ (0.03s)** — unitarios puros, sin dependencias externas.
**Instalado en:** `/usr/local/lib/libseedclient.so.1.0.0`

**Makefile — nueva cadena de dependencias:**
```
libseedclient → crypto-transport → etcd-client → componentes
```
- `crypto-transport-build` depende de `seed-client-build`
- `clean-libs` y `test-libs` incluyen seed-client

**arXiv outreach DAY 96:**
Email enviado a Prof. Yisroel Mirsky (BGU) — yisroel@bgu.ac.il ✅

---

### Day 95 (23 Mar 2026) — Cryptographic Provisioning Infrastructure

- `tools/provision.sh`: 4 modos full/status/verify/reprovision
- Ed25519 keypairs + ChaCha20 seeds (32B) para 6 componentes
- Paths AppArmor-compatible: `/etc/ml-defender/{component}/` chmod 600
- Bloque `identity` en los 6 JSONs de componentes (ADR-013)
- `rag-config.json`: AES-256-CBC eliminado → ChaCha20-Poly1305 unificado
- `pipeline-start` depende de `provision-check` (fail-closed security)
- Vagrantfile: `cryptographic-provisioning` con `run: "once"`
- Consejo DAY 95 (6/7): unanimidad. DEBT-CRYPTO-001/002/003 identificados.

---

### Day 93 — ADR-012 PHASE 1: plugin-loader + ABI validation
- `libplugin_loader.so.1.0.0` → `/usr/local/lib/` ✅
- `libplugin_hello.so` → `/usr/lib/ml-defender/plugins/` ✅
- ABI: `plugin_api_version() = 1` via ctypes ✅

### Day 83 — Ground truth bigFlows + CSV E2E + MERGE TO MAIN
FPR ML = 0.0049% · **tag: v0.83.0-day83-main** ✅

### Days 76–82 — Proto3 Stability · Sentinel · F1=0.9985 · DEBT-FD-001
### Days 63–75 — Pipeline 6/6 · ChaCha20 · FAISS · HMAC · trace_id
### Days 1–62 — Foundation: eBPF/XDP · protobuf · ZMQ · RandomForest C++20

---

## 🔄 EN CURSO / INMEDIATO

### DAY 97 — crypto-transport HKDF + nonce + ADR-020 + integridad E2E

---

## 🔐 BACKLOG CRIPTOGRÁFICO

### DEBT-CRYPTO-002 — HKDF en crypto-transport (🔴 P1 — DAY 97)

**Implementación:** libsodium exclusivamente.
No OpenSSL, no implementación propia. No se reinventa la rueda.

```bash
vagrant ssh -c "pkg-config --modversion libsodium 2>/dev/null || echo 'apt-get install -y libsodium-dev'"
```

**Interfaz objetivo:**
```cpp
// Constructor nuevo — acepta SeedClient, depreca key directa
CryptoTransport(const SeedClient& seed_client,
                const std::string& context = "ml-defender:transport:v1");
// HKDF-SHA256(seed, salt=zeros, info=context, len=32) → session_key
// CryptoTransport: movible, no copiable (RAII de session keys)
```

**Context string — formato canónico (DAY 96):**
```
"ml-defender:{component}:{version}:{direction}"
// Ejemplos:
"ml-defender:sniffer:v1:tx"
"ml-defender:sniffer:v1:rx"
"ml-defender:ml-detector:v1:tx"
```
Incluir dirección (tx/rx) evita colisiones entre pares.
Incluir versión garantiza que un cambio de protocolo no mezcla claves.
Alineado con RFC 5869 (HKDF), Signal Protocol, TLS 1.3, Noise Protocol.

**Nota C++20:** El pipeline se mantiene en C++20 por dependencia con eBPF/XDP.
Migración a C++23 solo si lo exige el kernel Linux / nueva tecnología eBPF,
tras estudio serio en rama dedicada. `std::expected` diferido indefinidamente.
Error handling: `throw` consistente en todo el pipeline.

### DEBT-CRYPTO-001 — Nonce management (🔴 P1 — DAY 97)

**Decisión (Consejo DAY 96, Gemini):**
Contador monotónico de 96 bits por sesión. No aleatorio puro —
seeds persistentes implican riesgo de reúso entre reinicios.
Responsabilidad exclusiva de `CryptoTransport`, nunca de `SeedClient`.

### DEBT-CRYPTO-003a — mlock() sobre buffer del seed (🟡 P2 — DAY 97)

```cpp
// En seed_client.cpp, tras leer seed.bin:
mlock(seed_.data(), seed_.size());
// explicit_bzero en destructor ya implementado ✅
```
Evita que el material criptográfico llegue al swap (memoria virtual → disco).
Crítico en sistemas con RAM limitada (hospitales, escuelas).

### DEBT-CRYPTO-003b — Entropy check en provision.sh (🟡 P2 — DAY 97)

```bash
avail=$(cat /proc/sys/kernel/random/entropy_avail)
[ "$avail" -lt 256 ] && apt-get install -y haveged && systemctl start haveged
```

### FEAT-CRYPTO-1 — Rotación de claves sin downtime (P2 — DAY 98+)

⚠️ **Advertencia explícita:** La rotación de seeds con `make provision-reprovision`
**requiere reinicio ordenado de toda la pipeline**. Componentes en vuelo siguen
usando la clave derivada de la semilla antigua; componentes nuevos usarán la nueva.
Estado "split-brain" si se reprovision sin reiniciar. Documentar en SECURITY_MODEL.md.

### FEAT-CRYPTO-2 — Handshake efímero (Noise simplificado) (P3 — PHASE 2)
### FEAT-CRYPTO-3 — TPM 2.0 / HSM enterprise (P3 — ENT-8)

---

## 🔴 ADR-020 — Cifrado + compresión siempre obligatorios (P1 — DAY 97)

**Decisión (Alonso + Consejo DAY 96):**

Los JSONs de contrato tienen flags `encryption.enabled` y `compression.enabled`.
Esto abre la ventana a estados inconsistentes — un componente cifra, otro no →
caos en el pipeline, imposible de depurar.

**Resolución:**
- **Cifrado:** SIEMPRE obligatorio en cuanto seed disponible. Sin opción de desactivar.
- **Compresión:** SIEMPRE cuando sea posible. Sin opción de desactivar.
- Eliminar los flags `enabled` de los JSONs de contrato de los 6 componentes.
- Comportamiento **determinista, coherente y seguro** en todo momento.

**Estado del orden de operaciones:**
```
SUBIDA a etcd-server:     LZ4(json) → ChaCha20(compressed_bytes)
BAJADA desde etcd-server: ChaCha20_decrypt → LZ4_decompress → json

Orden correcto: comprimir ANTES de cifrar.
Datos ya cifrados tienen entropía máxima → LZ4 es inútil sobre ellos.
```

⚠️ **Este flujo está diseñado pero sin test E2E que lo verifique.**
Los tests unitarios pasan. El flujo completo (incluida lectura posterior
por rag-local) no está probado. Tarea: TEST-INTEG-2 (ver abajo).

La compresión ocurre solo 6 veces en el ciclo de vida del pipeline (una por
componente al subir JSON a etcd-server). No es un hot-path. La ganancia de
espacio es secundaria — la coherencia de comportamiento es primaria.

**Acción:** Crear ADR-020. Eliminar flags `enabled` de JSONs en DAY 97.

---

## 🧪 TESTS DE INTEGRACIÓN E2E (P1 — DAY 97-98)

### TEST-INTEG-1 — Pipeline completo: provision → seed → crypto → mensaje

Verifica la cadena de confianza de extremo a extremo:
1. `provision.sh` genera seed.bin
2. `SeedClient` carga seed, aplica mlock()
3. `CryptoTransport` deriva session_key via HKDF (libsodium)
4. Sniffer cifra mensaje y envía
5. ml-detector recibe, descifra, procesa
6. Verificar integridad: mensaje llegó byte-a-byte correcto

### TEST-INTEG-2 — Upload JSON a etcd-server: cifrado + compresión

Verifica el contrato de transporte del JSON de componente:
1. JSON de componente → LZ4 → ChaCha20 → etcd-server
2. etcd-server almacena blob cifrado
3. Lectura posterior → ChaCha20_decrypt → LZ4_decompress → JSON original
4. Verificar round-trip byte-a-byte idéntico

Este test es **prerequisito** para que rag-local pueda leer y actualizar
campos de un JSON en etcd-server (funcionalidad en backlog, no implementada).

---

## 📋 DOCUMENTACIÓN PENDIENTE (DAY 97-98)

### DOCS-1 — SECURITY_MODEL.md
- Límites PHASE 1 (chmod 0600, sin TPM)
- **Rotación requiere reinicio ordenado** — advertencia explícita
- Roadmap PHASE 2 (TPM 2.0, ENT-8)
- Referencia RFC 5869 como base del diseño HKDF

### DOCS-2 — Perfiles AppArmor por componente
```
apparmor/
  usr.sbin.ml-defender.sniffer
  usr.sbin.ml-defender.ml-detector
  ... (6 perfiles — rutas permitidas de lectura)
```
Si hace falta ADR para el enfoque, se crea.

---

## 📋 BACKLOG — COMMUNITY & FEATURES

### 🟥 P0 — Paper arXiv
- Draft v5 ✅ DAY 88 · LaTeX ✅ DAY 89
- Email Sebastian Garcia ✅ (sin respuesta)
- Email Yisroel Mirsky ✅ DAY 96 (esperando)
- Si sin respuesta 48h → Tier 3: Martin Grill (coautor CTU-13) o Battista Biggio

### 🟧 P1 — Fast Detector Config (DEBT-FD-001)
Migrar Path A a JSON. Prerequisito bloqueante para FEAT-RANSOM-*.

### 🟨 P2 — Expansión ransomware (prerequisito: DEBT-FD-001)
FEAT-RANSOM-1 (WannaCry) · FEAT-RANSOM-2 (Neris Extended) ·
FEAT-RANSOM-3 (DDoS) · FEAT-RANSOM-4 (Ryuk/Conti) · FEAT-RANSOM-5 (LockBit — BLOQUEADO PHASE2)

### 🟨 P2 — Pipeline reentrenamiento
FEAT-RETRAIN-1/2/3 (dataset balanceado · RF C++20 · A/B testing)

### 🟩 P3 — Enterprise
ENT-1 Federated · ENT-2 Attack Graph · ENT-3 P2P Seed ·
ENT-4 Hot-Reload · ENT-5 rag-world · ENT-7 OpenTelemetry · ENT-8 HSM

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:           ████████████████████ 100% ✅
HMAC Infrastructure:                  ████████████████████ 100% ✅
Proto3 Pipeline Stability:            ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
CSV Pipeline (ml-detector+firewall):  ████████████████████ 100% ✅
Paper arXiv (draft v5 + LaTeX):       ████████████████████ 100% ✅
Cryptographic Provisioning PHASE 1:   ████████████████████ 100% ✅  DAY 95
Identity blocks (6 JSONs):            ████████████████████ 100% ✅  DAY 95
seed-client (libseedclient):          ████████████████████ 100% ✅  DAY 96
Makefile dep order:                   ████████████████████ 100% ✅  DAY 96
plugin-loader ADR-012 PHASE 1:        ████████████████░░░░  80% 🟡  integración sniffer DAY 97
DEBT-CRYPTO-002 (HKDF libsodium):     ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 DAY 97
DEBT-CRYPTO-001 (nonce 96-bit):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 DAY 97
DEBT-CRYPTO-003a (mlock seed):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 97
DEBT-CRYPTO-003b (entropy check):     ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P2 DAY 97
crypto-transport refactor (HKDF):     ░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 DAY 97
ADR-020 (cifrado siempre obligatorio):░░░░░░░░░░░░░░░░░░░░   0% ⏳  P1 DAY 97
TEST-INTEG-1 (pipeline E2E):          ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 97-98
TEST-INTEG-2 (etcd JSON round-trip):  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 97-98
DOCS-1 (SECURITY_MODEL.md):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 97-98
DOCS-2 (AppArmor profiles):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 97-98
etcd-client deprecación seed:         ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 98
Fast Detector Config (DEBT-FD-001):   ████░░░░░░░░░░░░░░░░  20% 🟡  PHASE 2
trace_id correlación:                 ████████████████░░░░  80% 🟡  2 fallos pendientes
FEAT-RANSOM-*:                        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post DEBT-FD-001
ENT-*:                                ░░░░░░░░░░░░░░░░░░░░   0% ⏳  largo plazo
```

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| Sentinel correctness | -9999.0f fuera del dominio ✅ | 79 |
| Thresholds ML | Desde JSON ✅ | 80 |
| Fast Detector dual-path | Path A hardcodeado (DEBT-FD-001), Path B JSON ✅ | 82 |
| Algoritmo cifrado pipeline | ChaCha20-Poly1305 unificado ✅ | 95 |
| Keys AppArmor-compatible | `/etc/ml-defender/{component}/` chmod 600 ✅ | 95 |
| Fail-closed security | pipeline-start depende de provision-check ✅ | 95 |
| Seed como material base | seed → HKDF → session_key. Nunca directo ✅ | 95 |
| Dep order libs | libseedclient → crypto-transport → etcd-client ✅ | 96 |
| Naming convención Linux | libseedclient sin underscore ✅ | 96 |
| HKDF implementation | libsodium exclusivamente ✅ | 96 |
| HKDF context format | `"ml-defender:{component}:{version}:{tx\|rx}"` ✅ | 96 |
| Nonce policy | Contador monotónico 96-bit por sesión ✅ | 96 |
| C++20 permanente | Migración C++23 solo si kernel/eBPF lo exige + estudio serio ✅ | 96 |
| Error handling | `throw` en todo el pipeline. `std::expected` diferido indefinidamente ✅ | 96 |
| Cifrado obligatorio | SIEMPRE. Eliminar flag `enabled`. Sin opción de desactivar ✅ | 96 |
| Compresión obligatoria | SIEMPRE cuando posible. Sin opción de desactivar ✅ | 96 |
| Orden operaciones | LZ4 → ChaCha20 (comprimir antes de cifrar) ✅ | 96 |
| Rotación seeds | Requiere reinicio ordenado. Documentar explícitamente ✅ | 96 |
| Modelo fundacional | Un único RF ensemble que lo haya visto todo ✅ | 89 |
| etcd-client rol final | Transporte puro de blobs opacos — sin gestión de seed ✅ | 96 |

---

### Notas del Consejo de Sabios

> DAY 81: "El fast detector ya es el backbone que protege. El ml-detector es hoy un
> observador silencioso y una fábrica de datos."

> DAY 95: "El seed no es una clave — es material base. HKDF primero.
> Sin esta piedra angular, ENT-1, ENT-3 y ENT-5 serían inseguros por diseño."
> — ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (unanimidad 5/5)

> DAY 96: "El diseño sigue RFC 5869 (HKDF), Signal Protocol y TLS 1.3.
> El contrato en el .hpp es defensa en profundidad contra mal uso.
> Esto no es 'solo leer 32 bytes' — es diseño criptográfico con intención."
> — Grok · DeepSeek · Gemini · ChatGPT5 (unanimidad 4/4)

---

*Última actualización: Day 96 (post-Consejo) — 24 Mar 2026*
*Branch: feature/plugin-loader-adr012*
*Tests: 39/39 ✅*
*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic), Grok, ChatGPT5, DeepSeek, Qwen, Gemini, Parallel.ai*