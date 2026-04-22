# aRGus NDR — BACKLOG
*Última actualización: DAY 125 — 22 Abril 2026*

---

## 📐 Criterio de compleción

| Estado | Criterio |
|---|---|
| ✅ 100% | Implementado + probado en condiciones reales + resultado documentado |
| 🟡 80% | Implementado + compilando + smoke test pasado, sin validación E2E completa |
| 🟡 60% | Implementado parcialmente o con valores placeholder conocidos |
| ⏳ 0% | No iniciado |

---

## 📋 POLÍTICA DE DEUDA TÉCNICA

- **Bloqueante:** se cierra dentro de la feature en que se detectó. No hay merge a main sin test verde.
- **No bloqueante con feature natural:** se asigna a la feature destino. Documentada con ID de feature.
- **No bloqueante sin feature natural:** se acumula hasta abrir `feature/tech-debt-cleanup` (3+ DEBTs sin destino claro).
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA DE SCRIPTS:** Lógica compleja → `tools/script.sh`, nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().
- **REGLA PERMANENTE (DAY 124 — Consejo 7/7):** Ningún fix de seguridad en código de producción se mergea sin test de demostración RED→GREEN. El test debe fallar con el código antiguo y pasar con el nuevo. Sin excepciones.
- **REGLA PERMANENTE (DAY 125 — Consejo 8/8):** Todo fix de seguridad incluye: (1) unit test sintético, (2) property test de invariante, (3) test de integración en el componente real. Sin excepciones.

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa | x86-debug, imagen Vagrant con todas las herramientas, build-debug. Para investigación y desarrollo diario. |
| **aRGus-production** | 🟡 Pendiente de cocinar | x86-apparmor + arm64-apparmor. Imágenes Debian optimizadas, sin herramientas de desarrollo. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ No iniciada | Apéndice científico. Kernel seL4, libpcap (no eBPF/XDP), sniffer reescrito en monohilo. Branch independiente. Nunca se mergeará a main salvo sorpresa. |

---

## ✅ CERRADO DAY 125

### DEBT-GITIGNORE-TEST-SOURCES-001
- **Status:** ✅ CERRADO DAY 125
- **Fix:** `**/build/**/test_*` + excepciones `!test_*.cpp` / `!test_*.hpp`
- **Bonus:** 47 fuentes de test versionadas en commit atómico separado

### DEBT-INTEGER-OVERFLOW-TEST-001
- **Status:** ✅ CERRADO DAY 125
- **Fix:** `compute_memory_mb()` extraída a `memory_utils.hpp` header-only. Aritmética `double` directa — `int64_t` insuficiente para valores extremos (`LONG_MAX/4096 * 8192` desborda `int64_t`, no `double`).
- **Tests:** 4/4 RED→GREEN: unit sintético (A) + valor extremo (GREEN) + property nunca negativo (C) + property monotonicidad (C)
- **Hallazgo:** El property test encontró un bug que el unit test no cubría. Valida Opción C del Consejo DAY 124.
- **Pendiente DAY 126:** Añadir `MAX_REALISTIC_MEMORY_MB` como bound en property test + log warning en producción (ver DEBT-MEMORY-UTILS-BOUNDS-001)

### DEBT-SAFE-PATH-TEST-RELATIVE-001
- **Status:** ✅ CERRADO DAY 125
- **Fix:** Test 10 `RelativePathResolvesBeforePrefixCheck` en `contrib/safe-path/tests/test_safe_path.cpp`
- **Hallazgo colateral:** `SeedRejectSymlink` fallaba pre-existente → DEBT-SAFE-PATH-SEED-SYMLINK-001

### DEBT-SAFE-PATH-TEST-PRODUCTION-001 (rag-ingester)
- **Status:** ✅ CERRADO DAY 125 (rag-ingester)
- **Fix:** `test_config_parser_traversal.cpp` — 3 tests: `RejectNonExistentPath`, `RejectEmptyPath`, `ProductionConfigLoadsCorrectly`. Usa config real de producción (el JSON es la ley).
- **Hallazgo de diseño:** prefix derivado del input → DEBT-CONFIG-PARSER-FIXED-PREFIX-001 (promovido a bloqueante por Consejo 8/8)
- **Pendiente:** seed-client y firewall-acl-agent → DEBT-PRODUCTION-TESTS-REMAINING-001

### DEBT-CRYPTO-TRANSPORT-CTEST-001
- **Status:** ✅ CERRADO DAY 125
- **Causa raíz:** `test_crypto_transport.cpp` y `test_integ_contexts.cpp` creaban `seed.bin` con `0600` (owner_read|owner_write). `safe_path::resolve_seed` exige estrictamente `0400`.
- **Fix:** `perm_options::replace` con solo `owner_read` en ambos ficheros
- **Tests:** 5/5 ctest 100% PASSED

### Gate final DAY 125
- `vagrant halt → make up → make bootstrap → make test-all`: **ALL TESTS PASSED** desde VM fría

---

## ✅ CERRADO DAY 124

### ADR-037 — Static Analysis Security Hardening (safe_path)
- **Status:** ✅ CERRADO DAY 124 — mergeado a main (commit 8bf83b90)
- **Tag:** `v0.5.1-hardened`
- **Tests:** ALL TESTS PASSED · 6/6 RUNNING · TEST-PROVISION-1 8/8 VERDE

---

## 🔴 DEUDA ABIERTA — Bloqueante para merge a main

**Decisión Consejo 8/8 DAY 125 (Opción B):** La rama `fix/day125-debt-closure` NO se mergea a main hasta que las siguientes deudas estén cerradas con tests RED→GREEN.

---

### DEBT-SAFE-PATH-SEED-SYMLINK-001
**Severidad:** 🔴 Crítica | **Bloqueante para merge:** Sí | **Target:** DAY 126
**Origen:** DAY 125 — pre-existente, confirmado con `git stash`
**Descripción:** `resolve_seed()` no rechaza symlinks dentro del prefix. `SafePathTest.SeedRejectSymlink` falla. Un symlink es un vector clásico de TOCTOU (Time-of-Check to Time-of-Use): el atacante puede crear symlink legítimo dentro del prefix apuntando al seed real, esperar el check, y cambiar el destino entre check y `open()`.

**Veredicto Consejo 8/8:** ESTRICTO. Sin flag configurable. `seed.bin` es material criptográfico de nivel máximo. Si CI/CD necesita symlinks para seeds, el CI/CD está mal configurado, no el código. `provision.sh` ya genera seeds reales con `0400`.

**Fix:**
```cpp
// safe_path.hpp — resolve_seed()
struct stat st;
if (lstat(path.c_str(), &st) != 0)
    throw std::runtime_error("[safe_path] lstat failed: " + path);
if (S_ISLNK(st.st_mode))
    throw std::runtime_error("[safe_path] SECURITY VIOLATION — symlink rejected for seed: " + path);
// ... verificar 0400 ...
```

**Test de cierre:** `SafePathTest.SeedRejectSymlink` RED→GREEN — debe fallar con código antiguo, pasar con nuevo.

---

### DEBT-CONFIG-PARSER-FIXED-PREFIX-001
**Severidad:** 🔴 Alta | **Bloqueante para merge:** Sí | **Target:** DAY 126
**Origen:** DAY 125 — hallazgo de diseño en `test_config_parser_traversal`
**Descripción:** `config_parser` deriva el `allowed_prefix` de `safe_path` del directorio padre del propio `config_path` de entrada. Si el atacante controla el path, controla el prefix → bypass completo de la validación. Viola el principio de confianza mínima.

**Veredicto Consejo 8/8:** Añadir parámetro `allowed_prefix` explícito con default `/etc/ml-defender/`. El prefix nunca debe derivarse del input.

**Fix:**
```cpp
// config_parser.hpp
static Config load(const std::string& config_path,
                   const std::string& allowed_prefix = "/etc/ml-defender/");
```

**Implicaciones:**
- Producción: default funciona sin cambios
- Dev con symlinks (DEBT-DEV-PROD-SYMLINK-001): default funciona con `/etc/ml-defender/` → symlink
- Tests existentes: actualizar llamadas con prefix explícito
- Auditar `contrib/` y `tools/` por callers legacy

**Test de cierre:** `ConfigParserTraversal.RejectDotDotTraversal` RED→GREEN con prefix fijo.

---

### DEBT-PRODUCTION-TESTS-REMAINING-001
**Severidad:** 🔴 Alta | **Bloqueante para ADR-038:** Sí | **Target:** DAY 126
**Origen:** DAY 125 — Consejo 8/8 exige cobertura completa antes de ADR-038
**Descripción:** `seed-client` y `firewall-acl-agent` no tienen tests RED→GREEN de path traversal propios. `rag-ingester` está cubierto (DAY 125). El principio: cada punto de entrada externo debe tener su propio test de explotación. Un componente sin test es una promesa sin firma.

**Componentes pendientes:**

| Componente | Criticidad | Tests requeridos |
|-----------|------------|-----------------|
| `seed-client` | 🔴 Máxima — maneja material criptográfico | path traversal en carga de `seed.bin` |
| `firewall-acl-agent` | 🔴 Alta — controla reglas de red | config path traversal |

**Test de cierre:** `test_seed_client_traversal` PASSED · `test_firewall_config_traversal` PASSED · integrados en `make test-all`

---

### DEBT-MEMORY-UTILS-BOUNDS-001
**Severidad:** 🟡 Media | **Bloqueante para merge:** Sí (pequeño, DAY 126 mañana) | **Target:** DAY 126
**Origen:** DAY 125 — feedback Consejo 8/8 sobre P1
**Descripción:** `compute_memory_mb()` es `noexcept` pero no tiene guard de rango realista. Un bug upstream que pase valores absurdos produce métricas silenciosamente incorrectas. La función mantiene `noexcept` (componente de monitoring — mejor métrica incorrecta que crash), pero debe loguear warning si supera el bound realista.

**Fix:**
```cpp
// memory_utils.hpp
constexpr double MAX_REALISTIC_MEMORY_MB = 1024.0 * 1024.0; // 1 TB en MB

[[nodiscard]] inline double compute_memory_mb(long pages, long page_size) noexcept {
    const double result = (static_cast<double>(pages) * static_cast<double>(page_size))
                          / (1024.0 * 1024.0);
    // No throw (noexcept) pero log si absurdo — mejor métrica incorrecta que componente caído
    if (result > MAX_REALISTIC_MEMORY_MB || result < 0.0) {
        // spdlog::warn("[memory_utils] result out of realistic range: {} MB", result);
    }
    return result;
}
```

**Test de cierre:** `PropertyNeverNegative` actualizado con `EXPECT_LE(result, MAX_REALISTIC_MEMORY_MB)` · `RealisticBounds` test añadido.

---

## 🟡 DEUDA ABIERTA — Bloqueante (post-merge, pre-ADR-038)

### DEBT-SNYK-WEB-VERIFICATION-001
**Severidad:** 🟡 Media | **Bloqueante:** Sí (científicamente) | **Target:** DAY 126
**Origen:** DAY 124 — verificación solo con Snyk CLI macOS, no con Snyk web
**Descripción:** Los 23 findings originales de Snyk web no han sido re-verificados con Snyk web post-fix. No podemos afirmar cierre completo de ADR-037 hasta ejecutar Snyk web sobre `v0.5.1-hardened` / `v0.5.2`.

**Test de cierre:** Snyk web report → 0 findings en código C++ de producción

---

## 🟢 DEUDA ABIERTA — No bloqueante

### DEBT-PROPERTY-TESTING-RAPIDCHECK-001
**Severidad:** 🟢 Media | **Bloqueante:** No | **Target:** DAY 127
**Origen:** DAY 125 — Consejo 8/8 recomienda adopción sistémica
**Descripción:** El hallazgo de DAY 125 (property test encontró bug que unit test no cubría) valida la adopción de property testing sistémico. `rapidcheck` es la librería recomendada por el Consejo (8/8): header-only, C++20, integración Google Test, shrinking automático, sin dependencias problemáticas en Debian Bookworm.

**Plan de adopción:**
- Fase 1 (DAY 127): `rapidcheck` como submódulo en `third_party/`, property tests para `safe_path` y `memory_utils`
- Fase 2 (post-ADR-038): expandir a `crypto-transport`, `plugin-loader`
- Gate CI: fallo en property test = bloqueo de merge

**Regla permanente (Consejo 8/8):** Todo fix de seguridad incluye property test de invariante si aplica.

**Test de cierre:** `rapidcheck` integrado · `SafePathProps.ResolveNeverEscapesPrefix` PASSED · `MemoryUtils.NeverNegative` PASSED

---

### DEBT-DEV-PROD-SYMLINK-001
**Severidad:** 🟢 Media | **Bloqueante:** No | **Target:** DAY 127
**Origen:** DAY 124 — asimetría dev/prod resuelta provisionalmente con `weakly_canonical`
**Descripción:** En dev, configs en `/vagrant/component/config/`. En prod, en `/etc/ml-defender/`. La solución correcta es que dev replique la estructura de prod mediante symlinks en el Vagrantfile. Esto se vuelve imprescindible cuando DEBT-CONFIG-PARSER-FIXED-PREFIX-001 esté cerrado (prefix fijo `/etc/ml-defender/`).

**Veredicto Consejo (6/7):** Opción B — symlinks en Vagrantfile. Código siempre ve `/etc/ml-defender/`.

**Test de cierre:** `make bootstrap` con symlinks → `resolve()` usa siempre `/etc/ml-defender/` → ALL TESTS VERDE

---

### DEBT-PROVISION-PORTABILITY-001
**Severidad:** 🟢 Media | **Bloqueante:** No | **Target:** DAY 128
**Origen:** DAY 124 — `vagrant` hardcodeado en `chown` de `provision.sh`
**Descripción:** En producción bare metal o cualquier hipervisor distinto de Vagrant, el service user será diferente.

**Fix:** `ARGUS_SERVICE_USER="${ARGUS_SERVICE_USER:-vagrant}"` al inicio de `provision.sh`.

**Test de cierre:** `provision.sh` con `ARGUS_SERVICE_USER=testuser` → seeds con permisos `0400 testuser:testuser`

---

## 🔵 BACKLOG — Deuda de seguridad crítica (pre-producción)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF. SecureBuffer C++20. | Valgrind/ASan: seed no permanece en heap | feature/crypto-hardening |
| **DEBT-SNIFFER-SEED** | Unificar sniffer bajo SeedClient | sniffer arranca con SeedClient | feature/crypto-hardening |
| **docs/CRYPTO-INVARIANTS.md** | Tabla invariantes criptográficos + tests | Fichero existe con tabla completa | feature/crypto-hardening |

---

## 📋 BACKLOG — P3 Features futuras

### PHASE 5 — Loop Adversarial (→ feature/adr038-acrl)

| ID | Tarea | Gates mínimos |
|----|-------|--------------|
| **DEBT-PENTESTER-LOOP-001** | ACRL: Caldera → eBPF capture → XGBoost warm-start → Ed25519 sign → hot-swap | G1: reproducibilidad · G2: ground-truth flow · G3: ≥3 ATT&CK · G4: RFC-válido · G5: sandbox |
| **ADR-038** | ACRL ADR formal | Aprobado por Consejo |

### Feature enterprise — Reentrenamiento distribuido

| ID | Tarea |
|----|-------|
| **FEAT-CLOUD-RETRAIN-001** | Reentrenamiento → cloud aRGus + CSVs anonimizados → validación → transferencia a flota. |

### Variantes de producción (ADR-029)

| Variante | Tarea | Feature destino |
|----------|-------|----------------|
| **aRGus-production x86** | Imagen Debian cocinada apparmor x86 + Vagrantfile | feature/production-images |
| **aRGus-production arm64** | Imagen Debian cocinada apparmor arm64 + Vagrantfile | feature/production-images |
| **aRGus-seL4** | kernel seL4, libpcap, sniffer monohilo reescrito. Branch independiente. | feature/sel4-research |

### Paper arXiv:2604.04952

| Tarea | Target |
|-------|--------|
| §5 actualizar con lecciones DAY 124-125 (TDH, property testing, dev/prod parity) | Draft v17 · DAY 126-127 |
| §5.3 "Property Testing as a Security Fix Validator" — hallazgo F17 | Draft v17 |

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| **Test RED→GREEN obligatorio** | Todo fix de seguridad requiere test de demostración antes del merge. Sin excepciones. | Consejo 7/7 · DAY 124 |
| **Property test obligatorio** | Todo fix de seguridad incluye property test de invariante si aplica. | Consejo 8/8 · DAY 125 |
| **double para compute_memory_mb** | Aritmética double directa — int64_t insuficiente para valores extremos. | DAY 125 |
| **Symlinks en seeds: NO** | resolve_seed() rechaza symlinks estrictamente. Sin flag. CI/CD se adapta, no el código. | Consejo 8/8 · DAY 125 |
| **ConfigParser prefix fijo** | allowed_prefix explícito, default /etc/ml-defender/. Nunca derivado del input. | Consejo 8/8 · DAY 125 |
| **rapidcheck para property testing** | Adoptar rapidcheck como submodule. Fallo en property test = bloqueo de merge. | Consejo 8/8 · DAY 125 |
| **Paper §5 — lecciones TDH** | Incluir en §5 del paper actual, no reservar para follow-up. | Consejo 8/8 · DAY 125 |
| **ARGUS_SERVICE_USER** | Variable de entorno para service user. Default `vagrant`. | Consejo 6/7 · DAY 124 |
| **Asimetría dev/prod** | Opción B: symlinks en Vagrantfile. Código siempre usa `/etc/ml-defender/`. | Consejo 6/7 · DAY 124 |
| **safe_path header-only** | `contrib/safe-path/` — cero dependencias, C++20 puro. | Consejo 7/7 · DAY 123 |
| **Seeds 0400** | Seeds deben tener permisos `0400` (solo owner, solo lectura). | Consejo 7/7 · DAY 124 |
| **Paper — honestidad** | Incluir limitaciones en §5. La honestidad fortalece credibilidad científica. | Consejo 7/7 · DAY 124 |
| **Tres variantes** | aRGus-dev · aRGus-production (x86+ARM apparmor) · aRGus-seL4 (apéndice científico). | DAY 124 |

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:             ████████████████████ 100% ✅
HMAC Infrastructure:                    ████████████████████ 100% ✅
F1=0.9985 (CTU-13 Neris):              ████████████████████ 100% ✅
CryptoTransport (HKDF+AEAD):            ████████████████████ 100% ✅
ADR-025 Plugin Integrity (Ed25519):     ████████████████████ 100% ✅
TEST-INTEG-4a/4b/4c/4d/4e + SIGN:      ████████████████████ 100% ✅
AppArmor 6/6 enforce:                   ████████████████████ 100% ✅
arXiv:2604.04952 PUBLICADO:             ████████████████████ 100% ✅
PHASE 3 v0.4.0:                         ████████████████████ 100% ✅
PHASE 4 v0.5.0-preprod:                 ████████████████████ 100% ✅
ADR-026 XGBoost Prec=0.9945:            ████████████████████ 100% ✅
Wednesday OOD finding:                  ████████████████████ 100% ✅
make bootstrap idempotente:             ████████████████████ 100% ✅
ADR-037 safe_path v0.5.1-hardened:      ████████████████████ 100% ✅  DAY 124
DEBT-GITIGNORE-TEST-SOURCES-001:        ████████████████████ 100% ✅  DAY 125
DEBT-INTEGER-OVERFLOW-TEST-001:         ████████████████████ 100% ✅  DAY 125
DEBT-SAFE-PATH-TEST-RELATIVE-001:       ████████████████████ 100% ✅  DAY 125
DEBT-SAFE-PATH-TEST-PRODUCTION-001:     ██████████░░░░░░░░░░  50% 🟡  DAY 125 (rag-ingester ✅, pendientes seed-client+firewall)
DEBT-CRYPTO-TRANSPORT-CTEST-001:        ████████████████████ 100% ✅  DAY 125

DEBT-SAFE-PATH-SEED-SYMLINK-001:        ░░░░░░░░░░░░░░░░░░░░   0% 🔴 DAY 126 (bloqueante merge)
DEBT-CONFIG-PARSER-FIXED-PREFIX-001:    ░░░░░░░░░░░░░░░░░░░░   0% 🔴 DAY 126 (bloqueante merge)
DEBT-PRODUCTION-TESTS-REMAINING-001:    ░░░░░░░░░░░░░░░░░░░░   0% 🔴 DAY 126 (bloqueante ADR-038)
DEBT-MEMORY-UTILS-BOUNDS-001:           ░░░░░░░░░░░░░░░░░░░░   0% 🟡 DAY 126
DEBT-SNYK-WEB-VERIFICATION-001:         ░░░░░░░░░░░░░░░░░░░░   0% 🟡 DAY 126
DEBT-PROPERTY-TESTING-RAPIDCHECK-001:   ░░░░░░░░░░░░░░░░░░░░   0% 🟢 DAY 127
DEBT-DEV-PROD-SYMLINK-001:              ░░░░░░░░░░░░░░░░░░░░   0% 🟢 DAY 127
DEBT-PROVISION-PORTABILITY-001:         ░░░░░░░░░░░░░░░░░░░░   0% 🟢 DAY 128
DEBT-CRYPTO-003a (mlock+bzero):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳
DEBT-PENTESTER-LOOP-001 (ACRL):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-DEUDA
ADR-029 aRGus-production images:        ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-DEUDA
ADR-029 aRGus-seL4:                     ░░░░░░░░░░░░░░░░░░░░   0% ⏳ branch independiente
FEAT-CLOUD-RETRAIN-001:                 ░░░░░░░░░░░░░░░░░░░░   0% ⏳ post-ACRL
```

---

## 📝 Notas del Consejo de Sabios — DAY 125 (8/8)

> "DAY 125 es un éxito metodológico, no solo técnico.
>
> El hallazgo más importante del día: el property test PropertyNeverNegative
> encontró un bug latente en el propio fix F17 (int64_t insuficiente para
> LONG_MAX/4096 * 8192) que el unit test sintético no detectó. Esto no es
> anecdótico — es la validación empírica de que la diversidad de técnicas
> de testing es una defensa en profundidad.
>
> Decisiones unánimes (8/8):
> - Symlinks en seeds: RECHAZAR ESTRICTAMENTE. Sin flag configurable.
    >   El material criptográfico no admite compromiso de ergonomía.
> - ConfigParser prefix fijo: BLOQUEANTE DAY 126. Derivar prefix del input
    >   viola el principio de confianza mínima.
> - Tests de producción: seed-client y firewall-acl-agent ANTES de ADR-038.
    >   No hay ACRL sobre componentes sin cobertura de explotación demostrada.
> - rapidcheck: adoptar como submodule. Property test = gate de merge.
> - Paper §5: incluir lecciones TDH ahora. La honestidad metodológica
    >   es una contribución científica per se.
>
> Regla permanente añadida DAY 125:
> 'Todo fix de seguridad incluye: (1) unit test sintético,
>  (2) property test de invariante, (3) test de integración en componente real.'
>
> La rama fix/day125-debt-closure NO se mergea a main hasta cerrar:
> DEBT-SAFE-PATH-SEED-SYMLINK-001, DEBT-CONFIG-PARSER-FIXED-PREFIX-001,
> DEBT-PRODUCTION-TESTS-REMAINING-001. Tag v0.5.2 al merge final.
>
> Frase del día — DeepSeek: 'Un escudo que no se prueba contra su propio filo
> es un escudo que ya está roto.'"
> — Consejo de Sabios (8/8) · DAY 125

---
---

## BACKLOG-FEDER-001 — Convocatoria FEDER: Presentación a Andrés Caro Lindo (UEx/INCIBE)

**Estado:** PENDIENTE — bloqueado por prerequisites técnicos  
**Contacto:** Andrés Caro Lindo — UEx/INCIBE, endorser arXiv:2604.04952  
**Objetivo:** Obtener financiación europea para las Fases 5 y 6 del proyecto

---

### Gate de entrada (prerequisites mínimos antes de contactar)

- [ ] ADR-026 mergeado a main (DEBT-XGBOOST-TEST-REAL-001 cerrada)
- [ ] ADR-029 Variant A (x86 + AppArmor + eBPF/XDP) estable y reproducible
- [ ] ADR-029 Variant B (ARM64 + AppArmor + libpcap) estable y reproducible
- [ ] pcap relay funcional end-to-end en Vagrant (ambas arquitecturas)
- [ ] `make bootstrap` + `make pipeline-status` 6/6 RUNNING verde y reproducible
- [ ] Demo técnica grabable en menos de 10 minutos

---

### Argumento central para la convocatoria

Durante aproximadamente un año de desarrollo en solitario, un investigador
independiente y un Consejo de Sabios compuesto por 8 modelos de IA han
construido un sistema NDR open source orientado a infraestructura crítica
(hospitales, centros educativos, municipios) con las siguientes capacidades
demostradas y documentadas científicamente (arXiv:2604.04952):

- Pipeline de 6 componentes en C++20 con cifrado ChaCha20-Poly1305
- Detección ML con XGBoost (F1=0.9978, ROC-AUC=1.0000 vs CIC-IDS-2017)
- Integridad de plugins via Ed25519 + TOCTOU-safe dlopen
- AppArmor enforce en los 6 componentes (0 denials)
- Metodología TDH (Test-Driven Hardening) reproducible y documentada
- Validado en x86 y ARM64

Este es el límite físico de lo que un investigador independiente puede
producir sin financiación externa. Los fondos FEDER desbloquean lo que
viene a continuación.

Nota de contexto: el encarecimiento del acceso agéntico en plataformas
cloud (Claude Code, GPT-4, etc.) refuerza el argumento de soberanía
tecnológica — un sistema NDR crítico no puede depender de servicios
externos de pago para su inferencia.

---

### FASE 5 — Lo que los fondos desbloquean [REQUIRES-FUNDING] [REQUIRES-HW]

#### 5.1 — Componente de Telemetría y Datos Soberanos
- Nuevo componente: `telemetry-collector`
- Recogida de tráfico real anonimizado desde entornos hospitalarios/municipales
- Pipeline de anonimización de datos (GDPR-compliant) antes de cualquier
  almacenamiento o procesamiento
- Los datos nunca salen del edificio — soberanía total

#### 5.2 — Entrenamiento Local de Modelos
- Entrenamiento y reentrenamiento de modelos ML en hardware propio
- Sin dependencia de cloud externo para la inferencia ni el entrenamiento
- Ciclo cerrado: datos reales → anonimización → entrenamiento → plugin firmado

#### 5.3 — Argus Cloud (Federación de Inteligencia)
- Componente de sincronización federada de inteligencia de amenazas
- Las organizaciones participantes comparten patrones de ataque anonimizados
- Modelo de confianza: ninguna organización expone datos en bruto
- Arquitectura: compatible con ADR-035 (etcd HA, Raft + mTLS)

#### 5.4 — Validación en Hardware de Bajos Recursos [REQUIRES-HW]
Hardware necesario (financiable vía FEDER):
- Raspberry Pi 4 y/o 5 (ARM64) — validar pipeline en edge deployment
- Mini PCs de bajo consumo — perfil hospitalario/municipal realista
- Servidor de entrenamiento local — entrenamiento de modelos sin cloud

Objetivo: determinar el perfil mínimo de hardware para cada modalidad
de despliegue (full pipeline / sensor only / inference only).
Esto es en sí mismo un resultado científico publicable.

---

### FASE 6 — Ecosistema de Plugins y Componente Enterprise [POST-FEDER]

#### 6.1 — Integración Wazuh
- Plugin de correlación con Wazuh (HIDS/SIEM)
- Visibilidad completa: tráfico de red (NDR) + comportamiento de host (EDR)
- El NDR ve lo que entra y sale; Wazuh ve lo que ocurre dentro

#### 6.2 — Generación de Flujos
- Plugin de exportación NetFlow / sFlow / IPFIX
- Compatibilidad con infraestructura de red existente en hospitales y municipios

#### 6.3 — Grafos de Ataque
- Visualización de kill chain en tiempo real
- Grafos de propagación lateral
- Exportación compatible con MITRE ATT&CK

#### 6.4 — Plugins ML Especializados
- Detección de anomalías de comportamiento (UEBA ligero)
- Modelos específicos por sector (sanitario, educativo, municipal)
- Arquitectura de plugins ya diseñada — cada modelo es un .so firmado

#### 6.5 — Componente Enterprise
- Dashboard de gestión centralizada
- SLA y soporte para despliegues institucionales
- API REST para integración con SOC existentes
- Modelo de sostenibilidad económica del proyecto open source

---

### Notas estratégicas

- La Variante C (seL4/Genode, ADR-029) queda como investigación futura —
  argumento de "trabajo adicional financiable" para una segunda convocatoria
- ADR-036 (Formal Verification con TLA+/Coq/Isabelle) ídem — requiere
  equipo especializado, no es abordable en solitario
- El benchmark x86 vs ARM64 vs seL4 (delta de coste de seguridad formal)
  es en sí mismo una contribución científica publicable — segunda paper
- El Consejo de Sabios (metodología multi-AI peer review) puede documentarse
  como contribución metodológica independiente

---

### Frase para la presentación

"Hemos llegado al límite físico de lo que un investigador independiente
y ocho modelos de inteligencia artificial pueden construir solos en un año.
El paper documenta ese límite con rigor. Los fondos FEDER construyen lo
que viene después: datos reales, hardware soberano, e inteligencia
compartida entre los hospitales y municipios de Extremadura."

---

*DAY 125 — 22 Abril 2026 · branch fix/day125-debt-closure · Tag pendiente: v0.5.2 post-merge*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*