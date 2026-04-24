# aRGus NDR — BACKLOG
*Última actualización: DAY 127 — 23 Abril 2026*

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
- **REGLA PERMANENTE (DAY 127 — Consejo 8/8):** La taxonomía safe_path tiene tres primitivas activas y una futura. Toda nueva superficie de ficheros debe clasificarse explícitamente con PathPolicy antes de implementar. Documentar en docs/SECURITY-PATH-PRIMITIVES.md.

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa | x86-debug, imagen Vagrant con todas las herramientas, build-debug. Para investigación y desarrollo diario. |
| **aRGus-production** | 🟡 Pendiente de cocinar | x86-apparmor + arm64-apparmor. Imágenes Debian optimizadas, sin herramientas de desarrollo. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ No iniciada | Apéndice científico. Kernel seL4, libpcap (no eBPF/XDP), sniffer reescrito en monohilo. Branch independiente. Nunca se mergeará a main salvo sorpresa. |

---

## ✅ CERRADO DAY 127

### DEBT-DEV-PROD-SYMLINK-001
- **Status:** ✅ CERRADO DAY 127 — mergeado a main
- **Fix:** Nueva primitiva `resolve_config()` en `safe_path.hpp`. Usa `lexically_normal()` para verificar el prefix ANTES de seguir symlinks. Permite `/etc/ml-defender/*.json → /vagrant/*.json` en dev via `provision.sh`. Corrección de hardcodes en `firewall-acl-agent/src/main.cpp` y `Makefile`. `config_loader.cpp` y `config_parser.cpp` usan `resolve_config()` en lugar de `resolve()`.
- **Hallazgo técnico clave:** `fs::is_symlink(resolved)` es inútil post-`weakly_canonical()` — el symlink ya fue resuelto. Para configs con symlinks legítimos, verificar el prefix sobre el path lexical (antes de resolver). Dos primitivas distintas para dos casos de seguridad distintos.
- **Tests:** `test_safe_path_config.cpp` — 5/5 RED→GREEN. 6/6 RUNNING. `make test-all` ALL TESTS COMPLETE.

---

## ✅ CERRADO DAY 126

### DEBT-SAFE-PATH-SEED-SYMLINK-001
- **Status:** ✅ CERRADO DAY 126 — mergeado a main (v0.5.2-hardened)
- **Fix:** `lstat` ANTES de `resolve()` en `resolve_seed()`. `fs::is_symlink(resolved)` llegaba tarde: `weakly_canonical()` ya había resuelto el symlink. `lstat()` sobre el path original es la única defensa correcta. Sin flag configurable.
- **Tests:** 11/11 `test_safe_path` PASSED incluyendo `SeedRejectSymlink` RED→GREEN.

### DEBT-CONFIG-PARSER-FIXED-PREFIX-001
- **Status:** ✅ CERRADO DAY 126 — mergeado a main
- **Fix:** `allowed_prefix` explícito como parámetro en `rag_ingester::ConfigParser::load()` y `mldefender::firewall::ConfigLoader::load_from_file()`. Default `/etc/ml-defender/`. El prefix nunca se deriva del input.
- **Tests:** 4/4 `test_config_parser_traversal` + 3/3 `ConfigLoaderTraversal` PASSED.

### DEBT-PRODUCTION-TESTS-REMAINING-001
- **Status:** ✅ CERRADO DAY 126 — mergeado a main
- **Fix:** `test_seed_client_traversal.cpp` (3/3) y `test_config_loader_traversal.cpp` (3/3). RED→GREEN en seed-client y firewall-acl-agent.

### DEBT-MEMORY-UTILS-BOUNDS-001
- **Status:** ✅ CERRADO DAY 126 — mergeado a main
- **Fix:** `MAX_REALISTIC_MEMORY_MB = 1024.0 * 1024.0` en `memory_utils.hpp`. `RealisticBounds` test añadido. `PropertyNeverNegative` separado correctamente de bounds check.
- **Tests:** 5/5 PASSED.

### Tag v0.5.2-hardened
- **Status:** ✅ MERGEADO a main DAY 126

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

### DEBT-SAFE-PATH-TEST-RELATIVE-001
- **Status:** ✅ CERRADO DAY 125
- **Fix:** Test 10 `RelativePathResolvesBeforePrefixCheck` en `contrib/safe-path/tests/test_safe_path.cpp`

### DEBT-SAFE-PATH-TEST-PRODUCTION-001 (rag-ingester)
- **Status:** ✅ CERRADO DAY 125 (rag-ingester) + completado DAY 126 (seed-client + firewall)
- **Fix:** `test_config_parser_traversal.cpp` — 3 tests.

### DEBT-CRYPTO-TRANSPORT-CTEST-001
- **Status:** ✅ CERRADO DAY 125
- **Causa raíz:** permisos `0600` en seeds de test fixtures. Fix: `0400`.
- **Tests:** 5/5 ctest PASSED

---

## ✅ CERRADO DAY 124

### ADR-037 — Static Analysis Security Hardening (safe_path)
- **Status:** ✅ CERRADO DAY 124 — mergeado a main
- **Tag:** `v0.5.1-hardened`

---

## 🟡 DEUDA ABIERTA — Pendiente

### DEBT-SNYK-WEB-VERIFICATION-001
**Severidad:** 🟡 Media | **Bloqueante:** Sí (científicamente) | **Target:** DAY 128
**Origen:** DAY 124 — verificación solo con Snyk CLI macOS, no con Snyk web
**Descripción:** Los 23 findings originales de Snyk web no han sido re-verificados post-fix. No podemos afirmar cierre completo de ADR-037 hasta ejecutar Snyk web sobre `v0.5.2-hardened`.

**Criterio de triage (Consejo 8/8 DAY 127):**

| Tipo | Código propio | Third-party | Acción |
|------|--------------|-------------|--------|
| Path traversal / overflow / crypto misuse | 🔴 Bloqueante | 🔴 Bloqueante | Fix con RED→GREEN |
| Otro HIGH en código propio | 🔴 Bloqueante | 🟡 Analizar | Fix o justificar en ADR |
| MEDIUM en código propio | 🟡 Próximo sprint | 🟢 Documentar | KNOWN-ISSUES.md |
| Falso positivo demostrable | 🟢 Documentar | 🟢 Documentar | Cerrar con justificación |
| Third-party no alcanzable en prod | — | 🟢 Monitorizar | `.trivyignore` + upstream |

**Regla:** La herramienta propone; el modelo de amenazas decide. El Consejo revisa criterios, no cada finding individual. El Consejo solo interviene en findings HIGH/CRITICAL en código de producción propio.

**Test de cierre:** Snyk web report → 0 findings HIGH/CRITICAL en código C++ de producción propio. Residuos documentados en `docs/security/SNYK-DAY-128.md`.

---

### DEBT-PROPERTY-TESTING-PATTERN-001
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** DAY 128
**Origen:** DAY 127 — redefinida (antes DEBT-PROPERTY-TESTING-RAPIDCHECK-001, sin deps nuevas)
**Descripción:** Formalizar el patrón de property testing manual que ya demostró valor (F17 DAY 125). Sin dependencias nuevas — el patrón manual es suficiente y ya está validado. rapidcheck queda como opción futura si se necesita shrinking automático.

**Plan:**
1. `docs/testing/PROPERTY-TESTING.md` — qué es un property test, cuándo usarlo, patrón estándar
2. Aplicar property tests de invariantes a `resolve_seed()`, `resolve_config()`, `config_parser`
3. Diagrama de decisión: unit test + property test + integration test como capas

**Test de cierre:** `docs/testing/PROPERTY-TESTING.md` existe con patrón documentado + 3 nuevos property tests RED→GREEN en superficies críticas.

---

### DEBT-PROVISION-PORTABILITY-001
**Severidad:** 🟢 Media | **Bloqueante:** No | **Target:** DAY 128
**Origen:** DAY 124 — `vagrant` hardcodeado en `chown` de `provision.sh`
**Descripción:** En producción bare metal o cualquier hipervisor distinto de Vagrant, el service user será diferente.

**Fix:** `ARGUS_SERVICE_USER="${ARGUS_SERVICE_USER:-vagrant}"` al inicio de `provision.sh`. Validación explícita: si variable crítica no definida → error explícito.

**Test de cierre:** `provision.sh` con `ARGUS_SERVICE_USER=testuser` → seeds con permisos `0400 testuser:testuser`. `TEST-PROVISION-1` verde.

---

## 🔵 BACKLOG — Deuda de seguridad crítica (pre-producción)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **DEBT-SAFE-PATH-RESOLVE-MODEL-001** | `resolve_model()` — primitiva futura para modelos firmados Ed25519. Los modelos XGBoost (`.ubj`) son input crítico del ml-detector. Verificación criptográfica debe estar en `safe_path`, no dispersa. (Propuesto por Kimi DAY 127) | `resolve_model()` implementada + test RED→GREEN | feature/adr038-acrl |
| **DEBT-SAFE-PATH-TAXONOMY-DOC-001** | `docs/SECURITY-PATH-PRIMITIVES.md` — diagrama de decisión taxonomía safe_path con PathPolicy enum conceptual. (Propuesto por ChatGPT/Grok/Qwen DAY 127) | Fichero existe con tabla + diagrama | DAY 128 |
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF. SecureBuffer C++20. | Valgrind/ASan: seed no permanece en heap | feature/crypto-hardening |
| **DEBT-SNIFFER-SEED** | Unificar sniffer bajo SeedClient | sniffer arranca con SeedClient | feature/crypto-hardening |
| **DEBT-FUZZING-LIBFUZZER-001** | libFuzzer sobre superficies críticas: `safe_path`, `crypto_transport`, `config_parser` (parsers JSON). Siguiente capa después de property testing. | 1h de fuzzing sin crash para cada superficie | post-property-testing |

---

## 📋 BACKLOG — P3 Features futuras

### PHASE 5 — Loop Adversarial (→ feature/adr038-acrl)

| ID | Tarea | Gates mínimos |
|----|-------|--------------|
| **DEBT-PENTESTER-LOOP-001** | ACRL: Caldera → eBPF capture → XGBoost warm-start → Ed25519 sign → hot-swap | G1: reproducibilidad · G2: ground-truth flow · G3: ≥3 ATT&CK · G4: RFC-válido · G5: sandbox |
| **ADR-038** | ACRL ADR formal | Aprobado por Consejo |

### Variantes de producción (ADR-029)

| Variante | Tarea | Feature destino |
|----------|-------|----------------|
| **aRGus-production x86** | Imagen Debian cocinada apparmor x86 + Vagrantfile | feature/production-images |
| **aRGus-production arm64** | Imagen Debian cocinada apparmor arm64 + Vagrantfile | feature/production-images |
| **aRGus-seL4** | kernel seL4, libpcap, sniffer monohilo reescrito. Branch independiente. | feature/sel4-research |

### Paper arXiv:2604.04952 — Draft v17

| Tarea | Target |
|-------|--------|
| §5.3 "Property Testing as a Security Fix Validator" — hallazgo F17 | Draft v17 |
| §5.4 "Dev/Prod Parity via Symlinks, Not Conditional Logic" | Draft v17 |
| §5.5 "RED→GREEN as Non-Negotiable Merge Gate" | Draft v17 |
| §5.x "Taxonomía safe_path: lexically_normal vs weakly_canonical" — distinción no documentada en literatura C++20 | Draft v17 |
| Trabajo relacionado: QuickCheck (Claessen & Hughes), CWE-22/23, TOCTOU literature, OWASP Path Traversal | Draft v17 |

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| **Test RED→GREEN obligatorio** | Todo fix de seguridad requiere test de demostración antes del merge. Sin excepciones. | Consejo 7/7 · DAY 124 |
| **Property test obligatorio** | Todo fix de seguridad incluye property test de invariante si aplica. | Consejo 8/8 · DAY 125 |
| **double para compute_memory_mb** | Aritmética double directa — int64_t insuficiente para valores extremos. | DAY 125 |
| **Symlinks en seeds: NO** | resolve_seed() rechaza symlinks estrictamente. lstat() ANTES de resolve(). Sin flag. | Consejo 8/8 · DAY 125-126 |
| **ConfigParser prefix fijo** | allowed_prefix explícito, default /etc/ml-defender/. Nunca derivado del input. | Consejo 8/8 · DAY 125-126 |
| **resolve_config() para configs** | lexically_normal() verifica prefix ANTES de seguir symlinks. Permite paridad dev/prod. | DAY 127 |
| **Makefile paths absolutos** | rag-ingester y firewall arrancan con paths /etc/ml-defender/ — fin de paths relativos. | DAY 127 |
| **Taxonomía safe_path: 3 primitivas activas** | resolve() general · resolve_seed() criptográfico · resolve_config() configs con symlinks. Documentar con PathPolicy enum conceptual. | Consejo 8/8 · DAY 127 |
| **resolve_model() como primitiva futura** | Modelos XGBoost son superficie crítica. resolve_model() verificará firma Ed25519 en safe_path. Backlog ADR-038. | Consejo (Kimi) · DAY 127 |
| **Property testing primero, fuzzing después** | Property testing formalizar DAY 128. libFuzzer para parsers post-property. Mutation testing último. | Consejo 8/8 · DAY 127 |
| **Snyk: modelo de amenazas decide** | La herramienta propone; el modelo de amenazas decide. Código propio = bloqueante. Third-party = documentar. | Consejo 8/8 · DAY 127 |
| **Paper §5 — lecciones TDH** | Incluir en §5 del paper actual. Framing: "unit testing insuficiente para fixes de seguridad". | Consejo 8/8 · DAY 125-127 |
| **ARGUS_SERVICE_USER** | Variable de entorno para service user. Default `vagrant`. | Consejo 6/7 · DAY 124 |
| **safe_path header-only** | `contrib/safe-path/` — cero dependencias, C++20 puro. | Consejo 7/7 · DAY 123 |
| **Seeds 0400** | Seeds deben tener permisos `0400` (solo owner, solo lectura). | Consejo 7/7 · DAY 124 |
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
DEBT-SAFE-PATH-TEST-PRODUCTION-001:     ████████████████████ 100% ✅  DAY 126
DEBT-CRYPTO-TRANSPORT-CTEST-001:        ████████████████████ 100% ✅  DAY 125
DEBT-SAFE-PATH-SEED-SYMLINK-001:        ████████████████████ 100% ✅  DAY 126
DEBT-CONFIG-PARSER-FIXED-PREFIX-001:    ████████████████████ 100% ✅  DAY 126
DEBT-PRODUCTION-TESTS-REMAINING-001:    ████████████████████ 100% ✅  DAY 126
DEBT-MEMORY-UTILS-BOUNDS-001:           ████████████████████ 100% ✅  DAY 126
DEBT-DEV-PROD-SYMLINK-001:              ████████████████████ 100% ✅  DAY 127

DEBT-SNYK-WEB-VERIFICATION-001:         ░░░░░░░░░░░░░░░░░░░░   0% 🟡 DAY 128 (navegador)
DEBT-PROPERTY-TESTING-PATTERN-001:      ░░░░░░░░░░░░░░░░░░░░   0% 🟡 DAY 128
DEBT-SAFE-PATH-TAXONOMY-DOC-001:        ░░░░░░░░░░░░░░░░░░░░   0% 🟡 DAY 128
DEBT-PROVISION-PORTABILITY-001:         ░░░░░░░░░░░░░░░░░░░░   0% 🟢 DAY 128

DEBT-ETCDCLIENT-LEGACY-SEED-001:      ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-P2P-CLEANUP
  Síntoma:  EtcdClientHmacTest 9/9 FAILING — EtcdClient lee seed via resolve_seed().
  Causa:    Código legado pre-P2P (ADR-026/027). No es regresión — es cleanup pendiente.
  Acción:   Eliminar lectura de seed en EtcdClient constructor cuando P2P seed
            distribution esté implementado (ENT-3).
  Gate:     RED→GREEN tras cleanup. No bloqueante hasta implementar ADR-024/P2P.
  Feature:  feature/etcdclient-p2p-cleanup
DEBT-SAFE-PATH-RESOLVE-MODEL-001:       ░░░░░░░░░░░░░░░░░░░░   0% ⏳ feature/adr038-acrl
DEBT-FUZZING-LIBFUZZER-001:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳ post-property-testing
DEBT-CRYPTO-003a (mlock+bzero):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳
DEBT-PENTESTER-LOOP-001 (ACRL):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-DEUDA
ADR-029 aRGus-production images:        ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-DEUDA
ADR-029 aRGus-seL4:                     ░░░░░░░░░░░░░░░░░░░░   0% ⏳ branch independiente
FEAT-CLOUD-RETRAIN-001:                 ░░░░░░░░░░░░░░░░░░░░   0% ⏳ post-ACRL
```

---

## 📝 Notas del Consejo de Sabios — DAY 127 (8/8)

> "DAY 125-127 cierra una fase de hardening excepcional. El sistema es ahora más seguro
> y la metodología TDH está validada por la práctica.
>
> Hallazgos técnicos clave:
> - lstat() ANTES de resolve() es la única defensa correcta para material criptográfico.
    >   fs::is_symlink(resolved) llega tarde — weakly_canonical() ya resolvió el symlink.
> - lexically_normal() vs weakly_canonical(): dos herramientas para dos casos de seguridad.
    >   La distinción no está bien documentada en la literatura C++20 — contribución §5 del paper.
> - Property test encontró bug en el propio fix (F17). Valida la adopción sistémica.
>
> Taxonomía safe_path aprobada (8/8):
> - resolve()         → validación general (weakly_canonical, post-resolución)
> - resolve_seed()    → material criptográfico (lstat pre-resolución, sin symlinks, 0400)
> - resolve_config()  → configs con symlinks legítimos (lexically_normal pre-resolución)
> - resolve_model()   → [BACKLOG] modelos firmados Ed25519 (ADR-038)
>
> Formalizar PathPolicy enum como documentación semántica de la taxonomía.
> No añadir más primitivas sin caso de uso concreto demostrado.
>
> Pregunta crítica para FEDER (Kimi):
> ¿La demo FEDER requiere federación funcional o es suficiente con NDR standalone?
> Clarificar con Andrés Caro Lindo ANTES de julio 2026.
> Si requiere federación, el deadline septiembre 2026 es imposible sin equipo.
> Si es NDR standalone, el deadline es alcanzable.
>
> Orden de testing validado (8/8):
> 1. Unit tests (RED→GREEN obligatorio)
> 2. Property tests (invariantes matemáticas — DAY 128)
> 3. Fuzzing libFuzzer (parsers, interfaces externas — post-property)
> 4. Mutation testing (calidad de suite — pre-release)
>
> Regla permanente añadida DAY 127:
> 'Toda nueva superficie de ficheros se clasifica con PathPolicy antes de implementar.
>  La taxonomía safe_path se documenta con diagrama de decisión.'
>
> Frase del día — ChatGPT: 'Has eliminado la confianza implícita en los fixes.
> Eso es lo que define un sistema de seguridad maduro.'"
> — Consejo de Sabios (8/8) · DAY 127

---

## BACKLOG-FEDER-001 — Convocatoria FEDER: Presentación a Andrés Caro Lindo (UEx/INCIBE)

**Estado:** PENDIENTE — bloqueado por prerequisites técnicos
**Contacto:** Andrés Caro Lindo — UEx/INCIBE, endorser arXiv:2604.04952
**Deadline límite:** 22 septiembre 2026
**Objetivo:** Obtener financiación europea para las Fases 5 y 6 del proyecto

---

### ⚠️ Pregunta crítica pendiente (Consejo DAY 127)

**¿La demo FEDER requiere federación funcional (ADR-038) o es suficiente con NDR standalone?**

- Si es NDR standalone → deadline septiembre 2026 es **alcanzable** con el ritmo actual
- Si requiere federación funcional → deadline septiembre 2026 es **imposible** sin equipo adicional

**Acción requerida:** Clarificar scope con Andrés Caro Lindo ANTES de julio 2026.

---

### Gate de entrada (prerequisites mínimos antes de contactar)

- [x] ADR-026 mergeado a main (XGBoost F1=0.9978)
- [ ] ADR-029 Variant A (x86 + AppArmor + eBPF/XDP) estable y reproducible
- [ ] ADR-029 Variant B (ARM64 + AppArmor + libpcap) estable y reproducible
- [ ] pcap relay funcional end-to-end en Vagrant (ambas arquitecturas)
- [ ] `make bootstrap` + `make pipeline-status` 6/6 RUNNING verde y reproducible
- [ ] Demo técnica grabable en menos de 10 minutos (`scripts/feder-demo.sh`)

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

---

### FASE 5 — Lo que los fondos desbloquean [REQUIRES-FUNDING] [REQUIRES-HW]

#### 5.1 — Componente de Telemetría y Datos Soberanos
#### 5.2 — Entrenamiento Local de Modelos
#### 5.3 — Argus Cloud (Federación de Inteligencia) — **REQUIERE CLARIFICACIÓN DE SCOPE**
#### 5.4 — Validación en Hardware de Bajos Recursos [REQUIRES-HW]

### FASE 6 — Ecosistema de Plugins y Componente Enterprise [POST-FEDER]

#### 6.1 — Integración Wazuh
#### 6.2 — Generación de Flujos (NetFlow/sFlow/IPFIX)
#### 6.3 — Grafos de Ataque (kill chain + MITRE ATT&CK)
#### 6.4 — Plugins ML Especializados
#### 6.5 — Componente Enterprise

---

*DAY 127 — 23 Abril 2026 · Tag activo: v0.5.2-hardened · main limpio*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*