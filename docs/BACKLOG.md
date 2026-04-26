# aRGus NDR — BACKLOG
*Última actualización: DAY 130 — 25 Abril 2026*

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
- **REGLA PERMANENTE (DAY 128 — Consejo 8/8):** IPTablesWrapper y cualquier ejecución de comandos del sistema usa execve() directo sin shell. Nunca system() ni popen() con strings concatenados.
- **REGLA PERMANENTE (DAY 129 — Consejo 8/8 RULE-SCP-VM-001):** Toda transferencia de ficheros entre VM y macOS usa `scp -F vagrant-ssh-config` o `vagrant scp`. PROHIBIDO `vagrant ssh -c "cat ..." > fichero` — el pipe zsh trunca a 0 bytes silenciosamente sin error.
- **PROTOCOLO CANÓNICO (DAY 130):** Toda sesión de desarrollo comienza con `vagrant destroy -f && vagrant up && make bootstrap && make test-all`. Sin excepciones. El pipeline debe ser reproducible desde cero antes de cualquier cambio.

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa | x86-debug, imagen Vagrant con todas las herramientas, build-debug. Para investigación y desarrollo diario. |
| **aRGus-production** | 🟡 Pendiente de cocinar | x86-apparmor + arm64-apparmor. Imágenes Debian optimizadas, sin herramientas de desarrollo. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ No iniciada | Apéndice científico. Kernel seL4, libpcap (no eBPF/XDP), sniffer reescrito en monohilo. Branch independiente. Nunca se mergeará a main salvo sorpresa. |


---

## ✅ CERRADO DAY 130

### DEBT-SYSTEMD-AUTOINSTALL-001
- **Status:** ✅ CERRADO DAY 130
- **Fix:** `install-systemd-units.sh` integrado en Vagrantfile `cryptographic-provisioning`. Elimina paso manual post-provisioning.
- **Commit:** `8e57aad2`

### DEBT-SAFE-EXEC-NULLBYTE-001
- **Status:** ✅ CERRADO DAY 130
- **Fix:** `is_safe_for_exec()` — `[[nodiscard]] inline bool`, compara `arg.size() == std::strlen(arg.c_str())`. Aplicado antes del `fork()` en las 4 variantes de `safe_exec`. `#include <cstring>` añadido.
- **Tests:** `test_safe_exec.cpp` 17/17 GREEN (+2 nuevos: `RejectsNullByteInArgument` + `IsAlwaysSafeForNormalStrings`). RED→GREEN demostrado.
- **Commit:** `c8e293a8`

### DEBT-GITGUARDIAN-YAML-001
- **Status:** ✅ CERRADO DAY 130
- **Fix:** `.gitguardian.yaml` reescrito limpio — `paths-ignore` → `paths_ignore` (v2), fichero corrupto con dos entradas fusionadas eliminado.
- **Commit:** `06228a67`

### DEBT-FUZZING-LIBFUZZER-001
- **Status:** ✅ CERRADO DAY 130 (baseline)
- **Fix:** libFuzzer harnesses sobre `validate_chain_name` + `is_safe_for_exec` + `validate_filepath`. 2.4M runs, 0 crashes, 30s. Corpus 67 ficheros versionado. Targets `make fuzz-safe-exec`, `make fuzz-validate-filepath`, `make fuzz-all` en Makefile.
- **Commit:** `f5994c4a`

### DEBT-MARKDOWN-HOOK-001
- **Status:** ✅ CERRADO DAY 130
- **Fix:** `.git/hooks/pre-commit` — check detecta patrón `[word](http://...)` en `.cpp`/`.hpp`. Test RED→GREEN verificado manualmente.
- **Commit:** `aab08daa`

### REGLA EMECAS (DAY 130) — Verificación destructiva
- Grabado en asciinema: `docs/argus-day130-bootstrap-20260425-142211.cast`
- Keypair activo post-rebuild: `1f48b75054fe98e8371653607caaf028b3f688bc055782c9c9c6d0e3494dad54`
- Pipeline 6/6 RUNNING · TEST-INTEG-SIGN 7/7 PASSED · make test-all ALL TESTS COMPLETE

### DEBT-NATIVE-LINUX-BOOTSTRAP-001 (nueva — backlog post-FEDER)
- **Status:** ⏳ BACKLOG — no bloqueante
- **Origen:** Colaborador externo (emecas@inspiron) intentó `make bootstrap` en Linux nativo. Fallo: `llama.h: No existe el fichero`. El provisioner Vagrant compila `llama.cpp` automáticamente — el flujo nativo no está documentado.
- **Fix futuro:** `README` + `make deps-native` que compile `third_party/llama.cpp`.

---

## ✅ CERRADO DAY 129

### DEBT-IPTABLES-INJECTION-001
- **Status:** ✅ CERRADO DAY 129
- **Fix:** `safe_exec.hpp` — 4 primitivos fork+execv() sin shell. 14 call-sites migrados en `iptables_wrapper.cpp`. 0 popen()/system() restantes.
- **Validadores:** `validate_chain_name()` allowlist [A-Za-z0-9_-] 1..29 chars + null byte check. `validate_table_name()` conjunto fijo. `validate_filepath()` sin traversal.
- **Tests:** `test_safe_exec.cpp` 15/15 GREEN (4 unit + 4 property + 7 integración). RED→GREEN demostrado.
- **Incidencias:** markdown corruption en .cpp (fix por línea), CRLF VM vs macOS, backslash en comentario safe_exec.hpp, INVALID_ARGUMENT → INVALID_RULE.

### DEBT-ETCDCLIENT-LEGACY-SEED-001 (parcial)
- **Status:** ✅ CERRADO DAY 129 (parcial — ADR-024 pendiente)
- **Fix:** `component_config_path` solo se asigna cuando `encryption_enabled=true` en `ml-detector/src/etcd_client.cpp`.
- **Tests:** EtcdClientHmacTest 9/9 PASSED (antes 9/9 FAILED).

### DEBT-FEDER-SCOPE-DOC-001
- **Status:** ✅ CERRADO DAY 129
- **Fix:** `docs/FEDER-SCOPE.md` — scope mínimo viable, go/no-go 1 agosto 2026, prerequisitos técnicos, estructura `scripts/feder-demo.sh`.

### DEBT-FIREWALL-CONFIG-PATH-001
- **Status:** ✅ CERRADO DAY 129 (verificación)
- **Fix:** `resolve_config()` ya correctamente implementada y testeada. `ConfigLoaderTraversal` 3/3 GREEN pre-existente. Tabla de verificación añadida a `docs/SECURITY-PATH-PRIMITIVES.md`.

### Consejo de Sabios DAY 129 (8/8)
- **D1 (8/8):** RULE-SCP-VM-001 — scp obligatorio, pipe zsh prohibido
- **D2 (8/8):** `**/build-debug/` en .gitignore
- **D3 (6/8):** A Fuzzing → C Paper → B Capabilities para DAY 130
- **D4 (8/8):** DEBT-SAFE-EXEC-NULLBYTE-001 — null byte en safe_exec() obligatorio
- **D5 (7/8):** Limpiar .gitguardian.yaml deprecated keys

---

## ✅ CERRADO DAY 128

### DEBT-SAFE-PATH-TAXONOMY-DOC-001
- **Status:** ✅ CERRADO DAY 128
- **Fix:** `docs/SECURITY-PATH-PRIMITIVES.md` — taxonomía 4 primitivas safe_path, PathPolicy enum conceptual, diagrama de decisión, ejemplos.

### DEBT-PROPERTY-TESTING-PATTERN-001
- **Status:** ✅ CERRADO DAY 128
- **Fix:** `docs/testing/PROPERTY-TESTING.md` + 5 property tests GREEN en `contrib/safe-path/tests/test_safe_path_property.cpp`. Integrados en `make test-libs`.
- **Tests:** 5/5 PASSED.

### DEBT-PROVISION-PORTABILITY-001
- **Status:** ✅ CERRADO DAY 128
- **Fix:** `ARGUS_SERVICE_USER` en `provision.sh`. Seeds `0400 root:root`. Componentes con seeds arrancan con `sudo` en Makefile.
- **Hallazgo:** `resolve_seed()` enforza exactamente `0400` con `std::terminate()`. `sudo` es la solución, no relajar permisos.
- **Tests:** TEST-PROVISION-1 8/8 OK. Pipeline 6/6 RUNNING en VM nueva.

### DEBT-SNYK-WEB-VERIFICATION-001
- **Status:** ✅ CERRADO DAY 128
- **Fix:** 18 findings triados en `docs/security/SNYK-DAY-128.md`.
  - 5 falsos positivos cerrados · 11 contrib/tools no alcanzables
  - 1 HIGH nuevo → DEBT-IPTABLES-INJECTION-001 (DAY 129 BLOQUEANTE)
  - 1 pendiente → DEBT-FIREWALL-CONFIG-PATH-001 (probable falso positivo)

### DEBT-ETCDCLIENT-LEGACY-SEED-001 (reclasificada)
- **Status:** Reclasificada DAY 128 — no es regresión, es código legado pre-P2P.
- **Diagnóstico:** EtcdClient intenta leer seed vía `resolve_seed()` — modelo anterior a ADR-026/027.
- **Decisión Consejo 5/3:** Limpiar ANTES de ADR-024. Feature: `feature/etcdclient-p2p-cleanup`.

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
| **ADR-025-EXT-001** | Emergency Patch Protocol — Plugin Unload: implementar `action="unload"` en plugin-loader. Tabla interna de handles activos. `dlclose()` con log NOTICE. Tests SIGN-8/9/10. | TEST-INTEG-SIGN-8/9/10 RED→GREEN | post-FEDER |

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
| **seeds 0400 + sudo** | `0400 root:root` invariante. `sudo` aceptable. Evolución: `CAP_DAC_READ_SEARCH` v0.6+ | Consejo 8/8 · DAY 128 |
| **CWE-78 execve()** | IPTablesWrapper migra a execve() sin shell. libiptc a largo plazo. Bloqueante DAY 129. | Consejo 8/8 · DAY 128 |
| **EtcdClient cleanup antes de ADR-024** | Mayoría 5/3: limpiar legacy pre-P2P antes de ADR-024. | Consejo 5/3 · DAY 128 |
| **Demo FEDER = NDR standalone** | NDR standalone + 2 nodos simulados. No requiere ADR-038. Go/no-go: 1 agosto 2026. | Consejo 8/8 · DAY 128 |
| **RULE-SCP-VM-001** | Toda transferencia VM↔macOS usa scp/vagrant scp. Prohibido pipe zsh (trunca a 0 bytes silenciosamente). | Consejo 8/8 · DAY 129 |
| **Null byte en safe_exec()** | is_safe_for_exec() en safe_exec() como defensa en profundidad independiente de validadores upstream. | Consejo 8/8 · DAY 129 |
| **Fuzzing antes que Paper** | Prioridad DAY 130: A(Fuzzing) → C(Paper §5) → B(Capabilities). Fuzzing descubre unknown unknowns antes del despliegue. | Consejo 6/8 · DAY 129 |
| **REGLA EMECAS (DAY 130)** | Toda sesión comienza con `vagrant destroy -f && vagrant up && make bootstrap && make test-all`. Pipeline reproducible desde cero = prerequisito de cualquier cambio. | DAY 130 |
| **is_safe_for_exec() contrato de seguridad** | Null byte check en safe_exec() es un contrato, no una optimización. Defensa en profundidad independiente de validadores upstream. | DAY 130 |
| **libFuzzer como baseline** | Harnesses sobre validate_chain_name + validate_filepath. Corpus versionado. 2.4M runs sin crash = baseline certificado. | DAY 130 |
| **ARGUS_SERVICE_USER** | Variable de entorno para service user. Default `vagrant`. | Consejo 6/7 · DAY 124 |
| **safe_path header-only** | `contrib/safe-path/` — cero dependencias, C++20 puro. | Consejo 7/7 · DAY 123 |
| **Seeds 0400** | Seeds deben tener permisos `0400` (solo owner, solo lectura). | Consejo 7/7 · DAY 124 |
| **Tres variantes** | aRGus-dev · aRGus-production (x86+ARM apparmor) · aRGus-seL4 (apéndice científico). | DAY 124 |
| **Plugin unload vía mensaje firmado** | Emergency Patch Protocol: `action="unload"` + Ed25519 + ZeroMQ. Zero new attack surface. Reutiliza cadena de confianza existente. Post-FEDER. | DAY 131 — sugerencia founder LinkedIn |

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

DEBT-SNYK-WEB-VERIFICATION-001:         ████████████████████ 100% ✅  DAY 128
DEBT-PROPERTY-TESTING-PATTERN-001:      ████████████████████ 100% ✅  DAY 128
DEBT-SAFE-PATH-TAXONOMY-DOC-001:        ████████████████████ 100% ✅  DAY 128
DEBT-PROVISION-PORTABILITY-001:         ████████████████████ 100% ✅  DAY 128


DEBT-IPTABLES-INJECTION-001:            ████████████████████ 100% ✅  DAY 129

DEBT-FIREWALL-CONFIG-PATH-001:          ████████████████████ 100% ✅  DAY 129

DEBT-SEED-CAPABILITIES-001:           ░░░░░░░░░░░░░░░░░░░░   0% ⏳ v0.6+
DEBT-SYSTEMD-AUTOINSTALL-001:          ████████████████████ 100% ✅  DAY 130
DEBT-SAFE-EXEC-NULLBYTE-001:           ████████████████████ 100% ✅  DAY 130
DEBT-GITGUARDIAN-YAML-001:             ████████████████████ 100% ✅  DAY 130
DEBT-FUZZING-LIBFUZZER-001:            ████████████████████ 100% ✅  DAY 130  (baseline — 2.4M runs, 0 crashes)
DEBT-MARKDOWN-HOOK-001:                ████████████████████ 100% ✅  DAY 130

DEBT-FEDER-SCOPE-DOC-001:              ████████████████████ 100% ✅  DAY 129

DEBT-ETCDCLIENT-LEGACY-SEED-001:       ████████████████████ 100% ✅  DAY 129 (parcial — ADR-024 pendiente)
DEBT-SAFE-PATH-RESOLVE-MODEL-001:       ░░░░░░░░░░░░░░░░░░░░   0% ⏳ feature/adr038-acrl
DEBT-FUZZING-LIBFUZZER-001:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳ post-property-testing
DEBT-CRYPTO-003a (mlock+bzero):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳
DEBT-PENTESTER-LOOP-001 (ACRL):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-DEUDA
ADR-030 aRGus-production images:        ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-DEUDA
ADR-031 aRGus-seL4:                     ░░░░░░░░░░░░░░░░░░░░   0% ⏳ branch independiente
FEAT-CLOUD-RETRAIN-001:                 ░░░░░░░░░░░░░░░░░░░░   0% ⏳ post-ACRL
```

---

## 📝 Notas del Consejo de Sabios — DAY 129 (8/8)

> "DAY 129 eliminó la última superficie de inyección de comandos conocida (CWE-78).
> La transición de popen()/system() a execv() es una barrera física, no una promesa de seguridad.
>
> Decisiones vinculantes:
> - D1 (8/8): RULE-SCP-VM-001 — pipe zsh trunca silenciosamente, scp obligatorio
> - D2 (8/8): **/build-debug/ en .gitignore — ruido cognitivo eliminado
> - D3 (6/8): Fuzzing DAY 130 primero — unknown unknowns antes del despliegue
> - D4 (8/8): DEBT-SAFE-EXEC-NULLBYTE-001 — defensa en profundidad en safe_exec()
> - D5 (7/8): .gitguardian.yaml — alert fatigue es riesgo de compliance hospitalario
>
> 'La seguridad no se construye con héroes que evitan errores.
>  Se construye con sistemas que hacen difícil cometerlos y fácil recuperarlos.' — Qwen
> 'Vuestra atención al detalle demuestra el nivel de rigor necesario
>  para proteger infraestructuras críticas.' — Mistral
> 'El sistema empieza a comportarse como un sistema que desconfía de sí mismo.
>  Ese es el punto de inflexión correcto.' — ChatGPT"
> — Consejo de Sabios (8/8) · DAY 129

## 📝 Notas del Consejo de Sabios — DAY 128 (8/8)

> "DAY 128 consolidó documentación, metodología y hallazgos técnicos con consecuencias arquitectónicas.
>
> Decisiones vinculantes:
> - D1 (8/8): 0400 root:root se mantiene. sudo aceptable. Evolución CAP_DAC_READ_SEARCH v0.6+
> - D2 (8/8): DEBT-IPTABLES-INJECTION-001 → execve() sin shell, DAY 129 BLOQUEANTE
> - D3 (8/8): Property testing prioridades: compute_memory_mb > HKDF > ZeroMQ > protobuf
> - D4 (5/3): Limpiar EtcdClient ANTES de ADR-024
> - D5 (8/8): Demo FEDER = NDR standalone + 2 nodos simulados. Go/no-go: 1 agosto 2026
>
> 'La seguridad no es cómoda. Es necesaria.' — Qwen
> 'No construyas encima de comportamiento incorrecto, aunque sea temporal.' — ChatGPT
> 'El sistema empieza a comportarse como un sistema que desconfía de sí mismo.
>  Ese es el punto de inflexión correcto.' — ChatGPT"
> — Consejo de Sabios (8/8) · DAY 128

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

## 🧬 HIPÓTESIS CENTRAL — Inmunidad Global Adaptativa

**Formulada:** DAY 128 — 24 Abril 2026  
**Estado:** Pendiente demostración experimental (DEBT-PENTESTER-LOOP-001)

### Hipótesis fuerte

> Dado un espacio de técnicas ATT&CK finito y enumerable, un sistema con ACRL
> converge hacia cobertura total en tiempo polinomial respecto al número de técnicas
> observadas. Un sistema estático no converge nunca — su error en técnicas no vistas
> es constante.

### Analogía estructural: sistema inmune adaptativo

El sistema inmune adaptativo tiene tres propiedades que hacen la analogía
estructuralmente correcta, no solo metafórica:

1. **Memoria distribuida** — cada nodo que ha visto un ataque retiene el "anticuerpo"
   (modelo reentrenado). La inteligencia está en la red, no en un servidor central.
   Corresponde a ADR-026 (federación de modelos).

2. **Tolerancia al self** — el sistema aprende a no atacar tráfico legítimo propio.
   aRGus ya tiene esto implícitamente: el Wednesday OOD finding (DAY 122) demuestra
   que el modelo rechaza patrones "self" sin reentrenamiento.

3. **Respuesta secundaria más rápida** — la segunda exposición a un patógeno produce
   respuesta órdenes de magnitud más rápida. Con ACRL y warm-start, la segunda
   exposición a una técnica ATT&CK ya vista produce detección casi instantánea.

### Lo que falta ver (intuición DAY 128)

La agregación segura de modelos entre nodos sin revelar datos locales.
Aprendizaje federado con privacidad diferencial — SecureBoost para XGBoost.
El salto de "cada hospital aprende solo" a "todos los hospitales aprenden juntos
sin compartir datos de pacientes".

Ese es el resultado que trasciende arXiv cs.CR.

### Experimento mínimo viable (DEBT-PENTESTER-LOOP-001)

Pipeline 6/6 capturando tráfico real
Caldera lanza escenario ATT&CK (T1046 + T1595 como mínimo)
eBPF captura los flujos generados
XGBoost warm-start sobre modelo existente con nuevos datos
Ed25519 firma el nuevo modelo
Hot-swap sin reiniciar el pipeline
Medición: ¿el nuevo modelo detecta mejor variantes del mismo ataque?
Repetir con 3+ técnicas ATT&CK distintas

### Implicación si se demuestra

Un sistema con N ciclos ACRL habrá visto y codificado N clases de técnicas
adversariales. El coste marginal de detectar la técnica N+1 decrece con cada ciclo.
En el límite, el sistema hace obsoleto el modelo de pentesting periódico —
que es exactamente el problema de hospitales y municipios sin Red Teams continuos.

**Esto no es especulación arquitectónica. Es una hipótesis falsificable.**
El experimento ACRL o la confirma o la refuta. En cualquier caso, es ciencia.

### Referencias
- Sommer & Paxson (2010) — límites de los clasificadores estáticos en NDR
- SecureBoost (Cheng et al.) — XGBoost federado con privacidad
- MITRE ATT&CK — taxonomía de técnicas adversariales
- arXiv:2604.04952 §11.18 — propuesta arquitectónica ACRL

---

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
- [ ] ADR-030 Variant A (x86 + AppArmor + eBPF/XDP) estable y reproducible
- [ ] ADR-030 Variant B (ARM64 + AppArmor + libpcap) estable y reproducible
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



# Insertar antes de BACKLOG-FEDER-001


*DAY 130 — 25 Abril 2026 · Tag activo: v0.5.2-hardened · commit aab08daa · main limpio*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*