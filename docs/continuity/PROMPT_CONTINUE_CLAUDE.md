# ML Defender (aRGus NDR) — DAY 126 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA SCRIPTS:** Lógica compleja → `tools/script.sh`. Nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().
- **REGLA macOS/sed:** Nunca `sed -i` sin `-e ''`. Usar Python3 heredoc para ediciones de ficheros en macOS.
- **REGLA PERMANENTE (Consejo 7/7 DAY 124):** Ningún fix de seguridad en código de producción se mergea sin test de demostración RED→GREEN. El test debe fallar con el código antiguo y pasar con el nuevo. Sin excepciones.
- **REGLA PERMANENTE (Consejo 8/8 DAY 125):** Todo fix de seguridad incluye: (1) unit test sintético, (2) property test de invariante, (3) test de integración en componente real. Sin excepciones.

---

## Estado al cierre de DAY 125

### Branch activa
`fix/day125-debt-closure` — NO mergeada a main. Decisión Consejo 8/8 + Alonso (Opción B).
Merge bloqueado por 4 deudas críticas que se cierran HOY (DAY 126).
Tag pendiente: `v0.5.2` al merge final.

### Último commit en la rama
`af0d8b89` — fix(debt): DEBT-CRYPTO-TRANSPORT-CTEST-001

### Hitos completados DAY 125
- **DEBT-GITIGNORE-TEST-SOURCES-001** ✅ — `.gitignore` arreglado, 47 fuentes de test versionadas
- **DEBT-INTEGER-OVERFLOW-TEST-001** ✅ — `memory_utils.hpp` + 4 tests RED→GREEN (property test encontró bug latente en int64_t fix)
- **DEBT-SAFE-PATH-TEST-RELATIVE-001** ✅ — Test 10 en `safe_path`
- **DEBT-SAFE-PATH-TEST-PRODUCTION-001** ✅ (rag-ingester) — `test_config_parser_traversal` en ctest
- **DEBT-CRYPTO-TRANSPORT-CTEST-001** ✅ — permisos `0400` en test fixtures (era `0600`)
- **Consejo 8/8 DAY 125** ✅ — feedback completo recibido y sintetizado

### Hallazgo metodológico DAY 125 (Consejo 8/8)
`PropertyNeverNegative` encontró un bug latente en el propio fix F17: `int64_t` desborda para `LONG_MAX/4096 * 8192`. Fix correcto: aritmética `double` directa. Esto valida la Opción C del Consejo DAY 124 y justifica la adopción sistémica de property testing.

---

## PASO 0 — DAY 126: verificar entorno

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout fix/day125-debt-closure && git status
make pipeline-status
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE"
```

Si la VM está parada: `make up && make bootstrap`

---

## PASO 1 — DEBT-MEMORY-UTILS-BOUNDS-001 (rápido, 15 min)

### Qué hacer
Añadir `MAX_REALISTIC_MEMORY_MB` como constante en `memory_utils.hpp` y actualizar el property test.

La función mantiene `noexcept` (componente de monitoring — mejor métrica incorrecta que crash), pero debe loguear warning si supera el bound.

### Fix en memory_utils.hpp
```cpp
// memory_utils.hpp
constexpr double MAX_REALISTIC_MEMORY_MB = 1024.0 * 1024.0; // 1 TB en MB

[[nodiscard]] inline double compute_memory_mb(long pages, long page_size) noexcept {
    const double result = (static_cast<double>(pages) * static_cast<double>(page_size))
                          / (1024.0 * 1024.0);
    // No throw (noexcept) — mejor métrica incorrecta que componente caído
    // En producción añadir: if (result > MAX_REALISTIC_MEMORY_MB || result < 0.0) log_warning(...)
    return result;
}
```

### Fix en test_zmq_memory_overflow.cpp
Actualizar `PropertyNeverNegative` para añadir:
```cpp
EXPECT_LE(result, MAX_REALISTIC_MEMORY_MB)
    << "Result exceeds realistic memory bound for pages=" << pages
    << " page_size=" << page_size;
```

Añadir test adicional:
```cpp
TEST(ZmqMemoryOverflow, RealisticBounds) {
    // 1 TB de RAM = 256M páginas de 4KB
    const long max_pages_realistic = (1024LL * 1024 * 1024 * 1024) / 4096;
    double result = compute_memory_mb(max_pages_realistic, 4096);
    EXPECT_NEAR(result, 1024.0 * 1024.0, 1.0); // 1 TB en MB
    EXPECT_LE(result, MAX_REALISTIC_MEMORY_MB);
}
```

Gate: `test_zmq_memory_overflow` — 5 tests PASSED

---

## PASO 2 — DEBT-SAFE-PATH-SEED-SYMLINK-001 (30 min)

### Qué hacer
`resolve_seed()` en `contrib/safe-path/include/safe_path/safe_path.hpp` no rechaza symlinks. `SafePathTest.SeedRejectSymlink` falla. Fix: `lstat` + `S_ISLNK` + throw. Sin flag configurable.

**Veredicto Consejo 8/8:** ESTRICTO. El material criptográfico no admite compromiso de ergonomía. Si CI/CD necesita symlinks para seeds, el CI/CD está mal configurado, no el código.

### Localizar el código
```bash
grep -n "resolve_seed\|S_ISLNK\|lstat\|symlink" contrib/safe-path/include/safe_path/safe_path.hpp | head -20
```

### Fix a implementar
```cpp
// En resolve_seed(), ANTES de verificar permisos:
struct stat st;
if (lstat(path.c_str(), &st) != 0) {
    throw std::runtime_error(
        "[safe_path] SECURITY VIOLATION — lstat failed: " + path);
}
if (S_ISLNK(st.st_mode)) {
    throw std::runtime_error(
        "[safe_path] SECURITY VIOLATION — symlink rejected for seed material: " + path);
}
```

### Verificar que el test ya existía (RED pre-fix)
```bash
grep -n "SeedRejectSymlink" contrib/safe-path/tests/test_safe_path.cpp
```

Gate: `test_safe_path` — SeedRejectSymlink PASSED (era FAILED antes del fix)

---

## PASO 3 — DEBT-CONFIG-PARSER-FIXED-PREFIX-001 (45 min)

### Qué hacer
`config_parser.cpp` deriva el prefix de `safe_path` del `parent_path` del propio `config_path`. Si el atacante controla el path, controla el prefix → bypass. Fix: `allowed_prefix` como parámetro explícito con default `/etc/ml-defender/`.

**Veredicto Consejo 8/8:** El prefix nunca debe derivarse del input.

### Localizar el código
```bash
grep -n "weakly_canonical\|parent_path\|config_prefix\|allowed_prefix" rag-ingester/src/common/config_parser.cpp
grep -n "load\|allowed_prefix" rag-ingester/include/common/config_parser.hpp
```

### Fix
```cpp
// config_parser.hpp
static Config load(const std::string& config_path,
                   const std::string& allowed_prefix = "/etc/ml-defender/");

// config_parser.cpp
Config ConfigParser::load(const std::string& config_path,
                          const std::string& allowed_prefix) {
    const auto safe_config_path =
        argus::safe_path::resolve(config_path, allowed_prefix);
    // ... resto igual ...
}
```

### Actualizar test existente para usar prefix fijo
```bash
# test_config_parser_traversal.cpp — añadir caso:
TEST(ConfigParserTraversal, RejectDotDotWithFixedPrefix) {
    // Con prefix fijo, ../etc/passwd debe ser rechazado aunque el parent sea "/"
    EXPECT_THROW(
        rag_ingester::ConfigParser::load("../etc/passwd", "/etc/ml-defender/"),
        std::runtime_error
    ) << "Fixed prefix MUST reject ../ traversal";
}
```

Gate: `test_config_parser_traversal` — 4 tests PASSED (incluyendo nuevo RED→GREEN)

---

## PASO 4 — DEBT-PRODUCTION-TESTS-REMAINING-001 (1-2 horas)

### 4.1 seed-client — test path traversal

```bash
ls libs/seed-client/tests/
grep -n "add_executable\|test_" libs/seed-client/CMakeLists.txt | tail -20
cat libs/seed-client/src/seed_client.cpp | grep -n "safe_path\|resolve_seed\|keys_dir" | head -20
```

Crear `libs/seed-client/tests/test_seed_client_traversal.cpp`:
- RED: path con `../` en `keys_dir_` → SECURITY VIOLATION
- RED: symlink como seed → SECURITY VIOLATION
- GREEN: path legítimo dentro de `keys_dir_` → OK

### 4.2 firewall-acl-agent — test path traversal

```bash
ls firewall-acl-agent/tests/unit/
grep -n "safe_path\|resolve\|config" firewall-acl-agent/src/config_loader.cpp | head -20
grep -n "add_executable\|test_" firewall-acl-agent/CMakeLists.txt | tail -10
```

Crear `firewall-acl-agent/tests/unit/test_config_loader_traversal.cpp`:
- RED: `../etc/passwd` → runtime_error SECURITY VIOLATION
- GREEN: path legítimo → OK

### Verificar que ambos están en ctest
```bash
vagrant ssh defender -c "cd /vagrant/libs/seed-client/build && ctest -N"
vagrant ssh defender -c "cd /vagrant/firewall-acl-agent/build-debug && ctest -N"
```

Gate: todos los tests de path traversal PASSED en seed-client y firewall-acl-agent

---

## PASO 5 — DEBT-SNYK-WEB-VERIFICATION-001 (cuando estés en el navegador)

Ejecutar Snyk web sobre `fix/day125-debt-closure` (o main post-merge):
- URL: https://app.snyk.io
- Target: repositorio `alonsoir/argus`
- Filtro: código C++ de producción
- Gate: 0 findings HIGH/CRITICAL en código propio (no third_party)

Documentar resultado en `docs/security/SNYK-DAY-126.md`

---

## PASO 6 — Commit, tag y merge a main

Una vez todos los gates verdes:

```bash
git add -A
git commit -F - << 'EOF'
fix(debt): DAY 126 — seed symlink + config prefix + remaining component tests

- DEBT-SAFE-PATH-SEED-SYMLINK-001: lstat+S_ISLNK en resolve_seed, estricto sin flag
- DEBT-CONFIG-PARSER-FIXED-PREFIX-001: allowed_prefix explicito, default /etc/ml-defender/
- DEBT-PRODUCTION-TESTS-REMAINING-001: RED->GREEN para seed-client + firewall-acl-agent
- DEBT-MEMORY-UTILS-BOUNDS-001: MAX_REALISTIC_MEMORY_MB en property test

Consejo 8/8 DAY 125: el material criptografico no admite compromiso de ergonomia.
EOF

make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE"
git push origin fix/day125-debt-closure
```

Si todos los tests verdes, mergear:
```bash
git checkout main
git merge --no-ff fix/day125-debt-closure -m "merge(day125-126): v0.5.2 — debt closure complete

5 debts closed DAY 125 + 4 debts closed DAY 126.
property test found latent bug in F17 fix (int64_t overflow).
seed symlink: strict rejection, no flag.
config_parser: fixed prefix, never derived from input.
seed-client + firewall-acl-agent: RED->GREEN traversal tests.
make test-all: ALL TESTS PASSED from cold VM."

git tag -a v0.5.2-hardened -m "v0.5.2-hardened: DAY 125-126 debt closure complete"
git push origin main --tags
```

---

## Contexto permanente

### Secuencia canónica
```bash
make up           # vagrant up
make bootstrap    # 8 pasos, todo automático
make test-all     # verificación completa
```

### Branch activa
`fix/day125-debt-closure` — pendiente de merge a main post-cierre DAY 126

### Estado de deuda al inicio de DAY 126
```
🔴 DEBT-SAFE-PATH-SEED-SYMLINK-001       → DAY 126 (HOY) — PASO 2
🔴 DEBT-CONFIG-PARSER-FIXED-PREFIX-001   → DAY 126 (HOY) — PASO 3
🔴 DEBT-PRODUCTION-TESTS-REMAINING-001   → DAY 126 (HOY) — PASO 4
🟡 DEBT-MEMORY-UTILS-BOUNDS-001          → DAY 126 (HOY) — PASO 1
🟡 DEBT-SNYK-WEB-VERIFICATION-001        → DAY 126 (navegador) — PASO 5
🟢 DEBT-PROPERTY-TESTING-RAPIDCHECK-001  → DAY 127
🟢 DEBT-DEV-PROD-SYMLINK-001             → DAY 127
🟢 DEBT-PROVISION-PORTABILITY-001        → DAY 128
⏳ DEBT-PENTESTER-LOOP-001               → POST-DEUDA
```

### Modelos firmados activos
```
/vagrant/ml-detector/models/production/level1/
  xgboost_cicids2017_v2.ubj + .sig  (DAY 122 — IN-DISTRIBUTION)
  wednesday_eval_report.json        (OOD finding sealed)
```

### Paper arXiv:2604.04952
Draft v16 activo. https://arxiv.org/abs/2604.04952
Pendiente: Draft v17 con §5 actualizado:
- §5.3 "Property Testing as a Security Fix Validator" (hallazgo F17 DAY 125)
- §5.4 "Dev/Prod Parity via Symlinks, Not Conditional Logic"
- §5.5 "RED→GREEN as Non-Negotiable Merge Gate"

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

### REGLA DE ORO DAY 126
Un symlink en material criptográfico no es ergonomía — es un vector de ataque.
El prefix derivado del input no es conveniencia — es un bypass de seguridad.
Ambos se cierran HOY con test RED→GREEN antes del merge.

---

*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*"Un fix sin test de demostración es una promesa sin firma." — Qwen, Consejo DAY 124*
*"Un escudo que no se prueba contra su propio filo es un escudo que ya está roto." — DeepSeek, Consejo DAY 125*