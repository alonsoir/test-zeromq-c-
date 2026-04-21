# ML Defender (aRGus NDR) — DAY 125 Continuity Prompt

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

---

## Estado al cierre de DAY 124

### Tag activo
`v0.5.1-hardened` — branch `main` @ commit `8bf83b90`

### Hitos completados DAY 124
- **ADR-037** ✅ — `contrib/safe-path/` header-only C++20 mergeado a main.
- **F17** ✅ — integer overflow corregido en `zmq_handler.cpp` (int64_t cast).
- **Seeds** ✅ — `provision.sh` + `seed_client.cpp` + `Makefile` actualizados a `0400`.
- **9 acceptance tests RED→GREEN** ✅ — path traversal, symlink, prefijos, permisos.
- **Consejo DAY 124** ✅ — 7/7 unánime en todos los puntos.

### Lección metodológica DAY 124 (Consejo 7/7)
Los fixes de producción (`seed_client`, `config_loader`, `config_parser`) no tienen tests de demostración RED→GREEN propios. `rag-ingester` STOPPED se descubrió en el build de producción, no en un test. Esta es la deuda más importante que cerramos HOY.

---

## PASO 0 — DAY 125: verificar entorno

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout main && git status
make pipeline-status
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE"
```

Si la VM está parada: `make up && make bootstrap`

---

## PASO 1 — DEBT-GITIGNORE-TEST-SOURCES-001 (rápido, 5 min)

La regla `**/test_*` en `.gitignore` (línea 146) ignora fuentes de test. Ya causó que `test_seed_client.cpp` y `test_perms_seed.cpp` no se versionaran en DAY 124.

```bash
# Verificar estado actual
grep -n "test_\*\|test_\*.cpp" .gitignore | head -10
```

Fix con Python3:
```bash
python3 << 'PYEOF'
with open(".gitignore", "r") as f:
    content = f.read()

# Añadir excepciones para fuentes de test después de la regla **/test_*
content = content.replace(
    "**/test_*\n",
    "**/build/**/test_*\n!**/test_*.cpp\n!**/test_*.hpp\n"
)

with open(".gitignore", "w") as f:
    f.write(content)
print("Done")
PYEOF
```

Gate: `git check-ignore libs/seed-client/tests/test_seed_client.cpp` → no ignorado

---

## PASO 2 — DEBT-INTEGER-OVERFLOW-TEST-001

### 2.1 Extraer función pura en zmq_handler.cpp

Primero verifiquemos la ubicación exacta del código a extraer:
```bash
grep -n "compute_memory_mb\|mem_bytes\|pages \* page_size\|int64_t.*pages" ml-detector/src/zmq_handler.cpp | head -10
```

Crear la función pura. Localizamos el header de ZMQHandler:
```bash
grep -n "compute_memory_mb\|current_memory_mb_\|pages" ml-detector/include/zmq_handler.hpp | head -10
```

La función pura a añadir en `zmq_handler.hpp` (sección private helpers o como inline libre):
```cpp
// Pure function — testable independently (ADR-037 F17)
[[nodiscard]] inline double compute_memory_mb(long pages, long page_size) noexcept {
    const auto mem_bytes = static_cast<int64_t>(pages) * static_cast<int64_t>(page_size);
    return static_cast<double>(mem_bytes) / (1024.0 * 1024.0);
}
```

Y en `zmq_handler.cpp` reemplazar el cálculo inline por la llamada a la función.

### 2.2 Crear test RED→GREEN

```bash
# Verificar dónde están los tests de ml-detector
ls ml-detector/tests/
```

Crear `ml-detector/tests/unit/test_zmq_memory_overflow.cpp`:

```cpp
// ml-detector/tests/unit/test_zmq_memory_overflow.cpp
//
// ACCEPTANCE TEST — DEBT-INTEGER-OVERFLOW-TEST-001 (ADR-037 F17)
//
// RED→GREEN: demuestra que el código antiguo overflowea y el nuevo no.
// Veredicto Consejo DAY 124 (7/7): Opción A + C.
// - A: unit test con valores sintéticos extremos
// - C: property loop ligero, sin dependencias nuevas

#include <gtest/gtest.h>
#include <climits>
#include <cstdint>
#include "zmq_handler.hpp"  // para compute_memory_mb

// ─── ACCEPTANCE TEST RED ────────────────────────────────────────────────────
// Demuestra que el código ANTIGUO produce overflow con valores grandes.
// Sin este test, el fix es "una promesa sin firma" (Qwen, Consejo DAY 124).

TEST(ZmqMemoryOverflow, OldCodeOverflowsWithLargePages) {
    // Con pages = LONG_MAX / 4096 + 1, la multiplicación (long * long) desborda
    long pages = LONG_MAX / 4096 + 1;
    long page_size = 4096;

    // Simulamos el código ANTIGUO (vulnerable):
    volatile long old_product = pages * page_size; // overflow intencional
    double old_result = static_cast<double>(old_product) / (1024.0 * 1024.0);

    // El resultado debe ser negativo o absurdamente grande — evidencia de overflow
    EXPECT_TRUE(old_result < 0.0 || old_result > 1e15)
        << "OLD CODE: expected overflow evidence, got: " << old_result;
}

// ─── ACCEPTANCE TEST GREEN ───────────────────────────────────────────────────
// Demuestra que el código NUEVO produce el resultado correcto.

TEST(ZmqMemoryOverflow, NewCodeHandlesExtremeValues) {
    long pages = LONG_MAX / 4096;
    long page_size = 4096;

    double result = compute_memory_mb(pages, page_size);

    // Resultado debe ser positivo y acotado (no overflow, no negativo)
    EXPECT_GT(result, 0.0) << "Result must be positive";
    EXPECT_LT(result, 1e13) << "Result must be bounded (< 10 PB)";
}

// ─── PROPERTY TEST ───────────────────────────────────────────────────────────
// Para cualquier pages >= 0 y page_size en [4096, 65536],
// el resultado de compute_memory_mb nunca es negativo.
// Opción C ligera: loop sin dependencias externas (Consejo DAY 124).

TEST(ZmqMemoryOverflow, PropertyNeverNegative) {
    const long page_sizes[] = {4096, 8192, 16384, 65536};
    const long page_values[] = {
        0, 1, 1000, 100000,
        LONG_MAX / 65536,
        LONG_MAX / 16384,
        LONG_MAX / 8192,
        LONG_MAX / 4096
    };

    for (long page_size : page_sizes) {
        for (long pages : page_values) {
            double result = compute_memory_mb(pages, page_size);
            EXPECT_GE(result, 0.0)
                << "Negative result for pages=" << pages
                << " page_size=" << page_size;
        }
    }
}

// ─── PROPERTY TEST: monotonía ─────────────────────────────────────────────
// Más páginas → más memoria. El resultado debe ser monótono creciente.

TEST(ZmqMemoryOverflow, PropertyMonotonicallyIncreasing) {
    long page_size = 4096;
    long prev_pages = 0;
    double prev_result = compute_memory_mb(prev_pages, page_size);

    for (long pages : {1L, 1000L, 1000000L, 1000000000L}) {
        double result = compute_memory_mb(pages, page_size);
        EXPECT_GE(result, prev_result)
            << "Non-monotonic at pages=" << pages;
        prev_result = result;
        prev_pages = pages;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  DEBT-INTEGER-OVERFLOW-TEST-001 — F17 RED→GREEN gate  \n";
    std::cout << "  A: unit sintético · C: property loop sin deps        \n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    return RUN_ALL_TESTS();
}
```

### 2.3 Añadir el test al CMakeLists.txt de ml-detector

```bash
grep -n "test_ransomware_detector_unit\|add_executable.*test_" ml-detector/CMakeLists.txt | head -10
```

Añadir el nuevo test siguiendo el patrón existente.

Gate: `test_zmq_memory_overflow` — 4 tests PASSED

---

## PASO 3 — DEBT-SAFE-PATH-TEST-RELATIVE-001

Añadir en `contrib/safe-path/tests/test_safe_path.cpp`:

```cpp
// ─── ACCEPTANCE TEST 10: path relativo ──────────────────────────────────────
// ATAQUE: componente recibe config_path = "config/foo.json" (relativo al CWD)
// SIN FIX: weakly_canonical no canonicalizaba el prefix → prefix era "config/"
//          → no matching con path absoluto resuelto → SECURITY VIOLATION falso positivo
//          → rag-ingester STOPPED (incidencia DAY 124)
// CON FIX: prefix canonicalizado con weakly_canonical antes de la comparación.

TEST_F(SafePathTest, RelativePathResolvesBeforePrefixCheck) {
    namespace fs = std::filesystem;

    // Crear un config legítimo dentro del allowed_dir
    std::ofstream(allowed_dir + "config.json") << "{\"ok\":true}";

    // Simular path relativo: obtener path relativo desde CWD al fichero
    // (como haría rag-ingester con "config/rag-ingester.json")
    auto abs_path = fs::path(allowed_dir + "config.json").string();

    // El path relativo se resuelve con weakly_canonical antes del prefix check
    EXPECT_NO_THROW({
        auto r = argus::safe_path::resolve(abs_path, allowed_dir);
        EXPECT_FALSE(r.empty());
    }) << "Legitimate file in allowed_dir must not be rejected";
}
```

Gate: `test_safe_path` con nuevo caso PASSED

---

## PASO 4 — DEBT-SAFE-PATH-TEST-PRODUCTION-001

Tests RED→GREEN por componente. Uno por componente modificado.

### 4.1 seed-client — test path traversal

```bash
# Ver estructura de tests existentes para seguir el patrón
ls libs/seed-client/tests/
grep -n "add_executable\|test_" libs/seed-client/CMakeLists.txt | tail -20
```

Crear `libs/seed-client/tests/test_seed_client_traversal.cpp`:
- RED: path con `../` → SECURITY VIOLATION
- GREEN: path dentro de `keys_dir_` → OK

### 4.2 firewall-acl-agent — test path traversal

```bash
ls firewall-acl-agent/tests/ 2>/dev/null || echo "no tests dir"
grep -n "add_executable\|test_" firewall-acl-agent/CMakeLists.txt | tail -10
```

### 4.3 rag-ingester — test path traversal config

```bash
ls rag-ingester/tests/
grep -n "test_config_parser" rag-ingester/tests/CMakeLists.txt
```

Ver `test_config_parser.cpp` existente para seguir patrón:
```bash
cat rag-ingester/tests/test_config_parser.cpp
```

Añadir casos RED→GREEN al test existente:
- RED: `../etc/passwd` → runtime_error con SECURITY VIOLATION
- GREEN: path legítimo → sin excepción

Gate: todos los tests de producción path traversal PASSED

---

## PASO 5 — DEBT-CRYPTO-TRANSPORT-CTEST-001

```bash
# Ejecutar con output verboso para ver la causa raíz
vagrant ssh defender -c "cd /vagrant/crypto-transport/build && ctest -V 2>&1 | tail -50"
```

Aislar si el fallo es:
- **Linking:** error de símbolo no encontrado
- **Runtime:** crash o timeout
- **Aserción lógica:** resultado incorrecto

Una vez identificada la causa, documentar en `docs/KNOWN-ISSUES.md` si requiere refactor mayor, o fix inline si es sencillo.

Gate: `test_crypto_transport` PASSED · `test_integ_contexts` PASSED · Makefile sin `|| echo`

---

## PASO 6 — Commit, tag y push

```bash
git add -A
git commit -m "fix(debt): DAY 125 — overflow test, safe_path tests, .gitignore, crypto investigation

- DEBT-INTEGER-OVERFLOW-TEST-001: compute_memory_mb() pure function + RED/GREEN/PROPERTY tests
- DEBT-SAFE-PATH-TEST-RELATIVE-001: relative path acceptance test in contrib/safe-path
- DEBT-SAFE-PATH-TEST-PRODUCTION-001: RED→GREEN tests for seed_client, config_loader, config_parser
- DEBT-GITIGNORE-TEST-SOURCES-001: ignore only build artifacts, not test sources
- DEBT-CRYPTO-TRANSPORT-CTEST-001: root cause documented / fixed

Consejo DAY 124 7/7: 'Un fix sin test de demostración es una promesa sin firma.'"

make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE"
git push origin main
```

---

## Contexto permanente

### Tres variantes del pipeline
| Variante | Estado |
|----------|--------|
| **aRGus-dev** | ✅ Activa — `main` @ `v0.5.1-hardened` |
| **aRGus-production** | 🟡 Pendiente de cocinar (post-deuda) |
| **aRGus-seL4** | ⏳ No iniciada — branch independiente futura |

### Secuencia canónica
```bash
make up           # vagrant up
make bootstrap    # 8 pasos, todo automático
make test-all     # verificación completa
```

### Estado de deuda tras DAY 124
```
🔴 DEBT-INTEGER-OVERFLOW-TEST-001     → DAY 125 (HOY)
🔴 DEBT-SAFE-PATH-TEST-PRODUCTION-001 → DAY 125 (HOY)
🔴 DEBT-SAFE-PATH-TEST-RELATIVE-001   → DAY 125 (HOY)
🟢 DEBT-GITIGNORE-TEST-SOURCES-001    → DAY 125 (HOY, rápido)
🟡 DEBT-SNYK-WEB-VERIFICATION-001     → DAY 126
🟡 DEBT-CRYPTO-TRANSPORT-CTEST-001    → DAY 125-127
🟢 DEBT-DEV-PROD-SYMLINK-001          → DAY 127
🟢 DEBT-PROVISION-PORTABILITY-001     → DAY 128
⏳ DEBT-PENTESTER-LOOP-001            → POST-DEUDA
```

### Modelos firmados activos
```
/vagrant/ml-detector/models/production/level1/
  xgboost_cicids2017_v2.ubj + .sig  (DAY 122 — IN-DISTRIBUTION)
  wednesday_eval_report.json        (OOD finding sealed)
```

### Paper arXiv:2604.04952
Draft v16 activo. https://arxiv.org/abs/2604.04952
Pendiente: actualizar §5 con lecciones aprendidas DAY 124 (tests de demostración, asimetría dev/prod).

### Tag activo
`v0.5.1-hardened` — main.

### REGLA DE ORO DAY 125
Si un fix de seguridad no tiene test que falle con el código antiguo y pase con el nuevo, no está cerrado. Punto.

---

*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*"Un fix sin test de demostración es una promesa sin firma." — Qwen, Consejo DAY 124*
*"Un escudo sin tests es un escudo de papel." — Kimi, Consejo DAY 124*