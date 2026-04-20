# ML Defender (aRGus NDR) — DAY 124 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA SCRIPTS:** Lógica compleja → `tools/script.sh`. Nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().
- **REGLA macOS/sed:** Nunca `sed -i` sin `-e ''`. Usar Python3 heredoc para ediciones de ficheros en macOS.

---

## Estado al cierre de DAY 123

### Hitos completados DAY 123
- **Opción A** ✅ — `pandas scikit-learn` añadidos al Vagrantfile (commit `e88e4bf8`). Cierra DEBT-PANDAS-001.
- **ADR-037** ✅ — Redactado, enviado al Consejo, veredicto recibido y tabulado.
- **Consejo ADR-037** ✅ — **7/7 UNÁNIME** — APROBADO con cambios.

### Veredicto Consejo ADR-037 (síntesis ejecutiva)
El Consejo aprobó la solución `safe_path` (header-only, C++20, cero dependencias)
rechazando unánimemente las librerías externas propuestas por Snyk.

**Cambios mandatorios post-Consejo:**

| # | Cambio | Origen |
|---|--------|--------|
| 1 | Normalizar trailing slash en `resolve()` | ChatGPT, Gemini, Kimi, Qwen (4+) — CRÍTICO |
| 2 | TOCTOU documentado explícitamente en ADR | ChatGPT, Grok, Qwen, DeepSeek |
| 3 | `seed_client`: `O_NOFOLLOW | O_CLOEXEC` + check permisos `0400` + check symlink | Kimi, ChatGPT |
| 4 | Test symlink malicioso en `test_safe_path.cpp` | DeepSeek, ChatGPT |
| 5 | Comentario `// SAFE: n <= BUF_SIZE` en inotify | Gemini, Grok |
| 6 | `ml-detector` añadido al mapa de prefijos: `/etc/ml-defender/models/` | Mistral |

### Mapa de prefijos aprobado por el Consejo

| Componente | Prefijo | Nivel |
|-----------|---------|-------|
| `seed-client` | `/etc/ml-defender/keys/` | 🔴 Criptográfico |
| `firewall-acl-agent` | `/etc/ml-defender/` | 🟡 Config |
| `rag-ingester` | `/etc/ml-defender/` | 🟡 Config |
| `ml-detector` | `/etc/ml-defender/models/` | 🟡 Modelos firmados |
| `tools/` | `/vagrant/` o `/shared/` | 🟢 Dev |
| `contrib/` | `/shared/` | 🟢 Investigación |

### Tag activo
`v0.5.0-preproduction` — sin cambios DAY 123.
Branch de trabajo DAY 124: `feature/adr037-snyk-hardening` (AÚN NO CREADA).

---

## PASO 0 — DAY 124: verificar entorno

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout main && git status
make pipeline-status
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE"
```

Si la VM está parada: `make up && make bootstrap`

---

## PASO 1 — Crear la branch de trabajo

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout -b feature/adr037-snyk-hardening
git push -u origin feature/adr037-snyk-hardening
```

---

## PASO 2 — Plan de implementación completo ADR-037

### 2.1 Crear `contrib/safe-path/` — la minilibrería

Estructura a crear:
```
contrib/safe-path/
  include/
    safe_path/
      safe_path.hpp
  tests/
    test_safe_path.cpp
  CMakeLists.txt
  README.md
```

### `safe_path.hpp` — versión final post-Consejo

```cpp
// contrib/safe-path/include/safe_path/safe_path.hpp
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>

namespace argus::safe_path {

// Uses weakly_canonical (not canonical) to allow paths to non-existent files
// (e.g., output files created later). The prefix check remains secure because
// unresolved trailing components cannot contain ".." after the last existing
// directory. Any ".." that would escape the prefix is caught by the comparison.
// TOCTOU note: a window exists between resolve() and ifstream/open().
// Mitigation: AppArmor enforces write paths at kernel level for production
// components. For seed material, use resolve_seed() which opens with O_NOFOLLOW.

[[nodiscard]] inline std::string resolve(
    const std::string& path,
    const std::string& allowed_prefix)
{
    namespace fs = std::filesystem;

    if (path.empty()) {
        throw std::runtime_error("[safe_path] Empty path rejected");
    }

    const auto canonical = fs::weakly_canonical(fs::path(path)).string();

    // Normalise trailing slash to prevent bypass:
    // /etc/ml-defender would otherwise match /etc/ml-defender-evil/
    std::string prefix = allowed_prefix;
    if (!prefix.empty() && prefix.back() != '/') {
        prefix += '/';
    }

    if (canonical.rfind(prefix, 0) != 0) {
        throw std::runtime_error(
            "[safe_path] SECURITY VIOLATION — path traversal rejected\n"
            "  requested : '" + path + "'\n"
            "  resolved  : '" + canonical + "'\n"
            "  allowed   : '" + prefix + "'\n"
            "  ACTION    : Pipeline halt. Administrator notified.");
    }
    return canonical;
}

[[nodiscard]] inline std::string resolve_writable(
    const std::string& path,
    const std::string& allowed_prefix)
{
    const auto resolved = resolve(path, allowed_prefix);
    namespace fs = std::filesystem;
    const auto parent = fs::path(resolved).parent_path();
    if (!fs::is_directory(parent)) {
        throw std::runtime_error(
            "[safe_path] Parent directory does not exist: " + parent.string());
    }
    return resolved;
}

// Hardened variant for cryptographic material (seed.bin).
// Adds: symlink check + permission check (0400) + O_NOFOLLOW | O_CLOEXEC.
// Returns open file descriptor — caller is responsible for close().
[[nodiscard]] inline int resolve_seed(
    const std::string& path,
    const std::string& allowed_prefix = "/etc/ml-defender/keys/")
{
    namespace fs = std::filesystem;
    const auto resolved = resolve(path, allowed_prefix);

    // Explicit symlink check post-resolution (TOCTOU mitigation for seeds)
    if (fs::is_symlink(fs::path(resolved))) {
        throw std::runtime_error(
            "[safe_path] SECURITY VIOLATION — symlink rejected for seed material: "
            + resolved);
    }

    // Verify permissions: must be 0400 (read-only, owner only)
    struct stat st{};
    if (stat(resolved.c_str(), &st) != 0 || (st.st_mode & 0777) != 0400) {
        throw std::runtime_error(
            "[safe_path] SECURITY VIOLATION — seed file permissions must be 0400: "
            + resolved);
    }

    // Open with O_NOFOLLOW | O_CLOEXEC — kernel-level symlink protection
    const int fd = open(resolved.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(
            "[safe_path] Cannot open seed file: " + resolved);
    }
    return fd;
}

} // namespace argus::safe_path
```

### 2.2 Tests de aceptación — filosofía RED→GREEN

**REGLA FUNDAMENTAL DAY 124:**
Los tests de aceptación deben demostrar dos cosas:
1. **RED:** El código actual SIN el fix es vulnerable (el test falla o el ataque tiene éxito)
2. **GREEN:** Con el fix aplicado, el ataque es rechazado con error explícito

Este es el contrato de seguridad: no basta con que el código "funcione" —
debe demostrar que *la vulnerabilidad existía* y que *el fix la cierra*.

### `test_safe_path.cpp` — tests de aceptación completos

```cpp
// contrib/safe-path/tests/test_safe_path.cpp
//
// ACCEPTANCE TESTS — ADR-037 safe_path
//
// Filosofía RED→GREEN:
// - Cada test documenta un ataque real
// - Sin safe_path: el ataque tendría éxito (apertura de fichero arbitrario)
// - Con safe_path: el ataque es rechazado con std::runtime_error
//
// Ejecutar: ./test_safe_path
// Gate: ALL TESTS PASSED

#include <gtest/gtest.h>
#include <safe_path/safe_path.hpp>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// ─── Fixtures ──────────────────────────────────────────────────────────────

class SafePathTest : public ::testing::Test {
protected:
    std::string tmp_dir;
    std::string allowed_dir;
    std::string forbidden_dir;

    void SetUp() override {
        tmp_dir     = fs::temp_directory_path() / "argus_test_safe_path";
        allowed_dir = tmp_dir + "/allowed/";
        forbidden_dir = tmp_dir + "/forbidden/";
        fs::create_directories(allowed_dir);
        fs::create_directories(forbidden_dir);
        // Crear fichero legítimo
        std::ofstream(allowed_dir + "legit.json") << "{\"ok\":true}";
        // Crear fichero prohibido
        std::ofstream(forbidden_dir + "secret.bin") << "FORBIDDEN_CONTENT";
    }

    void TearDown() override {
        fs::remove_all(tmp_dir);
    }
};

// ─── P1: weakly_canonical resuelve correctamente ───────────────────────────

TEST_F(SafePathTest, LegitimatePathResolvesCorrectly) {
    // GREEN: path normal dentro del prefijo → OK
    EXPECT_NO_THROW({
        auto r = argus::safe_path::resolve(allowed_dir + "legit.json", allowed_dir);
        EXPECT_FALSE(r.empty());
    });
}

// ─── ACCEPTANCE TEST 1: Path traversal con ".." ────────────────────────────
// ATAQUE: Un componente recibe como config_path el valor
//   "/allowed/../forbidden/secret.bin"
// SIN FIX: std::ifstream abriría el fichero sin problema.
// CON FIX: safe_path::resolve lanza runtime_error → pipeline halt.

TEST_F(SafePathTest, RejectDotDotTraversal) {
    const std::string attack = allowed_dir + "../forbidden/secret.bin";

    // Demostrar que SIN safe_path el fichero sería accesible
    {
        std::ifstream f(attack);
        EXPECT_TRUE(f.good()) << "Prerequisite: without safe_path, file is accessible";
    }

    // CON safe_path: debe rechazarlo
    EXPECT_THROW(
        argus::safe_path::resolve(attack, allowed_dir),
        std::runtime_error
    ) << "safe_path MUST reject ../ traversal";
}

// ─── ACCEPTANCE TEST 2: Path traversal absoluto ────────────────────────────
// ATAQUE: config_path = "/etc/passwd"
// CON FIX: rechazado porque no empieza por allowed_prefix.

TEST_F(SafePathTest, RejectAbsolutePathOutsidePrefix) {
    EXPECT_THROW(
        argus::safe_path::resolve("/etc/passwd", allowed_dir),
        std::runtime_error
    );
}

// ─── ACCEPTANCE TEST 3: Bypass por prefijo sin trailing slash ──────────────
// ATAQUE: allowed_prefix = "/tmp/argus_test_safe_path/allowed" (sin slash)
//         path = "/tmp/argus_test_safe_path/allowed_evil/secret.bin"
// SIN normalización: rfind matchearía porque "allowed" es prefijo de "allowed_evil"
// CON normalización de trailing slash: rechazado.

TEST_F(SafePathTest, RejectPrefixBypassWithoutTrailingSlash) {
    // Crear directorio "evil" al mismo nivel que allowed
    const std::string evil_dir = tmp_dir + "/allowed_evil/";
    fs::create_directories(evil_dir);
    std::ofstream(evil_dir + "secret.bin") << "EVIL";

    const std::string prefix_without_slash = tmp_dir + "/allowed"; // sin /

    EXPECT_THROW(
        argus::safe_path::resolve(evil_dir + "secret.bin", prefix_without_slash),
        std::runtime_error
    ) << "Trailing slash normalisation MUST prevent prefix bypass";
}

// ─── ACCEPTANCE TEST 4: Symlink apuntando fuera del prefijo ─────────────────
// ATAQUE: symlink dentro del prefijo apunta a fichero prohibido.
// CON FIX: weakly_canonical resuelve el symlink → destino fuera del prefijo → rechazado.

TEST_F(SafePathTest, RejectSymlinkPointingOutsidePrefix) {
    const std::string symlink_path = allowed_dir + "evil_link";
    fs::create_symlink(forbidden_dir + "secret.bin", symlink_path);

    EXPECT_THROW(
        argus::safe_path::resolve(symlink_path, allowed_dir),
        std::runtime_error
    ) << "Symlink pointing outside prefix MUST be rejected";
}

// ─── ACCEPTANCE TEST 5: Path vacío ─────────────────────────────────────────

TEST_F(SafePathTest, RejectEmptyPath) {
    EXPECT_THROW(
        argus::safe_path::resolve("", allowed_dir),
        std::runtime_error
    );
}

// ─── ACCEPTANCE TEST 6: resolve_writable — directorio padre inexistente ─────

TEST_F(SafePathTest, RejectWritableWithNonExistentParent) {
    const std::string ghost = allowed_dir + "nonexistent/output.bin";
    EXPECT_THROW(
        argus::safe_path::resolve_writable(ghost, allowed_dir),
        std::runtime_error
    );
}

// ─── ACCEPTANCE TEST 7: Mensaje de error contiene información de alerta ─────
// El mensaje debe indicar: qué se intentó, qué se resolvió, qué está permitido,
// y que el pipeline se para.

TEST_F(SafePathTest, ErrorMessageContainsSecurityAlert) {
    const std::string attack = allowed_dir + "../forbidden/secret.bin";
    try {
        argus::safe_path::resolve(attack, allowed_dir);
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("SECURITY VIOLATION"), std::string::npos)
            << "Error must contain SECURITY VIOLATION";
        EXPECT_NE(msg.find("Pipeline halt"), std::string::npos)
            << "Error must mention pipeline halt";
        EXPECT_NE(msg.find("Administrator notified"), std::string::npos)
            << "Error must mention admin notification";
    }
}

// ─── ACCEPTANCE TEST 8 (seed): resolve_seed rechaza symlinks ────────────────

TEST_F(SafePathTest, SeedRejectSymlink) {
    // Crear seed falsa con permisos correctos
    const std::string real_seed = allowed_dir + "seed.bin";
    std::ofstream(real_seed) << "FAKESEED";
    chmod(real_seed.c_str(), 0400);

    // Crear symlink hacia ella
    const std::string sym = allowed_dir + "seed_link.bin";
    fs::create_symlink(real_seed, sym);

    EXPECT_THROW(
        argus::safe_path::resolve_seed(sym, allowed_dir),
        std::runtime_error
    ) << "resolve_seed MUST reject symlinks even within the prefix";
}

// ─── ACCEPTANCE TEST 9 (seed): resolve_seed rechaza permisos incorrectos ────

TEST_F(SafePathTest, SeedRejectWrongPermissions) {
    const std::string seed = allowed_dir + "seed.bin";
    std::ofstream(seed) << "FAKESEED";
    chmod(seed.c_str(), 0644); // permisos incorrectos (debería ser 0400)

    EXPECT_THROW(
        argus::safe_path::resolve_seed(seed, allowed_dir),
        std::runtime_error
    ) << "resolve_seed MUST reject seed files with permissions != 0400";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  ADR-037 ACCEPTANCE TESTS — safe_path security gate  \n";
    std::cout << "  RED→GREEN: cada test documenta un ataque real        \n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    return RUN_ALL_TESTS();
}
```

### 2.3 Aplicar safe_path en código de producción

**Orden de aplicación:**

**a) `seed-client/src/seed_client.cpp` — PRIORIDAD MÁXIMA**
```cpp
// ANTES (línea 99):
std::ifstream seed_file(seed_path, std::ios::binary);

// DESPUÉS:
#include <safe_path/safe_path.hpp>
const int seed_fd = argus::safe_path::resolve_seed(seed_path);
// usar seed_fd con fdopen() o read() directo
// close(seed_fd) al finalizar
```

**b) `firewall-acl-agent/src/core/config_loader.cpp` — línea 80**
```cpp
// ANTES:
std::ifstream file(config_path);

// DESPUÉS:
#include <safe_path/safe_path.hpp>
const auto safe = argus::safe_path::resolve(config_path, "/etc/ml-defender/");
std::ifstream file(safe);
```

**c) `rag-ingester/src/common/config_parser.cpp` — línea 9**
```cpp
// ANTES:
std::ifstream file(config_path);

// DESPUÉS:
#include <safe_path/safe_path.hpp>
const auto safe = argus::safe_path::resolve(config_path, "/etc/ml-defender/");
std::ifstream file(safe);
```

**d) `ml-detector/src/zmq_handler.cpp` — línea 988 (F17 integer overflow)**
```cpp
// ANTES:
long pages = 0;
long page_size = sysconf(_SC_PAGESIZE);
current_memory_mb_.store((pages * page_size) / (1024.0 * 1024.0));

// DESPUÉS:
long pages = 0;
long page_size = sysconf(_SC_PAGESIZE);
const auto mem_bytes = static_cast<int64_t>(pages) * static_cast<int64_t>(page_size);
current_memory_mb_.store(static_cast<double>(mem_bytes) / (1024.0 * 1024.0));
```

**e) Comentarios FP inotify — `csv_dir_watcher.cpp:168` y `csv_file_watcher.cpp:112`**
```cpp
// F15/F16 — Falso positivo Snyk documentado (ADR-037)
// SAFE: n <= BUF_SIZE = 4096 garantizado por POSIX read().
// ptr < buf + n nunca desborda. Snyk no traza acotación de read() → BUF_SIZE.
while (ptr < buf + n) {  // NOLINT(safe — see ADR-037)
```

**f) Contrib/ y tools/ — Categoría B**
Misma función `safe_path::resolve()`, prefijo `/shared/` o `/vagrant/`.

### 2.4 Comportamiento de seguridad en producción — FAIL CLOSED

Cuando `safe_path::resolve()` lanza una excepción en un componente de producción,
el comportamiento debe ser:

```
🔴 SECURITY VIOLATION DETECTED
   Component: [nombre]
   Attempted path: [path]
   Resolved to: [canonical]
   Allowed prefix: [prefix]
   ACTION: PIPELINE HALT — all components stopping
   Administrator: check /var/log/ml-defender/security.log
```

El componente que detecta la violación debe:
1. Loguear con nivel CRITICAL en spdlog
2. Enviar señal de shutdown al pipeline (via etcd o señal SIGTERM a todos)
3. Terminar con exit code != 0

Esto implementa el contrato: **O todo bien y autorizado, o nada.**

---

## PASO 3 — Actualizar ADR-037 en el repo

Copiar el ADR-037 generado en DAY 123 a:
```bash
docs/adr/ADR-037-snyk-hardening.md
```
Incorporar los cambios post-Consejo (trailing slash, TOCTOU, seed hardening, mapa de prefijos completo).

---

## PASO 4 — make test-all VERDE

```bash
make up && make bootstrap
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE"
```

Gate: `safe_path` tests PASSED + todos los tests anteriores siguen PASSED.

---

## PASO 5 — Segunda pasada Snyk

Tras los fixes, ejecutar Snyk sobre los ficheros modificados.
Gate: **0 findings en código de producción.**

---

## PASO 6 — Merge y tag

```bash
git checkout main
git merge feature/adr037-snyk-hardening
git tag v0.5.1-hardened
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

### Regla permanente de code review (añadida DAY 123)
> Todo nuevo uso de `std::ifstream`/`std::ofstream` con input no-constante
> en código de producción **debe** pasar por `argus::safe_path::resolve()`.

### Estado de modelos firmados
```
/vagrant/ml-detector/models/production/level1/
  xgboost_cicids2017_v2.ubj + .sig  (DAY 122 — IN-DISTRIBUTION)
  wednesday_eval_report.json        (OOD finding sealed)
```

### Paper arXiv:2604.04952
Draft v16 activo. https://arxiv.org/submit/7495855/view

### Tag activo al cierre DAY 123
`v0.5.0-preproduction` — branch `feature/adr037-snyk-hardening` pendiente de crear.

### DEBT-PENTESTER-LOOP-001 (próxima frontera tras ADR-037)
ACRL: Caldera → captura eBPF → reentrenamiento XGBoost → hot-swap.

---

*"Via Appia Quality — la superficie mínima es la superficie más segura."*
*"O todo bien y autorizado, o nada."*