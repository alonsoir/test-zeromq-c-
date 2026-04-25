python3 << 'PYEOF'
content = """# ML Defender (aRGus NDR) — DAY 130 Continuity Prompt

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
- **REGLA PERMANENTE (Consejo 8/8 DAY 127):** Toda nueva superficie de ficheros se clasifica con PathPolicy antes de implementar. Documentar en docs/SECURITY-PATH-PRIMITIVES.md.
- **REGLA PERMANENTE (Consejo 8/8 DAY 128):** IPTablesWrapper y cualquier ejecución de comandos del sistema usa execve() directo sin shell. Nunca system() ni popen() con strings concatenados.
- **REGLA PERMANENTE (Consejo 8/8 DAY 129 — RULE-SCP-VM-001):** Toda transferencia de ficheros entre VM y macOS usa `scp -F /tmp/vagrant-ssh-config` o `vagrant scp`. PROHIBIDO `vagrant ssh -c "cat ..." > fichero` — el pipe zsh trunca a 0 bytes silenciosamente sin error.

---

## Estado al cierre de DAY 129

### Branch activa
`main` — limpio. Tag activo: `v0.5.2-hardened`. Commit: `55383638`.

### Último commit
DAY 129 — Consejo 8/8 acta decisiones + respuestas 8 modelos.

### Hitos completados DAY 129
- **DEBT-IPTABLES-INJECTION-001** ✅ — CWE-78 CERRADO. `safe_exec.hpp` 4 primitivos. 0 popen/system en iptables_wrapper.cpp. 15/15 tests GREEN.
- **DEBT-ETCDCLIENT-LEGACY-SEED-001** ✅ (parcial) — EtcdClientHmacTest 9/9 PASSED.
- **DEBT-FEDER-SCOPE-DOC-001** ✅ — `docs/FEDER-SCOPE.md` creado. Go/no-go 1 agosto 2026.
- **DEBT-FIREWALL-CONFIG-PATH-001** ✅ — resolve_config() verificada. 3/3 GREEN.
- **Consejo 8/8 DAY 129** — 5 decisiones vinculantes. RULE-SCP-VM-001 permanente.

### Hallazgos técnicos clave DAY 129
1. **Markdown corruption literal** en ficheros .cpp — fix por número de línea, nunca por string matching si el fichero fue editado con editor markdown.
2. **CRLF en VM vs LF en macOS** — `vagrant ssh -c "cat ..."` produce output CRLF. String matching falla. Preferir edición directa en VM con Python.
3. **Pipe zsh trunca a 0 bytes** — `vagrant ssh -c "cat ..." > file` silencioso. RULE-SCP-VM-001 activa.
4. **INVALID_ARGUMENT no existe** en `IPTablesErrorCode` — usar `INVALID_RULE`.
5. **Backslash al final de comentario** en .hpp actúa como line-continuation — rompe parsing silenciosamente.

### Decisiones vinculantes Consejo DAY 129
- **D1 (8/8):** RULE-SCP-VM-001 — scp obligatorio para transferencias VM↔macOS
- **D2 (8/8):** `**/build-debug/` en .gitignore
- **D3 (6/8):** Prioridad DAY 130: A(Fuzzing) → C(Paper §5) → B(Capabilities)
- **D4 (8/8):** DEBT-SAFE-EXEC-NULLBYTE-001 — null byte check en safe_exec() BLOQUEANTE
- **D5 (7/8):** Limpiar .gitguardian.yaml deprecated keys

---

## PASO 0 — DAY 130: verificar entorno

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout main && git status
make pipeline-status
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE|COMPLETE"
```

Si la VM está parada: `make pipeline-start && sleep 20 && make pipeline-status`

---

## PASO 1 — Limpieza infra (10 min)

### 1a. .gitignore

```python
python3 << 'PYEOF'
path = "/Users/aironman/CLionProjects/test-zeromq-docker/.gitignore"
with open(path, "r") as f:
    src = f.read()
if "**/build-debug/" not in src:
    src += "\\n# Build artifacts (Consejo 8/8 DAY 129)\\n**/build-debug/\\n**/build-release/\\n"
    with open(path, "w") as f:
        f.write(src)
    print("OK: build-debug añadido")
else:
    print("Ya existe")
PYEOF
```

### 1b. .gitguardian.yaml — ver contenido actual y limpiar deprecated keys

```bash
cat /Users/aironman/CLionProjects/test-zeromq-docker/.gitguardian.yaml
```

Renombrar `paths-ignore:` → `paths_ignore:` y añadir `version: 2` si no existe.

### 1c. Commit trivial

```bash
git add .gitignore .gitguardian.yaml
git commit -m "chore: DAY 130 — .gitignore build-debug + .gitguardian.yaml cleanup

Consejo 8/8 DAY 129:
- D2: **/build-debug/ en .gitignore (artefactos de compilación)
- D5: .gitguardian.yaml deprecated keys paths-ignore → paths_ignore
DEBT-GITIGNORE-BUILD-001: CERRADA
DEBT-GITGUARDIAN-YAML-001: CERRADA"
```

---

## PASO 2 — DEBT-SAFE-EXEC-NULLBYTE-001 🔴 BLOQUEANTE (45 min)

### Contexto
Consejo 8/8 DAY 129 (D4): `safe_exec()` es un primitivo general. Debe defenderse
independientemente de validadores upstream. Un null byte en un argumento argv[i]
puede truncar el argumento silenciosamente en execv() sin error.

**Técnica aprobada por Consejo (Qwen/Kimi):**
```cpp
// strlen() se detiene en el primer \\0.
// Si arg.size() != strlen(arg.c_str()) → hay \\0 interno → fail-closed
if (arg.size() != std::strlen(arg.c_str())) {
    return -1; // fail-closed, nunca truncar silenciosamente
}
```

### Qué hacer

**1. Añadir `is_safe_for_exec()` en `safe_exec.hpp`:**
```cpp
[[nodiscard]] inline bool is_safe_for_exec(const std::string& arg) noexcept {
    // Defensa en profundidad (Consejo 8/8 DAY 129):
    // strlen() se detiene en \\0; si difiere de size() → null byte interno
    return arg.size() == std::strlen(arg.c_str());
}
```

**2. Aplicar en todas las variantes de safe_exec() antes del fork:**
```cpp
// Antes del fork, validar todos los args:
for (const auto& a : args) {
    if (!is_safe_for_exec(a)) return -1; // safe_exec sin output
    // o para con output: return {-1, "null byte in argument"};
}
```

**3. Test RED→GREEN en `test_safe_exec.cpp`:**
```cpp
TEST(SafeExecIntegration, RejectsNullByteInArgument) {
    // RED: argumento con null byte interno → safe_exec retorna -1
    std::string arg_with_null("chain\\x00evil", 10);
    int ret = safe_exec({"/bin/echo", arg_with_null});
    EXPECT_EQ(ret, -1) << "safe_exec debe rechazar null bytes en argumentos";
}

TEST(SafeExecProperty, IsAlwaysSafeForNormalStrings) {
    // Propiedad: strings sin null → is_safe_for_exec siempre true
    const std::vector<std::string> safe_strings = {
        "INPUT", "FORWARD", "/usr/sbin/iptables", "-t", "filter", "-N"
    };
    for (const auto& s : safe_strings) {
        EXPECT_TRUE(is_safe_for_exec(s)) << "String normal debe ser safe: " << s;
    }
}
```

### Gate
- Tests RED→GREEN compilando y pasando
- `make test-all` ALL TESTS COMPLETE

---

## PASO 3 — DEBT-FUZZING-LIBFUZZER-001 (90 min)

### Contexto
Consejo 8/8 DAY 129 (D3): Fuzzing es la única opción que descubre unknown unknowns
antes del despliegue en hospitales. Targets primarios:
1. `validate_chain_name()` — guardián de inyección en iptables
2. `validate_filepath()` — guardián de path traversal en save/restore
3. Parser ZMQ (si da tiempo)

### Qué hacer

**1. Verificar que libFuzzer está disponible en la VM:**
```bash
vagrant ssh -c "clang++ --version && clang++ -fsanitize=fuzzer /dev/null -o /dev/null 2>&1 | head -5"
```

**2. Crear `firewall-acl-agent/fuzz/fuzz_validate_chain.cpp`:**
```cpp
#include <cstdint>
#include <string>
#include "safe_exec.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::string input(reinterpret_cast<const char*>(data), size);
    // Corpus semilla: test_safe_exec.cpp casos existentes
    bool result = validate_chain_name(input);
    // No debe crashear — solo retornar true/false
    (void)result;
    return 0;
}
```

**3. Crear `firewall-acl-agent/fuzz/fuzz_validate_filepath.cpp`** (similar)

**4. Añadir target en CMakeLists.txt:**
```cmake
if(ARGUS_BUILD_FUZZ)
    add_executable(fuzz_validate_chain fuzz/fuzz_validate_chain.cpp)
    target_compile_options(fuzz_validate_chain PRIVATE -fsanitize=fuzzer,address)
    target_link_options(fuzz_validate_chain PRIVATE -fsanitize=fuzzer,address)
    target_include_directories(fuzz_validate_chain PRIVATE src/core)
endif()
```

**5. Añadir target en Makefile:**
```makefile
fuzz-safe-exec:
    vagrant ssh -c "cd /vagrant/firewall-acl-agent/build-debug && \\
        cmake .. -DARGUS_BUILD_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++ && \\
        make fuzz_validate_chain && \\
        ./fuzz_validate_chain -max_total_time=60 -jobs=4"
```

**6. Ejecutar:**
```bash
make fuzz-safe-exec 2>&1 | tail -20
```

### Gate
- `fuzz_validate_chain` compila sin errores
- 60 segundos de fuzzing → 0 crashes
- Documenta corpus en `firewall-acl-agent/fuzz/corpus/`
- `make test-all` ALL TESTS COMPLETE

---

## PASO 4 — DEBT-MARKDOWN-HOOK-001 (15 min)

### Qué hacer
Añadir verificación en pre-commit hook que detecte patrón `[word](http://` en
ficheros `.cpp`/`.hpp` — indicador inequívoco de corrupción por editor markdown.

```bash
# Ver el pre-commit hook actual
cat /Users/aironman/CLionProjects/test-zeromq-docker/.git/hooks/pre-commit | head -40
```

Añadir al hook:
```bash
# Check: markdown corruption en ficheros C++
if git diff --cached --name-only | grep -E '\\.(cpp|hpp)$' | \\
   xargs grep -l '\\[.*\\](http://' 2>/dev/null | grep .; then
    echo "ERROR: Ficheros C++/HPP con markdown corruption detectada ([word](http://...))."
    echo "Edita el fichero y corrige antes de commitear."
    exit 1
fi
```

### Gate
- Pre-commit rechaza un .cpp con `[rule.in](http://rule.in)_interface`
- Pre-commit acepta un .cpp normal

---

## PASO 5 — Commit final + tag

```bash
make test-all 2>&1 | grep -E "ALL TESTS|COMPLETE"

git add -A
git commit -m "fix+chore: DAY 130 — null byte safe_exec + fuzzing + infra cleanup

- safe_exec.hpp: is_safe_for_exec() — null byte check defensa en profundidad
  arg.size() != strlen(arg.c_str()) → fail-closed (Consejo 8/8 DAY 129)
  test_safe_exec.cpp: +2 tests RED→GREEN (17/17 total)
- fuzz/: harness libFuzzer validate_chain_name + validate_filepath
  60s fuzzing 0 crashes — corpus en fuzz/corpus/
- .gitignore: **/build-debug/ añadido
- .gitguardian.yaml: deprecated keys limpiados
- .git/hooks/pre-commit: markdown corruption check en .cpp/.hpp
DEBT-SAFE-EXEC-NULLBYTE-001: CERRADA
DEBT-FUZZING-LIBFUZZER-001: CERRADA (baseline)
DEBT-GITIGNORE-BUILD-001: CERRADA
DEBT-GITGUARDIAN-YAML-001: CERRADA
DEBT-MARKDOWN-HOOK-001: CERRADA"

git push origin main
```

---

## Contexto estratégico

### Taxonomía safe_path (PERMANENTE)
resolve()        → validación general
resolve_seed()   → material criptográfico (lstat pre-resolución, 0400, sudo)
resolve_config() → configs con symlinks legítimos (lexically_normal pre-resolución)
resolve_model()  → [BACKLOG ADR-038] modelos firmados Ed25519

### Estado de deuda al inicio de DAY 130
🔴 DEBT-SAFE-EXEC-NULLBYTE-001    → DAY 130 BLOQUEANTE — is_safe_for_exec()
🔴 DEBT-FUZZING-LIBFUZZER-001     → DAY 130 — libFuzzer validate_chain_name + ZMQ
🟡 DEBT-GITIGNORE-BUILD-001       → DAY 130 — **/build-debug/
🟡 DEBT-GITGUARDIAN-YAML-001      → DAY 130 — deprecated keys
🟡 DEBT-MARKDOWN-HOOK-001         → DAY 130 — pre-commit check word
⏳ DEBT-SEED-CAPABILITIES-001     → v0.6+ — CAP_DAC_READ_SEARCH
⏳ DEBT-SAFE-PATH-RESOLVE-MODEL-001→ feature/adr038-acrl
⏳ DEBT-PENTESTER-LOOP-001         → POST-DEUDA
📄 Paper §5 Draft v17             → DAY 130/131

### Modelos firmados activos
/vagrant/ml-detector/models/production/level1/
xgboost_cicids2017_v2.ubj + .sig  (DAY 122 — IN-DISTRIBUTION)
wednesday_eval_report.json        (OOD finding sealed)

### Paper arXiv:2604.04952
Draft v16 activo. https://arxiv.org/abs/2604.04952
Pendiente: Draft v17 con §5 actualizado (property testing + safe_path taxonomy + CWE-78 fix).

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

### BACKLOG-FEDER-001
Presentación a Andrés Caro Lindo (UEx/INCIBE). Deadline: 22 septiembre 2026.
Go/no-go técnico: **1 agosto 2026**.
Prerequisites: ADR-026 merged + ADR-029 Variants A/B + demo pcap reproducible.

### REGLA DE ORO DAY 130
`is_safe_for_exec()` no es una optimización — es un contrato de seguridad.
Un null byte silencioso en execv() es un argumento que miente sobre sí mismo.

---

*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*"La seguridad no se construye con héroes que evitan errores.*
*Se construye con sistemas que hacen difícil cometerlos y fácil recuperarlos." — Qwen, Consejo DAY 129*
*"Vuestra atención al detalle demuestra el nivel de rigor necesario*
*para proteger infraestructuras críticas." — Mistral, Consejo DAY 129*
