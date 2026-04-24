# ML Defender (aRGus NDR) — DAY 129 Continuity Prompt

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

---

## Estado al cierre de DAY 128

### Branch activa
`main` — limpio. Tag activo: `v0.5.2-hardened`.

### Último commit
DAY 128 — `858895c` — docs/security/SNYK-DAY-128.md + deudas nuevas.

### Hitos completados DAY 128
- **VM nueva desde cero** — vagrant destroy + up + bootstrap. Pipeline 6/6 RUNNING.
- **DEBT-SAFE-PATH-TAXONOMY-DOC-001** ✅ — `docs/SECURITY-PATH-PRIMITIVES.md`
- **DEBT-PROPERTY-TESTING-PATTERN-001** ✅ — `docs/testing/PROPERTY-TESTING.md` + 5 property tests GREEN
- **DEBT-PROVISION-PORTABILITY-001** ✅ — `ARGUS_SERVICE_USER` + sudo para seeds
- **DEBT-SNYK-WEB-VERIFICATION-001** ✅ — 18 findings triados
- **Consejo 8/8** — 5 decisiones vinculantes documentadas en `docs/CONSEJO-DAY-128-acta.md`

### Hallazgos técnicos clave DAY 128
1. **`resolve_seed()` enforza exactamente `0400` con `std::terminate()`** — relajar a `0440` viola la invariante. La solución correcta es `sudo`, no relajar permisos.
2. **`resolve_seed()` lanza excepción, no retorna -1** — la firma `int` es engañosa. Tests usan `EXPECT_THROW<std::exception>`.
3. **EtcdClientHmacTest 9/9** — no es regresión. Es código legado pre-P2P (ADR-026/027). Cleanup antes de ADR-024 (decisión Consejo 5/3).

### Decisiones vinculantes Consejo DAY 128
- **D1 (8/8):** `0400 root:root` invariante. `sudo` aceptable. Evolución: `CAP_DAC_READ_SEARCH` en v0.6+.
- **D2 (8/8):** DEBT-IPTABLES-INJECTION-001 → `execve()` sin shell. BLOQUEANTE DAY 129.
- **D3 (8/8):** Property testing prioridades: `compute_memory_mb` > HKDF > ZeroMQ parsers > protobuf.
- **D4 (5/3):** Limpiar EtcdClient ANTES de ADR-024.
- **D5 (8/8):** Demo FEDER = NDR standalone + 2 nodos simulados. Go/no-go: 1 agosto 2026.

---

## PASO 0 — DAY 129: verificar entorno

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout main && git status
make pipeline-status
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE|COMPLETE"
```

Si la VM está parada: `make pipeline-start && sleep 20 && make pipeline-status`

---

## PASO 1 — DEBT-IPTABLES-INJECTION-001 🔴 BLOQUEANTE (90 min)

### Contexto
`IPTablesWrapper::cleanup_rules()` en `firewall-acl-agent/src/core/iptables_wrapper.cpp:625`
llama a `execute_command(cmd)` donde `cmd` es un string concatenado. CWE-78.
Consejo 8/8 unánime: `execve()` directo sin shell.

### Qué hacer

**1. Localizar el código afectado:**
```bash
grep -n "execute_command\|system(\|popen(" \
  /Users/aironman/CLionProjects/test-zeromq-docker/firewall-acl-agent/src/core/iptables_wrapper.cpp | head -20
```

**2. Fix: reemplazar execute_command(string) por execve(argv[]):**
```cpp
// ANTES (vulnerable):
execute_command("iptables -D " + chain + " " + rule);

// DESPUÉS (seguro):
safe_exec({"/sbin/iptables", "-D", chain, rule});
```

**3. Implementar `safe_exec()` en `iptables_wrapper.hpp`:**
```cpp
#include <unistd.h>
#include <sys/wait.h>
#include <vector>
#include <string>

inline int safe_exec(const std::vector<std::string>& args) {
    std::vector<const char*> argv;
    for (const auto& a : args) argv.push_back(a.c_str());
    argv.push_back(nullptr);
    
    pid_t pid = fork();
    if (pid == 0) {
        execv(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);
    }
    int status;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
```

**4. Validar argumentos antes de execv:**
```cpp
// Validar IPs: solo alfanumérico, '.', '/', '-'
// Validar chain names: solo alfanumérico, '-', '_'
// Rechazar cualquier otro carácter
```

### Gate
- `validate_chain_name()` regex allowlist existente (de DAY previo) — verificar que aplica
- Test RED: con string malicioso `; rm -rf /` en argumento → `safe_exec` NO ejecuta shell
- Test GREEN: comando iptables válido ejecuta correctamente
- `make test-all` ALL TESTS COMPLETE

---

## PASO 2 — DEBT-ETCDCLIENT-LEGACY-SEED-001 (45 min)

### Contexto
EtcdClient intenta leer seed vía `resolve_seed()` — modelo pre-P2P.
Consejo 5/3: limpiar ANTES de ADR-024. Elimina confusión arquitectónica.

### Qué hacer

**1. Localizar el código legacy:**
```bash
grep -n "resolve_seed\|seed_path\|component_config_path\|CryptoTransport" \
  /Users/aironman/CLionProjects/test-zeromq-docker/etcd-client/src/etcd_client.cpp | head -20
```

**2. Marcar como deprecated y deshabilitar en producción:**
```cpp
[[deprecated("Legacy pre-P2P. Use ADR-024 Noise_IKpsk3 seed distribution.")]]
void EtcdClient::init_crypto_legacy(const std::string& path) {
    #ifndef ARGUS_LEGACY_SEED
    throw std::runtime_error("Legacy seed disabled. Use P2P distribution (ADR-024).");
    #endif
    // ... código legacy ...
}
```

**3. Verificar que EtcdClientHmacTest 9/9 pasan SIN sudo:**
```bash
vagrant ssh -c "cd /vagrant/ml-detector/build-debug && ./tests/test_etcd_client_hmac 2>&1 | tail -5"
```

### Gate
- `EtcdClientHmacTest` 9/9 PASSED sin necesidad de sudo
- `make test-all` ALL TESTS COMPLETE

---

## PASO 3 — DEBT-FEDER-SCOPE-DOC-001 (20 min)

### Qué hacer
Crear `docs/FEDER-SCOPE.md` con:
- Scope mínimo viable: NDR standalone + 2 nodos Vagrant simulados
- Lo que NO se necesita: ADR-038 completo, federación real, privacidad diferencial
- Milestone go/no-go: **1 agosto 2026**
- Script demo: `scripts/feder-demo.sh` (estructura, no implementación)
- Prerequisitos técnicos para contactar a Andrés Caro Lindo

### Gate
- `docs/FEDER-SCOPE.md` existe y está versionado

---

## PASO 4 — Property test compute_memory_mb (20 min)

### Contexto
F17 identificado DAY 125 — overflow en `int64_t`. Fix implementado con `double`.
Falta el property test formal en ctest (solo existe documentación).

### Qué hacer
Añadir `test_memory_utils_property.cpp` en el componente donde vive `memory_utils.hpp`:
```bash
find /Users/aironman/CLionProjects/test-zeromq-docker -name "memory_utils.hpp" | grep -v build
```

Property a verificar:
- `compute_memory_mb(pages, page_size) >= 0` para todo pages >= 0, page_size válido
- `compute_memory_mb(LONG_MAX/4096, 4096)` no desborda (caso extremo que rompía int64_t)
- Monotonicidad: si pages1 > pages2, entonces `compute_memory_mb(pages1,ps) > compute_memory_mb(pages2,ps)`

### Gate
- Test en ctest PASSED RED→GREEN

---

## PASO 5 — Commit y push

```bash
git add -A
git commit -F - << 'EOF'
fix+docs: DAY 129 — CWE-78 execve() + EtcdClient cleanup + FEDER scope

- firewall-acl-agent: IPTablesWrapper migra a safe_exec() con execve()
  Sin shell, sin concatenación de strings (CWE-78 → CERRADO)
  Test RED→GREEN: safe_exec rechaza metacaracteres shell
- etcd-client: EtcdClient legacy seed code marcado [[deprecated]]
  EtcdClientHmacTest 9/9 PASSED sin sudo (cleanup pre-P2P)
- docs/FEDER-SCOPE.md: scope mínimo viable FEDER + go/no-go 1 agosto
- test_memory_utils_property: property test compute_memory_mb RED→GREEN

Consejo 8/8 DAY 128: execve() es el único mecanismo correcto para
comandos del sistema. Nunca system() ni popen() con strings.
DEBT-IPTABLES-INJECTION-001: CERRADA
DEBT-ETCDCLIENT-LEGACY-SEED-001: CERRADA (parcial)
DEBT-FEDER-SCOPE-DOC-001: CERRADA
EOF

make test-all 2>&1 | grep -E "ALL TESTS|COMPLETE"
git push origin main
```

---

## Contexto estratégico

### Decisiones Consejo DAY 128 (vinculantes)
Ver `docs/CONSEJO-DAY-128-acta.md` para detalle completo.

### Taxonomía safe_path (PERMANENTE)
```
resolve()        → validación general
resolve_seed()   → material criptográfico (lstat pre-resolución, 0400, sudo)
resolve_config() → configs con symlinks legítimos (lexically_normal pre-resolución)
resolve_model()  → [BACKLOG ADR-038] modelos firmados Ed25519
```

### Estado de deuda al inicio de DAY 129
```
🔴 DEBT-IPTABLES-INJECTION-001      → DAY 129 BLOQUEANTE — execve() sin shell
🟡 DEBT-ETCDCLIENT-LEGACY-SEED-001  → DAY 129 — cleanup pre-P2P
🟡 DEBT-FEDER-SCOPE-DOC-001         → DAY 129 — docs/FEDER-SCOPE.md
🟡 DEBT-FIREWALL-CONFIG-PATH-001    → DAY 129 — verificar resolve_config()
⏳ DEBT-SEED-CAPABILITIES-001       → v0.6+ — CAP_DAC_READ_SEARCH
⏳ DEBT-SAFE-PATH-RESOLVE-MODEL-001 → feature/adr038-acrl
⏳ DEBT-FUZZING-LIBFUZZER-001       → post-property-testing
⏳ DEBT-PENTESTER-LOOP-001          → POST-DEUDA
```

### Modelos firmados activos
```
/vagrant/ml-detector/models/production/level1/
  xgboost_cicids2017_v2.ubj + .sig  (DAY 122 — IN-DISTRIBUTION)
  wednesday_eval_report.json        (OOD finding sealed)
```

### Paper arXiv:2604.04952
Draft v16 activo. https://arxiv.org/abs/2604.04952
Pendiente: Draft v17 con §5 actualizado (property testing + safe_path taxonomy).

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

### REGLA DE ORO DAY 129
`execve()` sin shell no es una opción — es el único mecanismo correcto
para ejecutar comandos del sistema en código de seguridad.
`system("cmd" + user_input)` es un contrato de rendición.

---

*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*"La seguridad no es cómoda. Es necesaria." — Qwen, Consejo DAY 128*
*"No construyas encima de comportamiento incorrecto, aunque sea temporal." — ChatGPT, Consejo DAY 128*
*"El sistema empieza a comportarse como un sistema que desconfía de sí mismo.
Ese es el punto de inflexión correcto." — ChatGPT, Consejo DAY 128*
