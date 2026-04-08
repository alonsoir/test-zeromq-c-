# ML Defender (aRGus NDR) — DAY 112 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## Estado al cierre de DAY 111

### Hito del día
**arXiv:2604.04952 [cs.CR] PUBLICADO** — Viernes 3 Apr 2026, 05:20:13 UTC.
DOI: https://doi.org/10.48550/arXiv.2604.04952
"ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and
Anomalous Traffic Detection in Resource-Constrained Organizations"
— Alonso Isidoro Román. 28 páginas, cs.CR, MIT license.

### Completado DAY 111

**FIX-C — D8-pre inverso OBLIGATORIO ✅**
- `plugin_loader.cpp`: PLUGIN_MODE_NORMAL + payload==nullptr → std::terminate()
- Contrato D8-pre bidireccional completo (READONLY y NORMAL cubiertos)

**FIX-D — MAX_PLUGIN_PAYLOAD_SIZE OBLIGATORIO ✅**
- `plugin_loader.hpp`: constexpr MAX_PLUGIN_PAYLOAD_SIZE = 65536 (64KB)
- `plugin_loader.cpp`: payload_len > MAX → std::terminate()

**TEST-INTEG-4c — 3/3 PASSED ✅**
- Caso A: NORMAL + payload real → PLUGIN_OK, errors==0
- Caso B: NORMAL + plugin modifica campo read-only → D8 VIOLATION detectada
- Caso C: NORMAL + result_code=-1 → error registrado, no crash

**PHASE 2d — ml-detector ✅**
- `zmq_handler.hpp`: set_plugin_loader() setter + plugin_loader_ member + include
- `zmq_handler.cpp`: invoke_all(ctx) post-inferencia, payload=evento serializado,
  mode=PLUGIN_MODE_NORMAL, early return si result_code!=0
- `main.cpp`: set_plugin_loader(&plugin_loader_) tras zmq_handler.start()
- Compilación limpia. 6/6 RUNNING.

**ADR-029 — g_plugin_loader + async-signal-safe ✅**
- Documenta patrón global obligatorio para rag-security (único componente con este patrón)
- D1: static PluginLoader* g_plugin_loader = nullptr
- D2: signal handler solo async-signal-safe (write(), dlclose(), signal(), raise())
- D3: orden inicialización obligatorio (loader → asignación → signal handlers)
- D4: invoke_all en rag-security → modo READONLY
- D5: invoke_all NUNCA desde signal handler
- TEST-INTEG-4e definido (3 casos)

**Commits DAY 111:**
- Commit 1 (b23eca66): FIX-C + FIX-D + TEST-INTEG-4c
- Commit 2 (58d73c04): PHASE 2d ml-detector
- Commit 3: ADR-029
- Branch: feature/plugin-crypto

---

## Consejo DAY 111 — Preguntas para DAY 112

**Q1-112 — PHASE 2e: ¿invoke_all READONLY o NORMAL en rag-security?**
ADR-029 D4 establece READONLY. ¿Hay algún caso donde NORMAL tenga sentido
dado que rag-security es guardián de la memoria semántica?

**Q2-112 — TEST-INTEG-4e Caso C: ¿cómo testear SIGTERM sin fork()?**
Caso C requiere verificar que shutdown limpio ocurre bajo señal.
¿Subprocess con kill()? ¿alarm() + signal handler en el test?
¿O simplemente simular la lógica del handler sin señal real?

**Q3-112 — arXiv Replace v13: ¿subir ahora o esperar indexación completa?**
v1 está anunciada (2604.04952). Draft v13 está listo.
¿Riesgo de Replace antes de que Google Scholar/Semantic Scholar indexen v1?

---

## Orden DAY 112 (no saltarse)

### PASO 1 — Verificar estado
```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/plugin-crypto
git pull origin feature/plugin-crypto
make pipeline-status
```

### PASO 2 — PHASE 2e: rag-security

Siguiendo ADR-029 D1-D5 estrictamente.

**Archivos a tocar:**
- `rag-security/src/main.cpp`

**Patrón D1-D3:**
```cpp
// ADR-029 D1: global requerido para async-signal-safe signal handler
static ml_defender::PluginLoader* g_plugin_loader = nullptr;

// En main(), orden obligatorio (ADR-029 D3):
// 1. PluginLoader construido
ml_defender::PluginLoader plugin_loader(config_path);
plugin_loader.load_plugins();
// 2. Asignación ANTES de signal handlers
g_plugin_loader = &plugin_loader;
// 3. Instalar signal handlers
signal(SIGTERM, signal_handler);
signal(SIGINT,  signal_handler);
```

**Signal handler D2:**
```cpp
static void signal_handler(int sig) {
    const char msg[] = "[rag-security] signal received — shutting down\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);
    if (g_plugin_loader != nullptr) {
        g_plugin_loader->shutdown();
    }
    signal(sig, SIG_DFL);
    raise(sig);
}
```

**invoke_all D4 — READONLY:**
```cpp
if (g_plugin_loader != nullptr) {
    MessageContext ctx{};
    ctx.payload     = nullptr;  // READONLY: sin payload
    ctx.payload_len = 0;
    ctx.mode        = PLUGIN_MODE_READONLY;
    ctx.result_code = 0;
    // ... campos de red del evento
    g_plugin_loader->invoke_all(ctx);
    // READONLY: result_code ignorado (D4 — guardián semántico)
}
```

Gate: compilar + make plugin-integ-test verde (4a+4b+4c) + TEST-INTEG-4e.

### PASO 3 — TEST-INTEG-4e

3 casos según ADR-029:
- Caso A: READONLY + evento real → PLUGIN_OK, result_code ignorado
- Caso B: g_plugin_loader=nullptr → invoke_all no llamado, no crash
- Caso C: simulación lógica signal handler → shutdown limpio

Gate: make plugin-integ-test verde (4a+4b+4c+4e).

### PASO 4 — Commit + actualizar README/BACKLOG

### PASO 5 (opcional) — arXiv Replace v13
Solo si Consejo DAY 112 da luz verde en Q3.

---

## Deuda pendiente (no bloqueante)

- REC-2: noclobber + check 0-bytes CI (P2)
- ADR-025 (Plugin Integrity Ed25519) — post PHASE 2 completa
- TEST-PROVISION-1 como gate CI formal (post PHASE 2)
- arXiv Replace v13 — pendiente decisión Consejo Q3-112
- DEBT-SNIFFER-SEED — unificar sniffer bajo SeedClient

---

## Contexto permanente

### Proyecto
- **aRGus NDR (ML Defender)**: C++20 NDR para hospitales, escuelas, municipios.
- **arXiv**: arXiv:2604.04952 [cs.CR] — PUBLICADO 3 Apr 2026 ✅
- **Branch activa**: feature/plugin-crypto
- **Repositorio**: https://github.com/alonsoir/argus

### Comandos VM críticos
- Editar ficheros en VM: python3 << 'PYEOF' (nunca sed -i sin -e '' en macOS)
- vagrant ssh -c '...' con comillas simples para CMAKE_FLAGS
- NUNCA > fichero para escribir código — usar python3 heredoc
- CMake: NO_DEFAULT_PATH para libsodium
- CI: .github/workflows/ci.yml

### PHASE 2 estado
- 2a ✅ firewall (TEST-INTEG-4a)
- 2b ✅ rag-ingester READONLY (TEST-INTEG-4b)
- 2c ✅ sniffer NORMAL payload real (TEST-INTEG-4c)
- 2d ✅ ml-detector post-inferencia
- 2e ⏳ rag-security (ADR-029 documentado, listo para implementar)

### Consejo de Sabios (7 miembros)
Claude, Grok, ChatGPT, DeepSeek, Qwen (Alibaba), Gemini, Parallel.ai.
Qwen se auto-identifica como DeepSeek — registrar como Qwen en actas.
ADR-029 aprobado implícitamente por unanimidad DAY 110 (REC-1 DeepSeek).