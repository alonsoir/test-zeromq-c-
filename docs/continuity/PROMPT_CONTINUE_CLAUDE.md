# ML Defender (aRGus NDR) — DAY 111 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## Estado al cierre de DAY 110

### Completado hoy

**PASO 2 — PluginMode + mode field (Q1 Consejo DAY 109) ✅**
- `plugin_api.h`: enum `PluginMode` (NORMAL=0, READONLY=1), `annotation[64]` restaurado,
  `mode uint8_t` consume 1 byte de `reserved[60]` → `reserved[59]`
- `plugin_loader.cpp`: D8-pre coherence check — READONLY+payload!=nullptr → std::terminate()
  snap_mode añadido a snapshot D8 y al invariant check
- `rag-ingester/src/main.cpp`: PHASE 2b reconstruida — PluginLoader init + ctx_readonly
  con mode=PLUGIN_MODE_READONLY antes de embed_chronos(), early return si result_code!=0
- `test_integ_4b.cpp`: TEST-INTEG-4b PASSED (Caso A + Caso B)

**PASO 3 — PHASE 2c sniffer (Q2 Consejo DAY 109) ✅**
- `ring_consumer.hpp`: set_plugin_loader() setter + plugin_loader_ member
- `ring_consumer.cpp`: invoke_all(ctx_msg) en process_raw_event() con payload real,
  mode=PLUGIN_MODE_NORMAL, result_code!=0 → events_dropped++ + return
- `sniffer/src/userspace/main.cpp`: set_plugin_loader(&plugin_loader_) tras set_stats_interval()
- `sniffer.json`: hello plugin active=false (D1 Graceful Degradation OK)
- Pipeline: 6/6 RUNNING con binarios actualizados

**PASO 4 — Paper v13 (Q3 Consejo DAY 109) ✅**
- §4 Integration Philosophy: 4 argumentos como enumerate LaTeX
- Compilación limpia Overleaf confirmada

**Incidente DAY 110:**
Tres ficheros críticos estaban vacíos (0 bytes) en la rama. Backups .backup intactos.
El backup de rag-ingester/src/main.cpp era pre-PHASE 2b — reconstruida desde cero.
Lección: noclobber en scripts + check de ficheros 0 bytes pendiente de implementar.

**Commits DAY 110:**
- Commit 1: feat(plugin-api): PluginMode + mode + PHASE 2b reconstruida + TEST-INTEG-4b
- Commit 2: feat(sniffer): PHASE 2c — plugin_process_message con payload real
- Push: ebc1d0e7..360faf8b feature/plugin-crypto

**Estado pipeline:** 6/6 RUNNING. Branch: feature/plugin-crypto.

---

## Consejo DAY 110 — Decisiones (5/5 respondieron)

**Nota de acta:** Qwen (chat.qwen.ai) se auto-identifica como DeepSeek — patrón
consolidado DAY 103-110. Registrado como Qwen en todas las actas.

**Q1-111 — PHASE 2d antes de 2e: UNANIMIDAD 5/5**
ml-detector sigue patrón limpio (set_plugin_loader + member). rag-security tiene
patrón especial (g_plugin_loader global + signal handler async-signal-safe) que
merece atención dedicada después.

**FIX-C — D8-pre inverso OBLIGATORIO (ChatGPT5)**
El contrato actual solo valida READONLY+payload!=nullptr. Falta la inversa:
NORMAL + payload == nullptr → std::terminate().
El contrato es bidireccional. Sin este fix, PHASE 2c no está completamente cerrada.

**FIX-D — MAX_PLUGIN_PAYLOAD_SIZE OBLIGATORIO (ChatGPT5)**
Sniffer recibe payload real de red — datos arbitrarios sin límite de tamaño.
Hard limit en plugin_loader antes de invocar: payload_len > MAX → std::terminate().
Valor propuesto: 64KB. Esto es D8 extendido, no opcional.

**REC-1 — ADR-029 antes de PHASE 2e (DeepSeek)**
Documentar g_plugin_loader + restricciones async-signal-safe antes de implementar
PHASE 2e. No bloqueante para 2d.

**REC-2 — noclobber + check 0-bytes en CI (ChatGPT5 + Gemini)**
set -o noclobber en scripts de cierre. find . -name "*.cpp" -size 0 en CI o make check.
No bloqueante, P2.

**Contenido TEST-INTEG-4c (Grok):**
- Caso A: payload no-nulo, longitud correcta → PLUGIN_OK, errors==0
- Caso B: plugin intenta modificar payload → D8-light VIOLATION detectada
- Caso C: result_code!=0 → paquete descartado, no llega a ml-detector, no crash

---

## Orden DAY 111 (no saltarse)

### PASO 1 — Verificar estado
```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/plugin-crypto
git pull origin feature/plugin-crypto
make pipeline-status
```

### PASO 2 — FIX-C: D8-pre inverso

Archivo: `plugin-loader/src/plugin_loader.cpp`
En `invoke_all(MessageContext& ctx)`, añadir tras el check READONLY:
```cpp
// D8-pre inverso (FIX-C, Consejo DAY 110 — ChatGPT5 obligatorio)
// PLUGIN_MODE_NORMAL garantiza payload presente. payload==nullptr = violación.
if (ctx.mode == PLUGIN_MODE_NORMAL &&
    (ctx.payload == nullptr || ctx.payload_len == 0)) {
    std::cerr << "[plugin-loader] SECURITY: PLUGIN_MODE_NORMAL violado — "
              << "payload es nullptr antes de invocar plugin '"
              << p->name << "' — std::terminate()\n";
    std::terminate();
}
```

Gate: compilar + make plugin-integ-test verde (4a + 4b no deben romperse).

### PASO 3 — FIX-D: MAX_PLUGIN_PAYLOAD_SIZE

Archivo: `plugin-loader/include/plugin_loader/plugin_loader.hpp`
Añadir constexpr:
```cpp
static constexpr size_t MAX_PLUGIN_PAYLOAD_SIZE = 65536; // 64KB — D8 extendido
```

Archivo: `plugin-loader/src/plugin_loader.cpp`
En `invoke_all(MessageContext& ctx)`, tras FIX-C:
```cpp
// D8-pre size limit (FIX-D, Consejo DAY 110 — ChatGPT5 obligatorio)
if (ctx.payload != nullptr && ctx.payload_len > MAX_PLUGIN_PAYLOAD_SIZE) {
    std::cerr << "[plugin-loader] SECURITY: payload_len=" << ctx.payload_len
              << " excede MAX_PLUGIN_PAYLOAD_SIZE=" << MAX_PLUGIN_PAYLOAD_SIZE
              << " — std::terminate()\n";
    std::terminate();
}
```

Gate: compilar + make plugin-integ-test verde (4a + 4b).
Instalar header actualizado en VM: vagrant ssh -c 'sudo cp ...'

### PASO 4 — TEST-INTEG-4c

Escribir plugins/test-message/test_integ_4c.cpp con los 3 casos de Grok:
- Caso A: NORMAL + payload real presente → PLUGIN_OK, errors==0
- Caso B: NORMAL + plugin modifica campo read-only → D8 VIOLATION
- Caso C: NORMAL + result_code=-1 → error registrado, no crash, no llega a next stage

Añadir al Makefile target plugin-integ-test (tras 4b).
Gate: make plugin-integ-test verde (4a + 4b + 4c).

### PASO 5 — PHASE 2d: ml-detector

Solo si PASOS 2-4 están verdes.
Archivos: ml-detector/src/main.cpp, ml-detector/CMakeLists.txt
Mismo patrón que sniffer: set_plugin_loader() + member + invoke_all()
Contrato: payload post-inferencia, mode=PLUGIN_MODE_NORMAL, D8-v2 CRC32 activo.
Gate: TEST-INTEG-4d.

---

## Deuda pendiente (no bloqueante)

- ADR-028: escribir antes del primer plugin write-capable
- ADR-029: g_plugin_loader + async-signal-safe (antes de PHASE 2e)
- REC-2: noclobber + check 0-bytes CI (P2)
- PHASE 2e (rag-security) — tras 2d
- ADR-025 (Plugin Integrity Ed25519) — post PHASE 2 completa
- TEST-PROVISION-1 como gate CI formal (post PHASE 2)
- arXiv Replace v13 cuando submit/7438768 sea anunciado

---

## Contexto permanente

### Proyecto
- **aRGus NDR (ML Defender)**: C++20 NDR para hospitales, escuelas, municipios.
- **Branch activa**: feature/plugin-crypto
- **Repositorio**: https://github.com/alonsoir/argus
- **arXiv**: submit/7438768 — pendiente moderación (cs.CR).
  Draft v13 listo para Replace cuando v1 sea anunciada.

### Comandos VM críticos
- Editar ficheros en VM: python3 << 'PYEOF' (nunca sed -i sin -e '' en macOS)
- vagrant ssh -c '...' con comillas simples para CMAKE_FLAGS
- Restaurar backup: cp fichero.cpp.backup fichero.cpp — SIEMPRE verificar con wc -l
- NUNCA > fichero para escribir código — usar python3 heredoc o cat << 'EOF'
- CMake: NO_DEFAULT_PATH para libsodium
- CI: .github/workflows/ci.yml

### Consejo de Sabios (7 miembros)
Claude, Grok, ChatGPT, DeepSeek, Qwen (Alibaba), Gemini, Parallel.ai.
Qwen se auto-identifica como DeepSeek — registrar como Qwen en actas.
FIX-C y FIX-D son OBLIGATORIOS antes de cerrar PHASE 2c (ChatGPT5 DAY 110).