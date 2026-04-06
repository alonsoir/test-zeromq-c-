# ML Defender (aRGus NDR) — DAY 110 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## Estado al cierre de DAY 109

### Completado hoy

**FIX-A** ✅ — `MLD_ALLOW_UNCRYPTED` escape hatch en 3 adaptadores `etcd_client.cpp`
(ml-detector, sniffer, firewall-acl-agent). En producción: `std::terminate()`. En dev:
`cerr FATAL[DEV] + return` (constructor void, no `return false`).

**FIX-B** ✅ — `provision.sh`: `mkdir -p /vagrant/rag-security/config` + symlinks JSON
automáticos. Eliminado warning "Config dir no existe aún". Idempotente, verificado.

**PHASE 2b** ✅ — `plugin_process_message()` en `rag-ingester/src/main.cpp`.
Contrato READ-ONLY: `ctx_readonly.payload = nullptr`, `ctx_readonly.payload_len = 0`.
Inserción antes de `embed_chronos()`. Early return si `result_code != 0`.

**D8-light READ-ONLY fix** ✅ — `plugin-loader/src/plugin_loader.cpp`:
`is_readonly = (payload == nullptr && payload_len == 0)`. CRC32 protegido también.

**TEST-INTEG-4b** ✅ PASSED — variantes A (accept) y B (reject, D8 VIOLATION).
Integrado en `make plugin-integ-test` (4a + 4b secuencial).

**Paper Draft v12** ✅ — §3.4 Out-of-Scope expandido (vector físico/USB consciente),
§4 Integration Philosophy nueva subsección, §11.16 referencia corta. Compilación
limpia Overleaf. Pendiente: Replace en arXiv cuando submit/7438768 sea anunciado.

**Commits DAY 109:** da7355d8 → 81ab2101 → d13b35d1 → [hash cierre]
**Estado pipeline:** 6/6 RUNNING. Branch: `feature/plugin-crypto`.

---

## Consejo DAY 109 — Decisiones (5/5 respondieron)

**Nota de acta:** Qwen (chat.qwen.ai) se auto-identifica como DeepSeek — patrón
consolidado DAY 103-109. Registrado como Qwen en todas las actas.

**Q1 — Flag explícito READ-ONLY: APROBADO 4/5** (Grok en minoría)
Añadir `uint8_t mode` a `MessageContext` en `plugin_api.h` (1 byte de `reserved[60]`).
`PLUGIN_MODE_NORMAL=0`, `PLUGIN_MODE_READONLY=1`. D8-light valida coherencia:
`mode==READONLY → payload DEBE ser nullptr`. Convierte contrato implícito en verificable.

**Q2 — PHASE 2c sniffer: payload real, UNANIMIDAD 5/5**
El sniffer es el punto más privilegiado del pipeline. Sin payload, plugins inútiles.
El payload del sniffer son headers IP/TCP/UDP capturados por XDP — no TLS plaintext.
Contrato: payload presente, D8-v2 CRC32 activo, result_code!=0 → paquete descartado.

**Q3 — §4 Integration Philosophy: expandir, UNANIMIDAD 5/5**
Cuatro argumentos a añadir al paper v13:
1. Latencia determinista (HTTP/Kafka = jitter inaceptable para respuesta <10ms)
2. Superficie de ataque (parsers HTTP = fuente histórica de CVEs; raw TCP reduce >90%)
3. Sin broker = sin SPOF (Kafka/Redis incompatibles con host único 150-200 USD)
4. Footprint mínimo (sin librdkafka, sin libcurl, sin boost.asio)

**Q4 — ADR-028: diferir hasta primer plugin write-capable (Grok + DeepSeek 2/2)**
Gemini + Qwen querían aprobarlo antes de PHASE 2c. Grok + DeepSeek: diferir evita
documentación prematura. Decisión: diferir, pero documentar en backlog.

---

## ADR-028 — Qué es y cuándo escribirlo

**ADR-028 NO existe aún.** Fue nombrado por ChatGPT5 en Consejo DAY 108 como
"RAG Ingestion Trust Model" pero nunca se ha escrito. Solo existe como ítem en backlog.

**Lo que debe cubrir ADR-028:**
- Qué recursos puede leer un plugin (MessageContext fields, payload según mode)
- Qué recursos puede escribir (result_code, annotation — y nada más por ahora)
- Qué recursos son absolutamente protegidos (índice FAISS, SQLite, etcd)
- Auditoría de operaciones de escritura
- Política de rollback ante corrupción de contexto
- Distinción formal entre PLUGIN_MODE_READONLY y PLUGIN_MODE_READWRITE

**Relación con Q1:** ADR-028 es el documento que hace el flag `mode` legalmente
vinculante en la arquitectura. Sin ADR-028, el flag es un mecanismo sin contrato.
Con ADR-028, el flag tiene semántica formal.

**Cuándo escribirlo:** Antes del primer plugin write-capable. No es bloqueante para
PHASE 2c (sniffer es read-only en cuanto a modificación de payload).

---

## Orden DAY 110 (no saltarse)

### PASO 1 — Verificar estado
```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/plugin-crypto
git pull origin feature/plugin-crypto
make pipeline-status
```

### PASO 2 — Añadir `mode` a MessageContext (Q1 Consejo)

Archivo: `plugin-loader/include/plugin_loader/plugin_api.h`

```c
// Añadir enum y campo mode a MessageContext
typedef enum {
    PLUGIN_MODE_NORMAL   = 0,  // acceso normal al payload
    PLUGIN_MODE_READONLY = 1   // rag-ingester: payload=nullptr garantizado
} PluginMode;

// En MessageContext, sustituir primer byte de reserved[60] por mode:
uint8_t  mode;        // PluginMode — consume 1 byte de reserved[60]
uint8_t  reserved[59]; // era [60]
```

Actualizar `plugin_loader.cpp`: validar coherencia
`mode==READONLY && (payload!=nullptr || payload_len!=0) → std::terminate()`.

Actualizar `rag-ingester/src/main.cpp`:
`ctx_readonly.mode = PLUGIN_MODE_READONLY;` (explícito).

Actualizar test `test_integ_4b.cpp`: verificar que `mode` se propaga correctamente.

Gate: `make plugin-integ-test` verde (4a + 4b).

### PASO 3 — PHASE 2c: sniffer + plugin_process_message()

Contrato: payload real, `mode = PLUGIN_MODE_NORMAL`, D8-v2 CRC32 activo.
Archivos: `sniffer/src/userspace/main.cpp`, `sniffer/CMakeLists.txt`,
`sniffer/config/sniffer.json`.
Gate: TEST-INTEG-4c.

### PASO 4 — Paper v13 (§4 Integration Philosophy expandida)

Añadir los 4 argumentos del Consejo al §4. Actualizar draft.
Pendiente también: Replace en arXiv cuando submit/7438768 sea anunciado.

---

## Deuda pendiente (no bloqueante)

- ADR-028: escribir antes del primer plugin write-capable
- Unificar sniffer bajo SeedClient (eliminar `get_encryption_seed` manual)
- ADR-025 (Plugin Integrity Ed25519) — post PHASE 2 completa
- TEST-PROVISION-1 como gate CI formal (post PHASE 2)
- PHASE 2d (ml-detector) y 2e (rag-security) tras 2c

---

## Contexto permanente

### Proyecto
- **aRGus NDR (ML Defender)**: C++20 NDR para hospitales, escuelas, municipios.
- **Branch activa**: `feature/plugin-crypto`
- **Repositorio**: `https://github.com/alonsoir/argus`
- **arXiv**: `submit/7438768` — pendiente moderación (cs.CR).
  Draft v12 listo para Replace cuando v1 sea anunciada.

### Comandos VM críticos
- Editar ficheros en VM: `vagrant ssh -- python3 << 'PYEOF'`
- Comandos simples: `vagrant ssh -c '...'`
- NUNCA `sed -i` sin `-e ''` en macOS
- Firewall config: `/vagrant/firewall-acl-agent/config/firewall.json`
- CMake: `NO_DEFAULT_PATH` para libsodium
- CI: `.github/workflows/ci.yml`

### Consejo de Sabios (7 miembros)
Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI), DeepSeek, Qwen (Alibaba),
Gemini (Google), Parallel.ai. Revisores rigurosos e implacables. Demostrar
problemas con tests compilables o matemáticas antes de proponer fixes.
Verificar que test de vulnerabilidad retorna negativo tras aplicar fix.
**Qwen se auto-identifica como DeepSeek — comportamiento de entrenamiento,
no identidad real. Registrar como Qwen en todas las actas.**