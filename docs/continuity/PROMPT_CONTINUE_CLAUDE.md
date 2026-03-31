# ML Defender — Prompt de Continuidad DAY 104
## 1 abril 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 25/25 suites ✅
**Rama activa:** `feature/bare-metal-arxiv` → **merge a main HOY, primer acto**
**Último commit:** DAY 103 cierre — README + paper v8 + Consejo consolidado

---

## Lo realizado en DAY 103 (completo)

| Tarea | Estado |
|-------|--------|
| MAKEFILE-RAG — 6 fixes (cmake PROFILE, log path, rag-attach, test-components, build-unified, banner) | ✅ |
| PAPER §5 — HKDF Context Symmetry case study | ✅ |
| PAPER v8 — corrección crítica bug HKDF (ChatGPT5): contextos distintos → claves distintas → MAC failures | ✅ |
| BACKLOG.md — DAY 103 + BARE-METAL replanificado | ✅ |
| README.md — actualizado DAY 103 | ✅ |
| Consejo de Sabios — sesión ADR-023 + ADR-024, 5 revisores, decisiones consolidadas | ✅ |
| git push origin feature/bare-metal-arxiv | ✅ |

**NOTA BARE-METAL:** Bloqueado por hardware físico. No bloquea arXiv ni feature/plugin-crypto.
Resultado conocido: >33 Mbps VirtualBox, CPU/RAM con amplio margen.

---

## ⚡ PRIMER ACTO DAY 104 — merge + nueva rama

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker

# 1 — Merge a main
git checkout main
git pull origin main
git merge feature/bare-metal-arxiv
git push origin main

# 2 — Nueva rama feature/plugin-crypto
git checkout -b feature/plugin-crypto
git push -u origin feature/plugin-crypto
```

---

## Endorser arXiv — estado

| Endorser | Estado |
|----------|--------|
| Sebastian Garcia (CTU Prague) | ✅ respondió, recibió PDF v5 |
| Yisroel Mirsky (BGU) | ⏳ enviado DAY 96, sin respuesta |
| Andrés Caro Lindo (UEx/INCIBE) | ✅ endorsement confirmado — llamada HOY jueves 2 abril |

**Llamada Andrés:** 657 33 10 10. Él llama o tú llamas.
**Punto clave:** El endorser no necesita hacer nada antes de que enviemos.
Indicamos su email en el formulario arXiv → arXiv le manda email de confirmación.
El envío puede ocurrir en cualquier momento independientemente.

---

## Consejo DAY 103 — Decisiones consolidadas

### Q1 — MessageContext (ADR-023) — unanimidad 5/0

```c
typedef struct {
    uint8_t     version;       // MESSAGE_CONTEXT_VERSION = 1
    uint8_t     direction;     // MLD_TX = 0, MLD_RX = 1
    uint8_t     nonce[12];     // 96-bit monotonic counter
    uint8_t     tag[16];       // Poly1305 tag (16 bytes)
    uint8_t*    payload;       // buffer (in/out)
    size_t      length;        // longitud actual
    size_t      max_length;    // capacidad — siempre >= length + 16 (margen Poly1305)
    const char* channel_id;    // "sniffer-to-ml-detector" — selector contexto HKDF
    int32_t     result_code;   // 0=OK, -1=MAC failure, -2=buffer overflow
    uint8_t     reserved[8];   // para sequence_number / timestamp futuro
} MessageContext;
```

**Regla crítica (DeepSeek/Gemini):** `max_length` siempre >= `length + 16`.
El componente host es responsable del margen. El plugin no aloca.
**Campo crítico (Qwen):** `channel_id` — sin él el plugin no puede seleccionar
el contexto HKDF correcto. Directamente relacionado con el bug ADR-022.

### Q2 — plugin_process_message() — 4/1 OPCIONAL en PHASE 2a

- PHASE 2a: opcional vía `dlsym`, `PLUGIN_API_VERSION=1` sin bump
- Log INFO cuando se detecta que un plugin implementa el hook
- PHASE 2b: bump a VERSION=2 cuando JSON descriptor declare `"layer": "transport"`
- Gemini (minoría): bump inmediato — mérito técnico registrado, desestimado

### Q3 — ADR-024: Noise Protocol IK — unanimidad 5/0

```
PSK = HKDF(seed_family, "noise-ik-psk")  ← seed de familia (ADR-021)

Handshake (1-RTT, solo en arranque):
  Initiator (nuevo componente): -> e, es
  Responder (miembro existente): <- e, ee, se

Output: clave de sesión con forward secrecy + autenticación mutua
Sin PKI central. Compatible con libsodium 1.0.19.
Implementación: Noise-c (C puro).
```

### Q4 — Secuenciación — 4/1 diseño en paralelo

- Diseñar ADR-024 ahora (suficiente para mencionar en paper como Future Work)
- Implementar ADR-023 primero (prerequisito técnico de FEAT-PLUGIN-CRYPTO-1)
- ADR-024 implementación post-ADR-023

---

## DAY 104 — tareas en orden

### TAREA 1 — merge + nueva rama (ver PRIMER ACTO arriba)

### TAREA 2 — Redactar ADR-023 formal

Fichero: `docs/adr/ADR-023-multi-layer-plugin-architecture.md`

Contenido mínimo:
- Contexto: ADR-012 PHASE 1b → único hook `plugin_process_packet(PacketContext*)`
- Decisión: tres capas, tres contextos, tres hooks independientes
- `MessageContext` con campos aprobados por el Consejo (ver arriba)
- Estrategia PHASE 2a/2b/2c + gates TEST-INTEG-4a/4b/4c
- Regla DeepSeek: core CryptoTransport read-only durante PHASE 2a
- Minoría Gemini registrada

### TAREA 3 — Redactar ADR-024 borrador

Fichero: `docs/adr/ADR-024-dynamic-group-key-agreement.md`

Contenido mínimo:
- Problema: componente nuevo que se une a familia en runtime sin redeploy
- Protocolo: Noise IK + PSK derivado de seed_family (ADR-021)
- Handshake descrito
- Implementación: Noise-c + libsodium 1.0.19
- Estado: DISEÑO — implementación post-arXiv

### TAREA 4 — Llamada Andrés Caro Lindo

---

## Patrón ADR-012 PHASE 1b (establecido)

1. `CMakeLists.txt`: find_library(PLUGIN_LOADER_LIB) + find_path + target_include +
   target_link(dl) + target_compile_definitions(PLUGIN_LOADER_ENABLED)
2. `src/main.cpp`: `unique_ptr<PluginLoader>` local en main()
   (global `g_plugin_loader` solo si hay signal handler)
3. `config/*.json`: sección `plugins` con hello plugin `active:true`
4. Smoke test: `MLD_DEV_MODE=1 ./component 2>&1 | grep -i plugin`

---

## Constantes
```
Raíz:    /Users/aironman/CLionProjects/test-zeromq-docker
VM:      vagrant ssh -c '...'   ← SIEMPRE -c
         vagrant ssh -- python3 << 'PYEOF' ... PYEOF  ← para scripts Python
Binarios: /vagrant/{component}/build-debug/{component}
RAG bin:  /vagrant/rag/build/rag-security  ← build/ no build-debug/
Plugins:  /usr/lib/ml-defender/plugins/   ← plugins individuales
Libs:     /usr/local/lib/                 ← libplugin_loader.so
Keys:     /etc/ml-defender/{component}/seed.bin
dev:      MLD_DEV_MODE=1 → sin seed.bin

macOS:   NUNCA sed -i sin -e '' → Python3 heredoc
zsh:     NUNCA Python inline con paréntesis → heredoc 'PYEOF'
zsh:     NUNCA assert(!x) → usar assert(x == false)
cmake:   NO_DEFAULT_PATH para libsodium — priorizar /usr/local
rag:     comillas simples en vagrant ssh -c para CMAKE_FLAGS
```

---

## Consejo de Sabios

Revisores: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI),
DeepSeek, Qwen (Alibaba — se autoidentifica como DeepSeek),
Gemini (Google), Parallel.ai

---

*DAY 103 cierre — 31 marzo 2026*
*Tests: 25/25 ✅ · Paper: Draft v8 · Makefile rag: alineado*
*Consejo DAY 103: MessageContext aprobada, Noise IK aprobado*
*BARE-METAL: bloqueado por hardware, no bloquea arXiv*
*Endorser Andrés Caro Lindo: llamada HOY jueves 2 abril*
*Mañana: merge main + feature/plugin-crypto como PRIMER ACTO*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*