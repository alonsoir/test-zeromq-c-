# ML Defender — Prompt de Continuidad DAY 103
## 31 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 25/25 suites ✅ (nuevo récord — TEST-PLUGIN-INVOKE-1 añadido DAY 102)
**Rama activa:** `feature/bare-metal-arxiv`
**Último commit:** ADR-012 PHASE 1b COMPLETA — rag-security plugin-loader (DAY 102)

---

## Lo realizado en DAY 102 (completo)

| Tarea | Estado |
|-------|--------|
| TEST-PLUGIN-INVOKE-1 — invoke_all() smoke test | ✅ |
| ADR-012 PHASE 1b — firewall-acl-agent integrado + validado | ✅ |
| ADR-012 PHASE 1b — rag-ingester integrado + validado | ✅ |
| ADR-012 PHASE 1b — rag-security integrado + validado | ✅ |
| Andrés Caro Lindo (UEx) — respuesta recibida + reply enviado | ✅ |
| git push origin feature/bare-metal-arxiv | ✅ |

**ADR-012 PHASE 1b: 5/5 componentes ✅ COMPLETA**

---

## Endorser arXiv — estado

| Endorser | Estado |
|----------|--------|
| Sebastian Garcia (CTU Prague) | ✅ respondió, recibió PDF |
| Yisroel Mirsky (BGU) | ⏳ enviado DAY 96, sin respuesta |
| Andrés Caro Lindo (UEx) | ✅ confirmado — llamada telefónica jueves 2 abril |

**Nota:** Andrés ofrece colaboración futura (revistas, congresos INCIBE).
Su número: 657 33 10 10. Él llama o tú llamas el jueves.

---

## Patrón ADR-012 PHASE 1b (establecido — replicar si se añaden componentes)

1. `CMakeLists.txt`: find_library(PLUGIN_LOADER_LIB) + find_path + target_include +
   target_link(dl) + target_compile_definitions(PLUGIN_LOADER_ENABLED)
2. `src/main.cpp`: `std::unique_ptr<ml_defender::PluginLoader>` global o local
   (global si hay signal handler que necesita acceso) +
   `#ifdef PLUGIN_LOADER_ENABLED` guards en include + init + shutdown
3. `config/*.json`: sección `plugins` con hello plugin `active:true`
4. Smoke test: `MLD_DEV_MODE=1 ./component 2>&1 | grep -i plugin`

**Nota rag-security:** usa `g_plugin_loader` global (signal handler necesita acceso).
Resto de componentes: `unique_ptr` local en `main()`.

---

## DAY 103 — tareas en orden estricto

### TAREA 1 — Revisión Makefile rag alignment (URGENTE)

El Makefile tiene inconsistencias con el componente `rag`:
- `rag-build` delega a `cd /vagrant/rag && make build` (Makefile interno) → siempre Release
- Resto de componentes usan `cmake $(CMAKE_FLAGS)` directamente → perfil configurable
- `pipeline-start` / `pipeline-stop` / `pipeline-status` → ¿está rag-security incluido?
- No hay tarea `rag-attach` para attachear al proceso arrancado en tmux
- Los tests del rag (test_faiss_basic, test_embedder, test_onnx_basic) no están
  en `test-components` ni en `test-all`

```bash
# Exploración antes de tocar nada
grep -n "rag" Makefile | grep -v "rag-ingester\|#" | head -30
grep -n "pipeline-start\|pipeline-stop\|pipeline-status" Makefile | head -20
vagrant ssh -c 'cat /vagrant/rag/Makefile 2>/dev/null | head -40'
```

Decisiones a tomar:
- ¿Alinear `rag-build` al patrón de los demás (cmake directo con PROFILE)?
- ¿Añadir `rag-attach` (tmux attach -t rag-security)?
- ¿Añadir rag a `test-components`?
- ¿Añadir `rag-build` a `build-unified`?

### TAREA 2 — PAPER-ADR022 §6

Subsección "HKDF Context Symmetry: A Pedagogical Case Study in
Test-Driven Hardening". Ubicación: §6 (después de §6.4 TDH o §6.7).

Estructura:
- El error: contexto HKDF = componente (mal) vs canal (correcto)
- Por qué es invisible al type-checker (ambos son std::string)
- Cómo TEST-INTEG-3 lo detectó (regresión intencional → MAC failure)
- La lección: correctness criptográfica requiere tests E2E de protocolo

### TAREA 3 — Actualizar BACKLOG.md en repo

Reflejar DAY 102: ADR-012 PHASE 1b COMPLETA (5/5), tests 25/25,
mover PLUGIN-LOADER-FW y PLUGIN-LOADER-RAG de P2 a ✅ COMPLETADO.

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
```

---

## Consejo de Sabios — práctica establecida

Revisores: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI),
DeepSeek, Qwen (Alibaba — se autoidentifica como DeepSeek),
Gemini (Google), Parallel.ai

---

# ML Defender — Prompt de Continuidad DAY 103
## 31 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 25/25 suites ✅ (nuevo récord — TEST-PLUGIN-INVOKE-1 añadido DAY 102)
**Rama activa:** `feature/bare-metal-arxiv`
**Último commit:** ADR-012 PHASE 1b COMPLETA — rag-security plugin-loader (DAY 102)

---

## Lo realizado en DAY 102 (completo)

| Tarea | Estado |
|-------|--------|
| TEST-PLUGIN-INVOKE-1 — invoke_all() smoke test | ✅ |
| ADR-012 PHASE 1b — firewall-acl-agent integrado + validado | ✅ |
| ADR-012 PHASE 1b — rag-ingester integrado + validado | ✅ |
| ADR-012 PHASE 1b — rag-security integrado + validado | ✅ |
| Andrés Caro Lindo (UEx) — respuesta recibida + reply enviado | ✅ |
| git push origin feature/bare-metal-arxiv | ✅ |

**ADR-012 PHASE 1b: 5/5 componentes ✅ COMPLETA**

---

## Endorser arXiv — estado

| Endorser | Estado |
|----------|--------|
| Sebastian Garcia (CTU Prague) | ✅ respondió, recibió PDF |
| Yisroel Mirsky (BGU) | ⏳ enviado DAY 96, sin respuesta |
| Andrés Caro Lindo (UEx) | ✅ confirmado — llamada telefónica jueves 2 abril |

**Nota:** Andrés ofrece colaboración futura (revistas, congresos INCIBE).
Su número: 657 33 10 10. Él llama o tú llamas el jueves.

---

## Patrón ADR-012 PHASE 1b (establecido — replicar si se añaden componentes)

1. `CMakeLists.txt`: find_library(PLUGIN_LOADER_LIB) + find_path + target_include +
   target_link(dl) + target_compile_definitions(PLUGIN_LOADER_ENABLED)
2. `src/main.cpp`: `std::unique_ptr<ml_defender::PluginLoader>` global o local
   (global si hay signal handler que necesita acceso) +
   `#ifdef PLUGIN_LOADER_ENABLED` guards en include + init + shutdown
3. `config/*.json`: sección `plugins` con hello plugin `active:true`
4. Smoke test: `MLD_DEV_MODE=1 ./component 2>&1 | grep -i plugin`

**Nota rag-security:** usa `g_plugin_loader` global (signal handler necesita acceso).
Resto de componentes: `unique_ptr` local en `main()`.

---

## DAY 103 — tareas en orden estricto

### TAREA 1 — Revisión Makefile rag alignment (URGENTE)

El Makefile tiene inconsistencias con el componente `rag`:
- `rag-build` delega a `cd /vagrant/rag && make build` (Makefile interno) → siempre Release
- Resto de componentes usan `cmake $(CMAKE_FLAGS)` directamente → perfil configurable
- `pipeline-start` / `pipeline-stop` / `pipeline-status` → ¿está rag-security incluido?
- No hay tarea `rag-attach` para attachear al proceso arrancado en tmux
- Los tests del rag (test_faiss_basic, test_embedder, test_onnx_basic) no están
  en `test-components` ni en `test-all`

```bash
# Exploración antes de tocar nada
grep -n "rag" Makefile | grep -v "rag-ingester\|#" | head -30
grep -n "pipeline-start\|pipeline-stop\|pipeline-status" Makefile | head -20
vagrant ssh -c 'cat /vagrant/rag/Makefile 2>/dev/null | head -40'
```

Decisiones a tomar:
- ¿Alinear `rag-build` al patrón de los demás (cmake directo con PROFILE)?
- ¿Añadir `rag-attach` (tmux attach -t rag-security)?
- ¿Añadir rag a `test-components`?
- ¿Añadir `rag-build` a `build-unified`?

### TAREA 2 — PAPER-ADR022 §6

Subsección "HKDF Context Symmetry: A Pedagogical Case Study in
Test-Driven Hardening". Ubicación: §6 (después de §6.4 TDH o §6.7).

Estructura:
- El error: contexto HKDF = componente (mal) vs canal (correcto)
- Por qué es invisible al type-checker (ambos son std::string)
- Cómo TEST-INTEG-3 lo detectó (regresión intencional → MAC failure)
- La lección: correctness criptográfica requiere tests E2E de protocolo

### TAREA 3 — Actualizar BACKLOG.md en repo

Reflejar DAY 102: ADR-012 PHASE 1b COMPLETA (5/5), tests 25/25,
mover PLUGIN-LOADER-FW y PLUGIN-LOADER-RAG de P2 a ✅ COMPLETADO.

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
```

---

## Consejo de Sabios — práctica establecida

Revisores: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI),
DeepSeek, Qwen (Alibaba — se autoidentifica como DeepSeek),
Gemini (Google), Parallel.ai

---

*DAY 102 cierre — 30 marzo 2026*
*Tests: 25/25 ✅ · ADR-012 PHASE 1b: 5/5 COMPLETA ✅*
*Endorser Andrés Caro Lindo confirmado ✅ · Llamada jueves 2 abril*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*

---

## Consejo de Sabios — FEAT-PLUGIN-CRYPTO-1 (decisiones consolidadas)

Consulta realizada DAY 102 cierre. Unanimidad 5/0 en las tres preguntas.

### Q1 — Opción A (MessageContext) — unanimidad 5/0

`PacketContext` = capa de red. `MessageContext` = capa de transporte.
Mezclarlos en Opción B es el mismo *model mental error* que ADR-022.

```c
// PHASE 2 — nuevo hook en plugin_api.h
PluginResult plugin_process_message(MessageContext* ctx);
// MessageContext: payload, length, max_length, direction tx/rx,
//                nonce[12], tag[16], result_code
```

### Q2 — Símbolo opcional primero, bump después — unanimidad 5/0

```
PHASE 2a: plugin_process_message() OPCIONAL
          dlsym() → si existe: plugin de transporte
                  → si no:    plugin de red (PHASE 1, sigue funcionando)
          PLUGIN_API_VERSION = 1 (sin bump)

PHASE 2b: plugin_process_message() OBLIGATORIO para plugins de transporte
          PLUGIN_API_VERSION = 2
```

### Q3 — Dual-mechanism aprobada con gates adicionales

| Gate | Descripción |
|------|-------------|
| TEST-INTEG-4a | Round-trip idéntico byte a byte |
| TEST-INTEG-4b | Equivalencia semántica — ml-detector ve features idénticas en ambos paths |
| TEST-INTEG-4c | Fail-closed ante MAC failure → SIGABRT confirmado |

**Regla adicional (DeepSeek):** core `CryptoTransport` read-only durante PHASE 2a.
Solo se hacen cambios en el plugin. Validación unidireccional: plugin → core.

### Decisiones adicionales

- **ADR-023** a redactar antes de implementar FEAT-PLUGIN-CRYPTO-1:
  ```
  PacketContext  → plugin_process_packet()   [red]
  MessageContext → plugin_process_message()  [transporte]
  SkillContext   → plugin_execute_skill()    [aplicación — futuro]
  ```
- **Fail-closed confirmado:** MAC failure → `std::terminate()`. Sin modo degradado.
- **ADR-012 compatible:** plugin crypto *transforma*, no *decide*. No viola restricción de bloqueo.
- **Insight Gemini:** Opción A = agnositicismo de transporte. ZMQ → QUIC sin tocar sniffer.cpp.

**Todo esto es post-arXiv. DAY 103 arranca con Makefile rag alignment.**

*DAY 102 cierre — 30 marzo 2026*
*Tests: 25/25 ✅ · ADR-012 PHASE 1b: 5/5 COMPLETA ✅*
*Endorser Andrés Caro Lindo confirmado ✅ · Llamada jueves 2 abril*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*