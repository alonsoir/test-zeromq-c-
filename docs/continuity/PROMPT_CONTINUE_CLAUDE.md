# ML Defender — Prompt de Continuidad DAY 102
## 30 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 24/24 suites ✅
**Rama activa:** `feature/bare-metal-arxiv`
**Último commit:** ADR-012 PHASE 1b — ml-detector plugin-loader (DAY 101)

---

## Lo realizado en DAY 101 (completo)

| Tarea | Estado |
|-------|--------|
| fix(plugin-loader): extract_enabled_objects | ✅ |
| ADR-012 PHASE 1b — sniffer validado (bug fix) | ✅ |
| ADR-012 PHASE 1b — ml-detector integrado + validado | ✅ |
| Email andresc@unex.es (endorser arXiv) + PDF v6 | ✅ enviado |
| Consejo de Sabios DAY 101 — 5/5 respuestas | ✅ |
| BACKLOG.md actualizado | ✅ |

---

## Bug fix DAY 101 — extract_enabled_objects

`extract_enabled_list()` iteraba sobre claves del objeto JSON.
Reemplazada por `extract_enabled_objects()`:
- Itera objetos `{}` dentro del array `enabled`
- Filtra `active:false` antes de cargar
- Lee `name` y `path` explícitos del descriptor JSON
- Canonical plugin dir: `/usr/lib/ml-defender/plugins/`
- System lib dir: `/usr/local/lib/` (libplugin_loader.so — NO plugins individuales)

---

## ADR-012 PHASE 1b — estado exacto

| Componente | Estado | Config |
|------------|--------|--------|
| `sniffer` | ✅ DAY 101 | `sniffer/config/sniffer.json` |
| `ml-detector` | ✅ DAY 101 | `ml-detector/config/ml_detector_config.json` |
| `firewall-acl-agent` | ⏳ P1 DAY 102 | — |
| `rag-ingester` | ⏳ P2 | — |
| `rag-security` | ⏳ P3 | — |

**Nota:** `rag-security` añadido al roadmap DAY 101 — 5 componentes
total (no 4). Orden revisado por Consejo DAY 101:
sniffer ✅ → ml-detector ✅ → **firewall** → rag-ingester → rag-security

Patrón establecido (replicar en todos):
1. `CMakeLists.txt`: find_library + find_path + target_include +
   target_link + target_compile_definitions(PLUGIN_LOADER_ENABLED)
2. `src/main.cpp`: `#ifdef PLUGIN_LOADER_ENABLED` →
   include + instanciar + load_plugins() + shutdown()
3. `config/*.json`: sección `plugins` con hello plugin `active:true`
4. Smoke test: `MLD_DEV_MODE=1 ./component -c config.json 2>&1 | grep -i plugin`

---

## Consejo de Sabios DAY 101 — decisiones consolidadas

| Pregunta | Decisión | Origen |
|----------|----------|--------|
| Orden plugin-loader | firewall → rag-ingester → rag-security | Unanimidad 5/5 |
| HKDF paper ubicación | **§6 subsección independiente** | Grok + Qwen (árbitro: Alonso) |
| TEST-PLUGIN-INVOKE-1 | Necesario antes de seguir con firewall | Unanimidad 5/5 |

**Sobre Q2 (divergencia 3/2):**
ChatGPT5 + DeepSeek + Gemini → §5.5 con referencia cruzada
Grok + Qwen → §6 subsección independiente
Argumento ganador: el bug es un error de *modelo mental*
(contexto = componente vs canal), invisible al type-checker,
detectado por TDH. Es metodológico, no solo técnico. §6 lo eleva.

**Nota Consejo:** El archivo `qwen.md` se autoidentifica como DeepSeek.
Hipótesis del autor: Qwen es una versión modificada del código fuente
de DeepSeek, entrenada con datasets y pesos distintos. Los razonamientos
son distinguibles — el Consejo los trata como voces independientes.

---

## DAY 102 — tareas en orden estricto

### TAREA 1 — TEST-PLUGIN-INVOKE-1 (unanimidad Consejo)

Antes de integrar en firewall, validar el hot path de ejecución.
Crear test unitario en `plugin-loader/tests/` o en la suite del sniffer:
```cpp
// Objetivo: PacketContext sintético → invoke_all() → invocations > 0
// Verificar: PLUGIN_OK, invocations==1, overruns==0, errors==0
// El hello plugin no modifica contexto — verificar solo contadores
```
```bash
# Ver si existe suite de tests en plugin-loader
ls plugin-loader/tests/ 2>/dev/null || echo "no tests dir"
grep -n "test\|TEST\|gtest\|catch" plugin-loader/CMakeLists.txt | head -10
# Ver PacketContext — estructura necesaria para el test
grep -rn "PacketContext\|struct Packet" plugin-loader/include/ | head -10
grep -rn "struct PacketContext" sniffer/src/ | head -5
```

### TAREA 2 — PLUGIN-LOADER-FW (P1)

Integrar plugin-loader en `firewall-acl-agent`. Mismo patrón.
```bash
grep -n "add_executable\|target_link\|target_compile\|message(STATUS" \
  firewall-acl-agent/CMakeLists.txt | head -30
grep -n "main\|config\|shutdown\|return 0" \
  firewall-acl-agent/src/main.cpp | head -30
ls firewall-acl-agent/config/
```

### TAREA 3 — PAPER-ADR022 §6

Subsección "HKDF Context Symmetry: A Pedagogical Case Study in
Test Driven Hardening". Ubicación: §6 (después de §6.4 TDH o §6.7).

Estructura sugerida:
- El error: contexto HKDF = componente (mal) vs canal (correcto)
- Por qué es invisible al type-checker
- Cómo TEST-INTEG-3 lo detectó (regresión intencional → MAC failure)
- La lección: correctness criptográfica requiere tests E2E de protocolo

---

## Endorser arXiv — estado

| Endorser | Estado |
|----------|--------|
| Sebastian Garcia (CTU Prague) | ✅ respondió, recibió PDF |
| Yisroel Mirsky (BGU) | ⏳ enviado DAY 96, sin respuesta |
| Andrés Caro Lindo (UEx) | ⏳ enviado DAY 101, esperando |

---

## Constantes
```
Raíz:    /Users/aironman/CLionProjects/test-zeromq-docker
VM:      vagrant ssh -c '...'   ← SIEMPRE -c
Binarios: /vagrant/{component}/build-debug/{component}
Plugins:  /usr/lib/ml-defender/plugins/   ← plugins individuales
Libs:     /usr/local/lib/                 ← libplugin_loader.so
Keys:     /etc/ml-defender/{component}/seed.bin
dev:      MLD_DEV_MODE=1 → sin seed.bin

macOS:   NUNCA sed -i sin -e '' → Python3 heredoc
zsh:     NUNCA Python inline con paréntesis → heredoc 'PYEOF'
cmake:   NO_DEFAULT_PATH para libsodium — priorizar /usr/local
```

---

## Consejo de Sabios — práctica establecida

Revisores: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI),
DeepSeek, Qwen (Alibaba — se autoidentifica como DeepSeek),
Gemini (Google), Parallel.ai

---

*DAY 101 cierre — 29 marzo 2026*
*Tests: 24/24 ✅ · ADR-012 PHASE 1b: sniffer ✅ + ml-detector ✅*
*Bug fix: extract_enabled_objects ✅ · Endorser andresc@unex.es ✅*
*Consejo 5/5 ✅ · TEST-PLUGIN-INVOKE-1 pendiente DAY 102*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*