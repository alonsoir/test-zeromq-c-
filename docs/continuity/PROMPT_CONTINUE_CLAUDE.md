# ML Defender — Prompt de Continuidad DAY 107
## 3 abril 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 25/25 suites ✅ + TEST-INTEG-4a-PLUGIN (3/3 variantes) ✅
**Rama activa:** `feature/plugin-crypto`
**Último commit:** DAY 106 — PHASE 2a CLOSED + TEST-INTEG-4a-PLUGIN PASSED + paper arXiv submitted

---

## Lo realizado en DAY 106 (completo)

| Tarea | Estado |
|-------|--------|
| 1c — nonce/tag NULL contract en `plugin_api.h` | ✅ |
| 1d — Makefile deps: `plugin-loader-build` en sniffer, ml-detector, rag-ingester, rag-build | ✅ |
| 1a — CRC32 D8-v2 en `plugin_loader.cpp` (`crc32_fast()` + snapshot antes/después) | ✅ |
| 1b — `plugins/test-message/plugin_test_message.cpp` variantes A/B/C | ✅ |
| TEST-INTEG-4a-PLUGIN — 3/3 variantes PASSED | ✅ |
| Paper Draft v11 — UEx eliminada, 3 figuras TikZ añadidas | ✅ |
| arXiv submit — `submit/7438768` STATUS: submitted | ✅ |

---

## arXiv — estado

| Item | Estado |
|------|--------|
| Cuenta | `alonsoir` / `alonsoir@gmail.com` |
| Submission ID | `7438768` |
| Status | **submitted** — pendiente moderación (1-2 días hábiles) |
| Endorsers | Sebastian Garcia (CTU Prague) ✅, Andrés Caro Lindo (UEx) ✅ |

Cuando llegue el arXiv ID definitivo (`2504.XXXXX`) → actualizar README.md y paper.

---

## ⚡ PRIMER ACTO DAY 107

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/plugin-crypto
git pull origin feature/plugin-crypto
make pipeline-status
```

---

## PHASE 2b — rag-ingester (DESBLOQUEADA)

PHASE 2a completamente cerrada. Siguiente: integrar `plugin_process_message()` en `rag-ingester`.

### Patrón a seguir (igual que firewall-acl-agent DAY 105):

**Archivos a tocar:**
- `rag-ingester/src/main.cpp` — añadir `PluginLoader`, llamar `invoke_all(MessageContext&)`
- `rag-ingester/CMakeLists.txt` — linkear `libplugin_loader.so`
- `rag-ingester/config/rag-ingester.json` — añadir sección `plugins`

**Gate:** TEST-INTEG-4b — `plugin_process_message()` invocado en `rag-ingester` con `MessageContext`, post-invocation invariants verified, `result_code=0`.

### Después de rag-ingester: 3 componentes restantes
- `sniffer` (PHASE 2c)
- `ml-detector` (PHASE 2d)
- `rag-security` (PHASE 2e) — usa `g_plugin_loader` global para signal handler

---

## ADR-025 — Plugin Integrity Verification (Ed25519 + TOCTOU-safe dlopen)

Estado: **APPROVED** por Consejo (DAY 102). Implementación post-PHASE 2 completa.

Archivos a tocar cuando llegue el momento:
- `plugin_loader.cpp`
- `plugin-loader/CMakeLists.txt`
- `tools/provision.sh` (--reset flag)
- JSON config schemas (6 componentes)
- systemd units
- Tests: TEST-INTEG-SIGN-1 → TEST-INTEG-SIGN-7

---

## Archivos nuevos DAY 106

```
plugin-loader/include/plugin_loader/plugin_api.h   ← nonce/tag NULL contract (1c)
plugin-loader/src/plugin_loader.cpp                 ← crc32_fast() + D8-v2 (1a)
plugins/test-message/plugin_test_message.cpp        ← variantes A/B/C (1b)
plugins/test-message/CMakeLists.txt
plugins/test-message/test_variants.cpp
plugins/test-message/test_config.json
Makefile                                            ← plugin-loader-build deps (1d)
```

---

## Constantes

```
Raíz:    /Users/aironman/CLionProjects/test-zeromq-docker
VM:      vagrant ssh -c '...'   ← SIEMPRE -c
         vagrant ssh -- python3 << 'PYEOF' ... PYEOF  ← para scripts Python
Binarios: /vagrant/{component}/build-debug/{component}
RAG bin:  /vagrant/rag/build/rag-security  ← build/ no build-debug/
Plugins:  /usr/lib/ml-defender/plugins/
Libs:     /usr/local/lib/  ← libplugin_loader.so
Keys:     /etc/ml-defender/{component}/seed.bin
dev:      MLD_DEV_MODE=1 / MLD_ALLOW_DEV_MODE → solo Debug builds
Config fw: /vagrant/firewall-acl-agent/config/firewall.json  ← NO firewall-acl-agent.json

macOS:   NUNCA sed -i sin -e '' → Python3 heredoc
zsh:     NUNCA Python inline con paréntesis → heredoc 'PYEOF'
cmake:   NO_DEFAULT_PATH para libsodium
```

---

## Consejo de Sabios

Revisores: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI),
DeepSeek, Qwen (Alibaba), Gemini (Google), Parallel.ai

---

*DAY 106 cierre — 3 abril 2026*
*Tests: 25/25 ✅ + TEST-INTEG-4a-PLUGIN 3/3 ✅*
*PHASE 2a: COMPLETAMENTE CERRADA*
*arXiv: submit/7438768 SUBMITTED ✅*
*Próximo: PHASE 2b — rag-ingester*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*