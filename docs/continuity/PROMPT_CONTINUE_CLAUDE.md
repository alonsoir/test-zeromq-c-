# ML Defender — Prompt de Continuidad DAY 94
## 22 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING (etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall)
**Test suite:** 33/31 ✅ (crypto 3/3, etcd-hmac 12/12, ml-detector 9/9, rag-ingester 7/7, sniffer 1/1)
**Rama activa:** `feature/plugin-loader-adr012`
**Último tag:** DAY92 (rama no mergeada a main todavía)

---

## Lo que se hizo en DAY 93

### ADR-012 PHASE 1 — plugin-loader minimalista

**Ficheros creados y commiteados:**

| Fichero | Descripción |
|---|---|
| `common/include/sentinel.hpp` | `MISSING_FEATURE_SENTINEL = -9999.0f` centralizado (ADR-012 §4) |
| `plugin-loader/include/plugin_loader/plugin_api.h` | Contrato C puro, ABI estable, `PLUGIN_API_VERSION=1` |
| `plugin-loader/include/plugin_loader/plugin_loader.hpp` | Interfaz C++ `PluginLoader` + `PluginStats` |
| `plugin-loader/src/plugin_loader.cpp` | `dlopen`/`dlsym` lazy, sin crypto, sin seed-client |
| `plugin-loader/CMakeLists.txt` | Patrón idéntico a `crypto-transport` |
| `plugins/hello/hello_plugin.cpp` | Hello world plugin — validación contrato end-to-end |
| `plugins/hello/CMakeLists.txt` | Build del hello world plugin |
| `Makefile` | Targets `plugin-loader-build/clean/test` + `plugin-hello-build/clean` |

**Artefactos desplegados en VM:**
- `libplugin_loader.so.1.0.0` → `/usr/local/lib/` (53K) ✅
- `libplugin_hello.so` → `/usr/lib/ml-defender/plugins/` (16K) ✅

**ABI validation via Python3/ctypes:**
```
plugin_name()        = hello
plugin_version()     = 0.1.0
plugin_api_version() = 1  ✅
ABI version match    : True ✅
```

**Fixes aplicados durante el build:**
- `plugin_api.h`: añadir `#include <stdint.h>` y `#include <stddef.h>` (uint8_t/size_t en C puro)
- `plugin-loader/CMakeLists.txt`: `BUILD_TESTS=OFF` hasta que `tests/` exista (DAY 94)

**Documentación:**
- README badge `Plugin Loader ADR-012 PHASE 1` + living contracts links ✅
- `docs/BACKLOG.md` DAY 93 completado ✅

---

## Criterios ADR-012 hello world — estado actual

```
✅ dlopen/dlsym funcionan correctamente
✅ plugin_api_version() retorna PLUGIN_API_VERSION=1
✅ plugin_name() / plugin_version() resueltos
⏳ plugin_init() recibe config JSON           — test integración DAY 94
⏳ plugin_process_packet() en cada paquete   — test integración DAY 94
⏳ plugin_shutdown() limpio                  — test integración DAY 94
⏳ Si se elimina el .so, host no aborta      — validación manual DAY 94
⏳ Budget overrun → warning en log           — validación manual DAY 94
```

---

## Objetivo principal DAY 94 — integración sniffer + test suite

### Tarea A — Integrar plugin-loader en sniffer

El sniffer es el componente host natural (ADR-012). Mínimo cambio necesario:

**`sniffer/CMakeLists.txt`:**
```cmake
target_link_libraries(sniffer PRIVATE plugin-loader crypto-transport ...)
```

**`sniffer/src/` — punto de integración (tras el fast path):**
```cpp
#include "plugin_loader/plugin_loader.hpp"

// Init (una vez):
ml_defender::PluginLoader plugin_loader_("config/sniffer.json");
plugin_loader_.load_plugins();

// Por paquete/flujo (tras feature extraction):
PacketContext ctx{};
ctx.src_ip   = flow.src_ip;
ctx.dst_ip   = flow.dst_ip;
// ...
plugin_loader_.invoke_all(ctx);

// Shutdown:
plugin_loader_.shutdown();
```

**`sniffer/config/sniffer.json` — añadir sección plugins:**
```json
"plugins": {
  "directory": "/usr/lib/ml-defender/plugins",
  "budget_us": 100,
  "enabled": ["hello"],
  "hello": {}
}
```

### Tarea B — Test suite plugin-loader

Crear `plugin-loader/tests/` con al menos:
1. `test_plugin_loader_no_plugins.cpp` — JSON sin sección plugins → no abort
2. `test_plugin_loader_missing_so.cpp` — .so inexistente → warning, no abort
3. `test_plugin_loader_hello.cpp` — carga hello, init, process, shutdown OK
4. `test_plugin_loader_api_version_mismatch.cpp` — version≠1 → skip con warning
5. `test_plugin_loader_budget_overrun.cpp` — plugin lento → overrun counter++

Activar `BUILD_TESTS=ON` en `plugin-loader/CMakeLists.txt` cuando `tests/` exista.

### Tarea C (opcional DAY 94, si queda tiempo)

- Validar manualmente: borrar `libplugin_hello.so`, arrancar sniffer → no abort
- Validar manualmente: `budget_us=1` → overrun warnings en log

---

## Secuencia de diagnóstico al arrancar DAY 94

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git status
git log --oneline -5

# Verificar artefactos en VM
vagrant ssh defender -c "ls -lh /usr/local/lib/libplugin_loader.so* /usr/lib/ml-defender/plugins/"

# Verificar que plugin-loader está instalado
vagrant ssh defender -c "ldconfig -p | grep plugin_loader"
```

---

## Backlog activo — estado actualizado

| ID | Descripción | Estado |
|---|---|---|
| **ADR-012 PHASE 1** | plugin-loader + libplugin_hello.so | ✅ DONE — DAY 93 |
| **ADR-012 PHASE 1b** | Integración sniffer + test suite | **P1 — DAY 94** |
| **SYN-1** | `rst_ratio` extractor en sniffer | ✅ DONE — DAY 92 |
| **SYN-2** | `syn_ack_ratio` extractor en sniffer | ✅ DONE — DAY 92 |
| **DEBT-SMB-001** | `MIN_SYN_THRESHOLD` empírico para rst/syn_ack ratios | DAY 97+ tras SYN-3 |
| provision.sh | Script bash generación keypairs + seeds | DAY 95-96 |
| seed-client | Mini-componente libs/seed-client | DAY 95-96 |
| etcd refactor | Eliminar responsabilidades criptográficas | DAY 97+ |
| SYN-3 | Generador sintético Python/Scapy | DAY 97+ |
| SYN-4..7 | Validación + reentrenamiento + F1 update | tras SYN-3 |
| SYN-8..11 | Features P2 (port_445, flow_duration, port_diversity) | P2 |
| DEBT-FD-001 | Fast Detector Path A — leer sniffer.json | PHASE2 |
| ADR-007 | AND-consensus firewall — implementación | PHASE2 |

---

## arXiv — Estado

- Paper draft v4 listo: `docs/Ml defender paper draft v4.md`
- Email enviado a Sebastian Garcia — **esperando respuesta**
- Deadline: DAY 96 — si no responde, email a Yisroel Mirsky (Tier 2)

---

## Constantes del proyecto

```
Raíz:          /Users/aironman/CLionProjects/test-zeromq-docker
VM:            vagrant ssh defender
Logs:          /vagrant/logs/lab/
F1 log:        /vagrant/docs/experiments/f1_replay_log.csv
Paper:         /vagrant/docs/Ml defender paper draft v4.md
Proto:         /vagrant/protobuf/network_security.proto
Plugin dir:    /usr/lib/ml-defender/plugins/
macOS CRÍTICO: NUNCA usar sed -i sin -e '' — usar Python3 o editar en VM
```

---

*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic)*
*DAY 93 — 21 marzo 2026*
*Consejo de Sabios — ML Defender (aRGus NDR)*