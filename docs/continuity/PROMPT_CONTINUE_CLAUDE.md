# ML Defender — Prompt de Continuidad DAY 101
## 29 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 24/24 suites ✅
**Rama activa:** `feature/bare-metal-arxiv`
**Último merge:** PR #33 → main (DAY 100 completo)

---

## Lo realizado en DAY 100 (completo)

| Tarea | Estado |
|-------|--------|
| ADR-021 deployment.yml SSOT + seed families | ✅ |
| ADR-022 threat model + Opción 2 descartada | ✅ |
| set_terminate() en 6 main() | ✅ |
| CI reescrito honesto (ubuntu-latest) | ✅ |
| BACKLOG.md + ARCHITECTURE.md v7.0.0 | ✅ |
| README badges actualizados | ✅ |
| PR #33 mergeado → main | ✅ |
| ADR-012 PHASE 1b: plugin-loader en sniffer | ✅ |
| Consejo de Sabios DAY 100 (5/7 respuestas) | ✅ |
| Endorser identificado: Andrés Caro Lindo (UEx) | ✅ |

---

## ADR-012 PHASE 1b — estado exacto

Plugin-loader integrado en sniffer con guard `#ifdef PLUGIN_LOADER_ENABLED`:
- `sniffer/CMakeLists.txt`: find_library + link + define
- `sniffer/src/userspace/main.cpp`: PluginLoader instanciado + load/shutdown
- `sniffer/config/sniffer.json`: sección `plugins` con hello plugin

```json
"plugins": {
  "enabled": [
    {
      "name": "hello",
      "path": "/usr/local/lib/libplugin_hello.so",
      "active": false,
      "comment": "ADR-012 PHASE 1b — validation plugin. Set active:true to enable."
    }
  ]
}
```

**hello plugin activado DAY 101 — pendiente validar en runtime.**

**TODO pendiente en main.cpp:**
```cpp
// TODO: make plugin-loader always-link in PHASE 2 (ADR-012 PHASE 2)
```

---

## DAY 101 — primera tarea: activar hello plugin

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git status && git log --oneline -3

# Activar hello plugin
python3 << 'PYEOF'
import json
path = "sniffer/config/sniffer.json"
with open(path) as f:
    c = json.load(f)
c["plugins"]["enabled"][0]["active"] = True
with open(path, "w") as f:
    json.dump(c, f, indent=2)
print("hello plugin activado")
PYEOF

# Build en VM
vagrant ssh -c 'cd /vagrant && make sniffer-build 2>&1 | grep -i plugin'

# Tests completos
vagrant ssh -c 'cd /vagrant && make test 2>&1 | tail -10'

# Smoke test: arrancar sniffer en modo dev
vagrant ssh -c 'MLD_DEV_MODE=1 /vagrant/sniffer/build/sniffer -c /vagrant/sniffer/config/sniffer.json 2>&1 | grep -i plugin | head -5'
```

---

## DAY 101 — segunda tarea: email endorser

**Redactar y enviar email a Andrés Caro Lindo.**
(Ver sección Endorser abajo para contexto completo.)

---

## Consejo de Sabios DAY 100 — síntesis de decisiones

| Pregunta | Decisión consolidada | Origen |
|----------|---------------------|--------|
| ADR-022 en paper | Subsección dedicada, no nota al pie | Unanimidad 5/5 |
| #ifdef vs always-link | Correcto PHASE 1b. TODO: always-link PHASE 2 | Unanimidad 5/5 |
| Orden plugin-loader | sniffer ✅ → ml-detector → **firewall** → rag-ingester | ChatGPT+DeepSeek+Gemini |
| Endorser arXiv | Andrés Caro Lindo (UEx) — primera opción | Grok + confirmado |

---

## Endorser arXiv — Andrés Caro Lindo

**Email:** `andresc@unex.es`
**Cargo:** Profesor Titular, Dpto. Ingeniería de Sistemas Informáticos y Telemáticos
**Rol actual:** Investigador Principal, Cátedra INCIBE-UEx-EPCC (Ciberseguridad)
**Áreas:** Cybersecurity, Machine Learning, Pattern Recognition
**Google Scholar:** https://scholar.google.com/citations?user=Eq0Cvb0AAAAJ
**Grupo:** GIM (Grupo de Ingeniería de Medios) — http://gim.unex.es

**Contexto personal:** Fue profesor de Laboratorio de Programación 2 de Alonso.
**Framing del email:** Ex-alumno extremeño, proyecto open-source para hospitales
y escuelas de la región, 100 días de trabajo, F1=0.9985, 24/24 tests, paper
LaTeX listo. No pedir validación científica — pedir endorsement arXiv cs.CR.

---

## Backlog P1 activo DAY 101

| ID | Tarea | Prioridad |
|----|-------|-----------|
| PLUGIN-HELLO | Activar + validar hello plugin en sniffer real | P1 hoy |
| ARXIV-ENDORSER | Email a andresc@unex.es | P1 hoy |
| PAPER-ADR022 | Subsección "HKDF Context Symmetry" en paper | P1 |
| TODO-PLUGIN | Añadir TODO always-link en main.cpp | 5 min |
| PLUGIN-LOADER-ML | Integrar plugin-loader en ml-detector | P1 siguiente |
| PLUGIN-LOADER-FW | Integrar plugin-loader en firewall-acl-agent | P2 |
| PLUGIN-LOADER-RAG | Integrar plugin-loader en rag-ingester | P3 |
| BARE-METAL | Stress test sin VirtualBox (cuando haya hardware) | P1 milestone |
| DEBT-CRYPTO-003a | mlock() seed_client.cpp | P2 |

---

## Próximos componentes para plugin-loader

Orden revisado por Consejo DAY 100:
1. `sniffer` ✅ DAY 100
2. `ml-detector` — plugins de inferencia + threat intel
3. `firewall-acl-agent` — plugins de respuesta (notificaciones, logging)
4. `rag-ingester` — plugins de parseo de fuentes externas

Para cada uno: mismo patrón que sniffer.
CMakeLists + main (#ifdef guard) + JSON config.

---

## Constantes

```
Raíz:    /Users/aironman/CLionProjects/test-zeromq-docker
VM:      vagrant ssh -c '...'   ← SIEMPRE -c
Logs:    /vagrant/logs/lab/
Keys:    /etc/ml-defender/{component}/seed.bin
Libs:    /usr/local/lib/ — prioridad sobre /lib/x86_64-linux-gnu/

macOS:   NUNCA sed -i sin -e '' → Python3 heredoc
zsh:     NUNCA Python inline con paréntesis → heredoc 'PYEOF'
cmake:   NO_DEFAULT_PATH para libsodium — priorizar /usr/local
dev:     MLD_DEV_MODE=1 → permite arranque sin seed.bin
```

---

## Consejo de Sabios — práctica establecida

Revisores: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI),
DeepSeek, Qwen (Alibaba), Gemini (Google), Parallel.ai

---

*DAY 100 cierre — 28 marzo 2026*
*Tests: 24/24 ✅ · PR #33 → main ✅ · ADR-012 PHASE 1b ✅*
*Consejo 5/5 ✅ · Endorser: andresc@unex.es ✅*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*