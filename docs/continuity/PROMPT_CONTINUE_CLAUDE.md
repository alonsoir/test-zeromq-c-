# ML Defender — Prompt de Continuidad DAY 105
## 2 abril 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 25/25 suites ✅
**Rama activa:** `feature/plugin-crypto`
**Último commit:** DAY 104 cierre — Paper v9 + ADR-023 ACCEPTED + ADR-024 DISEÑO APROBADO

---

## Lo realizado en DAY 104 (completo)

| Tarea | Estado |
|-------|--------|
| Rama `feature/plugin-crypto` creada | ✅ |
| Paper v9 — P1/P2/P3/P4/P5/P6 (Gepeto) | ✅ |
| Paper v9 — corrección FP bare-metal (3 instancias) | ✅ |
| ADR-023 Multi-Layer Plugin Architecture — ACCEPTED | ✅ |
| ADR-024 Dynamic Group Key Agreement — DISEÑO APROBADO | ✅ |
| Consejo de Sabios — 2 rondas, 5/5 unanimidad | ✅ |
| BACKLOG.md — DAY 104 actualizado | ✅ |
| Commit + push a GitHub | ✅ |
| LinkedIn post (inglés) | ✅ |

**Qwen autoidentificado como DeepSeek en ambas rondas** — patrón registrado.

---

## ⚡ PRIMER ACTO DAY 105

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/plugin-crypto
git pull origin feature/plugin-crypto
```

---

## Endorser arXiv — estado

| Endorser | Estado |
|----------|--------|
| Sebastian Garcia (CTU Prague) | ✅ respondió, recibió PDF v5 |
| Yisroel Mirsky (BGU) | ⏳ enviado DAY 96, sin respuesta |
| Andrés Caro Lindo (UEx/INCIBE) | ✅ endorsement confirmado — **llamada HOY jueves 3 abril** |

**Llamada Andrés:** 657 33 10 10. Él llama o tú llamas.
**Punto clave:** indicar su email en el formulario arXiv → arXiv le envía
confirmación. El envío puede ocurrir en cualquier momento tras el endorsement.

---

## TAREA 0 — Llamada Andrés Caro Lindo (HOY)

Objetivo: confirmar proceso de endorsement arXiv cs.CR.
Preguntas clave:
1. ¿Puede hacer el endorsement desde su cuenta arXiv directamente?
2. ¿Prefiere revisar el paper v9 antes de endorsar?
3. ¿Hay algo del abstract o afiliación informal (UEx) que quiera ajustar?

---

## TAREA 1 — PHASE 2a: firewall-acl-agent + MessageContext

Implementar ADR-023 PHASE 2a según el contrato aprobado.

### Steps

1. Añadir `MessageContext` a `plugin-loader/include/plugin_api.h`
   (`PLUGIN_API_VERSION = 1`)
2. Implementar resolución de `plugin_process_message()` en `PluginLoader`
   (`dlsym`; aplicar Graceful Degradation Policy D1+D10)
3. Añadir post-invocation validation (snapshot + byte-wise comparison — D8)
4. Integrar en `firewall-acl-agent/src/main.cpp`
   (`#ifdef PLUGIN_LOADER_ENABLED`)
5. **Core `CryptoTransport` read-only** (no modificar paths de cifrado)
6. Actualizar `firewall-acl-agent/config/firewall.json` con sección `plugins`

### Gate: TEST-INTEG-4a

- `plugin_process_message()` invocado sobre al menos un `MessageContext` real
- Post-invocation invariants verificados
- `result_code == 0` confirmado
- `CryptoTransport` decryption path sin modificar (diff check)

---

## TAREA 2 — PAPER-FINAL: actualizar métricas DAY 104

Actualizar en Overleaf:
- Tests: 25/25 (ya estaba)
- ADR-023 ACCEPTED + ADR-024 DISEÑO APROBADO mencionados en §5
- Branch activa: `feature/plugin-crypto`
- Draft v9 → confirmar si se sube como v9 a arXiv o se espera a v10

---

## ADR-023 — Decisiones críticas a implementar en DAY 105

| ID | Decisión | Dónde implementar |
|----|----------|-------------------|
| D1 | fail-closed producción; DEV_MODE solo escape | `plugin_loader.cpp` |
| D2 | ownership/lifetime channel_id + payload contrato | `plugin_api.h` + docs |
| D3 | direction/nonce/tag read-only para plugin | `plugin_api.h` comentarios |
| D7 | trust model declarado | `plugin_api.h` header comment |
| D8 | post-invocation validation con snapshot | `plugin_loader.cpp` |
| D9 | TCB declaration | `plugin_api.h` header comment |
| D10 | MLD_DEV_MODE solo en Debug + MLD_ALLOW_DEV_MODE | `CMakeLists.txt` + `plugin_loader.cpp` |
| D11 | forward-compatibility ADR-024 | `plugin_api.h` comentario |

---

## ADR-024 — Open Questions a resolver antes de implementar (FASE 3)

| ID | Open Question |
|----|---------------|
| OQ-5 | Revocación de claves estáticas X25519 si nodo robado |
| OQ-6 | Rotación en reprovisionamiento — sesiones activas afectadas |
| OQ-7 | Replay primer mensaje — documentar en threat model |
| OQ-8 | Performance ARMv8 + comparación Noise_IKpsk3 vs Noise_KK |

**ADR-024 no bloquea DAY 105.** Implementación post-arXiv.

---

## Patrón ADR-012 PHASE 1b (establecido)

1. `CMakeLists.txt`: find_library + target_link(dl) + target_compile_definitions(PLUGIN_LOADER_ENABLED)
2. `src/main.cpp`: `unique_ptr<PluginLoader>` local (global solo si signal handler)
3. `config/*.json`: sección `plugins` con hello plugin `active:true`
4. Smoke test: `MLD_DEV_MODE=1 ./component 2>&1 | grep -i plugin`
   (solo válido en Debug + MLD_ALLOW_DEV_MODE builds)

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
dev:      MLD_DEV_MODE=1 → solo en builds Debug + MLD_ALLOW_DEV_MODE

macOS:   NUNCA sed -i sin -e '' → Python3 heredoc
zsh:     NUNCA Python inline con paréntesis → heredoc 'PYEOF'
zsh:     NUNCA assert(!x) → usar assert(x == false)
cmake:   NO_DEFAULT_PATH para libsodium — priorizar /usr/local
rag:     comillas simples en vagrant ssh -c para CMAKE_FLAGS
```

---

## Consejo de Sabios

Revisores: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI),
DeepSeek, Qwen (Alibaba — se autoidentifica como DeepSeek en ambas rondas,
patrón registrado DAY 103–104),
Gemini (Google), Parallel.ai

---

*DAY 104 cierre — 1 abril 2026*
*Tests: 25/25 ✅ · Paper: Draft v9 · Rama: feature/plugin-crypto*
*ADR-023: ACCEPTED · ADR-024: DISEÑO APROBADO*
*Consejo: 2 rondas, unanimidad 5/5*
*Endorser Andrés Caro Lindo: llamada HOY jueves 3 abril*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*