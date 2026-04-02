# ML Defender — Prompt de Continuidad DAY 106
## 3 abril 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 25/25 suites ✅
**Rama activa:** `feature/plugin-crypto`
**Último commit:** DAY 105 — PHASE 2a MessageContext + TEST-INTEG-4a PASSED + Paper v10

---

## Lo realizado en DAY 105 (completo)

| Tarea | Estado |
|-------|--------|
| PHASE 2a — `MessageContext` en `plugin_api.h` (D2,D3,D7,D9,D11) | ✅ |
| PHASE 2a — `invoke_all(MessageContext&)` en `plugin_loader.hpp/.cpp` | ✅ |
| PHASE 2a — Graceful Degradation D1 + snapshot D8 (pointer) en `plugin_loader.cpp` | ✅ |
| PHASE 2a — `firewall-acl-agent/src/main.cpp` + TEST-INTEG-4a smoke test | ✅ |
| Makefile — `firewall` depende de `plugin-loader-build` | ✅ |
| Paper v10 — §5.6 Plugin Architecture (ADR-023, ADR-024) | ✅ |
| arXiv — cuenta `alonsoir` creada, código `AFKRBO` enviado a Andrés | ✅ |
| Consejo de Sabios DAY 105 — 4/5 revisores, ACCEPTED CON CONDICIONES | ✅ |

---

## Endorser arXiv — estado

| Endorser | Estado |
|----------|--------|
| Sebastian Garcia (CTU Prague) | ✅ respondió, recibió PDF |
| Yisroel Mirsky (BGU) | ⏳ sin respuesta |
| Andrés Caro Lindo (UEx/INCIBE) | 📧 código AFKRBO enviado |

**Código endorsement:** `AFKRBO`
**URL para Andrés:** `https://arxiv.org/auth/endorse?x=AFKRBO`
**Cuenta arXiv:** `alonsoir` / `alonsoir@gmail.com`
**ZIP LaTeX listo:** `argus_ndr_v10.zip` (main.tex + references.bib)

Si no hay respuesta → llamar al **657 33 10 10**.

---

## ⚡ PRIMER ACTO DAY 106

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/plugin-crypto
git pull origin feature/plugin-crypto
```

---

## TAREA 1 — Cierre PHASE 2a: decisiones Consejo DAY 105 (BLOQUEANTE para 4b)

El Consejo (4/4 unanimidad en la mayoría) ha identificado 4 items obligatorios
antes de avanzar a PHASE 2b (rag-ingester):

### 1a — D8-v2: CRC32 del payload en debug builds (mayoría 3/4)

En `plugin_loader.cpp`, en `invoke_all(MessageContext&)`, añadir antes de la
invocación del plugin:

```cpp
// D8-v2: CRC32 snapshot del payload (debug builds)
#ifdef MLD_ALLOW_DEV_MODE
uint32_t crc_before = crc32_fast(ctx.payload, ctx.payload_len);
#endif
```

Y tras la invocación, comparar:

```cpp
#ifdef MLD_ALLOW_DEV_MODE
uint32_t crc_after = crc32_fast(ctx.payload, ctx.payload_len);
if (crc_after != crc_before) {
    std::cerr << "[plugin-loader] SECURITY D8: plugin '" << p->name
              << "' modificó contenido del payload (CRC mismatch)\n";
    stats_[i].errors++;
}
#endif
```

Implementar `crc32_fast()` como función estática simple en `plugin_loader.cpp`.

### 1b — TEST-INTEG-4a-PLUGIN: plugin de test con símbolo exportado (4/4)

Crear `plugins/test-message/plugin_test_message.cpp` con 3 variantes
seleccionables por variable de entorno:

- **Variante A** (`MLD_TEST_VARIANT=A`): exporta símbolo, result_code=0,
  no modifica nada → debe pasar sin errores
- **Variante B** (`MLD_TEST_VARIANT=B`): intenta `const_cast` sobre `direction`
  → D8-v2 debe detectarlo (CRC no cambia para direction, pero pointer check sí)
- **Variante C** (`MLD_TEST_VARIANT=C`): devuelve result_code=-1
  → host registra error en stats, no std::terminate()

Gate: smoke test con Variante A pasa, Variante B produce D8 log, Variante C
produce error en stats sin crash.

### 1c — nonce/tag NULL documentado en plugin_api.h (4/4)

Añadir en `plugin_api.h` antes de los campos nonce/tag:

```c
/* nonce: 12-byte ChaCha20 nonce.
 * tag:   16-byte Poly1305 MAC tag.
 *
 * Production guarantee: nonce != NULL && tag != NULL.
 * Test/config mode (--test-config, MLD_DEV_MODE): MAY be NULL.
 * Plugins MUST check for NULL before dereferencing.
 */
```

### 1d — Makefile deps: plugin-loader-build en 4 componentes restantes (4/4)

```bash
python3 << 'PYEOF'
path = "Makefile"
with open(path, "r") as f:
    content = f.read()

replacements = [
    ("sniffer: proto etcd-client-build",
     "sniffer: proto etcd-client-build plugin-loader-build"),
    ("ml-detector: proto etcd-client-build",
     "ml-detector: proto etcd-client-build plugin-loader-build"),
    ("rag-ingester: proto etcd-client-build crypto-transport-build",
     "rag-ingester: proto etcd-client-build crypto-transport-build plugin-loader-build"),
    ("rag-build:",
     "rag-build: plugin-loader-build"),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"OK: {old[:40]}...")
    else:
        print(f"NOT FOUND: {old[:40]}...")

with open(path, "w") as f:
    f.write(content)
PYEOF
```

---

## TAREA 2 — Llamada Andrés / submit arXiv

Si Andrés ha respondido o endorsado → continuar submission en `https://arxiv.org/user`.
Si no ha respondido → llamar al **657 33 10 10**.

Guión:
> *"Andrés, te reenvié un email de arXiv. Entras en
> arxiv.org/auth/endorse?x=AFKRBO con tu cuenta arXiv y confirmas.
> Son 2 minutos."*

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
DeepSeek, Qwen (Alibaba — se autoidentifica como DeepSeek, patrón consolidado DAY 103–105),
Gemini (Google), Parallel.ai
ChatGPT5: primera ausencia registrada DAY 105.

---

*DAY 105 cierre — 2 abril 2026*
*Tests: 25/25 ✅ · Paper: Draft v10 · Rama: feature/plugin-crypto*
*PHASE 2a: COMPLETA (TEST-INTEG-4a PASSED)*
*Consejo: ACCEPTED CON CONDICIONES — 4 items DAY 106 antes de PHASE 2b*
*arXiv: cuenta alonsoir ✅ · AFKRBO enviado a Andrés ✅*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*