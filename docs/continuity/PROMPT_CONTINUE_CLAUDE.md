# ML Defender — Prompt de Continuidad DAY 104
## 1 abril 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 25/25 suites ✅
**Rama activa:** `main` (feature/bare-metal-arxiv mergeada ✅)
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
| Revisión Gepeto (ChatGPT5 como revisor cs.CR) — 6 mejoras identificadas | ✅ |
| Merge feature/bare-metal-arxiv → main | ✅ |

**NOTA BARE-METAL:** Bloqueado por hardware físico. No bloquea arXiv ni feature/plugin-crypto.

---

## ⚡ PRIMER ACTO DAY 104 — nueva rama

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout main
git pull origin main
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
**Punto clave:** Indicamos su email en el formulario arXiv → arXiv le manda
confirmación. El envío puede ocurrir en cualquier momento.

---

## TAREA 0 — Paper v9 (Overleaf) — revisión Gepeto

Gepeto (ChatGPT5 como revisor cs.CR externo) revisó la v8 y detectó 6 mejoras.
Ninguna requiere código nuevo. Todas son framing y redacción.

### 🔴 Críticas (hacer antes de arXiv)

**P1 — HKDF §5.5: añadir causa→efecto explícito**

```text
Incorrect context → different derived keys → MAC verification fails
Correct context   → identical derived keys → successful authentication
```
Rematar con:
```text
This class of error is undetectable without end-to-end protocol validation.
```

**P2 — TDH §5.4 + Abstract: declararlo explícitamente como metodología propuesta**

```text
Test-Driven Hardening (TDH) is proposed as a methodology for building
security-critical distributed systems.
```

**P3 — Consejo §5.1: aclarar que NO es ensemble, ES adversarial validation**

```text
The Council does not average outputs. It introduces structured disagreement
as a mechanism to expose hidden assumptions and validate architectural decisions.
```

**P4 — §10 Limitations: añadir mini subsección Threats to Validity**

Tres puntos mínimos:
1. Seed compromise — qué pasa si seed.bin se filtra
2. Single-host assumption — sin distributed key agreement aún (ADR-024 pendiente)
3. No hardware root of trust — no TPM / secure enclave

### 🟡 Recomendadas

**P5 — §7 Throughput: claim técnico explícito**
```text
The observed throughput is limited by the virtualized NIC, not the pipeline.
```

**P6 — §5.5 ADR-023: frase conectando con bug HKDF**
```text
This design avoids semantic overloading of data structures across layers,
a common source of subtle security and correctness bugs.
```

---

## TAREA 1 — Redactar ADR-023 formal

Fichero: `docs/adr/ADR-023-multi-layer-plugin-architecture.md`

```c
typedef struct {
    uint8_t     version;       // MESSAGE_CONTEXT_VERSION = 1
    uint8_t     direction;     // MLD_TX = 0, MLD_RX = 1
    uint8_t     nonce[12];     // 96-bit monotonic counter
    uint8_t     tag[16];       // Poly1305 tag (16 bytes)
    uint8_t*    payload;       // buffer (in/out)
    size_t      length;        // longitud actual
    size_t      max_length;    // capacidad — siempre >= length + 16
    const char* channel_id;    // "sniffer-to-ml-detector" — selector HKDF
    int32_t     result_code;   // 0=OK, -1=MAC failure, -2=buffer overflow
    uint8_t     reserved[8];   // para sequence_number / timestamp futuro
} MessageContext;
```

- Estrategia PHASE 2a/2b/2c + gates TEST-INTEG-4a/4b/4c
- plugin_process_message() opcional vía dlsym, PLUGIN_API_VERSION=1
- Core CryptoTransport read-only durante PHASE 2a (DeepSeek)
- Minoría Gemini registrada (bump inmediato desestimado)

## TAREA 2 — Redactar ADR-024 borrador

Fichero: `docs/adr/ADR-024-dynamic-group-key-agreement.md`

```
PSK = HKDF(seed_family, "noise-ik-psk")  ← seed de familia (ADR-021)
Handshake (1-RTT, solo en arranque):
  Initiator: -> e, es
  Responder:  <- e, ee, se
Implementación: Noise-c + libsodium 1.0.19
Estado: DISEÑO — implementación post-arXiv
```

## TAREA 3 — Llamada Andrés Caro Lindo

---

## Patrón ADR-012 PHASE 1b (establecido)

1. `CMakeLists.txt`: find_library + target_link(dl) + target_compile_definitions(PLUGIN_LOADER_ENABLED)
2. `src/main.cpp`: `unique_ptr<PluginLoader>` local (global solo si signal handler)
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
Plugins:  /usr/lib/ml-defender/plugins/
Libs:     /usr/local/lib/  ← libplugin_loader.so
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
*Tests: 25/25 ✅ · Paper: Draft v8 · Rama: main*
*Consejo: MessageContext aprobada · Noise IK aprobado*
*Gepeto: 6 mejoras paper identificadas — P1/P2/P3/P4 críticas*
*Endorser Andrés Caro Lindo: llamada HOY jueves 2 abril*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*