Aquí van los dos documentos:

---

# ML Defender — Prompt de Continuidad DAY 99
## 27 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 22/22 suites ✅
- crypto-transport: 4/4
- seed-client: 1/1
- etcd-server: 1/1
- rag-ingester: 7/7
- ml-detector: 9/9
- etcd-client: 4/4
- sniffer: 1/1
- firewall: 26/27 (1 flaky `TimestampUniqueness` — pre-existente, VM VirtualBox)

**Rama:** `feature/plugin-loader-adr012`
**Último commit pendiente:** `feat(crypto): migrar 6 componentes CryptoManager → CryptoTransport (ADR-013 PHASE 2, DAY 98)` ← **NO comiteado aún — hacer antes de nada**

---

## Cadena de confianza (estado actual)

```
provision.sh → seed.bin (chmod 0600, /etc/ml-defender/{component}/)
    └► libseed_client    ✅ DAY 96
        └► CryptoTransport (HKDF-SHA256 + ChaCha20-Poly1305 IETF + nonce 96-bit)  ✅ DAY 97
            └► 6/6 componentes migrados  ✅ DAY 98
                ├── etcd-server (ComponentRegistry)
                ├── etcd-client (process_outgoing_data)
                ├── sniffer (RingBufferConsumer)
                ├── ml-detector (ZMQHandler + RAGLogger)
                ├── rag-ingester (EventLoader)
                └── firewall-acl-agent (ZMQSubscriber)
            └► Contextos HKDF: ⚠️ ASIMÉTRICOS — P1 DAY 99
            └► tools/ — PENDIENTE DAY 99
```

---

## ⚠️ ALERTA CRÍTICA — Contextos HKDF asimétricos

**El Consejo de Sabios lo confirmó unánimemente: los contextos actuales producen claves distintas en emisor y receptor. El sistema compila y pasa tests unitarios, pero fallará en E2E real.**

### Estado actual (INCORRECTO)
```cpp
// sniffer cifra con:
CryptoTransport tx(*sc, "ml-defender:sniffer:v1:tx");

// ml-detector descifra con:
CryptoTransport rx(*sc, "ml-defender:ml-detector:v1:rx");
// ↑ CLAVES DISTINTAS → descifrado falla con MAC error
```

### Arquitectura correcta (DAY 99)
```
El contexto HKDF pertenece al CANAL, no al componente.
Mismo contexto en emisor y receptor → misma clave → descifrado correcto.
```

### Tabla de contextos correctos (a implementar DAY 99)

```cpp
// crypto_transport/contexts.hpp — a crear

// Canal 1: sniffer → ml-detector
constexpr const char* CTX_SNIFFER_TO_ML = "ml-defender:sniffer-to-ml-detector:v1";

// Canal 2: ml-detector → firewall-acl-agent
constexpr const char* CTX_ML_TO_FIREWALL = "ml-defender:ml-detector-to-firewall:v1";

// Canal 3: etcd-server ↔ etcd-client
constexpr const char* CTX_ETCD_TX = "ml-defender:etcd:v1:tx";
constexpr const char* CTX_ETCD_RX = "ml-defender:etcd:v1:rx";

// Canal 4: artefactos RAGLogger ↔ rag-ingester
constexpr const char* CTX_RAG_ARTIFACTS = "ml-defender:rag-artifacts:v1";
```

---

## Decisiones cerradas DAY 98 (Consejo — no reabrir)

| Decisión | Resolución |
|---|---|
| Migración CryptoManager | **CryptoTransport en 6/6 componentes** — DAY 98 completo |
| CryptoManager legacy | **DEPRECATED** con banner — borrado sesión futura |
| Flags `enabled` JSON | **Mantenidos** — ADR-020 borrado sesión tranquila |
| Compresión | **LZ4 custom `[uint32_t orig_size LE]`** — estándar en 6 componentes |

## Decisiones cerradas DAY 98 por el Consejo

| Pregunta | Veredicto unánime |
|---|---|
| P1 Contextos HKDF | **Opción A: contexto por canal, idéntico en emisor y receptor** |
| P2 LZ4 formato | **Mantener custom `[uint32_t]`** — sistema cerrado, sin interoperabilidad necesaria |
| P3 Modo degradado | **FATAL en producción** — solo modo dev con flag explícito (`VAGRANT=1` o `--dev`) |
| P4 `tools/` | **Baja prioridad, pero antes de arXiv** |
| P5 TEST-INTEG-1/2 | **Gate obligatorio antes de arXiv submission** |

---

## Objetivos DAY 99 (orden estricto)

### PASO 0 — Commit DAY 98 (primero)
```bash
git add -A
git commit -m "feat(crypto): migrar 6 componentes CryptoManager → CryptoTransport (ADR-013 PHASE 2, DAY 98)

- etcd-server/etcd-client/sniffer/ml-detector/rag-ingester/firewall migrados
- CryptoManager DEPRECATED en todos los componentes
- LZ4 custom [uint32_t orig_size LE] estandarizado
- Tests: 22/22 suites green

DEBT-CRYPTO-004: resuelto"
```

### P1 — `contexts.hpp` + corregir contextos en 6 componentes

1. Crear `crypto-transport/include/crypto_transport/contexts.hpp` con constantes por canal
2. Reemplazar todos los contextos hardcodeados en los 6 componentes por las constantes
3. Verificar compilación: `make all-build`

### P1 — TEST-INTEG-1: round-trip sniffer → ml-detector

```cpp
// Estructura del test:
// 1. SeedClient desde seed.bin de prueba
// 2. CryptoTransport tx(sc, CTX_SNIFFER_TO_ML) — simula sniffer
// 3. Cifrar payload conocido
// 4. CryptoTransport rx(sc, CTX_SNIFFER_TO_ML) — simula ml-detector
// 5. Descifrar y verificar == original
```

### P1 — TEST-INTEG-2: round-trip JSON → LZ4 → ChaCha20 → descifrado

```cpp
// JSON → LZ4 compress (cabecera [uint32_t]) → CryptoTransport.encrypt
//   → wire → CryptoTransport.decrypt → LZ4 decompress → JSON original
// Verificar byte-a-byte
```

### P1 — Fail-closed en EventLoader y RAGLogger

```cpp
// ANTES (modo degradado silencioso):
} catch (const std::exception& e) {
    std::cerr << "[WARN] CryptoTransport no disponible — modo plaintext" << std::endl;
}

// DESPUÉS (fail-closed):
} catch (const std::exception& e) {
    throw std::runtime_error(
        "[FATAL] seed.bin no disponible — componente no puede arrancar sin cifrado. "
        "Ejecutar: sudo provision.sh. Error: " + std::string(e.what()));
}
// Excepción: if (getenv("MLD_DEV_MODE")) → modo degradado permitido
```

### P2 — `tools/` migración (si queda tiempo / antes de arXiv)

Ficheros:
- `tools/synthetic_sniffer_injector.cpp`
- `tools/synthetic_ml_output_injector.cpp`
- `tools/generate_synthetic_events.cpp`

Config path: `/etc/ml-defender/tools/tools.json`
Contexto: `CTX_SNIFFER_TO_ML` (simétrico al sniffer real)

### P3 — DEBT-CRYPTO-003a: mlock() en seed_client.cpp

```cpp
mlock(seed_.data(), seed_.size());
```

---

## Diagnóstico de arranque DAY 99

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker

# 0. Commit DAY 98 pendiente
git status && git log --oneline -3

# 1. Verificar contextos actuales (todos incorrectos)
grep -rn "ml-defender:sniffer\|ml-defender:ml-detector\|ml-defender:etcd\|rag-artifacts" \
  sniffer/ ml-detector/ rag-ingester/ firewall-acl-agent/ etcd-server/ etcd-client/ \
  --include="*.cpp" --include="*.hpp" | grep -v build | grep -v DEPRECATED

# 2. Estado tests
make test 2>&1 | tail -5

# 3. ¿Existe ya contexts.hpp?
ls crypto-transport/include/crypto_transport/contexts.hpp 2>/dev/null || echo "pendiente crear"
```

---

## Backlog P1 activo DAY 99

| ID | Tarea | Prioridad |
|---|---|---|
| ADR-013-CONTEXTS | `contexts.hpp` + reemplazar en 6 componentes | P1 — primero |
| TEST-INTEG-1 | sniffer → ml-detector round-trip E2E | P1 — gate arXiv |
| TEST-INTEG-2 | JSON → LZ4 → cifrado → descifrado round-trip | P1 — gate arXiv |
| FAIL-CLOSED | EventLoader + RAGLogger sin modo degradado silencioso | P1 |
| DEBT-CRYPTO-004b | tools/ migración a CryptoTransport | P2 — antes arXiv |
| DEBT-CRYPTO-003a | mlock() en seed_client.cpp | P2 |
| ADR-012 PHASE 1b | plugin-loader en sniffer | P3 |

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

Modo dev: MLD_DEV_MODE=1 → permite arranque sin seed.bin (EventLoader/RAGLogger)
```

---

*DAY 98 cierre — 26 marzo 2026*
*Tests: 22/22 suites ✅ · 6/6 componentes migrados · Contextos HKDF: P1 DAY 99*
*Consejo de Sabios: ChatGPT, DeepSeek, Gemini, Grok, Qwen — veredicto unánime*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*

---
---

