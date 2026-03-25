# ML Defender — Prompt de Continuidad DAY 98
## 26 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 22/22 suites ✅
- crypto-transport: 4/4 (test_crypto · test_compression · test_integration · test_crypto_transport)
- seed-client: 1/1
- etcd-server: 1/1
- rag-ingester: 7/7
- ml-detector: 9/9
  **Rama:** `feature/plugin-loader-adr012`
  **Último commit:** `feat(crypto): HKDF-SHA256 nativo + CryptoTransport + libsodium 1.0.19 (DAY 97)`

---

## Cadena de confianza (estado actual)

```
provision.sh → seed.bin (chmod 0600, /etc/ml-defender/{component}/)
    └► libseed_client    ✅ DAY 96 — instalado
        └► CryptoTransport (HKDF-SHA256 + ChaCha20-Poly1305 IETF + nonce 96-bit)  ✅ DAY 97
            └► etcd-client (transporte puro)   ← P1 DAY 98
                └► componentes + plugin-loader
```

---

## Decisiones cerradas DAY 97 (Consejo — no reabrir)

| Decisión | Resolución |
|---|---|
| HKDF implementation | **libsodium 1.0.19 nativo** — `crypto_kdf_hkdf_sha256_extract/expand` |
| libsodium versión | **1.0.19** compilada desde fuente — SHA-256 `018d79fe…` verificado |
| Bookworm dependency | `apt remove libsodium23` arrastra ZeroMQ — `provision.sh` reinstala `libzmq5 libzmq3-dev cppzmq-dev` |
| HKDF salt | 32 bytes cero (RFC 5869 default) |
| expand input | `prk[]` buffer directo — NO el state (API real de 1.0.19) |
| Nonce layout | `[0x00000000 \|\| uint64_LE_counter]` — upper 4 bytes fijos |
| CryptoManager | **DEPRECADO** — `CryptoTransport` lo sustituye en DAY 98 |
| etcd-server tests | `add_test()` añadido al CMakeLists.txt |
| libsodium path | `NO_DEFAULT_PATH` en CMake para priorizar `/usr/local` sobre sistema |

---

## Librerías instaladas en /usr/local

```bash
# Verificar estado post-DAY97
vagrant ssh -c "
ls -lh /usr/local/lib/libsodium* &&
ls -lh /usr/local/lib/libseed_client* &&
ls -lh /usr/local/lib/libcrypto_transport* &&
pkg-config --modversion libsodium
"
# Esperado:
#   libsodium.so.26 → 1.0.19
#   libseed_client.so.1.0.0
#   libcrypto_transport.so.1.0.0 (incluye transport.hpp instalado)
```

---

## Objetivos DAY 98

### P1 — Integrar CryptoTransport en los 6 componentes

Sustituir `CryptoManager` por `CryptoTransport` en cada componente.
Patrón de migración:

```cpp
// ANTES (CryptoManager — DEPRECADO)
CryptoManager crypto(seed_string);
auto encrypted = crypto.encrypt(plaintext);

// DESPUÉS (CryptoTransport — ADR-013 PHASE 2)
SeedClient sc("/etc/ml-defender/sniffer/sniffer.json");
sc.load();
CryptoTransport tx(sc, "ml-defender:sniffer:v1:tx");
CryptoTransport rx(sc, "ml-defender:sniffer:v1:rx");
auto encrypted = tx.encrypt(plaintext_bytes);
```

Componentes a migrar (en orden):
1. `etcd-server/src/crypto_manager.cpp` — ya tiene su propio CryptoManager
2. `sniffer` — mayor volumen de tráfico
3. `ml-detector`
4. `firewall-acl-agent`
5. `rag-ingester`
6. `rag-security`

### P1 — ADR-020: Eliminar flags enabled de los 6 JSONs

Eliminar `encryption.enabled` y `compression.enabled` de:
- `sniffer/config/sniffer.json`
- `ml-detector/config/ml_detector_config.json`
- `firewall-acl-agent/config/firewall.json`
- `rag-ingester/config/rag-ingester.json`
- `rag/config/rag-config.json`
- `etcd-server/config/etcd-server.json`

### P1 — etcd-client: integrar CryptoTransport

`etcd-client` actualmente usa `CryptoManager` para el transporte hacia etcd-server.
Migrar a `CryptoTransport` con contexto `"ml-defender:etcd-client:v1:tx"`.

### P2 — DEBT-CRYPTO-003a: mlock() en seed_client.cpp

```cpp
// Tras leer seed.bin, antes de limpiar buffer temporal:
mlock(seed_.data(), seed_.size());
```

### P2 — TEST-INTEG-1 + TEST-INTEG-2

TEST-INTEG-1: provision → seed → HKDF → cifrado → mensaje → descifrado → verificar.
TEST-INTEG-2: JSON → LZ4 → ChaCha20 → etcd → ChaCha20_dec → LZ4_dec → JSON (round-trip).

### P3 — DEBT-NAMING-001

`libs/seed-client/CMakeLists.txt` tiene `add_library(seed_client ...)` → genera
`libseed_client.so` (con underscore). El Consejo acordó `libseedclient` (sin underscore).
No bloquea nada — renombrar en DAY tranquilo.

---

## Diagnóstico de arranque DAY 98

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git status && git log --oneline -5

# Estado librerías
vagrant ssh -c "pkg-config --modversion libsodium && ls /usr/local/lib/libcrypto_transport* | head -3"

# Buscar usos de CryptoManager en los componentes
grep -r "CryptoManager" --include="*.cpp" --include="*.hpp" -l

# Ver cuántos JSONs aún tienen enabled flags
grep -r '"enabled"' --include="*.json" sniffer/ ml-detector/ firewall-acl-agent/ rag-ingester/ rag/ etcd-server/ 2>/dev/null | grep -v build

# arXiv: ¿respuesta Mirsky o Garcia?
```

---

## Backlog P1 activo DAY 98

| ID | Tarea |
|---|---|
| DEBT-CRYPTO-004 | Migrar CryptoManager → CryptoTransport en 6 componentes |
| ADR-020 | Eliminar flags enabled — JSONs de los 6 componentes |
| DEBT-ETCD-001 | etcd-client: integrar CryptoTransport |
| DEBT-CRYPTO-003a | mlock() en seed_client.cpp |
| TEST-INTEG-1/2 | Tests E2E pipeline + etcd JSON round-trip |
| ADR-012 PHASE 1b | plugin-loader integrado en sniffer |
| arXiv | Respuesta Mirsky / Tier 3 (Martin Grill) si silencio |

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
```

---

*DAY 97 cierre — 25 marzo 2026*
*Tests: 22/22 suites ✅ · libsodium 1.0.19 · CryptoTransport HKDF nativo*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*