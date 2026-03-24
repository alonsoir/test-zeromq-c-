# ML Defender — Prompt de Continuidad DAY 97
## 25 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 39/39 ✅ (crypto 3/3 · etcd-hmac 12/12 · ml-detector 9/9 · rag-ingester 7/7 · sniffer 1/1 · seed-client 6/6)
**Rama:** `feature/plugin-loader-adr012`
**Último commit:** `feat(crypto): add seed-client library — cryptographic base material reader (ADR-013 PHASE 1)`

---

## Cadena de confianza (estado actual)

```
provision.sh → seed.bin (chmod 0600, /etc/ml-defender/{component}/)
    └► libseedclient    ✅ DAY 96 — instalado, 6/6 tests
        └► crypto-transport (HKDF + ChaCha20 + LZ4)   ← P1 DAY 97
            └► etcd-client (transporte puro)           ← DAY 98
                └► componentes + plugin-loader
```

---

## Decisiones cerradas DAY 96 (Consejo — no reabrir)

| Decisión | Resolución |
|---|---|
| HKDF implementation | **libsodium** exclusivamente |
| HKDF context format | `"ml-defender:{component}:{version}:{tx\|rx}"` |
| Nonce policy | Contador monotónico **96-bit** por sesión |
| C++ standard | **C++20 permanente** — migración C++23 solo si kernel/eBPF lo exige |
| Error handling | `throw` en todo el pipeline — `std::expected` diferido |
| Cifrado | **SIEMPRE obligatorio** — eliminar flag `enabled` de JSONs |
| Compresión | **SIEMPRE** cuando posible — eliminar flag `enabled` de JSONs |
| Orden operaciones | **LZ4 → ChaCha20** (comprimir antes de cifrar) |
| Rotación seeds | Requiere **reinicio ordenado** de toda la pipeline |
| Library naming | `libseedclient` sin underscore (convención Linux) |

---

## Objetivos DAY 97

### P1 — DEBT-CRYPTO-002: HKDF en crypto-transport

**Diagnóstico inicial:**
```bash
vagrant ssh -c "pkg-config --modversion libsodium 2>/dev/null || echo 'apt-get install -y libsodium-dev'"
cat crypto-transport/include/crypto_transport/crypto.hpp | head -60
```

**Interfaz objetivo:**
```cpp
// Movible, no copiable (RAII de session keys)
CryptoTransport(const SeedClient& seed_client,
                const std::string& context = "ml-defender:transport:v1");
// HKDF-SHA256 via libsodium → session_key (nunca sale de CryptoTransport)
```

Ficheros: `crypto.hpp` · `crypto.cpp` · `CMakeLists.txt` (añadir libsodium) · tests backward compat.

### P1 — DEBT-CRYPTO-001: Nonce management

Contador monotónico 96-bit. Implementar en `CryptoTransport`, nunca en `SeedClient`.

### P1 — ADR-020: Eliminar flags enabled de JSONs

Eliminar `encryption.enabled` y `compression.enabled` de los 6 JSONs de componentes.
Cifrado y compresión son siempre obligatorios. Crear `docs/adr/ADR-020-crypto-mandatory.md`.

### P2 — DEBT-CRYPTO-003a: mlock() en seed_client.cpp

```cpp
// Tras leer seed.bin, antes de limpiar buffer temporal:
mlock(seed_.data(), seed_.size());
```

### P2 — DEBT-CRYPTO-003b: Entropy check en provision.sh

```bash
avail=$(cat /proc/sys/kernel/random/entropy_avail)
[ "$avail" -lt 256 ] && apt-get install -y haveged && systemctl start haveged
```

### P1 si tiempo — TEST-INTEG-1 + TEST-INTEG-2

TEST-INTEG-1: provision → seed → HKDF → cifrado → mensaje → descifrado → verificar.
TEST-INTEG-2: JSON → LZ4 → ChaCha20 → etcd → ChaCha20_dec → LZ4_dec → JSON (round-trip).

---

## Diagnóstico de arranque DAY 97

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git status && git log --oneline -5

# libsodium disponible?
vagrant ssh -c "pkg-config --modversion libsodium 2>/dev/null || echo 'INSTALAR'"

# seed-client sigue instalado?
vagrant ssh -c "ls -lh /usr/local/lib/libseedclient.so* 2>/dev/null || echo 'FALTA'"

# Ver crypto-transport actual
cat crypto-transport/include/crypto_transport/crypto.hpp | head -60

# arXiv: ¿respuesta Mirsky o Garcia?
```

---

## Backlog P1 activo DAY 97

| ID | Tarea |
|---|---|
| DEBT-CRYPTO-002 | HKDF en crypto-transport (libsodium) |
| DEBT-CRYPTO-001 | Nonce 96-bit monotónico |
| ADR-020 | Eliminar flags enabled — cifrado+compresión siempre |
| DEBT-CRYPTO-003a | mlock() en seed_client.cpp |
| DEBT-CRYPTO-003b | Entropy check provision.sh |
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

macOS:   NUNCA sed -i sin -e '' → Python3 heredoc
zsh:     NUNCA Python inline con paréntesis → heredoc 'PYEOF'
```

---

*DAY 96 post-Consejo — 24 marzo 2026*
*Tests: 39/39 ✅ · libseedclient instalado · Email Mirsky enviado*
*Consejo DAY 96: Grok · DeepSeek · Gemini · ChatGPT5 (unanimidad 4/4)*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*