# ML Defender — Prompt de Continuidad DAY 96
## 24 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING (etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall)
**Test suite:** 33/31 ✅ (crypto 3/3, etcd-hmac 12/12, ml-detector 9/9, rag-ingester 7/7, sniffer 1/1)
**Rama activa:** `feature/plugin-loader-adr012`
**Último tag:** DAY92

---

## Lo que se hizo en DAY 95

### tools/provision.sh — COMPLETADO ✅

6/6 componentes con keypairs Ed25519 + seeds ChaCha20 (32B).
Paths AppArmor-compatible: `/etc/ml-defender/{component}/`.
Makefile: `provision`, `provision-status`, `provision-check`, `provision-reprovision`.
Vagrantfile: bloque `cryptographic-provisioning` con `run: "once"`.
`pipeline-start` depende de `provision-check` — fail-closed security.

### JSONs (6/6) — bloque `identity` añadido ✅

Todos los componentes declaran `identity.keys_dir`, `public_key`, `private_key`, `seed_bin`.
`rag-config.json`: AES-256-CBC eliminado (letra muerta), unificado a ChaCha20-Poly1305.

### Acta Consejo DAY 95 (6/7 — Parallel.ai pendiente) ✅

Unanimidad en validación. Puntos críticos incorporados al backlog:
- `DEBT-CRYPTO-001`: nonce management con seeds persistentes
- `DEBT-CRYPTO-002`: HKDF para derivar session keys
- `DEBT-CRYPTO-003`: check entropy en provision.sh

---

## ⚠️ INSIGHT CRÍTICO DEL CONSEJO — leer antes de escribir código

**Afecta directamente al diseño de seed-client.**

ChatGPT + Grok (unanimidad): el seed de 32B es **material base**, no clave directa.

```
INCORRECTO:  seed.bin → chacha20_encrypt(data, key=seed)
CORRECTO:    seed.bin → HKDF(seed, context) → session_key → chacha20_encrypt(data, session_key)
```

Sin HKDF: no hay forward secrecy, compromiso de seed = descifrado histórico completo.

**Arquitectura de responsabilidades resultante:**
```
SeedClient        → lee seed.bin del disco, lo expone como array<uint8_t,32>
                    NO cifra, NO descifra, NO hace HKDF
                    Entrega: "esto es material criptográfico, úsalo con cuidado"
CryptoTransport   → recibe seed de SeedClient
                    aplica HKDF(seed, context="ml-defender:{component}:v1")
                    gestiona nonces
                    cifra/descifra con la session_key derivada
```

`SeedClient` sigue siendo igual de simple. El contrato debe quedar documentado
en el `.hpp`: "seed() devuelve material base para HKDF, no una clave de uso directo."

---

## Objetivo principal DAY 96 — libs/seed-client

### Tarea A — libs/seed-client (P1)

```cpp
// libs/seed-client/include/seed_client/seed_client.hpp

/**
 * SeedClient — Lector de material criptográfico base.
 *
 * Lee el seed.bin generado por tools/provision.sh y lo expone
 * para que CryptoTransport aplique HKDF antes de cualquier uso.
 *
 * CONTRATO:
 *   - seed() devuelve material base (32B). NO es una clave simétrica.
 *   - El llamador (CryptoTransport) es responsable de HKDF + nonce mgmt.
 *   - SeedClient no hace red, no genera seeds, no cifra, no distribuye.
 *
 * ADR refs: ADR-013 (PHASE 1), DEBT-CRYPTO-001, DEBT-CRYPTO-002
 */
class SeedClient {
public:
    explicit SeedClient(const std::string& component_json_path);
    void load();
    const std::array<uint8_t, 32>& seed() const;
    bool is_loaded() const;
    const std::string& component_id() const;
    const std::string& keys_dir() const;
private:
    std::string component_json_path_;
    std::string keys_dir_;
    std::string component_id_;
    std::array<uint8_t, 32> seed_{};
    bool loaded_ = false;
};
```

**Lógica interna de `load()`:**
1. Parsear JSON del componente → leer `identity.component_id` y `identity.keys_dir`
2. Construir path: `{keys_dir}/seed.bin`
3. Abrir fichero en modo binario
4. Leer exactamente 32 bytes — `std::runtime_error` si != 32
5. Copiar a `seed_` — limpiar buffer temporal con `explicit_bzero` o `memset`+barrera
6. Verificar que el fichero existe y tiene permisos 600 (advertencia si no)
7. `loaded_ = true`

**Estructura de ficheros:**
```
libs/seed-client/
  CMakeLists.txt          ← patrón idéntico a crypto-transport
  include/seed_client/
    seed_client.hpp
  src/
    seed_client.cpp
  tests/
    test_seed_client.cpp  ← CTest: load OK, file_not_found, wrong_size, component_id
```

**Dependencias:** nlohmann_json únicamente. Más primitivo que crypto-transport.

**Makefile (añadir):**
```makefile
seed-client-build:
    @vagrant ssh -c 'cd /vagrant/seed-client && rm -rf build && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4'
    @vagrant ssh -c 'cd /vagrant/seed-client/build && sudo make install && sudo ldconfig'

seed-client-clean:
    @vagrant ssh -c 'rm -rf /vagrant/seed-client/build'
    @vagrant ssh -c 'sudo rm -f /usr/local/lib/libseed_client.so*'

seed-client-test:
    @vagrant ssh -c 'cd /vagrant/seed-client/build && ctest --output-on-failure'
```

### Tarea B — DEBT-CRYPTO-003 (P2 — si queda tiempo)

Añadir check de entropy en `tools/provision.sh` antes de `openssl genpkey`:

```bash
# En provision.sh, función check_dependencies() o nueva check_entropy():
avail=$(cat /proc/sys/kernel/random/entropy_avail 2>/dev/null || echo "999")
if [ "$avail" -lt 256 ]; then
    log_warn "Entropy baja: ${avail} bits — instalando haveged"
    apt-get install -y haveged >/dev/null 2>&1
    systemctl start haveged 2>/dev/null || true
    sleep 1
fi
```

### Tarea C — ADR-012 PHASE 1b (P1 — si queda tiempo)

- Integrar plugin-loader en sniffer
- Test suite plugin-loader con CTest

---

## Secuencia de diagnóstico al arrancar DAY 96

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git status
git log --oneline -5

# Verificar estado de claves
make provision-status

# Ver si ya existe libs/seed-client
ls libs/ 2>/dev/null || echo "libs/ no existe — crear"

# Referencia de estructura
ls crypto-transport/include/crypto_transport/
ls crypto-transport/CMakeLists.txt | head -30

# arXiv — verificar si llegó respuesta de Sebastian Garcia
# Si no: preparar email Yisroel Mirsky (Tier 2)
```

---

## Backlog activo — estado actualizado DAY 95

| ID | Descripción | Estado |
|---|---|---|
| **provision.sh** | Script bash keypairs + seeds | ✅ DONE — DAY 95 |
| **identity blocks** | Bloque identity en 6 JSONs | ✅ DONE — DAY 95 |
| **rag-config.json** | AES→ChaCha20, flujo documentado | ✅ DONE — DAY 95 |
| **Vagrantfile** | cryptographic-provisioning block | ✅ DONE — DAY 95 |
| **seed-client** | libs/seed-client — material base para HKDF | **P1 — DAY 96** |
| **DEBT-CRYPTO-001** | Nonce management con seeds persistentes | **P1 — DAY 96 (diseño)** |
| **DEBT-CRYPTO-002** | HKDF en crypto-transport | **P1 — DAY 96 (diseño)** |
| **DEBT-CRYPTO-003** | Check entropy en provision.sh | P2 — DAY 96 |
| **ADR-012 PHASE 1b** | Integración sniffer + test suite | P1 — DAY 96 |
| **FEAT-CRYPTO-1** | Rotación de claves sin downtime | DAY 97+ |
| **FEAT-CRYPTO-2** | Handshake efímero (Noise simplificado) | PHASE 2 |
| **FEAT-CRYPTO-3** | TPM 2.0 / HSM enterprise | ENT-8 |
| **SYN-3..7** | Sintético + reentrenamiento + F1 | DAY 97+ |
| **DEBT-FD-001** | Fast Detector Path A → JSON | PHASE 2 |
| **ADR-007** | AND-consensus firewall | PHASE 2 |

---

## arXiv — Estado

- Paper draft v5: `docs/Ml defender paper draft v5.md`
- Email enviado a Sebastian Garcia — **esperando respuesta**
- **Deadline DAY 96:** si no responde → email Yisroel Mirsky (Tier 2)

---

## Constantes del proyecto

```
Raíz:          /Users/aironman/CLionProjects/test-zeromq-docker
VM:            vagrant ssh defender
Logs:          /vagrant/logs/lab/
Plugin dir:    /usr/lib/ml-defender/plugins/
Keys dir:      /etc/ml-defender/  ← CREADO EN DAY 95
macOS CRÍTICO: NUNCA usar sed -i sin -e '' — usar Python3 o editar en VM
zsh CRÍTICO:   NUNCA pegar Python inline con paréntesis — usar heredoc 'PYEOF'
```

---

*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*
*DAY 95 — 23 marzo 2026*
*Consejo de Sabios — ML Defender (aRGus NDR)*
*Acta Consejo #3: ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (unanimidad 5/5)*
*Insight crítico DAY 95: seed → HKDF → session_key. Nunca directo.*