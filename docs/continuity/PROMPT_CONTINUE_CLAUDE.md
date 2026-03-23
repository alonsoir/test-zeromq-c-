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

Script bash de provisioning criptográfico en `tools/` (no `scripts/`).
Cuatro modos: `full` | `status` | `verify` | `reprovision <component>`.

**Resultado verificado en VM:**
```
etcd-server        ✅ OK  ✅ OK  ✅ OK  5d45cfbbd7a4bf14...  2026-03-23
sniffer            ✅ OK  ✅ OK  ✅ OK  86df1c883bf196df...  2026-03-23
ml-detector        ✅ OK  ✅ OK  ✅ OK  74d6e63b64d81573...  2026-03-23
firewall-acl-agent ✅ OK  ✅ OK  ✅ OK  42b91b813356c318...  2026-03-23
rag-ingester       ✅ OK  ✅ OK  ✅ OK  5d77798b80410a9b...  2026-03-23
rag-security       ✅ OK  ✅ OK  ✅ OK  c3359aac0b900ac9...  2026-03-23
```

**Paths AppArmor-compatible (ADR-019):**
```
/etc/ml-defender/{component}/private.pem   chmod 600, root:root
/etc/ml-defender/{component}/public.pem    chmod 644, root:root
/etc/ml-defender/{component}/seed.bin      chmod 600, root:root  ← PHASE 1
/etc/ml-defender/{component}/seed.hex      chmod 600 (debug)
/etc/ml-defender/{component}/fingerprint.txt
/etc/ml-defender/{component}/provision_meta.json
/etc/ml-defender/plugins/                  ← listo, vacío (PHASE 2)
/etc/ml-defender/ebpf-plugins/             ← listo, vacío (ADR-018)
```

### JSONs de componentes — bloque `identity` añadido (6/6)

Todos los JSONs tienen ahora:
```json
"identity": {
  "component_id": "<nombre>",
  "keys_dir":    "/etc/ml-defender/<nombre>",
  "public_key":  "/etc/ml-defender/<nombre>/public.pem",
  "private_key": "/etc/ml-defender/<nombre>/private.pem",
  "seed_bin":    "/etc/ml-defender/<nombre>/seed.bin"
}
```

### rag-config.json — corrección arquitectural ✅

- `"encryption": "AES-256-CBC"` eliminado (era letra muerta)
- Unificado a `chacha20-poly1305` con `enabled: false`
- `_architecture_note` documenta el flujo real:
  - rag-security NO recibe tráfico ZMQ cifrado
  - Lee FAISS/SQLite poblados por rag-ingester
  - Los CSVs llegan con HMAC pero SIN cifrado
- Logging unificado a `/vagrant/logs/lab/rag-security.log`

### Makefile — 4 targets nuevos ✅

```makefile
make provision              # full provisioning
make provision-status       # tabla visual (con sudo)
make provision-check        # verificación CI (falla si faltan claves)
make provision-reprovision COMPONENT=sniffer
```

**pipeline-start ahora depende de provision-check:**
```makefile
pipeline-start: provision-check etcd-server-start
```
El pipeline nunca arranca sin claves válidas.

### Vagrantfile — bloque cryptographic-provisioning ✅

```ruby
defender.vm.provision "shell",
  name: "cryptographic-provisioning",
  run: "once",          # persiste entre reinicios
  inline: <<-CRYPTO_PROVISION
    bash /vagrant/tools/provision.sh full
  CRYPTO_PROVISION
```

También añadido a sudoers:
```
vagrant ALL=(ALL) NOPASSWD: /vagrant/tools/provision.sh
```

Y aliases en .bashrc:
```bash
alias provision-status='sudo bash /vagrant/tools/provision.sh status'
alias provision-verify='sudo bash /vagrant/tools/provision.sh verify'
```

---

## Objetivo principal DAY 96 — libs/seed-client

### Tarea A — libs/seed-client (P1)

Mini-librería C++20 al estilo `crypto-transport`. Lee y expone el seed
para que los componentes del pipeline puedan usarlo con `crypto-transport`.

**Interfaz mínima:**
```cpp
// libs/seed-client/include/seed_client/seed_client.hpp
class SeedClient {
public:
    explicit SeedClient(const std::string& config_json_path);
    void load();                                     // lee seed.bin del path del JSON
    const std::array<uint8_t, 32>& seed() const;    // seed listo para crypto-transport
    bool is_loaded() const;
    const std::string& component_id() const;
    // PHASE 2: seed_rotated() para rotación futura
private:
    std::string keys_dir_;
    std::string component_id_;
    std::array<uint8_t, 32> seed_;
    bool loaded_ = false;
};
```

**SeedClient NO hace:**
- Comunicación de red
- Generación de seeds
- Distribución de claves
- Cifrado/descifrado (eso es crypto-transport)

**SeedClient SÍ hace:**
- Leer `identity.keys_dir` del JSON del componente
- Abrir `{keys_dir}/seed.bin`
- Verificar que son exactamente 32 bytes
- Exponer el seed como `std::array<uint8_t, 32>`
- Fallar explícitamente si el fichero no existe o es inválido

**Estructura de ficheros:**
```
libs/seed-client/
  CMakeLists.txt
  include/seed_client/
    seed_client.hpp
  src/
    seed_client.cpp
  tests/
    test_seed_client.cpp    ← CTest
```

**Integración con Makefile:**
```makefile
seed-client-build:
    @vagrant ssh -c 'cd /vagrant/seed-client && ...'
seed-client-clean:
seed-client-test:
```

**Dependencias:** solo OpenSSL (para verificación futura) y nlohmann_json.
NO depende de crypto-transport ni etcd-client — es más primitivo que ambos.

### Tarea B — ADR-012 PHASE 1b (P1 — si queda tiempo)

- Integrar plugin-loader en sniffer
- Test suite plugin-loader con CTest

---

## Secuencia de diagnóstico al arrancar DAY 96

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git status
git log --oneline -5

# Verificar que las claves siguen en la VM
make provision-status

# Verificar pipeline
make pipeline-status

# Ver estructura de libs/ para situar seed-client
ls libs/ 2>/dev/null || echo "libs/ aún no existe — crear"
ls crypto-transport/   # referencia de estructura
```

---

## Backlog activo — estado actualizado DAY 95

| ID | Descripción | Estado |
|---|---|---|
| **provision.sh** | Script bash keypairs + seeds | ✅ DONE — DAY 95 |
| **identity blocks** | Bloque identity en 6 JSONs | ✅ DONE — DAY 95 |
| **rag-config.json** | AES→ChaCha20, flujo documentado | ✅ DONE — DAY 95 |
| **Vagrantfile** | cryptographic-provisioning block | ✅ DONE — DAY 95 |
| **seed-client** | Mini-componente libs/seed-client | **P1 — DAY 96** |
| **ADR-012 PHASE 1b** | Integración sniffer + test suite | **P1 — DAY 96** |
| **SYN-1/2** | rst_ratio + syn_ack_ratio | ✅ DONE — DAY 92 |
| **DEBT-SMB-001** | MIN_SYN_THRESHOLD empírico | DAY 97+ |
| **SYN-3..7** | Sintético + reentrenamiento + F1 | DAY 97+ |
| **DEBT-FD-001** | Fast Detector Path A → JSON | PHASE 2 |
| **ADR-007** | AND-consensus firewall | PHASE 2 |

---

## arXiv — Estado

- Paper draft v4: `docs/Ml defender paper draft v5.md`
- Email enviado a Sebastian Garcia — **esperando respuesta**
- Deadline: DAY 96 — si no responde, email a Yisroel Mirsky (Tier 2)

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