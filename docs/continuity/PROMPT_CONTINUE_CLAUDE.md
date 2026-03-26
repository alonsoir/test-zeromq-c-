# ML Defender — Prompt de Continuidad DAY 99
## 27 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING
**Tests:** 22/22 suites ✅ (1 flaky pre-existente en firewall VM)
**Rama:** `feature/plugin-loader-adr012`
**Commit pendiente DAY 98:** ⚠️ hacer como PASO 0

---

## Cadena de confianza

```
provision.sh → seed.bin (chmod 0600, /etc/ml-defender/{component}/)
    └► SeedClient → HKDF-SHA256 → CryptoTransport
        └► 6/6 componentes migrados ✅ DAY 98
        └► Contextos HKDF: ⚠️ ASIMÉTRICOS — P1 DAY 99
        └► tools/: pendiente DAY 99
```

---

## ⚠️ ALERTA CRÍTICA — Contextos HKDF asimétricos

```cpp
// INCORRECTO (claves distintas → MAC error en E2E):
// sniffer cifra:      "ml-defender:sniffer:v1:tx"
// ml-detector descifra: "ml-defender:ml-detector:v1:rx"

// CORRECTO (contexto pertenece al canal, idéntico en emisor/receptor):
// crypto-transport/include/crypto_transport/contexts.hpp — a crear
constexpr const char* CTX_SNIFFER_TO_ML  = "ml-defender:sniffer-to-ml-detector:v1";
constexpr const char* CTX_ML_TO_FIREWALL = "ml-defender:ml-detector-to-firewall:v1";
constexpr const char* CTX_ETCD_TX        = "ml-defender:etcd:v1:tx";
constexpr const char* CTX_ETCD_RX        = "ml-defender:etcd:v1:rx";
constexpr const char* CTX_RAG_ARTIFACTS  = "ml-defender:rag-artifacts:v1";
```

---

## Hoja de ruta — decisión cerrada

**No hay FASE 2.** La Opción 2 (instance_id en nonce) introduce deuda técnica con problemas de seguridad reales (replay cross-instance). Se documenta como camino descartado en ADR-022 y no se implementa.

```
FASE 1 (DAY 99)
  Single instance. Contexts + TEST-INTEG + fail-closed.
  Pipeline no distribuido estabilizado completamente.

FASE 3 (en cuanto FASE 1 esté estable)
  deployment.yml como SSOT de topología.
  Seeds por familia de canal.
  provision.sh refactorizado para leer manifiesto.
  Vagrantfile multi-VM con topología distribuida real.
  ADR-021: deployment manifest + families.
```

---

## Arquitectura de familias de canal (FASE 3 — diseño cerrado)

```
seed_family_A → canal captura→detección
                sniffer1 + ml-detector1 + ml-detector2

seed_family_B → canal detección→enforcement
                ml-detector1 + ml-detector2 + firewall1

seed_family_C → canal artefactos→RAG
                ml-detector1 + ml-detector2 + firewall1 + rag-ingester1
```

Un componente puede pertenecer a varias familias — recibe varios `seed.bin` en paths distintos. `provision.sh` lee `deployment.yml` y distribuye seeds a cada miembro de cada familia.

**Vagrantfile objetivo (FASE 3):**
```
VM sniffer1        (192.168.56.10)
VM ml-detector1    (192.168.56.11)
VM ml-detector2    (192.168.56.12)
VM firewall1       (192.168.56.13)
VM rag-ingester1   (192.168.56.14)
VM rag-local1      (192.168.56.15)

provision.sh --manifest deployment.yml → genera y distribuye seeds por familia
```

---

## Decisiones cerradas

| Decisión | Resolución |
|---|---|
| Migración CryptoManager | ✅ 6/6 DAY 98 |
| LZ4 formato | custom `[uint32_t orig_size LE]` — estándar interno |
| Modo degradado | **FATAL en producción** — solo `MLD_DEV_MODE=1` en dev |
| Contextos HKDF | por canal, idéntico emisor/receptor |
| Opción 2 (nonce instance_id) | **DESCARTADA** — replay cross-instance, deuda técnica |
| Multi-instancia | **FASE 3 directamente** — deployment.yml + families |
| TEST-INTEG-1/2 | gate obligatorio antes de arXiv |
| `tools/` | antes de arXiv, FASE 3 o antes si hay tiempo |

---

## Objetivos DAY 99 (orden estricto)

### PASO 0 — Commit DAY 98
```bash
git add -A
git commit -m "feat(crypto): migrar 6 componentes CryptoManager → CryptoTransport (ADR-013 PHASE 2, DAY 98)

- etcd-server/etcd-client/sniffer/ml-detector/rag-ingester/firewall migrados
- CryptoManager DEPRECATED en todos los componentes
- LZ4 custom [uint32_t orig_size LE] estandarizado
- Tests: 22/22 suites green

DEBT-CRYPTO-004: resuelto"
```

### PASO 1 — `contexts.hpp`
```bash
# 1. Crear crypto-transport/include/crypto_transport/contexts.hpp
# 2. Añadir como PUBLIC header en crypto-transport/CMakeLists.txt
# 3. make crypto-transport-build  ← instala en /usr/local/include/
# 4. Reemplazar strings hardcodeados en 6 componentes
# 5. make all-build  ← verificar compilación
```

### PASO 2 — TEST-INTEG-1
```
SeedClient(seed.bin) → tx(CTX_SNIFFER_TO_ML).encrypt(payload)
                     → rx(CTX_SNIFFER_TO_ML).decrypt()
                     → assert == payload original
```

### PASO 3 — TEST-INTEG-2
```
JSON → LZ4([uint32_t]+data) → tx.encrypt() → rx.decrypt()
     → LZ4_decompress() → assert == JSON original byte-a-byte
```

### PASO 4 — Fail-closed EventLoader + RAGLogger
```bash
# ANTES: provision.sh está corriendo y seeds en /etc/ml-defender/
vagrant ssh -c "ls /etc/ml-defender/rag-ingester/"  # verificar
# DESPUÉS: implementar fail-closed con MLD_DEV_MODE como escape
# make test  ← verificar 22/22 sigue verde
```

### PASO 5 — Test pendiente
```bash
# Identificar el test que está configurado pero no en ctest
vagrant ssh -c "cd /vagrant/etcd-server/build-debug && ctest -N"
vagrant ssh -c "cd /vagrant/etcd-client/build && ctest -N"
# Sospecha: test_hmac_integration en etcd-server
```

### PASO 6 — ADR-021 + ADR-022 (documentar, no implementar)
```
ADR-021: deployment manifest + families — diseño cerrado, implementación FASE 3
ADR-022: nonce collision multi-instancia — Opción 2 descartada, Opción 1 vía ADR-021
```

### PASO 7 — `tools/` migración (si queda tiempo)
```
tools/synthetic_sniffer_injector.cpp    — usa CTX_SNIFFER_TO_ML
tools/synthetic_ml_output_injector.cpp
tools/generate_synthetic_events.cpp
Config: /etc/ml-defender/tools/tools.json
```

---

## Ficheros a actualizar (P1 DAY 99)

⚠️ **Estos ficheros no se actualizaron en DAY 98 — pendiente:**
```
docs/BACKLOG.md           — añadir ADR-021, ADR-022, FASE 3, eliminar FASE 2
docs/ARCHITECTURE.md      — añadir arquitectura de familias de canal,
                            deployment.yml, Vagrantfile multi-VM,
                            decisión sobre Opción 2 descartada
```

---

## Diagnóstico de arranque DAY 99

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git status && git log --oneline -3

# Contextos a reemplazar
grep -rn "ml-defender:sniffer\|ml-defender:ml-detector\|ml-defender:etcd\|rag-artifacts" \
  sniffer/ ml-detector/ rag-ingester/ firewall-acl-agent/ etcd-server/ etcd-client/ \
  --include="*.cpp" --include="*.hpp" | grep -v build | grep -v DEPRECATED

# Estado tests
make test 2>&1 | tail -5

# contexts.hpp
ls crypto-transport/include/crypto_transport/contexts.hpp 2>/dev/null || echo "pendiente crear"
```

---

## Backlog P1 activo DAY 99

| ID | Tarea | Fase | Prioridad |
|---|---|---|---|
| DAY99-0 | Commit DAY 98 pendiente | 1 | P0 — primero |
| ADR-013-CONTEXTS | `contexts.hpp` + 6 componentes | 1 | P1 |
| TEST-INTEG-1 | sniffer→ml round-trip E2E | 1 | P1 gate arXiv |
| TEST-INTEG-2 | JSON→LZ4→cifrado round-trip | 1 | P1 gate arXiv |
| FAIL-CLOSED | EventLoader + RAGLogger | 1 | P1 |
| TEST-PENDING | identificar + correr test pendiente | 1 | P1 |
| DOCS-UPDATE | BACKLOG.md + ARCHITECTURE.md | 1 | P1 |
| ADR-021 | deployment.yml + families (diseño) | 3 | P1 documentar |
| ADR-022 | nonce multi-instancia (Opción 2 descartada) | — | P1 documentar |
| DEBT-CRYPTO-004b | tools/ migración | 1/3 | P2 antes arXiv |
| DEBT-CRYPTO-003a | mlock() seed_client.cpp | 1 | P2 |
| VAGRANTFILE-DIST | multi-VM topología distribuida | 3 | P3 post-arXiv |
| ADR-020 | borrar flags enabled JSON | — | DAY tranquilo |
| DEBT-NAMING-001 | libseedclient sin underscore | — | DAY tranquilo |

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

*DAY 98 cierre — 26 marzo 2026*
*Tests: 22/22 ✅ · 6/6 componentes migrados*
*Hoja de ruta limpia: FASE 1 (contexts + tests) → FASE 3 (deployment.yml + families + Vagrantfile distribuido)*
*Opción 2 descartada — sin deuda técnica*
*Co-authored-by: Alonso Isidoro Roman + Claude (Anthropic)*