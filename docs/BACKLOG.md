# aRGus NDR — BACKLOG
*Última actualización: DAY 124 — 21 Abril 2026*

---

## 📐 Criterio de compleción

| Estado | Criterio |
|---|---|
| ✅ 100% | Implementado + probado en condiciones reales + resultado documentado |
| 🟡 80% | Implementado + compilando + smoke test pasado, sin validación E2E completa |
| 🟡 60% | Implementado parcialmente o con valores placeholder conocidos |
| ⏳ 0% | No iniciado |

---

## 📋 POLÍTICA DE DEUDA TÉCNICA

- **Bloqueante:** se cierra dentro de la feature en que se detectó. No hay merge a main sin test verde.
- **No bloqueante con feature natural:** se asigna a la feature destino. Documentada con ID de feature.
- **No bloqueante sin feature natural:** se acumula hasta abrir `feature/tech-debt-cleanup` (3+ DEBTs sin destino claro).
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA DE SCRIPTS:** Lógica compleja → `tools/script.sh`, nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().
- **REGLA PERMANENTE (DAY 124 — Consejo 7/7):** Ningún fix de seguridad en código de producción se mergea sin test de demostración RED→GREEN. El test debe fallar con el código antiguo y pasar con el nuevo. Sin excepciones.

---

## 🏗️ Tres variantes del pipeline (DAY 124)

El proyecto ha madurado hasta tener tres variantes claramente diferenciadas que comparten el mismo codebase pero divergen en objetivos y entorno de despliegue:

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa | x86-debug, imagen Vagrant con todas las herramientas, build-debug. Para investigación y desarrollo diario. |
| **aRGus-production** | 🟡 Pendiente de cocinar | x86-apparmor + arm64-apparmor. Imágenes Debian optimizadas, sin herramientas de desarrollo. Una imagen por arquitectura, cada una con su Vagrantfile. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ No iniciada | Apéndice científico. Kernel seL4, libpcap (no eBPF/XDP), sniffer reescrito en monohilo. Branch independiente. Nunca se mergeará a main salvo sorpresa. Contribución científica publicable. |

**Implicación arquitectónica:** Toda la deuda técnica abierta debe cerrarse antes de cocinar las imágenes de producción. La rama main debe estar blindada antes de que nazcan las ramas de variante.

---

## ✅ CERRADO DAY 124

### ADR-037 — Static Analysis Security Hardening (safe_path)
- **Status:** ✅ CERRADO DAY 124 — mergeado a main (commit 8bf83b90)
- **Tag:** `v0.5.1-hardened`
- **Tests:** ALL TESTS PASSED · 6/6 RUNNING · TEST-PROVISION-1 8/8 VERDE
- **Implementado:**
  - `contrib/safe-path/` — librería header-only C++20, cero dependencias externas
  - `resolve()`, `resolve_writable()`, `resolve_seed()` (O_NOFOLLOW + 0400 + symlink check)
  - 9 acceptance tests RED→GREEN documentando ataques reales
  - `seed-client`: `resolve_seed()` con `keys_dir_` como prefijo dinámico
  - `firewall-acl-agent/config_loader.cpp`: `resolve()` con prefijo canonicalizado
  - `rag-ingester/config_parser.cpp`: `resolve()` con prefijo canonicalizado
  - `ml-detector/zmq_handler.cpp`: F17 integer overflow (`int64_t` cast)
  - `rag-ingester/csv_dir_watcher.cpp`: F15 falso positivo inotify documentado
  - `provision.sh`: seeds `0640` → `0400`
  - `Makefile`: CHECK 6 actualizado a `0400`
- **Deuda residual descubierta:** ver sección DEUDA ABIERTA DAY 124 abajo

### DEBT-PANDAS-001 — pandas scikit-learn en Vagrantfile
- **Status:** ✅ CERRADO DAY 123 (commit e88e4bf8)

---

## 🔴 DEUDA ABIERTA — Bloqueante para DEBT-PENTESTER-LOOP-001

El Consejo (7/7 unánime, DAY 124) ha determinado que las siguientes deudas deben cerrarse **antes** de iniciar DEBT-PENTESTER-LOOP-001 y antes de cocinar imágenes de producción.

---

### DEBT-INTEGER-OVERFLOW-TEST-001
**Severidad:** 🔴 Alta | **Bloqueante:** Sí
**Origen:** DAY 124 — F17 corregido en `zmq_handler.cpp` sin test de demostración RED→GREEN
**Descripción:** El fix de integer overflow (`int64_t` cast en cálculo de memoria) no tiene test que demuestre que el código antiguo overflowea con valores extremos y que el nuevo produce resultados correctos. Un integer overflow silencioso puede enmascarar degradación o producir comportamiento indefinido en un componente de detección crítico.

**Veredicto Consejo (7/7):** Opción A + C. Unit test sintético + property loop ligero. Sin dependencias nuevas. Extraer cálculo a función pura testeable independientemente.

**Plan de implementación:**
```cpp
// Paso 1: extraer en zmq_handler.cpp
double compute_memory_mb(long pages, long page_size);

// ml-detector/tests/test_zmq_memory_overflow.cpp
// TEST RED: versión antigua overflowea con pages = LONG_MAX / page_size + 1
// TEST GREEN: versión nueva produce resultado correcto y positivo
// TEST PROPERTY: loop sobre rangos realistas, resultado siempre >= 0
```
**Test de cierre:** `test_zmq_memory_overflow` PASSED

---

### DEBT-SAFE-PATH-TEST-PRODUCTION-001
**Severidad:** 🔴 Alta | **Bloqueante:** Sí
**Origen:** DAY 124 — `rag-ingester` STOPPED en build, no en test
**Descripción:** Los fixes de producción (`seed_client`, `config_loader`, `config_parser`) no tienen tests RED→GREEN propios. La incidencia del path relativo se descubrió contra el build de producción en lugar de contra un test. Violación directa del principio RED→GREEN.

**Plan de implementación:** Test de integración por componente que demuestre: (1) path de ataque es rechazado con runtime_error, (2) path legítimo pasa.

**Test de cierre:** `test_seed_client_path_traversal` PASSED · `test_config_loader_path_traversal` PASSED · `test_config_parser_path_traversal` PASSED

---

### DEBT-SAFE-PATH-TEST-RELATIVE-001
**Severidad:** 🔴 Alta | **Bloqueante:** Sí
**Origen:** DAY 124 — `test_safe_path.cpp` no cubre paths relativos
**Descripción:** La incidencia del path relativo (`config/rag-ingester.json`) no habría ocurrido con este test. `weakly_canonical` resuelve paths relativos antes del prefix check, pero no estaba verificado. Tests de librería no son suficientes sin este caso.

**Veredicto Consejo (6/7):** En `contrib/safe-path/tests/`. La capacidad de resolver paths relativos es propiedad de `resolve()`, no de ningún componente concreto.

**Plan de implementación:**
```cpp
// contrib/safe-path/tests/test_safe_path.cpp — añadir:
TEST_F(SafePathTest, RelativePathResolvesBeforePrefixCheck) {
    // input: path relativo dentro del allowed_dir
    // debe resolverse a absoluto antes de la comparación de prefijo
}
```
**Test de cierre:** `test_safe_path` con caso relativo PASSED

---

### DEBT-SNYK-WEB-VERIFICATION-001
**Severidad:** 🟡 Media | **Bloqueante:** Sí (científicamente)
**Origen:** DAY 124 — verificación solo con Snyk CLI macOS, no con Snyk web
**Descripción:** Los 23 findings originales de Snyk web no han sido re-verificados con Snyk web post-fix. No podemos afirmar cierre completo de ADR-037 hasta ejecutar Snyk web sobre `v0.5.1-hardened`.

**Test de cierre:** Snyk web report sobre `main@v0.5.1-hardened` → 0 findings en código C++ de producción

---

### DEBT-CRYPTO-TRANSPORT-CTEST-001
**Severidad:** 🟡 Media | **Bloqueante:** Sí (antes del pentester loop)
**Origen:** Pre-ADR-037 — preexistente, silenciado en Makefile con `|| echo "⚠️ crypto-transport has known LZ4 issues"`
**Descripción:** `test_crypto_transport` y `test_integ_contexts` fallan en CTest. Causa raíz desconocida. La capa de transporte criptográfico es el núcleo de confianza del sistema. No se puede avanzar con tests rotos ahí.

**Veredicto Consejo (7/7):** Investigar ahora. No más silenciar. Plan: `ctest -V`, aislar si es linking/runtime/aserción lógica, documentar causa raíz. Si requiere refactor mayor, documentar en `docs/KNOWN-ISSUES.md` y acotar tiempo (4h).

**Test de cierre:** `test_crypto_transport` PASSED · `test_integ_contexts` PASSED · Makefile sin `|| echo`

---

## 🟢 DEUDA ABIERTA — No bloqueante

### DEBT-DEV-PROD-SYMLINK-001
**Severidad:** 🟢 Media | **Bloqueante:** No
**Origen:** DAY 124 — asimetría dev/prod resuelta provisionalmente con `weakly_canonical`
**Descripción:** En dev, configs en `/vagrant/component/config/`. En prod, en `/etc/ml-defender/`. La solución actual funciona pero introduce asimetría. La solución correcta es que dev replique la estructura de prod mediante symlinks en el Vagrantfile.

**Veredicto Consejo (6/7):** Opción B — symlinks en Vagrantfile. Código siempre ve `/etc/ml-defender/`.

**Implementación aprobada:**
```ruby
# Todos los Vagrantfiles deben heredar esta convención
config.vm.provision "shell", inline: <<-SHELL
  mkdir -p /etc/ml-defender
  ln -sf /vagrant/rag-ingester/config /etc/ml-defender/rag-ingester
  ln -sf /vagrant/firewall-acl-agent/config /etc/ml-defender/firewall-acl-agent
SHELL
```

**Advertencia:** En despliegues sin Vagrant (bare metal, otros hipervisores), los configs deben ubicarse físicamente en `/etc/ml-defender/`. Documentar en `docs/DEV-ENV.md`. Cuando llegue el CICD on-premise, esta convención se traducirá al mecanismo de provisioning elegido.

**Test de cierre:** `make bootstrap` con symlinks → `resolve()` usa siempre `/etc/ml-defender/` → ALL TESTS VERDE

---

### DEBT-PROVISION-PORTABILITY-001
**Severidad:** 🟢 Media | **Bloqueante:** No
**Origen:** DAY 124 — `vagrant` hardcodeado en `chown` de `provision.sh`
**Descripción:** En producción bare metal o cualquier hipervisor distinto de Vagrant, el service user será diferente.

**Veredicto Consejo (6/7):** `ARGUS_SERVICE_USER="${ARGUS_SERVICE_USER:-vagrant}"` al inicio de `provision.sh`. En producción: `export ARGUS_SERVICE_USER=argus-ndr` antes de ejecutar.

**Test de cierre:** `provision.sh` con `ARGUS_SERVICE_USER=testuser` → seeds con permisos `0400 testuser:testuser`

---

### DEBT-GITIGNORE-TEST-SOURCES-001
**Severidad:** 🟢 Baja | **Bloqueante:** No
**Origen:** DAY 124 — `**/test_*` en `.gitignore` ocultó `test_seed_client.cpp` y `test_perms_seed.cpp`
**Descripción:** La regla global ignora fuentes de test. Anti-patrón que ya causó un problema real en DAY 124.

**Veredicto Consejo (7/7):** Refinar para ignorar solo artefactos de build.

**Fix:**
```gitignore
# Reemplazar **/test_*  por:
**/build/**/test_*
!**/test_*.cpp
!**/test_*.hpp
```
**Test de cierre:** `git check-ignore libs/seed-client/tests/test_seed_client.cpp` → no ignorado

---

### DEBT-TRIVY-THIRDPARTY-001
**Severidad:** 🟢 Baja | **Bloqueante:** No
**Origen:** DAY 124 — Trivy scan reveló 95 CVEs en `third_party/llama.cpp`
**Descripción:** 95 CVEs en dependencias Python y npm de llama.cpp. Upstream, no controlable desde aRGus. `.trivyignore` añadido.

**Test de cierre:** llama.cpp actualizado a versión sin CVEs CRITICAL/HIGH conocidos

---

## 🔵 BACKLOG — Deuda de seguridad crítica (pre-producción)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF. SecureBuffer C++20. | Valgrind/ASan: seed no permanece en heap | feature/crypto-hardening |
| **DEBT-SNIFFER-SEED** | Unificar sniffer bajo SeedClient | sniffer arranca con SeedClient | feature/crypto-hardening |
| **docs/CRYPTO-INVARIANTS.md** | Tabla invariantes criptográficos + tests | Fichero existe con tabla completa | feature/crypto-hardening |

---

## 📋 BACKLOG — P3 Features futuras

### PHASE 5 — Loop Adversarial (→ feature/adr038-acrl)

| ID | Tarea | Gates mínimos |
|----|-------|--------------|
| **DEBT-PENTESTER-LOOP-001** | ACRL: Caldera → eBPF capture → XGBoost warm-start → Ed25519 sign → hot-swap | G1: reproducibilidad · G2: ground-truth flow · G3: ≥3 ATT&CK · G4: RFC-válido · G5: sandbox |
| **ADR-038** | ACRL ADR formal | Aprobado por Consejo |

### Feature enterprise — Reentrenamiento distribuido

| ID | Tarea |
|----|-------|
| **FEAT-CLOUD-RETRAIN-001** | Reentrenamiento → cloud aRGus + CSVs anonimizados → validación → transferencia a flota. Requiere ACRL + CICD on-premise + imágenes producción cocinadas. |

### Variantes de producción (ADR-029)

| Variante | Tarea | Feature destino |
|----------|-------|----------------|
| **aRGus-production x86** | Imagen Debian cocinada apparmor x86 + Vagrantfile | feature/production-images |
| **aRGus-production arm64** | Imagen Debian cocinada apparmor arm64 + Vagrantfile | feature/production-images |
| **aRGus-seL4** | kernel seL4, libpcap, sniffer monohilo reescrito. Branch independiente. | feature/sel4-research |

### Infraestructura crypto y protocolo

| ID | Feature destino |
|----|----------------|
| ADR-024 Noise_IKpsk3 impl | feature/adr024-noise-p2p |
| ADR-032 Fase A+B HSM | feature/adr032-hsm |
| ADR-033 TPM Measured Boot | feature/crypto-hardening |
| ADR-034 deployment.yml SSOT | feature/bare-metal |
| ADR-035 etcd-server HA | feature/bare-metal |
| ADR-036 Formal Verification | feature/formal-verification |

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| **Test RED→GREEN obligatorio** | Todo fix de seguridad en producción requiere test de demostración antes del merge. Sin excepciones. | Consejo 7/7 · DAY 124 |
| **`ARGUS_SERVICE_USER`** | Variable de entorno para service user. Default `vagrant`. | Consejo 6/7 · DAY 124 |
| **Asimetría dev/prod** | Opción B: symlinks en Vagrantfile. Código siempre usa `/etc/ml-defender/`. | Consejo 6/7 · DAY 124 |
| **safe_path header-only** | `contrib/safe-path/` — cero dependencias, C++20 puro. | Consejo 7/7 · DAY 123 |
| **Seeds 0400** | Seeds deben tener permisos `0400` (solo owner, solo lectura). | Consejo 7/7 · DAY 124 |
| **Paper — honestidad** | Incluir limitaciones en §5. La honestidad fortalece credibilidad científica. | Consejo 7/7 · DAY 124 |
| **Tres variantes** | aRGus-dev · aRGus-production (x86+ARM apparmor) · aRGus-seL4 (apéndice científico). | DAY 124 |
| **CICD on-premise** | El pipeline CICD debe estar controlado on-premise. Diseño pendiente. | DAY 124 |
| Datasets académicos | INSUFICIENTES como fuente única. Covariate shift demostrado. | Consejo 7/7 · DAY 122 |
| ACRL | IA pentester → captura real → reentrenamiento → hot-swap firmado. | Consejo 7/7 · DAY 122 |
| Plugin integrity | Ed25519 + TOCTOU-safe dlopen + fail-closed std::terminate | ADR-025 · DAY 113 |
| Vagrantfile/Makefile SSOT | Vagrantfile = sistema. Makefile = build + tests + orquestación. | DAY 119 |

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:             ████████████████████ 100% ✅
HMAC Infrastructure:                    ████████████████████ 100% ✅
F1=0.9985 (CTU-13 Neris):              ████████████████████ 100% ✅
CryptoTransport (HKDF+AEAD):            ████████████████████ 100% ✅
ADR-025 Plugin Integrity (Ed25519):     ████████████████████ 100% ✅
TEST-INTEG-4a/4b/4c/4d/4e + SIGN:      ████████████████████ 100% ✅
AppArmor 6/6 enforce:                   ████████████████████ 100% ✅
arXiv:2604.04952 PUBLICADO:             ████████████████████ 100% ✅
PHASE 3 v0.4.0:                         ████████████████████ 100% ✅
PHASE 4 v0.5.0-preprod:                 ████████████████████ 100% ✅
ADR-026 XGBoost Prec=0.9945:            ████████████████████ 100% ✅
Wednesday OOD finding:                  ████████████████████ 100% ✅
make bootstrap idempotente:             ████████████████████ 100% ✅
ADR-037 safe_path v0.5.1-hardened:      ████████████████████ 100% ✅  DAY 124

DEBT-INTEGER-OVERFLOW-TEST-001:         ░░░░░░░░░░░░░░░░░░░░   0% 🔴 DAY 125
DEBT-SAFE-PATH-TEST-PRODUCTION-001:     ░░░░░░░░░░░░░░░░░░░░   0% 🔴 DAY 125
DEBT-SAFE-PATH-TEST-RELATIVE-001:       ░░░░░░░░░░░░░░░░░░░░   0% 🔴 DAY 125
DEBT-GITIGNORE-TEST-SOURCES-001:        ░░░░░░░░░░░░░░░░░░░░   0% 🟢 DAY 125
DEBT-SNYK-WEB-VERIFICATION-001:         ░░░░░░░░░░░░░░░░░░░░   0% 🟡 DAY 126
DEBT-CRYPTO-TRANSPORT-CTEST-001:        ░░░░░░░░░░░░░░░░░░░░   0% 🟡 DAY 126-127
DEBT-DEV-PROD-SYMLINK-001:              ░░░░░░░░░░░░░░░░░░░░   0% 🟢 DAY 127
DEBT-PROVISION-PORTABILITY-001:         ░░░░░░░░░░░░░░░░░░░░   0% 🟢 DAY 128
DEBT-CRYPTO-003a (mlock+bzero):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳
DEBT-PENTESTER-LOOP-001 (ACRL):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-DEUDA
ADR-029 aRGus-production images:        ░░░░░░░░░░░░░░░░░░░░   0% ⏳ POST-DEUDA
ADR-029 aRGus-seL4:                     ░░░░░░░░░░░░░░░░░░░░   0% ⏳ branch independiente
FEAT-CLOUD-RETRAIN-001:                 ░░░░░░░░░░░░░░░░░░░░   0% ⏳ post-ACRL
```

---

## 📝 Notas del Consejo de Sabios — DAY 124 (7/7)

> "ADR-037 mergeado y funcional. La implementación es sólida.
> La reflexión más importante del día no es el código escrito,
> sino el proceso identificado como incompleto:
> los fixes de producción deben tener tests de demostración, no solo tests de librería.
>
> Regla permanente: 'Ningún fix de seguridad en código de producción se mergea
> sin test de demostración RED→GREEN.'
>
> Frase del día — Qwen: 'Un fix sin test de demostración es una promesa sin firma.'
> Frase del día — Kimi: 'Un escudo sin tests es un escudo de papel.'
>
> El sistema es más seguro en código pero aún no completamente verificado en comportamiento.
> Eso se corrige en DAY 125-128 antes de iniciar DEBT-PENTESTER-LOOP-001."
> — Consejo de Sabios (7/7) · DAY 124

---

*DAY 124 — 21 Abril 2026 · main @ 8bf83b90 · Tag: v0.5.1-hardened*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*