# aRGus NDR — BACKLOG
*Última actualización: DAY 141 — 4 Mayo 2026*

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
- **REGLA PERMANENTE (DAY 124 — Consejo 7/7):** Ningún fix de seguridad en código de producción se mergea sin test de demostración RED→GREEN. Sin excepciones.
- **REGLA PERMANENTE (DAY 125 — Consejo 8/8):** Todo fix de seguridad incluye: (1) unit test sintético, (2) property test de invariante, (3) test de integración en el componente real. Sin excepciones.
- **REGLA PERMANENTE (DAY 127 — Consejo 8/8):** La taxonomía safe_path tiene tres primitivas activas y una futura. Toda nueva superficie de ficheros debe clasificarse con PathPolicy antes de implementar.
- **REGLA PERMANENTE (DAY 128 — Consejo 8/8):** IPTablesWrapper y cualquier ejecución de comandos del sistema usa execve() directo sin shell. Nunca system() ni popen() con strings concatenados.
- **REGLA PERMANENTE (DAY 129 — Consejo 8/8 RULE-SCP-VM-001):** Toda transferencia de ficheros entre VM y macOS usa `scp -F vagrant-ssh-config` o `vagrant scp`. PROHIBIDO pipe zsh — trunca a 0 bytes silenciosamente.
- **PROTOCOLO CANÓNICO (DAY 130 — REGLA EMECAS):** Toda sesión de desarrollo comienza con `vagrant destroy -f && vagrant up && make bootstrap && make test-all`. Sin excepciones.
- **REGLA PERMANENTE (DAY 133 — Consejo 8/8):** `cap_sys_admin` está prohibida en imágenes de producción si el kernel es ≥5.8. Usar `cap_bpf` para operaciones eBPF. Documentar fallback con DEBT-KERNEL-COMPAT-001 si necesario.
- **REGLA PERMANENTE (DAY 134 — Consejo 8/8):** `make hardened-full` es el EMECAS sagrado de la hardened VM — siempre incluye `vagrant destroy -f`. Para iteración de desarrollo usar `make hardened-redeploy` (sin destroy). Los gates `check-prod-all` se ejecutan siempre en ambos modos.
- **REGLA PERMANENTE (DAY 134 — Consejo 8/8):** Las semillas criptográficas NO se transfieren en el procedimiento EMECAS. La hardened VM arranca sin seeds. Target `prod-deploy-seeds` explícito para el momento del deploy real. Los WARNs de `seed.bin no existe` en `check-prod-permissions` son estado correcto por diseño.
- **REGLA PERMANENTE (DAY 134 — Consejo 8/8):** Falco .deb y artefactos binarios de terceros van en `dist/vendor/` (gitignored). El hash SHA-256 se committea en `dist/vendor/CHECKSUMS`. `make vendor-download` descarga y verifica. Si hash no coincide → abort.
- **REGLA PERMANENTE (DAY 134 — Consejo 8/8):** DEBT-ADR040-002 (`confidence_score` en ml-detector) es prerequisito bloqueante de DEBT-ADR040-006 (IPW). No implementar IPW sin verificar primero que el campo existe y varía en runtime.
- **REGLA PERMANENTE (DAY 138 — Consejo 8/8):** Variant B (libpcap) es monohilo por diseño de pcap_dispatch. Los campos de multihilo no aparecen en sniffer-libpcap.json — se hardcodean en el binario con comentario explícito. No configurable, no negociable.
- **REGLA PERMANENTE (DAY 138 — Consejo 8/8):** ODR violations en C++20 son Undefined Behaviour bloqueante. Sub-tarea P0 de DEBT-COMPILER-WARNINGS-CLEANUP-001. Ningún tag posterior sin resolver ODR primero.
- **REGLA PERMANENTE (DAY 140 — Consejo 8/8):** `-Werror` activo en todos los CMakeLists. 0 warnings es un invariante permanente — ningún merge sin `make all 2>&1 | grep -c 'warning:'` = 0.
- **REGLA PERMANENTE (DAY 140 — Consejo 8/8):** Código de terceros con API deprecated → suprimir por fichero en CMake + entrada en `docs/THIRDPARTY-MIGRATIONS.md`. Nunca suprimir warnings en código propio.
- **REGLA PERMANENTE (DAY 140 — Consejo 7/8):** En C++20, usar `[[maybe_unused]]` para parámetros no usados en interfaces virtuales y código nuevo. `/*param*/` solo en stubs temporales con DEBT asociada. Migrar progresivamente (DEBT-MAYBE-UNUSED-MIGRATION-001).
- **REGLA PERMANENTE (DAY 140 — Consejo 8/8):** Gate ODR pre-merge obligatorio: `make PROFILE=production all` antes de cualquier merge a main. Jenkinsfile documenta el gate CI cuando el servidor FEDER esté disponible (DEBT-ODR-CI-GATE-001).
- **REGLA PERMANENTE (DAY 141 — Consejo 8/8):** Variant A y Variant B nunca corren simultáneamente en el mismo hardware. Exclusión mutua via script de arranque (bash/python en Makefile), pre-FEDER. La lógica de detección NO entra en los binarios — separación de responsabilidades.
- **REGLA PERMANENTE (DAY 141 — Consejo 8/8):** `buffer_size_mb` es variable por diseño en sniffer-libpcap.json — permite trazar la curva de optimización de buffer en hardware real. Implementación pcap_create()+pcap_set_buffer_size() pre-FEDER obligatoria antes del benchmark ARM64.
- **REGLA PERMANENTE (DAY 141 — Consejo 8/8):** Clasificadores de warnings de build: script grep/awk determinista. Un LLM no determinista no hace trabajo determinista.

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa | x86-debug, imagen Vagrant completa. Para desarrollo diario. |
| **aRGus-production** | 🟡 En construcción | x86-apparmor + arm64-apparmor. Debian optimizado. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ No iniciada | Apéndice científico. Kernel seL4, libpcap. Branch independiente. |

---

## ✅ CERRADO DAY 141

### Bug Makefile — dependencia seed-client-build implícita
- **Status:** ✅ CERRADO DAY 141 — **Commit:** `63a37d9d`
- **Fix:** `firewall` y `pipeline-build` no declaraban `seed-client-build` como dependencia explícita. En VM limpia (sin binarios previos en `/vagrant`), `firewall-acl-agent` fallaba con `fatal error: seed_client/seed_client.hpp: No such file or directory`. El bug era invisible en sesiones normales porque los binarios sobreviven `vagrant destroy` al estar en carpeta compartida.
- **Invariante:** EMECAS verde — 0 errors tras el fix. 15 warnings restantes son todos third-party (FAISS/libtool). Ver `docs/THIRDPARTY-MIGRATIONS.md`.

### DEBT-PCAP-CALLBACK-LIFETIME-DOC-001 — Contrato lifetime PcapCallbackData
- **Status:** ✅ CERRADO DAY 141 — **Commit:** `63a37d9d` (junto a docs)
- **Fix:** Comentario de contrato añadido en `sniffer/include/pcap_backend.hpp` encima de `struct PcapCallbackData`: validez durante toda la sesión de captura, prohibición de destruir PcapBackend durante `pcap_dispatch()` activo, señalización asíncrona no soportada.

### DEBT-VARIANT-B-CONFIG-001 — JSON propio sniffer-libpcap + config-driven main
- **Status:** ✅ CERRADO DAY 141 — **Commit:** pendiente tag
- **Fix:** `sniffer/config/sniffer-libpcap.json` creado — derivado de `sniffer.json` eliminando todo lo irrelevante para libpcap (kernel_space, threading, af_xdp, deployment, profiles, feature_groups, ml_defender, fast_detector, backpressure, processing, auto_tuner). `main_libpcap.cpp` refactorizado: acepta `-c <config_path>`, lee interface y ZMQ endpoint desde JSON, SeedClient recibe `sniffer-libpcap.json`, stats periódicas cada 30s con `send_failures` + `drop_rate_alert`. Hardcodeado en binario con comentario: snaplen=65535, promiscuous=1, zmq_sender_threads=1 (monohilo ADR-029).
- **Test de cierre:** `make sniffer-libpcap` — 0 warnings. `make test-all` — 9/9 PASSED. ✅

---

## ✅ CERRADO DAY 138

### DEBT-CAPTURE-BACKEND-ISP-001 — CaptureBackend interfaz mínima (ISP)
- **Status:** ✅ CERRADO DAY 138 — **Commit:** `1a7f723a`

### DEBT-VARIANT-B-PCAP-IMPL-001 — Pipeline completo libpcap
- **Status:** ✅ CERRADO DAY 138 — **Commits:** `22df0099` + `da1badf7`
- **Suite 8 tests — 8/8 PASSED en make test-all.**

---

## ✅ CERRADO DAY 134

### Pipeline E2E en hardened VM — check-prod-all PASSED
- **Status:** ✅ CERRADO DAY 134 — **Commits:** `f256e6f0` + `2e9a5b39`

### DEBT-KERNEL-COMPAT-001 · DEBT-PAPER-FUZZING-METRICS-001 · ADR-040 + ADR-041
- **Status:** ✅ CERRADO DAY 134

---

## ✅ CERRADO DAY 133

### Paper Draft v18 · DEBT-PROD-APPARMOR-COMPILER-BLOCK-001 · DEBT-PROD-FALCO-EXOTIC-PATHS-001 · Linux Capabilities
- **Status:** ✅ CERRADO DAY 133

---

## ✅ CERRADO DAY 130–132

DAY 132: DEBT-PROD-COMPAT-BASELINE-001 · README Prerequisites
DAY 130: DEBT-SYSTEMD-AUTOINSTALL-001 · DEBT-SAFE-EXEC-NULLBYTE-001 · DEBT-FUZZING-LIBFUZZER-001 · REGLA EMECAS
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`

---

## ✅ CERRADO DAY 124–129

DAY 124: ADR-037 safe_path → v0.5.1-hardened
DAY 125-126: 8 deudas cerradas · lstat() pre-resolution · prefix fijo
DAY 127: resolve_config() · taxonomía safe_path
DAY 128: Snyk 18 findings · 5 property tests
DAY 129: CWE-78 CERRADO · EtcdClientHmac 9/9

---

## 🔴 DEUDAS ABIERTAS — Seguridad y arquitectura

### DEBT-VARIANT-B-BUFFER-SIZE-001
**Severidad:** 🔴 Alta — P1 pre-FEDER (elevada DAY 141)
**Estado:** ABIERTO — DAY 141
**Componente:** `sniffer/src/userspace/pcap_backend.cpp` + `sniffer/src/userspace/main_libpcap.cpp`

`pcap_open_live()` no permite configurar el buffer del kernel — requiere refactorizar a `pcap_create()+pcap_set_buffer_size()+pcap_activate()`. El campo `capture.buffer_size_mb` existe en `sniffer-libpcap.json` pero no se aplica aún.

**Motivación:** el buffer default del kernel en RPi5 puede ser 2MB. El script de barrido paramétrico necesita poder modificar el valor entre ejecuciones para trazar la curva de optimización. Sin curva empírica, los resultados ARM64/RPi estarán contaminados por una limitación artificial del software, no por el hardware real.

**Test de cierre:** `sniffer-libpcap -c sniffer-libpcap.json` arranca con buffer configurable. Script `tools/buffer-sweep.sh` traza curva drop_rate vs buffer_size_mb. `make test-all` 9/9 PASSED sin regresión.
**Estimación:** 1 sesión

---

### DEBT-VARIANT-B-MUTEX-001
**Severidad:** 🔴 Alta — pre-FEDER
**Estado:** ABIERTO — DAY 141
**Componente:** `tools/argus-sniffer-check.sh` (nuevo) + Makefile targets `sniffer-start`, `sniffer-libpcap-start`

Variant A y Variant B nunca deben correr simultáneamente en el mismo hardware. Un admin podría creer que correr ambos da alta disponibilidad — es un error que produce duplicación de eventos y comportamiento indefinido en etcd.

**Nivel 1 (pre-FEDER obligatorio):** Script bash/python ejecutado por el Makefile antes de arrancar cualquier variante:
1. Consultar etcd: ¿hay otro sniffer registrado en este nodo?
2. Si sí → warn explícito al admin + stop ambos + exit 1
3. Admin decide qué variante arrancar

**Nivel 2 (post-FEDER):** Hardware-aware. Variant A verifica `bpf()` syscall disponible + kernel ≥5.8. Variant B verifica ausencia de eBPF o arquitectura ARM64. Kill de la variante incorrecta + warn + continuar. Con Nivel 1 implementado, la mitad del trabajo estará hecha.

**AppArmor:** mismo perfil para ambas variantes — mismos permisos de red sin root. La distinción la hace etcd al registrarse.

**Test de cierre:** `make sniffer-start && make sniffer-libpcap-start` → script detecta coexistencia → warn + exit 1. Admin no puede arrancar dos variantes sin intervención explícita.
**Estimación:** 1 sesión

---

### DEBT-IRP-NFTABLES-001
**Severidad:** 🔴 Alta — post-merge, pre-demo FEDER
**Estado:** ABIERTO — DAY 136 (protocolo aprobado DAY 138)
**Componente:** /usr/local/bin/argus-network-isolate (pendiente implementar)

Protocolo aprobado por Consejo DAY 138 (8/8):
1. Snapshot: `nft list ruleset > /tmp/argus-backup-$$.nft`
2. Generar fichero de reglas de aislamiento
3. Validar: `nft -c -f /tmp/argus-isolate-$$.nft`
4. Aplicar atómico: `nft -f /tmp/argus-isolate-$$.nft`
5. Timer rollback automático 300s
6. Fallback emergencia: `ip link set eth0 down`

**ADR relacionado:** ADR-042 IRP enmienda E1
**Estimación:** 3 sesiones

---

### DEBT-IRP-QUEUE-PROCESSOR-001
**Severidad:** 🔴 Alta — post-merge
**Estado:** ABIERTO — DAY 136
**Componente:** ADR-042 IRP
**Descripción:** Cola irp-queue sin límites ni procesador systemd dedicado.
**Estimación:** 1 sesión (junto a IRP-NFTABLES)

---

### DEBT-EMECAS-AUTOMATION-001
**Severidad:** 🟡 Media
**Estado:** ABIERTO — DAY 140
**Componente:** Makefile raíz + directorio logs/
Targets `make emecas-dev/prod-x86/prod-arm64` con log automático fechado.
**Estimación:** 1 sesión

---

### DEBT-LLAMA-API-UPGRADE-001
**Severidad:** 🟡 Media — API deprecated, no CVE activo
**Estado:** ABIERTO — DAY 140
**Componente:** `rag/src/llama_integration_real.cpp:29`
**Estimación:** 1 sesión post-FEDER (salvo CVE)

---

### DEBT-ODR-CI-GATE-001
**Severidad:** 🔴 Alta
**Estado:** ABIERTO — DAY 140
**Componente:** Jenkinsfile + `make check-odr`
**Estimación:** 1 sesión post-hardware FEDER

---

### DEBT-GENERATED-CODE-CI-001
**Severidad:** 🟡 Media
**Estado:** ABIERTO — DAY 140
**Estimación:** 1 sesión post-hardware

---

### DEBT-MAYBE-UNUSED-MIGRATION-001
**Severidad:** 🟢 Baja
**Estado:** ABIERTO — DAY 140
**Estimación:** 1 sesión

---

### DEBT-JENKINS-SEED-DISTRIBUTION-001
**Severidad:** 🔴 Alta | **Estado:** ABIERTO — DAY 136

### DEBT-CRYPTO-MATERIAL-STORAGE-001
**Severidad:** 🔴 Alta | **Estado:** ABIERTO — DAY 136

### DEBT-PROD-APT-SOURCES-INTEGRITY-001
**Severidad:** 🔴 Crítica | **Estado:** ABIERTO

### DEBT-SEEDS-SECURE-TRANSFER-001 · DEBT-SEEDS-LOCAL-GEN-001 · DEBT-SEEDS-BACKUP-001
**Severidad:** 🔴 Alta | **Corrección:** post-FEDER

### DEBT-KEY-SEPARATION-001 · DEBT-DEBIAN13-UPGRADE-001 · DEBT-PROD-APPARMOR-PORTS-001
**Severidad:** 🟡 Media | **Target:** post-FEDER

### DEBT-PROD-FALCO-RULES-EXTENDED-001 · DEBT-APT-TIMEOUT-CONFIG-001 · DEBT-FEDER-DEMO-SCRIPT-001 · DEBT-CHECK-PROD-SEED-CONDITIONAL-001
**Severidad:** 🟡 Media | **Target:** varios

---

## 🔵 BACKLOG — Deuda de seguridad crítica (pre-producción)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **DEBT-SAFE-PATH-RESOLVE-MODEL-001** | `resolve_model()` para modelos firmados Ed25519 | test RED→GREEN | feature/adr038-acrl |
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF | Valgrind/ASan | feature/crypto-hardening |
| **DEBT-SNIFFER-SEED** | Unificar sniffer bajo SeedClient | sniffer arranca con SeedClient | feature/crypto-hardening |
| **DEBT-NATIVE-LINUX-BOOTSTRAP-001** | README + make deps-native sin Vagrant | make deps-native verde en Ubuntu 22.04 | post-FEDER |

---

## 📋 BACKLOG — ADR-040 y ADR-041

### DEBT-ADR040-001 a 012 — ML Plugin Retraining Contract
**Target:** post-FEDER (implementación Año 1) | **Consejo 8/8 DAY 134**

| ID | Descripción | Target |
|----|-------------|--------|
| DEBT-ADR040-001 | Golden set v1 (≥50K flows, Parquet, SHA-256 embebido en plugin) | v1.0 |
| DEBT-ADR040-002 | confidence_score ∈ [0,1] en salida ZeroMQ | v1.0 |
| DEBT-ADR040-003 | walk_forward_split.py — mín. 3 ventanas, KS drift | v1.1 |
| DEBT-ADR040-004 | check_guardrails.py — Recall −0.5pp / F1 −2pp → exit 1 | v1.1 |
| DEBT-ADR040-005 | Guardrail integrado en firma Ed25519 (ADR-025) | v1.1 |
| DEBT-ADR040-006 | IPW + uncertainty sampling (P≈0.5), ratio [3%-10%] | v1.2 |
| DEBT-ADR040-007 | Interfaz web revisión exploración en rag-security | v1.2 |
| DEBT-ADR040-008 | Informe diversidad por ciclo: Shannon entropy, ATT&CK coverage | v1.2 |
| DEBT-ADR040-009 | Competición algoritmos: XGBoost vs CatBoost vs LightGBM vs RF | pre-lock-in |
| DEBT-ADR040-010 | Dataset lineage en metadatos del plugin | v1.1 |
| DEBT-ADR040-011 | Canary deployment: 5-10% tráfico 24h antes de 100% | v1.2 |
| DEBT-ADR040-012 | docs/GOLDEN-SET-REGISTRY.md | v1.0 |

### DEBT-ADR041-001 a 006 — Hardware Acceptance Metrics FEDER
**Target:** pre-FEDER, deadline 22 sep 2026 | **Consejo 8/8 DAY 134**

| ID | Descripción | Estado |
|----|-------------|--------|
| DEBT-ADR041-001 | pcap CTU-13 benchmark versionado con SHA-256 | ⏳ |
| DEBT-ADR041-002 | make golden-set-eval ARCH=$(uname -m) | ⏳ |
| DEBT-ADR041-003 | make feder-demo — suite completa <30 min | ⏳ |
| DEBT-ADR041-004 | Compra hardware x86 (NUC, NIC XDP nativo) | ⏳ |
| DEBT-ADR041-005 | Compra Raspberry Pi 4/5 | ⏳ |
| DEBT-ADR041-006 | Ejecución protocolo completo en hardware físico | ⏳ |

---

## 📋 BACKLOG — Benchmarks Empíricos (FEDER Year 1)

### BACKLOG-ZMQ-TUNING-001
**Estado:** ⏳ BACKLOG | **Prioridad:** P1 — Prerequisito de BENCHMARK-CAPACITY
**Bloqueado por:** ADR-029 Variant A + Variant B estables
**Documento:** `docs/adr/BACKLOG-ZMQ-TUNING-001.md`

### BACKLOG-BENCHMARK-CAPACITY-001
**Estado:** ⏳ BACKLOG | **Prioridad:** P1 — FEDER Year 1 Deliverable
**Bloqueado por:** BACKLOG-ZMQ-TUNING-001 + hardware físico
**Documento:** `docs/adr/BACKLOG-BENCHMARK-CAPACITY-001.md`

### BACKLOG-BUILD-WARNING-CLASSIFIER-001
**Estado:** ⏳ BACKLOG | **Prioridad:** Post-FEDER
**Origen:** DAY 141 — 15 warnings third-party contaminando grep EMECAS
**Decisión Consejo DAY 141:** Implementar como **script grep/awk determinista**, no como clasificador TinyLlama. Un LLM no determinista no debe hacer trabajo determinista. Patrones conocidos: prefijo `defender:`, paths `/tmp/faiss/`, `libtool:`.
**Workaround actual:** `grep 'warning:' output.md | grep -v 'defender:'`

---

## 📋 BACKLOG — P3 Features futuras

### PHASE 5 — Loop Adversarial

| ID | Tarea | Gates mínimos |
|----|-------|--------------|
| **DEBT-PENTESTER-LOOP-001** | ACRL: Caldera → eBPF → XGBoost warm-start → Ed25519 → hot-swap | G1–G5 sandbox |
| **ADR-038** | ACRL ADR formal | Aprobado por Consejo |
| **ADR-025-EXT-001** | Emergency Patch Protocol — Plugin Unload vía mensaje firmado | TEST-INTEG-SIGN-8/9/10 RED→GREEN |

### Variantes de producción

| Variante | Tarea | Feature destino |
|----------|-------|----------------|
| **aRGus-production x86** | Pipeline E2E hardened · check-prod-all verde | feature/adr030-variant-a |
| **aRGus-production arm64** | Imagen Debian arm64 + AppArmor + Vagrantfile | feature/production-images |
| **aRGus-seL4** | Kernel seL4, libpcap, sniffer monohilo. Branch independiente. | feature/sel4-research |

---

## BACKLOG-FEDER-001

**Estado:** PENDIENTE — bloqueado por prerequisites técnicos
**Contacto:** Andrés Caro Lindo — UEx/INCIBE — andresc@unex.es
**Deadline límite:** 22 septiembre 2026 | **Go/no-go técnico:** 1 agosto 2026
**Emails enviados DAY 141:** hardware FEDER (RPi5+N100+switch) + scope standalone vs federado

### Gate de entrada

- [x] ADR-026 mergeado a main (XGBoost F1=0.9978)
- [x] ADR-030 Variant A infraestructura completa (DAY 133)
- [x] Pipeline E2E en hardened VM verde (`make check-prod-all`) — DAY 134 ✅
- [ ] ADR-030 Variant B (ARM64) estable
- [ ] DEBT-VARIANT-B-BUFFER-SIZE-001 implementada (pre-benchmark)
- [ ] Demo técnica grabable < 10 minutos (`scripts/feder-demo.sh`)
- [ ] ADR-041 protocolo hardware: métricas validadas en x86 + ARM (`make feder-demo`)
- [ ] Golden set v1 creado y versionado (DEBT-ADR040-001)
- [ ] BACKLOG-ZMQ-TUNING-001 concluido
- [ ] BACKLOG-BENCHMARK-CAPACITY-001 concluido (FEDER Year 1 Deliverable)
- [ ] Clarificación scope con Andrés: NDR standalone vs federación (antes julio 2026)

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| **Test RED→GREEN obligatorio** | Todo fix de seguridad requiere test antes del merge. | Consejo 7/7 · DAY 124 |
| **Property test obligatorio** | Todo fix de seguridad incluye property test si aplica. | Consejo 8/8 · DAY 125 |
| **Symlinks en seeds: NO** | resolve_seed(): lstat() ANTES de resolve(). | Consejo 8/8 · DAY 125-126 |
| **ConfigParser prefix fijo** | allowed_prefix explícito, default /etc/ml-defender/. | Consejo 8/8 · DAY 125-126 |
| **resolve_config() para configs** | lexically_normal() verifica prefix ANTES de seguir symlinks. | DAY 127 |
| **Taxonomía safe_path: 3 primitivas activas** | resolve() · resolve_seed() · resolve_config(). | Consejo 8/8 · DAY 127 |
| **CWE-78 execve()** | execv() sin shell. | Consejo 8/8 · DAY 128 |
| **RULE-SCP-VM-001** | scp/vagrant scp. Prohibido pipe zsh. | Consejo 8/8 · DAY 129 |
| **REGLA EMECAS** | vagrant destroy -f && vagrant up && make bootstrap && make test-all. | DAY 130 |
| **AppArmor como primera línea BSR** | AppArmor bloquea compiladores. check-prod-no-compiler es auditoría. | DAY 132 — founder |
| **cap_bpf reemplaza cap_sys_admin** | Linux ≥5.8: cap_bpf para eBPF. | Consejo 8/8 · DAY 133 |
| **cap_net_bind_service eliminada** | Puerto 2379 > 1024. Innecesaria. | Consejo 8/8 · DAY 133 |
| **LimitMEMLOCK en systemd** | etcd-server: LimitMEMLOCK=16M. | Consejo 8/8 · DAY 133 |
| **deny explícitos en AppArmor** | Mantener — claridad auditiva hospitalaria. | Founder · DAY 133 |
| **Walk-forward obligatorio (ADR-040)** | K-fold prohibido. Split sobre timestamp_first_packet. Mín. 3 ventanas. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Golden set inmutable (ADR-040)** | ≥50K flows, SHA-256 embebido en plugin firmado. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Guardrail asimétrico Ed25519 (ADR-040)** | Recall −0.5pp. F1 −2pp. FPR +1pp. Latencia p99 +10%. Exit 1 = no firma. | ADR-040 · Consejo 8/8 · DAY 134 |
| **IPW + uncertainty sampling (ADR-040)** | 5% exploración (P≈0.5). Ratio adaptativo [3%-10%]. | ADR-040 · Consejo 8/8 · DAY 134 |
| **CaptureBackend mínima (ISP)** | 5 métodos puros. EbpfBackend tiene métodos eBPF. main.cpp usa EbpfBackend directamente. | Consejo 5-2-1 · DAY 137 → Cerrado DAY 138 |
| **Variant B monohilo permanente** | libpcap no es thread-safe sobre mismo handle. zmq_sender_threads=1 hardcodeado, no configurable. | Consejo 8/8 · DAY 138 |
| **dontwait policy NDR** | Mejor perder paquete que bloquear loop captura. Exponer send_failures como métrica. | Consejo 8/8 · DAY 138 |
| **nftables transaccional para IRP** | nft -f atómico. Snapshot + rollback 300s. Fallback ip link down. iptables rechazado en Debian 12. | Consejo 8/8 · DAY 138 |
| **ODR es P0 bloqueante** | ODR violations en C++20 = UB. Bloqueante para cualquier tag posterior. | Consejo 8/8 · DAY 138 |
| **-Werror invariante permanente** | 0 warnings es invariante. Ningún merge sin grep -c warning: = 0. | Consejo 8/8 · DAY 140 |
| **Terceros deprecated: suprimir + doc** | APIs deprecated de terceros → suprimir por fichero + THIRDPARTY-MIGRATIONS.md. Nunca suprimir código propio. | Consejo 8/8 · DAY 140 |
| **[[maybe_unused]] en C++20** | Interfaces virtuales y código nuevo → [[maybe_unused]]. Stubs temporales → /*param*/ con DEBT. | Consejo 7/8 · DAY 140 |
| **Gate ODR pre-merge obligatorio** | make PROFILE=production all antes de merge a main. Jenkinsfile cuando haya servidor. | Consejo 8/8 · DAY 140 |
| **seL4 no diseñar ahora** | CaptureBackend (5 métodos) es reutilizable. Todo lo demás reescritura. YAGNI hasta equipo especializado. | Consejo 8/8 · DAY 138 |
| **seed-client-build dependencia explícita** | firewall y pipeline-build deben declarar seed-client-build. En VM limpia sin binarios previos el build falla silenciosamente. | DAY 141 |
| **Exclusión mutua Variant A/B** | Nunca simultáneas en el mismo hardware. Nivel 1: script bash/python en Makefile pre-FEDER. Nivel 2: hardware-aware post-FEDER. Lógica NO en binarios. | Consejo 8/8 · DAY 141 |
| **buffer_size_mb variable por diseño** | Permite trazar curva de optimización. Implementar pcap_create()+pcap_set_buffer_size() pre-FEDER antes del benchmark ARM64. | Consejo 8/8 · DAY 141 |
| **AppArmor: mismo perfil A y B** | Mismos permisos de red sin root. La distinción la hace etcd al registrarse. | Consejo 8/8 · DAY 141 |
| **Warning classifier: grep/awk** | Script determinista. Un LLM no determinista no hace trabajo determinista. | Consejo 8/8 · DAY 141 |

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:             100% ✅
HMAC Infrastructure:                    100% ✅
F1=0.9985 (CTU-13 Neris):              100% ✅
CryptoTransport (HKDF+AEAD):            100% ✅
ADR-025 Plugin Integrity (Ed25519):     100% ✅
TEST-INTEG-4a/4b/4c/4d/4e + SIGN:      100% ✅
arXiv:2604.04952 PUBLICADO:             100% ✅
PHASE 3 v0.4.0:                         100% ✅
PHASE 4 v0.5.0-preprod:                 100% ✅
ADR-026 XGBoost Prec=0.9945:            100% ✅
ADR-037 safe_path v0.5.1-hardened:      100% ✅  DAY 124
DEBT-PROD-APPARMOR-COMPILER-BLOCK-001:  100% ✅  DAY 133
DEBT-PROD-FALCO-EXOTIC-PATHS-001:       100% ✅  DAY 133
DEBT-PROD-FS-MINIMIZATION-001:           60% 🟡  DAY 133 (parcial)
vagrant/hardened-x86/ completo:         100% ✅  DAY 133
DEBT-PAPER-FUZZING-METRICS-001:         100% ✅  DAY 134
DEBT-KERNEL-COMPAT-001:                 100% ✅  DAY 134
ADR-040 ML Retraining Contract (def.):  100% ✅  DAY 134
ADR-041 HW Acceptance Metrics (def.):   100% ✅  DAY 134
make hardened-full EMECAS:              100% ✅  DAY 135
DEBT-PROD-APT-SOURCES-INTEGRITY-001:    100% ✅  DAY 135
DEBT-CONFIDENCE-SCORE-001:              100% ✅  DAY 135
arXiv replace v15→v18:                  100% ✅  DAY 135
v0.6.0-hardened-variant-a mergeado:     100% ✅  DAY 136
docs/KNOWN-DEBTS-v0.6.md:              100% ✅  DAY 136 (actualizado DAY 138)
DEBT-CAPTURE-BACKEND-ISP-001:           100% ✅  DAY 138
DEBT-VARIANT-B-PCAP-IMPL-001:          100% ✅  DAY 138 (8/8 tests)
DEBT-COMPILER-WARNINGS-CLEANUP-001:     100% ✅  DAY 140 (192→0 warnings, ODR limpio)
DEBT-PCAP-CALLBACK-LIFETIME-DOC-001:   100% ✅  DAY 141
DEBT-VARIANT-B-CONFIG-001:             100% ✅  DAY 141 (9/9 tests, 0 warnings)
Bug Makefile seed-client-build:         100% ✅  DAY 141 (commit 63a37d9d)
DEBT-VARIANT-B-BUFFER-SIZE-001:          0% ⏳  P1 pre-FEDER (pre-benchmark ARM64)
DEBT-VARIANT-B-MUTEX-001:               0% ⏳  P1 pre-FEDER (Nivel 1 script)
BACKLOG-ZMQ-TUNING-001:                  0% ⏳  pre-FEDER
BACKLOG-BENCHMARK-CAPACITY-001:           0% ⏳  FEDER Year 1 Deliverable
BACKLOG-BUILD-WARNING-CLASSIFIER-001:    0% ⏳  post-FEDER (grep/awk script)
DEBT-IRP-NFTABLES-001:                   0% ⏳  pre-FEDER
DEBT-IRP-QUEUE-PROCESSOR-001:            0% ⏳  post-merge
DEBT-LLAMA-API-UPGRADE-001:              0% ⏳  post-FEDER (salvo CVE)
DEBT-ODR-CI-GATE-001:                    0% ⏳  requiere servidor CI/CD
DEBT-GENERATED-CODE-CI-001:              0% ⏳  requiere servidor CI/CD
DEBT-MAYBE-UNUSED-MIGRATION-001:         0% ⏳  cosmético, post deudas P0
DEBT-EMECAS-AUTOMATION-001:              0% ⏳  post deudas P0
DEBT-JENKINS-SEED-DISTRIBUTION-001:      0% ⏳  pre-FEDER
DEBT-CRYPTO-MATERIAL-STORAGE-001:        0% ⏳  pre-FEDER
DEBT-KEY-SEPARATION-001:                 0% ⏳  post-FEDER
DEBT-ADR040-001..012:                    0% ⏳  post-FEDER Año 1
DEBT-ADR041-001..006:                    0% ⏳  pre-FEDER
ADR-031 aRGus-seL4:                      0% ⏳  branch independiente
```

---

## 📝 Notas del Consejo de Sabios — DAY 141 (8/8)

> "DAY 141 — Bug Makefile seed-client-build cerrado. DEBT-PCAP-CALLBACK-LIFETIME-DOC-001 cerrado. DEBT-VARIANT-B-CONFIG-001 cerrado — sniffer-libpcap.json propio + main_libpcap.cpp config-driven. 9/9 tests PASSED. 0 warnings. Emails FEDER enviados a Andrés Caro Lindo.
>
> Q1 (8/8 + founder): Identidad criptográfica compartida component_id='sniffer' correcta. PERO: exclusión mutua obligatoria. DEBT-VARIANT-B-MUTEX-001 registrada. Nivel 1 via script bash/python en Makefile, pre-FEDER. Nivel 2 hardware-aware post-FEDER. La lógica de detección NO entra en los binarios. AppArmor: mismo perfil para ambas variantes. La distinción la hace etcd.
>
> Q2 (8/8 + founder): buffer_size_mb pre-FEDER obligatorio. Variable por diseño — script de barrido paramétrico para trazar curva de optimización. Sin curva, los resultados ARM64/RPi estarán contaminados por limitación artificial del software. DEBT-VARIANT-B-BUFFER-SIZE-001 elevada a P1 pre-FEDER.
>
> Q3 (8/8 + founder): Script grep/awk determinista para clasificar warnings de build. Un LLM no determinista no hace trabajo determinista. BACKLOG-BUILD-WARNING-CLASSIFIER-001 actualizado a grep/awk.
>
> Q4 (8/8): EMECAS suficiente gate ahora. Jenkins cuando haya servidor FEDER.
>
> 'buffer_size_mb no es una opción de confort — es una variable experimental. Sin ella, el benchmark ARM64 mide el default del kernel, no el hardware.' — Claude
>
> 'La exclusión mutua en el script de arranque, no en el binario, es la decisión correcta: separación de responsabilidades, sin añadir lógica de orquestación a un componente de captura.' — Grok"
> — Consejo de Sabios (8/8) · DAY 141

---

## 📝 Notas del Consejo de Sabios — DAY 140 (8/8)

> "DAY 140 — 192 → 0 warnings. `-Werror` activo como invariante permanente. ODR limpio con LTO.
> 'Pasar de 192 a 0 warnings con -Werror no es cosmético, es un cambio estructural. Has eliminado una clase entera de fallos futuros.' — ChatGPT
> 'El proyecto está entrando en fase donde los errores ya no serán visibles, sino estadísticos o esporádicos.' — Qwen"
> — Consejo de Sabios (8/8) · DAY 140

---

## 📝 Notas del Consejo de Sabios — DAY 138 (8/8)

> "DAY 138 — ISP cerrado. Pipeline Variant B completo. ODR P0 bloqueante confirmado.
> 'Los warnings ODR en C++ son bombas de reloj. En infraestructura crítica, el comportamiento indefinido no es hipotético — es inevitable.' — Grok"
> — Consejo de Sabios (8/8) · DAY 138

---

## 📝 Notas del Consejo de Sabios — DAY 136 (8/8)

> "DAY 136 — v0.6.0-hardened-variant-a mergeado. DEBT-IRP-NFTABLES-001 es P0 pre-FEDER. argus-network-isolate inexistente = fail catastrófico en demo."
> — Consejo de Sabios (8/8) · DAY 136

---

## 📝 Notas del Consejo de Sabios — DAY 134 (8/8)

> "ADR-040 + ADR-041: contratos de calidad ML y métricas de aceptación hardware. Walk-forward obligatorio. Golden set inmutable. Temperatura ARM ≤75°C gate no negociable."
> — Consejo de Sabios (8/8) · DAY 134

---

## 📝 Notas del Consejo de Sabios — DAY 133 (8/8)

> "Transición de 'diseño correcto' a 'comportamiento real verificable'. cap_bpf. AppArmor 6/6. 'Un escudo que no se prueba contra el ataque real es un escudo de teatro.' — Qwen"
> — Consejo de Sabios (8/8) · DAY 133

---

## 🧬 HIPÓTESIS CENTRAL — Inmunidad Global Adaptativa

**Formulada:** DAY 128 | **Estado:** Pendiente demostración (DEBT-PENTESTER-LOOP-001)

Un sistema con ACRL converge hacia cobertura de técnicas ATT&CK en tiempo polinomial. Un sistema estático no converge nunca.

---

*DAY 141 — 4 Mayo 2026 · feature/variant-b-libpcap @ 63a37d9d*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*