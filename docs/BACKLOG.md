# aRGus NDR — BACKLOG
*Última actualización: DAY 142 — 5 Mayo 2026*

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
- **REGLA PERMANENTE (DAY 142 — Consejo 8/8 + founder):** El criterio de disparo del IRP nunca se basa en una señal única. Para FEDER: `threat_score >= 0.95 AND event_type IN (ransomware, lateral_movement, c2_beacon)`. En entornos hospitalarios, un falso positivo sobre un equipo médico crítico (ventilador, bomba de infusión) conectado a la intranet/DMZ es inaceptable. La señal debe ser explicable, auditable y multi-componente.
- **REGLA PERMANENTE (DAY 142 — Consejo 8/8 + founder):** `auto_isolate: true` por defecto en `isolate.json`. El sistema protege sin que el administrador toque nada. Desactivar el aislamiento automático es un acto explícito y consciente. Instalar y funcionar.
- **REGLA PERMANENTE (DAY 142 — Consejo 8/8):** Todo trigger de aislamiento automático usa `fork()+execv()`. El proceso padre (firewall-acl-agent) nunca muere. El agente debe sobrevivir al aislamiento para continuar registrando evidencia forense. Un agente muerto durante un ataque activo es exactamente lo que el atacante busca.
- **REGLA PERMANENTE (DAY 142 — Consejo 8/8):** AppArmor `enforce` desde el primer deploy de cualquier nuevo componente. La fase `complain` no es una característica de seguridad — es deuda de validación. Si el perfil bloquea algo legítimo, se descubre en dev, no en producción.
- **REGLA PERMANENTE (DAY 142 — macOS):** zsh intercepta `!` en heredocs. Para código C++ con emojis o caracteres especiales: siempre `vagrant ssh << 'SSHEOF'` con Python dentro. Nunca heredoc directo desde zsh para código complejo.

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa | x86-debug, imagen Vagrant completa. Para desarrollo diario. |
| **aRGus-production** | 🟡 En construcción | x86-apparmor + arm64-apparmor. Debian optimizado. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ No iniciada | Apéndice científico. Kernel seL4, libpcap. Branch independiente. |

---

## ✅ CERRADO DAY 142

### Regresión test_config_parser — safe_path path no compliant
- **Status:** ✅ CERRADO DAY 142 — **Commit:** `4bbc98ee`
- **Fix:** `test_config_parser` pasaba `/vagrant/rag-ingester/config/rag-ingester.json` a `ConfigParser::load()`. ADR-037 (safe_path) bloqueaba correctamente el path de dev. Fix: usar path de producción `/etc/ml-defender/rag-ingester/rag-ingester.json`. `test_config_parser_traversal` (ataques path traversal) ya pasaba — no tocado.
- **Invariante:** EMECAS verde — 8/8 tests rag-ingester PASSED.

### DEBT-IRP-NFTABLES-001 — sesiones 1/3 y 2/3
- **Status:** 🟡 60% — sesiones 1 y 2 cerradas — **Commits:** `6480e234` + `e8928612`
- **Sesión 1:** Binario `argus-network-isolate` C++20 creado en `tools/argus-network-isolate/`. Pasos 1-3: snapshot selectivo (solo tabla `argus_isolate`, excluye tablas iptables-managed con `xt match` incompatibles), generate_rules con whitelist IP/port configurable, validate_dry_run (`nft -c`). Config: `tools/argus-network-isolate/config/isolate.json`. Forense JSONL en `/var/log/argus/network-isolate-forensic.jsonl`.
- **Sesión 2:** Pasos 4-6: apply atómico (`nft -f`), timer `systemd-run --on-active=300s` idempotente (stop+reset-failed antes de crear), rollback robusto (elimina tabla `argus_isolate`, no toca tablas del sistema). Ciclo completo verificado en dev VM (eth2): NORMAL→ISOLATED→STATUS→ROLLBACK→NORMAL. SSH sobrevivió en todo momento (eth0 + whitelist).
- **Pendiente sesión 3:** integración `firewall-acl-agent` + AppArmor profile.
- **Makefile:** `argus-network-isolate-build`, `argus-network-isolate-install`, `argus-network-isolate-test`, `argus-network-isolate-clean`.

### EMECAS reproducibility — argus-network-isolate en pipeline-build + provision
- **Status:** ✅ CERRADO DAY 142 — **Commit:** `e3f5f9c4`
- **Fix:** Vagrantfile: `nftables` declarado explícitamente. `provision.sh`: instala `isolate.json` en `/etc/ml-defender/firewall-acl-agent/` + crea `/var/log/argus/`. Makefile: `argus-network-isolate-build` + `argus-network-isolate-install` en `pipeline-build`. `check-system-deps` verifica nftables + binario instalado.

### DEBT-VARIANT-B-BUFFER-SIZE-001 — pcap_create()+pcap_set_buffer_size()
- **Status:** ✅ CERRADO DAY 142 — **Commit:** `7c4dba58`
- **Fix:** `PcapBackend::open()` refactorizado de `pcap_open_live()` a `pcap_create()+pcap_set_buffer_size()+pcap_activate()`. `buffer_size_mb` del JSON ahora se aplica realmente. `CaptureBackend` interfaz actualizada con el parámetro. Crítico en ARM64/RPi donde el kernel default es 2MB vs 8MB configurado.
- **Test de cierre:** `[pcap] Variant B opened on eth1 buffer=8MB` verificado. `make test-all` sin regresión.

### DEBT-VARIANT-B-MUTEX-001 — exclusión mutua Variant A/B (Nivel 1)
- **Status:** ✅ CERRADO DAY 142 Nivel 1 — **Commit:** `9458a90d`
- **Fix:** `scripts/check-sniffer-mutex.sh` via sesiones tmux. Detecta si hay variant activa antes de arrancar otra. Variant A session: `sniffer`. Variant B session: `sniffer-libpcap`. Conflicto detectado → detiene variant activa + exit 1. Makefile: `sniffer-start` y `sniffer-libpcap-start` llaman al mutex. Nuevo target `sniffer-libpcap-start`.
- **NOTA:** Nivel 1 provisional. Ver `DEBT-MUTEX-ROBUST-001` post-FEDER.
- **Test de cierre:** Variant B activa + intento Variant A → violación detectada, Variant B detenida, exit 1. ✅

---

## ✅ CERRADO DAY 141

### Bug Makefile — dependencia seed-client-build implícita
- **Status:** ✅ CERRADO DAY 141 — **Commit:** `63a37d9d`

### DEBT-PCAP-CALLBACK-LIFETIME-DOC-001 — Contrato lifetime PcapCallbackData
- **Status:** ✅ CERRADO DAY 141 — **Commit:** `63a37d9d`

### DEBT-VARIANT-B-CONFIG-001 — JSON propio sniffer-libpcap + config-driven main
- **Status:** ✅ CERRADO DAY 141
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

### DEBT-IRP-NFTABLES-001 — sesión 3/3 pendiente
**Severidad:** 🔴 Alta — P0 pre-FEDER
**Estado:** 🟡 60% — sesiones 1/3 y 2/3 CERRADAS — DAY 142
**Componente:** `firewall-acl-agent` + `tools/argus-network-isolate/` + AppArmor

Pasos 1-6 implementados y verificados en dev VM. Pendiente sesión 3:
1. Añadir a `isolate.json`: `auto_isolate` (default true), `threat_score_threshold` (0.95), `auto_isolate_event_types` (ransomware, lateral_movement, c2_beacon).
2. En `firewall-acl-agent`: detectar umbral + tipo superado → `fork()+execv()` a `argus-network-isolate isolate --interface <iface>`.
3. Test integración: evento sintético score >= 0.95 + tipo correcto → aislamiento automático.
4. AppArmor profile `enforce` para `argus-network-isolate` (combinar perfiles Gemini + Kimi DAY 142).
5. Instalar binario en `provision.sh` para hardened VM.

**Decisiones de diseño aprobadas (Consejo 8/8 + founder DAY 142):**
- `auto_isolate: true` por defecto — instalar y funcionar.
- Criterio disparo: `threat_score >= 0.95 AND event_type IN (ransomware, lateral_movement, c2_beacon)` — señal multi-componente, nunca umbral único.
- `fork()+execv()` — el firewall-acl-agent nunca muere.
- AppArmor `enforce` desde el primer deploy.
- Rollback actual (eliminar solo `argus_isolate`) suficiente para FEDER.

**ADR relacionado:** ADR-042 IRP
**Estimación:** 1 sesión (sesión 3/3)

---

### DEBT-ETCD-HA-QUORUM-001 — etcd-server en HA con quorum
**Severidad:** 🔴 Alta — P0 post-FEDER (OBLIGATORIO, no opcional)
**Estado:** ABIERTO — DAY 142
**Componente:** `etcd-server/` — arquitectura multi-nodo

etcd-server actual es single-node. Si cae, ningún componente puede registrarse ni coordinarse — y ningún mecanismo de mutex entre componentes puede ser robusto. Diseño requerido:
- Múltiples instancias etcd-server con quorum (Raft o equivalente).
- Componentes se registran ante el primer etcd disponible al arrancar.
- Al recuperarse un nodo etcd caído, se une al quorum y sincroniza estado.
- Quorum garantiza que todos los componentes registrados y vivos compartan el mismo estado.
- Líder elegido — si cae, quorum inmediato para elegir nuevo líder.
- Nuevo etcd que llega se une al quorum y, cuando le toque, es el nuevo líder.

**Nota:** No es deuda "eterna" — es deuda crítica que hay que cerrar. Es prerequisito de `DEBT-MUTEX-ROBUST-001` y de cualquier coordinación fiable entre componentes en producción.

**Test de cierre:** `make hardened-full` con 3 instancias etcd. Kill del líder → quorum en < 5s → componentes siguen operativos → nuevo líder elegido.
**Estimación:** 3-4 sesiones post-FEDER

---

### DEBT-MUTEX-ROBUST-001 — Mutex robusto entre variantes sniffer
**Severidad:** 🟡 P1 post-FEDER
**Estado:** ABIERTO — DAY 142 (Nivel 1 via tmux cerrado)
**Componente:** `scripts/check-sniffer-mutex.sh` + coordinación etcd

La implementación actual via sesiones tmux (Nivel 1) es provisional. No es robusta en producción — depende de una herramienta de usuario, no de un mecanismo de coordinación del sistema. Alternativas a evaluar para Nivel 2: `flock` (lockfile), PID file en `/var/run/argus/`, o coordinación via etcd cuando esté en HA (`DEBT-ETCD-HA-QUORUM-001`). La solución definitiva no puede depender de una única fuente de verdad que pueda caer.

**Test de cierre:** exclusión mutua funciona incluso si tmux no está disponible o etcd está caído.
**Estimación:** 1 sesión post-FEDER (tras DEBT-ETCD-HA-QUORUM-001)

---

### DEBT-IRP-MULTI-SIGNAL-001 — Criterio de disparo multi-señal IRP
**Severidad:** 🟡 P1 post-FEDER
**Estado:** ABIERTO — DAY 142
**Componente:** `firewall-acl-agent` + `isolate.json`

Para FEDER: dos condiciones AND mínimas (score + event_type). Para producción hospitalaria real: señal más rica. Contexto: monitores de quirófano, bombas de infusión y ventiladores mecánicos pueden estar en la intranet/DMZ del hospital — `firewall-acl-agent` en esos nodos tiene sentido. Un falso positivo que aísle un equipo médico es inaceptable. El criterio de disparo debe ser explicable, auditable y resistente a falsos positivos transitorios.

**Diseño futuro (IDEA-IRP-DECISION-MATRIX-001):** matriz de decisión con score + tipo + ventana temporal + potencialmente whitelist de dispositivos críticos.

**Nota sobre Platt scaling:** Qwen (Consejo DAY 142) advierte que sin calibración del score (Platt scaling o isotonic regression), el valor 0.95 no tiene significado estadístico real. Registrar como sub-tarea de DEBT-ADR040-002.

**Estimación:** 2 sesiones post-FEDER

---

### DEBT-IRP-LAST-KNOWN-GOOD-001 — Rollback con estado persistente
**Severidad:** 🟢 Baja post-FEDER
**Estado:** ABIERTO — DAY 142
**Componente:** `tools/argus-network-isolate/isolate.cpp`

El rollback actual elimina solo la tabla `argus_isolate` — correcto y suficiente para FEDER. En entornos con rulesets nftables propios del cliente (hospitales con segmentación VLAN, QoS, reglas personalizadas), el rollback podría dejar el sistema en estado inconsistente. Solución: `/etc/ml-defender/firewall-acl-agent/last-known-good.nft` actualizado periódicamente, firmado Ed25519. Restauración selectiva en rollback.

**Estimación:** 1 sesión post-FEDER

---

### DEBT-IRP-QUEUE-PROCESSOR-001
**Severidad:** 🔴 Alta — post-merge
**Estado:** ABIERTO — DAY 136
**Componente:** ADR-042 IRP
**Descripción:** Cola irp-queue sin límites ni procesador systemd dedicado.
**Estimación:** 1 sesión (junto a IRP-NFTABLES sesión 3)

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
| DEBT-ADR040-002 | confidence_score ∈ [0,1] en salida ZeroMQ + Platt scaling | v1.0 |
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

### BACKLOG-BENCHMARK-CAPACITY-001
**Estado:** ⏳ BACKLOG | **Prioridad:** P1 — FEDER Year 1 Deliverable
**Bloqueado por:** BACKLOG-ZMQ-TUNING-001 + hardware físico

### BACKLOG-BUILD-WARNING-CLASSIFIER-001
**Estado:** ⏳ BACKLOG | **Prioridad:** Post-FEDER
**Decisión Consejo DAY 141:** script grep/awk determinista. Workaround actual: `grep 'warning:' output.md | grep -v 'defender:'`

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
- [x] DEBT-VARIANT-B-BUFFER-SIZE-001 implementada ✅ DAY 142
- [ ] ADR-030 Variant B (ARM64) estable
- [ ] DEBT-IRP-NFTABLES-001 sesión 3/3 — integración firewall-acl-agent
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
| **Exclusión mutua Variant A/B** | Nunca simultáneas en el mismo hardware. Nivel 1: script bash via tmux (pre-FEDER). Nivel 2: robusto post-FEDER (DEBT-MUTEX-ROBUST-001). Lógica NO en binarios. | Consejo 8/8 · DAY 141-142 |
| **buffer_size_mb variable por diseño** | Permite trazar curva de optimización. pcap_create()+pcap_set_buffer_size() implementado DAY 142. | Consejo 8/8 · DAY 141 → Cerrado DAY 142 |
| **Warning classifier: grep/awk** | Script determinista. Un LLM no determinista no hace trabajo determinista. | Consejo 8/8 · DAY 141 |
| **auto_isolate: true por defecto** | El sistema protege sin configuración manual. Desactivar es acto explícito. | Consejo 8/8 + founder · DAY 142 |
| **IRP criterio multi-señal** | score >= 0.95 solo no es suficiente. FEDER: score AND event_type. Producción: señal más rica. | Consejo 8/8 + founder · DAY 142 |
| **fork()+execv() en IRP** | firewall-acl-agent nunca muere al disparar aislamiento. Operación atómica. | Consejo 8/8 · DAY 142 |
| **AppArmor enforce desde primer deploy** | Nuevos componentes: enforce desde el commit inicial. complain máximo 1 día en dev. | Consejo 8/8 · DAY 142 |
| **etcd-server HA es deuda crítica** | Single-node etcd no es robusta. DEBT-ETCD-HA-QUORUM-001 obligatoria post-FEDER. | Founder · DAY 142 |

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
DEBT-VARIANT-B-BUFFER-SIZE-001:        100% ✅  DAY 142 (commit 7c4dba58)
DEBT-VARIANT-B-MUTEX-001 (Nivel 1):    100% ✅  DAY 142 (commit 9458a90d)
DEBT-IRP-NFTABLES-001:                  60% 🟡  DAY 142 sesiones 1-2/3 (sesión 3 pendiente)
DEBT-ETCD-HA-QUORUM-001:                0% ⏳  P0 post-FEDER (OBLIGATORIO)
DEBT-MUTEX-ROBUST-001:                   0% ⏳  post-FEDER (tras HA etcd)
DEBT-IRP-MULTI-SIGNAL-001:              0% ⏳  post-FEDER
DEBT-IRP-LAST-KNOWN-GOOD-001:           0% ⏳  post-FEDER
DEBT-IRP-QUEUE-PROCESSOR-001:           0% ⏳  post-merge
BACKLOG-ZMQ-TUNING-001:                  0% ⏳  pre-FEDER
BACKLOG-BENCHMARK-CAPACITY-001:           0% ⏳  FEDER Year 1 Deliverable
BACKLOG-BUILD-WARNING-CLASSIFIER-001:    0% ⏳  post-FEDER (grep/awk script)
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

## 📝 Notas del Consejo de Sabios — DAY 142 (8/8)

> "DAY 142 — Seis commits. Tres DEBTs cerradas. El IRP pasa de arquitectura a sistema ejecutable y verificable.
>
> P1 (8/8 + founder): Umbral único `score >= 0.95` para FEDER, pero nunca como señal única. Mínimo dos condiciones AND: score + event_type. En entornos hospitalarios, equipos médicos conectados a intranet/DMZ (monitores de quirófano, bombas de infusión) son activos que `firewall-acl-agent` debe proteger — un falso positivo que los aísle es inaceptable. La señal debe ser explicable, auditable y multi-componente. Platt scaling registrado como sub-tarea de DEBT-ADR040-002.
>
> P2 (8/8): `fork()+execv()` obligatorio. El firewall-acl-agent nunca puede morir durante un incidente. Es el único componente que puede registrar evidencia y ejecutar rollback. `FD_CLOEXEC` en descriptores heredados. `prctl(PR_SET_PDEATHSIG, SIGTERM)` en el hijo.
>
> P3 (8/8): AppArmor `enforce` desde el primer deploy. Perfiles aportados por Gemini y Kimi para combinar en sesión 3.
>
> P4 (8/8): Diseño actual de rollback correcto para FEDER. `DEBT-IRP-LAST-KNOWN-GOOD-001` registrada post-FEDER.
>
> Founder: el mutex via tmux es provisional — `DEBT-MUTEX-ROBUST-001` post-FEDER. La raíz del problema es etcd single-node — `DEBT-ETCD-HA-QUORUM-001` es deuda crítica obligatoria, no opcional. Un sistema de coordinación que depende de una única fuente de verdad que puede caer no es robusto en producción hospitalaria.
>
> 'auto_isolate: true por defecto. Instalar y funcionar. Un hospital que no toca la configuración debe estar protegido.' — Founder
>
> 'El agente de firewall debe sobrevivir al aislamiento. Un agente muerto durante un ataque activo es exactamente lo que el atacante busca.' — Claude, Grok, DeepSeek, Gemini, Kimi, Mistral, Qwen, ChatGPT (8/8)"
> — Consejo de Sabios (8/8) · DAY 142

---

## 📝 Notas del Consejo de Sabios — DAY 141 (8/8)

> "DAY 141 — Bug Makefile seed-client-build cerrado. DEBT-PCAP-CALLBACK-LIFETIME-DOC-001 cerrado. DEBT-VARIANT-B-CONFIG-001 cerrado — sniffer-libpcap.json propio + main_libpcap.cpp config-driven. 9/9 tests PASSED. 0 warnings. Emails FEDER enviados a Andrés Caro Lindo.
>
> Q1 (8/8 + founder): Exclusión mutua obligatoria. DEBT-VARIANT-B-MUTEX-001 registrada. Nivel 1 via script bash/python en Makefile, pre-FEDER. La lógica de detección NO entra en los binarios.
>
> Q2 (8/8 + founder): buffer_size_mb pre-FEDER obligatorio. Variable por diseño — script de barrido paramétrico para trazar curva de optimización.
>
> Q3 (8/8 + founder): Script grep/awk determinista para clasificar warnings de build.
>
> 'buffer_size_mb no es una opción de confort — es una variable experimental. Sin ella, el benchmark ARM64 mide el default del kernel, no el hardware.' — Claude"
> — Consejo de Sabios (8/8) · DAY 141

---

## 📝 Notas del Consejo de Sabios — DAY 140 (8/8)

> "DAY 140 — 192 → 0 warnings. `-Werror` activo como invariante permanente. ODR limpio con LTO."
> — Consejo de Sabios (8/8) · DAY 140

---

## 📝 Notas del Consejo de Sabios — DAY 138 (8/8)

> "DAY 138 — ISP cerrado. Pipeline Variant B completo. ODR P0 bloqueante confirmado."
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

*DAY 142 — 5 Mayo 2026 · feature/variant-b-libpcap @ 9458a90d*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*