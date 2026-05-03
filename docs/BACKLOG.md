# aRGus NDR — BACKLOG
*Última actualización: DAY 140 — 3 Mayo 2026*

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

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa | x86-debug, imagen Vagrant completa. Para desarrollo diario. |
| **aRGus-production** | 🟡 En construcción | x86-apparmor + arm64-apparmor. Debian optimizado. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ No iniciada | Apéndice científico. Kernel seL4, libpcap. Branch independiente. |

---

## ✅ CERRADO DAY 138

### DEBT-CAPTURE-BACKEND-ISP-001 — CaptureBackend interfaz mínima (ISP)
- **Status:** ✅ CERRADO DAY 138 — **Commit:** `1a7f723a`
- **Fix:** `CaptureBackend` refactorizada a 5 métodos puros (open/poll/close/get_fd/get_packet_count). Los 7 métodos eBPF-específicos (attach_skb, detach_skb, get_ringbuf_fd, 4× filter map fds) movidos a `EbpfBackend` como métodos públicos no-virtuales. `main.cpp` usa `EbpfBackend` directamente sin downcast.
- **Test de cierre:** `make sniffer && make sniffer-libpcap` compilando sin warnings nuevos. ✅

### DEBT-VARIANT-B-PCAP-IMPL-001 — Pipeline completo libpcap
- **Status:** ✅ CERRADO DAY 138 — **Commits:** `22df0099` + `da1badf7`
- **Fix:** Pipeline completo `pcap_dispatch(64 pkts) → ETH/IP/TCP/UDP parse → NetworkSecurityEvent proto → LZ4 → ChaCha20-Poly1305 → ZMQ PUSH tcp://127.0.0.1:5571`. Wire format idéntico a Variant A. Mismo SeedClient + CTX_SNIFFER_TO_ML. Mecanismo `PcapCallbackData{cb, ctx}` como `u_char* user`.
- **Suite 8 tests — 8/8 PASSED en make test-all:** lifecycle, poll_null, callback, error_handling, proto_parse_tcp, proto_parse_udp, stress (10K callbacks), regression.

---

## ✅ CERRADO DAY 134

### Pipeline E2E en hardened VM — check-prod-all PASSED
- **Status:** ✅ CERRADO DAY 134
- **Commits:** `f256e6f0` + `2e9a5b39`

### DEBT-KERNEL-COMPAT-001
- **Status:** ✅ CERRADO DAY 134 — **Commit:** `2e9a5b39`

### DEBT-PAPER-FUZZING-METRICS-001
- **Status:** ✅ CERRADO DAY 134

### ADR-040 + ADR-041 — Integración en BACKLOG + README
- **Status:** ✅ CERRADO DAY 134 — **Commit:** `87680d83`

---

## ✅ CERRADO DAY 133

### Paper Draft v18 — §6.12 métricas BSR reales
- **Status:** ✅ CERRADO DAY 133 — **Commit:** `c6e0c9f1` + post-Consejo

### Paper Draft v18 — §6.8 reformulación fuzzing (post-Consejo DAY 133)
- **Status:** ✅ CERRADO DAY 133

### DEBT-PROD-APPARMOR-COMPILER-BLOCK-001
- **Status:** ✅ CERRADO DAY 133 — **Commit:** `c6e0c9f1` + post-Consejo

### DEBT-PROD-FALCO-EXOTIC-PATHS-001
- **Status:** ✅ CERRADO DAY 133 — **Commit:** `c6e0c9f1` + post-Consejo

### DEBT-PROD-FS-MINIMIZATION-001 (parcial)
- **Status:** ✅ CERRADO DAY 133 (parcial — minbase es deuda futura)

### Makefile — targets de producción ADR-030 Variant A
- **Status:** ✅ CERRADO DAY 133 — **Commit:** `c6e0c9f1`

### Linux Capabilities — setcap mínimo (post-Consejo DAY 133)
- **Status:** ✅ CERRADO DAY 133

---

## ✅ CERRADO DAY 132

### DEBT-PROD-COMPAT-BASELINE-001
- **Status:** ✅ CERRADO DAY 132 — **Commit:** `9b3438fb`

### vagrant/hardened-x86/Vagrantfile — ADR-030 Variant A
- **Status:** ✅ COMPLETADO DAY 133 — **Commit:** `c6e0c9f1`

### Paper Draft v17 → v18
- **Status:** ✅ Draft v18 COMPLETADO DAY 133

### README — Prerequisites
- **Status:** ✅ CERRADO DAY 132 — **Commit:** `18d8e101` en `main`

---

## ✅ CERRADO DAY 130

### DEBT-SYSTEMD-AUTOINSTALL-001
- **Status:** ✅ CERRADO DAY 130 — **Commit:** `8e57aad2`

### DEBT-SAFE-EXEC-NULLBYTE-001
- **Status:** ✅ CERRADO DAY 130 — 17/17 GREEN — **Commit:** `c8e293a8`

### DEBT-GITGUARDIAN-YAML-001
- **Status:** ✅ CERRADO DAY 130 — **Commit:** `06228a67`

### DEBT-FUZZING-LIBFUZZER-001
- **Status:** ✅ CERRADO DAY 130 (baseline) — 2.4M runs, 0 crashes — **Commit:** `f5994c4a`

### DEBT-MARKDOWN-HOOK-001
- **Status:** ✅ CERRADO DAY 130 — **Commit:** `aab08daa`

### REGLA EMECAS — Keypair activo post-rebuild
`b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
Pipeline 6/6 RUNNING · TEST-INTEG-SIGN 7/7 PASSED · ALL TESTS COMPLETE

---

## ✅ CERRADO DAY 124–129

DAY 124: ADR-037 safe_path → v0.5.1-hardened
DAY 125-126: 8 deudas cerradas · lstat() pre-resolution · prefix fijo · v0.5.2-hardened
DAY 127: resolve_config() · dev/prod parity · taxonomía safe_path
DAY 128: Snyk 18 findings triados · 5 property tests · provision portability
DAY 129: CWE-78 CERRADO · EtcdClientHmac 9/9 · FEDER scope

---

## 🔴 DEUDAS ABIERTAS — Seguridad y arquitectura

### DEBT-COMPILER-WARNINGS-CLEANUP-001
**Severidad:** 🔴 Alta — ODR es P0 bloqueante (Consejo DAY 138: 8/8 unánime)
**Estado:** ABIERTO — DAY 136 (elevado a 🔴 DAY 138)
**Componente:** ml-detector, sniffer, rag-ingester, etcd-client

Sub-tareas en orden estricto:
1. [P0 BLOQUEANTE] ODR violations: `InternalNode` vs `TrafficNode` — UB en C++20. Diagnóstico: `nm -C build/ml-detector | grep InternalNode | c++filt`. Fix: unificar flags CMake, añadir `-Werror=odr`.
2. [P1] Protobuf dual-copy en ml-detector — `.pb.cc` compilado dos veces.
3. [P2] signed/unsigned en zmq_handler.cpp, rag_logger.cpp, feature_extractor.cpp.
4. [P3] OpenSSL SHA256_Init → EVP_DigestInit_ex.
5. [P4] -Wreorder en ring_consumer.hpp, zmq_handler.hpp, dual_nic_manager.hpp.
6. [P5] -Wunused-parameter en ml_defender_features.cpp stubs.

**Test de cierre:** `make hardened-full 2>&1 | grep -E "^.*warning:" | wc -l` = 0
**Estimación:** 3 sesiones. ODR sub-tarea P0 bloqueante para cualquier tag posterior.

---

### DEBT-VARIANT-B-CONFIG-001
**Severidad:** 🔴 Alta — pre-demo FEDER
**Estado:** ABIERTO — DAY 138
**Componente:** sniffer / main_libpcap.cpp, sniffer-libpcap.json (pendiente crear)

`sniffer-libpcap` tiene endpoint ZMQ, seed path e interface hardcodeados.
Necesita JSON propio simplificado. Campos multihilo DESAPARECEN del JSON
(hardcodeados en binario con comentario explícito — Variant B es monohilo
por diseño de libpcap, no configurable):
- Eliminar: `zmq_sender_threads`, `io_thread_pools`, `ring_buffer.*`, `threading.*`, `numa_node`, `cpu_affinity`, `batch_size`, `flush_interval`, `capture.snaplen` (→65535), `capture.promiscuous` (→1)
- Preservar: `capture.interface`, `capture.filter.*`, `capture.timeout_ms`, `output_socket.*`, `crypto.*`, `logging.*`
- Añadir: `capture.buffer_size_mb` (crítico en ARM64), `capture.sampling.mode/rate`
- Observabilidad: exponer `send_failures` como métrica. Alerta si drop_rate > 0.1%.
- Tests: e2e con `pcap_open_dead()` + inject (CI, sin root). `tcpreplay` manual (REQUIRES_ROOT).
- Validación en hardened-arm64 VM.

**Consejo DAY 138:** Ver actas Q2, Q3, Q4
**Estimación:** 1 sesión

---

### DEBT-IRP-NFTABLES-001
**Severidad:** 🔴 Alta — post-merge, pre-demo FEDER
**Estado:** ABIERTO — DAY 136 (protocolo aprobado DAY 138)
**Componente:** /usr/local/bin/argus-network-isolate (pendiente implementar)

ExecStartPre de argus-apt-integrity.service referencia argus-network-isolate
pero el script no está implementado. Protocolo aprobado por Consejo DAY 138 (8/8):
1. Snapshot: `nft list ruleset > /tmp/argus-backup-$$.nft`
2. Generar fichero de reglas de aislamiento
3. Validar: `nft -c -f /tmp/argus-isolate-$$.nft`
4. Aplicar atómico: `nft -f /tmp/argus-isolate-$$.nft`
5. Timer rollback automático 300s (si admin no confirma, restaurar)
6. Fallback emergencia: `ip link set eth0 down`

iptables rechazado — obsoleto en Debian 12 Bookworm.
**ADR relacionado:** ADR-042 IRP enmienda E1
**Estimación:** 2 sesiones

---

### DEBT-IRP-QUEUE-PROCESSOR-001
**Severidad:** 🔴 Alta — post-merge
**Estado:** ABIERTO — DAY 136
**Componente:** ADR-042 IRP
**Descripción:** La cola irp-queue no tiene límites de tamaño ni procesador
systemd dedicado. Requiere unidad systemd irp-queue-processor con límites
explícitos (enmienda E3 ADR-042).

---

### DEBT-PCAP-CALLBACK-LIFETIME-DOC-001
**Severidad:** 🟢 Baja — documentación
**Estado:** ABIERTO — DAY 138
**Componente:** sniffer/include/pcap_backend.hpp

Añadir comentario de contrato de lifetime de `PcapCallbackData`:
- Válido durante toda la sesión de captura
- No destruir PcapBackend durante pcap_dispatch() activo
- Señalización asíncrona no soportada (requeriría weak_ptr refactoring)
  **Estimación:** 10 minutos

---

### DEBT-EMECAS-AUTOMATION-001
**Severidad:** 🟡 Media — calidad de proceso y reproducibilidad FEDER
**Estado:** ABIERTO — DAY 140
**Componente:** Makefile raíz + directorio logs/

Targets `make emecas-dev`, `make emecas-prod-x86`, `make emecas-prod-arm64` con log automático fechado en `logs/emecas-<variant>-YYYYMMDD-HHMMSS.log`. Los logs son artefactos de reproducibilidad demostrables ante la comisión evaluadora FEDER. Resumen PASSED/FAILED + duración en última línea.
**Test de cierre:** `make emecas-dev && ls logs/emecas-dev-*.log | xargs tail -1` muestra `RESULT: PASSED`
**Estimación:** 1 sesión

---

### DEBT-JENKINS-SEED-DISTRIBUTION-001
**Severidad:** 🔴 Alta
**Estado:** ABIERTO — DAY 136
**Descripción:** El Mac del founder no puede ser parte de la cadena criptográfica
de producción hospitalaria. Jenkins gestiona: generación local en nodo hardened
→ distribución out-of-band → verificación permisos.
**Test de cierre:** `make ci-deploy-seeds` sin intervención del portátil del founder.

---

### DEBT-CRYPTO-MATERIAL-STORAGE-001
**Severidad:** 🔴 Alta
**Estado:** ABIERTO — DAY 136
**Descripción:** Seeds + keypairs sin solución de almacenamiento robusta.
Propuesta demo: HashiCorp Vault OSS (Vagrant-deployable). Objetivo final: TPM 2.0.
**Test de cierre:** `make vault-init && make vault-deploy-seeds` → check-prod-permissions PASSED.

---

### DEBT-PROD-APT-SOURCES-INTEGRITY-001
**Severidad:** 🔴 Crítica | **Bloqueante:** Sí | **Target:** feature/adr030-variant-a

SHA-256 de `sources.list` firmado en imagen. Si cambia → fail-closed.
AppArmor deny de escritura en `/etc/apt/`. Falco alerta si cualquier proceso escribe.
**Test de cierre:** Modificar `sources.list` en hardened VM → pipeline no arranca.

---

### DEBT-SEEDS-SECURE-TRANSFER-001
**Severidad:** 🔴 Alta — mitigado en Vagrant, inaceptable en producción real
**Corrección:** post-FEDER (protocolo out-of-band)

### DEBT-SEEDS-LOCAL-GEN-001
**Severidad:** 🔴 Alta
**Corrección:** post-FEDER

### DEBT-SEEDS-BACKUP-001
**Severidad:** 🔴 Alta
**Corrección:** post-FEDER

### DEBT-KEY-SEPARATION-001
**Severidad:** 🟡 Media | **Target:** post-FEDER
Separar keypairs Ed25519: pipeline-signing vs plugin-signing. Mismo keypair actualmente.

### DEBT-DEBIAN13-UPGRADE-001
**Severidad:** 🟡 Media | **Target:** post-FEDER

### DEBT-PROD-APPARMOR-PORTS-001
**Severidad:** 🟢 Baja | **Target:** post-JSON-estabilización

### DEBT-PROD-FALCO-RULES-EXTENDED-001
**Severidad:** 🟡 Media | **Target:** DAY 135+

### DEBT-APT-TIMEOUT-CONFIG-001
**Severidad:** 🟡 Media | **Target:** post-FEDER

### DEBT-FEDER-DEMO-SCRIPT-001
**Severidad:** 🟡 Media | **Target:** DAY 136+

### DEBT-CHECK-PROD-SEED-CONDITIONAL-001
**Severidad:** 🟡 Media | **Target:** post-merge

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


---

## 📋 BACKLOG — Benchmarks Empíricos (FEDER Year 1)

### BACKLOG-ZMQ-TUNING-001 — Optimización Empírica de Parámetros ZeroMQ y Pipeline
**Estado:** ⏳ BACKLOG
**Prioridad:** P1 — Prerequisito de BACKLOG-BENCHMARK-CAPACITY-001
**Bloqueado por:** ADR-029 Variant A estable + ADR-029 Variant B estable + DEBT-CAPTURE-BACKEND-ISP-001 ✅
**Estimación:** 2–4 días de sesión
**Documento:** `docs/adr/BACKLOG-ZMQ-TUNING-001.md`

Los parámetros ZeroMQ actuales (HWM, IO threads, batch size, linger, backpressure) fueron fijados bajo criterio de corrección, no de rendimiento. Los números del abstract de arXiv:2604.04952 (Draft v18) corresponden a una configuración no optimizada. Este experimento debe concluir **antes** de BACKLOG-BENCHMARK-CAPACITY-001 — de lo contrario el capacity benchmark mediría una mezcla de rendimiento real del backend y penalización artificial del tuning ZeroMQ.

**Prerequisitos:**
- [ ] ADR-029 Variant A (`EbpfBackend`) estable y mergeada a main
- [ ] ADR-029 Variant B (`PcapBackend`) estable y mergeada a main
- [x] DEBT-CAPTURE-BACKEND-ISP-001 ✅ cerrado DAY 138
- [ ] Volcado de valores JSON actuales de todo el pipeline (primer paso de la sesión)

**Test de cierre:** Tabla de parámetros optimizados por perfil de despliegue (x86 alto rendimiento / ARM64 recursos limitados) + curvas de sensibilidad por parámetro + JSONs de configuración actualizados con valores justificados + corrección del abstract arXiv post-experimentación.

---

### BACKLOG-BENCHMARK-CAPACITY-001 — Empirical Capacity Benchmark: eBPF vs libpcap vs ARM64
**Estado:** ⏳ BACKLOG
**Prioridad:** P1 — FEDER Year 1 Deliverable obligatorio
**Bloqueado por:** BACKLOG-ZMQ-TUNING-001 + ADR-029 Variant A estable + ADR-029 Variant B estable
**Estimación:** 3–5 días de sesión
**Documento:** `docs/adr/BACKLOG-BENCHMARK-CAPACITY-001.md`

Benchmark comparativo de cuatro configuraciones: **BM-A** (x86-64 eBPF/XDP high-end), **BM-B** (x86-64 libpcap high-end — control crítico para aislar coste de backend), **BM-C** (ARM64 libpcap / RPi5 — despliegue con recursos limitados), **BM-D** (x86-64 eBPF/XDP low-power / N100). Cuantifica el gap técnico entre organizaciones con más y menos recursos. Justifica directamente la frase central del prospecto FEDER: *"We need a server to know what server we need in production."* Incluye exploración multi-SBC ARM64 si BM-C muestra gap inaceptable.

**Prerequisitos:**
- [ ] BACKLOG-ZMQ-TUNING-001 concluido
- [ ] ADR-029 Variant A (`EbpfBackend`) mergeada a main
- [ ] ADR-029 Variant B (`PcapBackend`) mergeada a main
- [x] DEBT-CAPTURE-BACKEND-ISP-001 ✅ cerrado DAY 138
- [ ] Vagrantfile ARM64 operativo y reproducible
- [ ] pcap de tráfico mixto preparado y versionado en repositorio
- [ ] Hardware Fase 2: RPi5 (×1–3, ~80€/u) + Intel N100 board (~100–180€) adquiridos

**Test de cierre:** Tabla de saturation points por configuración y tasa de inyección + curvas drop rate vs Mbps + delta eBPF vs libpcap (hardware constante) + delta ARM64 vs x86 (backend constante) + recomendaciones de hardware mínimo por tipología de despliegue hospitalario.


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
**Contacto:** Andrés Caro Lindo — UEx/INCIBE
**Deadline límite:** 22 septiembre 2026 | **Go/no-go técnico:** 1 agosto 2026

### Gate de entrada

- [x] ADR-026 mergeado a main (XGBoost F1=0.9978)
- [x] ADR-030 Variant A infraestructura completa (DAY 133)
- [x] Pipeline E2E en hardened VM verde (`make check-prod-all`) — DAY 134 ✅
- [ ] ADR-030 Variant B (ARM64) estable
- [ ] Demo técnica grabable < 10 minutos (`scripts/feder-demo.sh`)
- [ ] ADR-041 protocolo hardware: métricas validadas en x86 + ARM (`make feder-demo`)
- [ ] Golden set v1 creado y versionado (DEBT-ADR040-001)
- [ ] BACKLOG-ZMQ-TUNING-001 concluido (parámetros ZMQ optimizados — inputs para benchmark)
- [ ] BACKLOG-BENCHMARK-CAPACITY-001 concluido (Empirical Capacity Benchmark — FEDER Year 1 Deliverable)
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
| **seL4 no diseñar ahora** | CaptureBackend (5 métodos) es reutilizable. Todo lo demás reescritura. YAGNI hasta equipo especializado. | Consejo 8/8 · DAY 138 |

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
DEBT-COMPILER-WARNINGS-CLEANUP-001:      60% 🟡  DAY 139 en curso (192→67 warnings)
BACKLOG-ZMQ-TUNING-001:                  0% ⏳  pre-FEDER (prerequisito BENCHMARK-CAPACITY)
BACKLOG-BENCHMARK-CAPACITY-001:           0% ⏳  FEDER Year 1 Deliverable obligatorio
DEBT-VARIANT-B-CONFIG-001:               0% ⏳  pre-FEDER
DEBT-IRP-NFTABLES-001:                   0% ⏳  pre-FEDER
DEBT-IRP-QUEUE-PROCESSOR-001:            0% ⏳  post-merge
DEBT-PCAP-CALLBACK-LIFETIME-DOC-001:     0% ⏳  trivial
DEBT-EMECAS-AUTOMATION-001:              0% ⏳  post deudas P0
DEBT-JENKINS-SEED-DISTRIBUTION-001:      0% ⏳  pre-FEDER
DEBT-CRYPTO-MATERIAL-STORAGE-001:        0% ⏳  pre-FEDER
DEBT-KEY-SEPARATION-001:                 0% ⏳  post-FEDER
DEBT-ADR040-001..012:                    0% ⏳  post-FEDER Año 1
DEBT-ADR041-001..006:                    0% ⏳  pre-FEDER
ADR-031 aRGus-seL4:                      0% ⏳  branch independiente
```

---

## 📝 Notas del Consejo de Sabios — DAY 138 (8/8)

> "DAY 138 — Dos deudas arquitectónicas cerradas. Dos nuevas registradas. El Consejo establece prioridades.
>
> Veredictos unánimes (8/8):
> Q1: PcapCallbackData lifetime seguro hoy. Documentar contrato. atomic<bool> opcional post-FEDER.
> Q2: dontwait correcto para NDR monohilo. Exponer send_failures como métrica. No backpressure.
> Q3: JSON propio simplificado. Campos multihilo hardcodeados en binario. buffer_size_mb necesario.
> Q4: pcap_open_dead() + inject para CI. tcpreplay para stress manual. Dos niveles.
> Q5: nft -f transaccional. Snapshot + rollback 300s. Fallback ip link down. iptables rechazado.
> Q6: ODR es P0 BLOQUEANTE. Ningún tag posterior sin resolver ODR primero. Sin discusión.
> Q7: No diseñar para seL4 ahora. CaptureBackend 5 métodos es reutilizable. Todo lo demás reescritura.
>
> Kimi (único con posición diferenciada en Q1): propone weak_ptr para blindaje futuro.
> Registrado como mejora opcional post-FEDER.
>
> 'Los warnings ODR en C++ son bombas de reloj. En infraestructura crítica,
>  el comportamiento indefinido no es hipotético — es inevitable.' — Grok
>
> 'Variant B monohilo ya es, de facto, mucho más cercana al modelo de
>  microkernel de seL4 que la Variant A multihilo.' — Gemini"
> — Consejo de Sabios (8/8) · DAY 138

---

## 📝 Notas del Consejo de Sabios — DAY 136 (8/8)

> "DAY 136 — v0.6.0-hardened-variant-a mergeado. El pipeline respira solo.
> D1: DEBT-IRP-NFTABLES-001 es P0 pre-FEDER. argus-network-isolate inexistente = fail catastrófico.
> D2: Jenkins para seeds — el Mac del founder no puede ser parte de la cadena criptográfica.
> D3: HashiCorp Vault (demo) + TPM 2.0 (objetivo final).
> D4: ODR violation en C++ es UB. UB en hospital es inaceptable.
> D5: DEBT-SEEDS-BACKUP-001 — el más preocupante para infraestructura crítica."
> — Consejo de Sabios (8/8) · DAY 136

---

## 📝 Notas del Consejo de Sabios — DAY 134 (8/8)

> "ADR-040 + ADR-041: contratos de calidad ML y métricas de aceptación hardware.
> Walk-forward obligatorio. Golden set inmutable. Guardrail asimétrico. IPW+uncertainty.
> Latencia end-to-end como métrica operacional primaria. Temperatura ARM ≤75°C gate no negociable."
> — Consejo de Sabios (8/8) · DAY 134

---

## 📝 Notas del Consejo de Sabios — DAY 133 (8/8)

> "DAY 133 — Transición de 'diseño correcto' a 'comportamiento real verificable'.
> cap_sys_admin → cap_bpf. cap_net_bind_service eliminada. LimitMEMLOCK=16M.
> 'Fuzzing misses nothing' — INCORRECTA, reformulada.
> 'Un escudo que no se prueba contra el ataque real es un escudo de teatro. Vosotros estáis construyendo acero.' — Qwen"
> — Consejo de Sabios (8/8) · DAY 133

---

## 🧬 HIPÓTESIS CENTRAL — Inmunidad Global Adaptativa

**Formulada:** DAY 128 | **Estado:** Pendiente demostración (DEBT-PENTESTER-LOOP-001)

Un sistema con ACRL converge hacia cobertura de técnicas ATT&CK en tiempo polinomial. Un sistema estático no converge nunca.

*DAY 139 — 2 Mayo 2026 · feature/variant-b-libpcap @ 91281005*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*