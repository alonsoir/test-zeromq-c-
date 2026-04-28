# aRGus NDR — BACKLOG
*Última actualización: DAY 133 — 27 Abril 2026*

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

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa | x86-debug, imagen Vagrant completa. Para desarrollo diario. |
| **aRGus-production** | 🟡 En construcción | x86-apparmor + arm64-apparmor. Debian optimizado. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ No iniciada | Apéndice científico. Kernel seL4, libpcap. Branch independiente. |

---

## ✅ CERRADO DAY 133

### Paper Draft v18 — §6.12 métricas BSR reales
- **Status:** ✅ CERRADO DAY 133
- **Fix:** Tabla BSR con métricas medidas: Dev VM (719 pkgs / 5.9 GB / compiladores) vs Hardened VM (304 pkgs / 1.3 GB / NONE). Reducción 58% paquetes, 78% disco. Nota al pie honesta sobre suelo Vagrant (~250 pkgs).
- **Commit:** `c6e0c9f1` + post-Consejo

### Paper Draft v18 — §6.8 reformulación fuzzing (post-Consejo DAY 133)
- **Status:** ✅ CERRADO DAY 133
- **Fix:** Eliminada "Fuzzing misses nothing within CPU time" (Consejo 8/8 unánime — científicamente incorrecta). Sustituida por formulación que reconoce naturaleza estocástica, guía por cobertura, sin garantía de completitud. Añadida `\cite{libfuzzer2016}`.
- **Commit:** post-Consejo DAY 133

### DEBT-PROD-APPARMOR-COMPILER-BLOCK-001
- **Status:** ✅ CERRADO DAY 133
- **Fix:** 6 perfiles AppArmor enforce en `security/apparmor/` (uno por componente). Default-deny con allowlists mínimas. `deny` explícitos mantenidos — claridad auditiva para auditores hospitalarios (decisión founder). Post-Consejo: `cap_sys_admin` → `cap_bpf` en sniffer.
- **Commit:** `c6e0c9f1` + post-Consejo

### DEBT-PROD-FALCO-EXOTIC-PATHS-001
- **Status:** ✅ CERRADO DAY 133
- **Fix:** Falco instalado en hardened VM con `modern_ebpf` driver. 10 reglas aRGus: 7 originales + 3 post-Consejo (`argus_config_modified_unexpected`, `argus_model_or_plugin_replaced`, `argus_apparmor_profile_modified`). Estrategia maduración 3 fases.
- **Commit:** `c6e0c9f1` + post-Consejo

### DEBT-PROD-FS-MINIMIZATION-001 (parcial)
- **Status:** ✅ CERRADO DAY 133 (parcial — minbase es deuda futura)
- **Fix:** Usuario `argus` (system, nologin, no home). `/opt/argus/` ownership estricto. `/tmp` y `/var/tmp` noexec,nosuid,nodev. `seed.bin` 0400. Sin SUID — capabilities via `setcap`. Gate `check-prod-permissions` automatizado.
- **Pendiente:** imagen minbase x86+ARM (DEBT-PROD-FS-MINIMIZATION-001 → post-FEDER)

### Makefile — targets de producción ADR-030 Variant A
- **Status:** ✅ CERRADO DAY 133
- **Fix:** `prod-build-x86`, `prod-collect-libs`, `prod-sign`, `prod-checksums`, `prod-deploy-x86`, `prod-full-x86`. Provisioning: `hardened-provision-all`. Gates: `check-prod-no-compiler` (dpkg+PATH), `check-prod-apparmor`, `check-prod-capabilities`, `check-prod-permissions`, `check-prod-falco`, `check-prod-all`. Scripts en `tools/prod/`.
- **Commit:** `c6e0c9f1`

### Linux Capabilities — setcap mínimo (post-Consejo DAY 133)
- **Status:** ✅ CERRADO DAY 133
- **Decisiones Consejo 8/8:** sniffer: `cap_bpf` reemplaza `cap_sys_admin` (Linux ≥5.8). etcd-server: `cap_net_bind_service` ELIMINADA (2379 > 1024). etcd-server: `cap_ipc_lock` + `LimitMEMLOCK=16M` en systemd. ml-detector, rag-ingester, rag-security: sin capabilities (no-root real).

---

## ✅ CERRADO DAY 132

### DEBT-PROD-COMPAT-BASELINE-001
- **Status:** ✅ CERRADO DAY 132
- **Fix:** `docs/HARDWARE-REQUIREMENTS.md` — especificaciones mínimas y recomendadas, compatibilidad XDP por driver NIC, paquetes runtime vs prohibidos en producción.
- **Commit:** `9b3438fb`

### vagrant/hardened-x86/Vagrantfile — ADR-030 Variant A
- **Status:** ✅ COMPLETADO DAY 133
- **Fix:** VM Debian 12 + AppArmor enforcing + sin compilador + BSR verificado. Makefile targets completos. AppArmor 6 perfiles. Falco 10 reglas.
- **Commit:** `c6e0c9f1`

### Paper Draft v17 → v18
- **Status:** ✅ Draft v18 COMPLETADO DAY 133
- **Fix:** v18 = v17 + tabla BSR métricas reales (§6.12) + reformulación fuzzing (§6.8 post-Consejo). Compilado Overleaf, 42 páginas.

### README — Prerequisites
- **Status:** ✅ CERRADO DAY 132
- **Commit:** `18d8e101` en `main`

---

## ✅ CERRADO DAY 130

### DEBT-SYSTEMD-AUTOINSTALL-001
- **Status:** ✅ CERRADO DAY 130 — **Commit:** `8e57aad2`

### DEBT-SAFE-EXEC-NULLBYTE-001
- **Status:** ✅ CERRADO DAY 130 — 17/17 GREEN — **Commit:** `c8e293a8`

### DEBT-GITGUARDIAN-YAML-001
- **Status:** ✅ CERRADO DAY 130 — **Commit:** `06228a67`

### DEBT-FUZZING-LIBFUZZER-001
- **Status:** ✅ CERRADO DAY 130 (baseline) — 2.4M runs, 0 crashes, corpus 67 ficheros — **Commit:** `f5994c4a`

### DEBT-MARKDOWN-HOOK-001
- **Status:** ✅ CERRADO DAY 130 — **Commit:** `aab08daa`

### REGLA EMECAS — Keypair activo post-rebuild DAY 133
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

## 🔴 DEUDA ABIERTA — Seguridad imagen de producción (ADR-030)

### DEBT-PROD-APT-SOURCES-INTEGRITY-001
**Severidad:** 🔴 Crítica | **Bloqueante:** Sí | **Target:** feature/adr030-variant-a

SHA-256 de `sources.list` firmado en imagen. Si cambia → fail-closed (default) o fail-warn configurable. AppArmor deny de escritura en `/etc/apt/`. Falco alerta si cualquier proceso escribe en `/etc/apt/`.

**Test de cierre:** Modificar `sources.list` en VM hardened → pipeline no arranca. Restaurar → arranca normalmente.

---

### DEBT-DEBIAN13-UPGRADE-001
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** post-FEDER

Documentar y validar upgrade path Debian 12 (bookworm) → Debian 13 (trixie) para bare-metal hospitalario.

---

### DEBT-PAPER-FUZZING-METRICS-001
**Severidad:** 🟡 Media | **Bloqueante:** Sí (pre-arXiv) | **Target:** DAY 134

Tabla completa de métricas §6.8 con datos reales DAY 130. Reformulación de la frase: CERRADA (post-Consejo DAY 133). Pendiente: recuperar métricas exactas de `validate_filepath` y `safe_exec`.

---

### DEBT-KEY-SEPARATION-001 *(nueva — DAY 133)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** post-FEDER
**Origen:** Consejo 8/8 DAY 133 — unánime

Separar keypairs Ed25519: `pipeline-signing.sk/pk` (binarios) vs `plugin-signing.sk/pk` (plugins). Actualmente mismo keypair para ambos dominios. Blast radius reducido. Rotación independiente.

**Fix:** Generar `pipeline-signing.sk/pk` en `provision.sh`. Documentar en `docs/SECURITY-KEY-MANAGEMENT.md`.

---

### DEBT-KERNEL-COMPAT-001 *(nueva — DAY 133)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** DAY 134
**Origen:** Consejo 8/8 DAY 133

Verificar que `cap_bpf` funciona correctamente para XDP en hardened VM (Debian bookworm kernel 6.1). `deploy-hardened.sh` tiene detección automática y fallback documentado.

**Test de cierre:** `ip link set dev eth1 xdp obj sniffer.bpf.o` exitoso con `cap_bpf+cap_net_admin` sin `cap_sys_admin`.

---

### DEBT-PROD-APPARMOR-PORTS-001 *(nueva — DAY 133)*
**Severidad:** 🟢 Baja | **Bloqueante:** No | **Target:** post-estabilización JSON
**Decisión founder:** No implementar hasta que los puertos sean fuente de verdad compartida entre JSON y perfil AA. "JSON es la ley."

---

### DEBT-PROD-FALCO-RULES-EXTENDED-001 *(nueva — DAY 133)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** DAY 135

Reglas Falco propuestas en Consejo DAY 133 no adoptadas: ptrace (Gemini), DNS tunneling (Kimi), conexiones salientes inesperadas (ChatGPT, Mistral), `/dev/mem` (Mistral), fork bombs (Mistral).

---


---

### DEBT-ADR040-001 a 012 — ML Plugin Retraining Contract *(nuevas — DAY 134)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** post-FEDER (implementación Año 1)
**Origen:** ADR-040 v2 — Consejo 8/8 DAY 134 (17 enmiendas, aprobado unánime)

| ID | Descripción | Target |
|----|-------------|--------|
| DEBT-ADR040-001 | Golden set v1 (≥50K flows, 70/30, Parquet, SHA-256 embebido en plugin) | v1.0 — pre-FEDER si posible |
| DEBT-ADR040-002 | Verificar que ml-detector emite `confidence_score ∈ [0,1]` en salida ZeroMQ | v1.0 |
| DEBT-ADR040-003 | `walk_forward_split.py` — `--split-field timestamp_first_packet`, mín. 3 ventanas, KS drift | v1.1 |
| DEBT-ADR040-004 | `check_guardrails.py` — Recall −0.5pp / F1 −2pp / FPR +1pp / latencia p99 +10% → exit 1 | v1.1 |
| DEBT-ADR040-005 | Integrar guardrail en proceso de firma Ed25519 (ADR-025) — `prod-sign` invoca guardrail | v1.1 |
| DEBT-ADR040-006 | IPW + uncertainty sampling (P≈0.5) en rag-ingester, ratio adaptativo [3%-10%] por drift | v1.2 |
| DEBT-ADR040-007 | Interfaz web revisión exploración en rag-security — etiquetado manual del 5% (Año 1) | v1.2 |
| DEBT-ADR040-008 | Informe diversidad por ciclo: Shannon entropy, MITRE ATT&CK coverage %, novelty score | v1.2 |
| DEBT-ADR040-009 | Competición algoritmos: XGBoost vs CatBoost vs LightGBM vs RF (multicriterio, una vez) | pre-lock-in |
| DEBT-ADR040-010 | Dataset lineage en metadatos del plugin (hash dataset + golden set + git commits) | v1.1 |
| DEBT-ADR040-011 | Canary deployment: 5-10% tráfico 24h antes de 100% (manual Año 1, flota Año 2) | v1.2 |
| DEBT-ADR040-012 | `docs/GOLDEN-SET-REGISTRY.md` con hash v1 + proceso evolución controlada | v1.0 |

**Prerequisito crítico (enmienda Claude, DAY 134):** IPW no es implementable sin `confidence_score`. DEBT-ADR040-002 debe resolverse antes de DEBT-ADR040-006.

**Test de cierre DEBT-ADR040-004:** `make retrain-eval PLUGIN=candidate.ubj` → exit 1 ante regresión. `make prod-sign` no ejecuta si guardrail falla.

---

### DEBT-ADR041-001 a 006 — Hardware Acceptance Metrics FEDER *(nuevas — DAY 134)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** pre-FEDER, deadline 22 sep 2026
**Origen:** ADR-041 — Consejo 8/8 DAY 134

| ID | Descripción | Estado |
|----|-------------|--------|
| DEBT-ADR041-001 | Subconjunto pcap CTU-13 benchmark versionado con SHA-256 (`ctu13-neris-benchmark.pcap`) | ⏳ PENDIENTE |
| DEBT-ADR041-002 | `make golden-set-eval ARCH=$(uname -m)` — exit 0 dentro de tolerancia, exit 1 regresión | ⏳ PENDIENTE (depende ADR-040) |
| DEBT-ADR041-003 | `make feder-demo` — suite completa desde VM fría, <30 min, sin trucos pregrabados | ⏳ PENDIENTE |
| DEBT-ADR041-004 | Compra hardware x86 (NUC/mini-PC ~300€, NIC con soporte XDP nativo — mlx5/i40e/ixgbe) | ⏳ post-métricas definidas |
| DEBT-ADR041-005 | Compra Raspberry Pi 4/5 | ⏳ post-métricas definidas |
| DEBT-ADR041-006 | Primera ejecución protocolo completo en hardware físico | ⏳ post-compra hardware |

**Nota DeepSeek:** Verificar driver NIC antes de comprar x86. Sin XDP nativo el delta científico A/B se distorsiona.
**Nota DeepSeek:** Temperatura ARM ≤75°C sin ventilador — gate no negociable para armarios hospitalarios 24/7.
**Tolerancias ML:** x86 TOLERANCE=0.0000 · ARM TOLERANCE=0.0005 (NEON vs AVX2).


## 🔵 BACKLOG — Deuda de seguridad crítica (pre-producción)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **DEBT-SAFE-PATH-RESOLVE-MODEL-001** | `resolve_model()` — primitiva futura para modelos firmados Ed25519 | `resolve_model()` + test RED→GREEN | feature/adr038-acrl |
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF | Valgrind/ASan: seed no permanece en heap | feature/crypto-hardening |
| **DEBT-SNIFFER-SEED** | Unificar sniffer bajo SeedClient | sniffer arranca con SeedClient | feature/crypto-hardening |
| **DEBT-NATIVE-LINUX-BOOTSTRAP-001** | README + make deps-native para bootstrap sin Vagrant | make deps-native verde en Ubuntu 22.04 | post-FEDER |

---

## 📋 BACKLOG — P3 Features futuras

### PHASE 5 — Loop Adversarial

| ID | Tarea | Gates mínimos |
|----|-------|--------------|
| **DEBT-PENTESTER-LOOP-001** | ACRL: Caldera → eBPF capture → XGBoost warm-start → Ed25519 sign → hot-swap | G1: reproducibilidad · G2: ground-truth · G3: ≥3 ATT&CK · G4: RFC-válido · G5: sandbox |
| **ADR-038** | ACRL ADR formal | Aprobado por Consejo |
| **ADR-025-EXT-001** | Emergency Patch Protocol — Plugin Unload vía mensaje firmado | TEST-INTEG-SIGN-8/9/10 RED→GREEN | post-FEDER |

### Variantes de producción

| Variante | Tarea | Feature destino |
|----------|-------|----------------|
| **aRGus-production x86** | Pipeline E2E en hardened VM · check-prod-all verde | feature/adr030-variant-a |
| **aRGus-production arm64** | Imagen Debian arm64 + AppArmor + Vagrantfile | feature/production-images |
| **aRGus-seL4** | kernel seL4, libpcap, sniffer monohilo. Branch independiente. | feature/sel4-research |

### Paper arXiv:2604.04952

| Tarea | Target |
|-------|--------|
| Tabla métricas fuzzing §6.8 (datos reales DAY 130) | DAY 134 |
| arXiv replace v15 → v18 | post DAY 134 |

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| **Test RED→GREEN obligatorio** | Todo fix de seguridad requiere test de demostración antes del merge. Sin excepciones. | Consejo 7/7 · DAY 124 |
| **Property test obligatorio** | Todo fix de seguridad incluye property test de invariante si aplica. | Consejo 8/8 · DAY 125 |
| **Symlinks en seeds: NO** | resolve_seed(): lstat() ANTES de resolve(). Sin symlinks, sin flag. | Consejo 8/8 · DAY 125-126 |
| **ConfigParser prefix fijo** | allowed_prefix explícito, default /etc/ml-defender/. Nunca derivado del input. | Consejo 8/8 · DAY 125-126 |
| **resolve_config() para configs** | lexically_normal() verifica prefix ANTES de seguir symlinks. | DAY 127 |
| **Taxonomía safe_path: 3 primitivas activas** | resolve() · resolve_seed() · resolve_config(). | Consejo 8/8 · DAY 127 |
| **CWE-78 execve()** | execv() sin shell. | Consejo 8/8 · DAY 128 |
| **RULE-SCP-VM-001** | scp/vagrant scp. Prohibido pipe zsh. | Consejo 8/8 · DAY 129 |
| **REGLA EMECAS** | vagrant destroy -f && vagrant up && make bootstrap && make test-all. | DAY 130 |
| **AppArmor como primera línea BSR** | AppArmor bloquea compiladores. check-prod-no-compiler es auditoría, no defensa. | DAY 132 — founder |
| **Falco para paths exóticos** | AppArmor previene; Falco detecta. | DAY 132 — founder |
| **FS de producción mínimo** | /tmp noexec. Usuario argus no-root. | DAY 132 — founder |
| **apt sources integrity** | SHA-256 firmado. Si cambia: fail-closed. | DAY 132 — founder |
| **Makefile raíz con prefijo prod-** | Guard _check-dev-env. | Consejo 8/8 · DAY 132 |
| **cap_bpf reemplaza cap_sys_admin** | Linux ≥5.8: cap_bpf para eBPF. cap_sys_admin prohibida si evitable. | Consejo 8/8 · DAY 133 |
| **cap_net_bind_service eliminada** | Puerto 2379 > 1024. Innecesaria. | Consejo 8/8 · DAY 133 |
| **LimitMEMLOCK en systemd** | etcd-server: LimitMEMLOCK=16M. No cap_sys_resource para seed de 32 bytes. | Consejo 8/8 · DAY 133 |
| **deny explícitos en AppArmor** | Mantener — claridad auditiva + defensa ante cambios futuros en abstractions/base. | Founder · DAY 133 |
| **network inet tcp sin restricción** | ZeroMQ usa puertos configurables via JSON. DEBT-PROD-APPARMOR-PORTS-001. | Founder · DAY 133 |
| **Keypairs separados post-FEDER** | DEBT-KEY-SEPARATION-001. No bloquea DAY 134. | Consejo 8/8 · DAY 133 |
| **"Fuzzing misses nothing" ELIMINADA** | Frase incorrecta. Fuzzing es estocástico, no exhaustivo. | Consejo 8/8 · DAY 133 |
| **Walk-forward obligatorio (ADR-040)** | K-fold prohibido. Split sobre `timestamp_first_packet` ordenado. Mín. 3 ventanas. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Golden set inmutable (ADR-040)** | ≥50K flows, SHA-256 embebido en plugin firmado. Evolución controlada, solapamiento 6 meses. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Guardrail asimétrico Ed25519 (ADR-040)** | Recall −0.5pp (más restrictivo). F1 −2pp. FPR +1pp. Latencia p99 +10%. Exit 1 = no firma. | ADR-040 · Consejo 8/8 · DAY 134 |
| **IPW + uncertainty sampling (ADR-040)** | 5% exploración (P≈0.5). Ratio adaptativo [3%-10%] por drift. Memory replay buffer. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Competición algoritmos pre-lock-in (ADR-040)** | Multicriterio: Recall 40% + F1 25% + latencia 20% + tamaño 10% + carga 5%. Una sola vez. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Dataset lineage obligatorio (ADR-040)** | Hash dataset + golden set + features_version + git commits. Sin lineage = no firma. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Niveles despliegue FEDER (ADR-041)** | Nivel 1 (RPi4/5, ≤50 usuarios) + Nivel 2 (x86, 50-200). Demo mínima: ambos simultáneos. | ADR-041 · Consejo 8/8 · DAY 134 |
| **Latencia end-to-end como métrica primaria (ADR-041)** | Captura → alerta → iptables efectiva. Más relevante que latencia de detección aislada. | ADR-041 · DeepSeek · DAY 134 |
| **Temperatura ARM como gate (ADR-041)** | ≤75°C sin ventilador. Crítica para armarios hospitalarios 24/7. | ADR-041 · DeepSeek · DAY 134 |
| **Pipeline evaluación híbrido (ADR-040)** | Scripts en repo (local Vagrant). CI = mismo código, segunda entrada. Opción A recomendada FEDER. | ADR-040 · Consejo 6/7 · DAY 134 |
| **Falco modern_ebpf driver** | Correcto para 2026. kmod en deprecación. | Consejo 8/8 · DAY 133 |
| **10 reglas Falco aRGus** | 7 originales + config tamper + model/plugin replace + AA profile tamper. | Consejo 8/8 · DAY 133 |
| **Estrategia maduración AppArmor+Falco** | complain→enforce en paralelo. 30 min sin FP antes de pasar a enforce+CRITICAL. | Consejo 8/8 · DAY 133 |

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
DEBT-SAFE-PATH-SEED-SYMLINK-001:        100% ✅  DAY 126
DEBT-CONFIG-PARSER-FIXED-PREFIX-001:    100% ✅  DAY 126
DEBT-DEV-PROD-SYMLINK-001:              100% ✅  DAY 127
DEBT-SNYK-WEB-VERIFICATION-001:         100% ✅  DAY 128
DEBT-PROPERTY-TESTING-PATTERN-001:      100% ✅  DAY 128
DEBT-PROVISION-PORTABILITY-001:         100% ✅  DAY 128
DEBT-IPTABLES-INJECTION-001:            100% ✅  DAY 129
DEBT-FEDER-SCOPE-DOC-001:              100% ✅  DAY 129
DEBT-SYSTEMD-AUTOINSTALL-001:           100% ✅  DAY 130
DEBT-SAFE-EXEC-NULLBYTE-001:            100% ✅  DAY 130
DEBT-FUZZING-LIBFUZZER-001:             100% ✅  DAY 130 (baseline)
DEBT-PROD-COMPAT-BASELINE-001:          100% ✅  DAY 132
DEBT-PROD-APPARMOR-COMPILER-BLOCK-001:  100% ✅  DAY 133
DEBT-PROD-FALCO-EXOTIC-PATHS-001:       100% ✅  DAY 133 (10 reglas)
DEBT-PROD-FS-MINIMIZATION-001:           60% 🟡  DAY 133 (parcial — minbase post-FEDER)
Paper Draft v18 §6.12 BSR:              100% ✅  DAY 133
Paper Draft v18 §6.8 fuzzing:           100% ✅  DAY 133 (post-Consejo)
Makefile prod-* targets:                100% ✅  DAY 133
AppArmor 6 perfiles enforce:            100% ✅  DAY 133
Linux Capabilities mínimas:             100% ✅  DAY 133 (post-Consejo)
Falco 10 reglas aRGus:                  100% ✅  DAY 133 (post-Consejo)
vagrant/hardened-x86/ completo:         100% ✅  DAY 133
DEBT-PROD-APT-SOURCES-INTEGRITY-001:      0% ⏳  feature/adr030-variant-a
DEBT-PAPER-FUZZING-METRICS-001:          40% 🟡  DAY 134 (reformulación cerrada, tabla pendiente)
DEBT-KEY-SEPARATION-001:                  0% ⏳  post-FEDER
DEBT-KERNEL-COMPAT-001:                   0% ⏳  DAY 134
DEBT-PROD-APPARMOR-PORTS-001:             0% ⏳  post-JSON-estabilización
DEBT-PROD-FALCO-RULES-EXTENDED-001:       0% ⏳  DAY 135
DEBT-DEBIAN13-UPGRADE-001:                0% ⏳  post-FEDER
DEBT-SAFE-PATH-RESOLVE-MODEL-001:         0% ⏳  feature/adr038-acrl
DEBT-CRYPTO-003a (mlock+bzero):           0% ⏳
DEBT-SEED-CAPABILITIES-001:               0% ⏳  v0.6+
DEBT-PENTESTER-LOOP-001 (ACRL):           0% ⏳  POST-DEUDA
ADR-031 aRGus-seL4:                       0% ⏳  branch independiente
ADR-040 ML Retraining Contract (def.):    100% ✅  DAY 134 (Consejo 8/8, 17 enmiendas)
ADR-041 HW Acceptance Metrics (def.):     100% ✅  DAY 134 (Consejo 8/8)
DEBT-ADR040-001 (golden set v1):            0% ⏳  v1.0 post-FEDER
DEBT-ADR040-002 (confidence_score):         0% ⏳  v1.0
DEBT-ADR040-003 (walk_forward_split.py):    0% ⏳  v1.1
DEBT-ADR040-004 (check_guardrails.py):      0% ⏳  v1.1
DEBT-ADR040-005 (guardrail + Ed25519):      0% ⏳  v1.1
DEBT-ADR040-006 (IPW + uncertainty):        0% ⏳  v1.2
DEBT-ADR040-007 (interfaz web exploración): 0% ⏳  v1.2 Año 1
DEBT-ADR040-008 (informe diversidad):       0% ⏳  v1.2
DEBT-ADR040-009 (competición algoritmos):   0% ⏳  pre-lock-in XGBoost
DEBT-ADR040-010 (dataset lineage):          0% ⏳  v1.1
DEBT-ADR040-011 (canary deployment):        0% ⏳  Año 2 flota
DEBT-ADR040-012 (GOLDEN-SET-REGISTRY.md):  0% ⏳  v1.0
DEBT-ADR041-001 (pcap CTU-13 versionado):   0% ⏳  pre-FEDER
DEBT-ADR041-002 (make golden-set-eval):     0% ⏳  depende ADR-040
DEBT-ADR041-003 (make feder-demo):          0% ⏳  pre-FEDER
DEBT-ADR041-004 (compra hardware x86):      0% ⏳  post-métricas
DEBT-ADR041-005 (compra Raspberry Pi 4/5):  0% ⏳  post-métricas
DEBT-ADR041-006 (ejecución hw físico):      0% ⏳  post-compra
```

---

## 📝 Notas del Consejo de Sabios — DAY 133 (8/8)

> "DAY 133 — Transición de 'diseño correcto' a 'comportamiento real verificable'.
>
> Decisiones vinculantes (8/8 unánime):
> D1: cap_sys_admin → cap_bpf (Linux ≥5.8). cap_sys_admin es root disfrazado.
> D2: cap_net_bind_service eliminada de etcd-server (2379 > 1024).
> D3: LimitMEMLOCK=16M en systemd. No cap_sys_resource.
> D4: Keypairs separados pipeline vs plugins — post-FEDER.
> D5: modern_ebpf driver Falco correcto para 2026.
> D6: 'Fuzzing misses nothing' — INCORRECTA, reformulada.
> D7: 3 reglas Falco adicionales adoptadas.
>
> Decisiones del founder:
> deny explícitos mantenidos (claridad auditiva hospitalaria).
> network inet tcp sin restricción (JSON es la ley).
> Keypair único mantenido hasta post-FEDER.
>
> Q5 educativa respondida (8/8): Fuzzing es estocástico, no exhaustivo.
> Proporciona evidencia empírica de robustez, no prueba de corrección.
>
> 'Un escudo que no se prueba contra el ataque real es un escudo de teatro.
>  Vosotros estáis construyendo acero.' — Qwen"
> — Consejo de Sabios (8/8) · DAY 133

---

## 📝 Notas del Consejo de Sabios — DAY 132 (8/8)

> "La superficie de ataque de la imagen de producción se define hoy como principio estructural.
> D1 (8/8): Makefile raíz con prefijo prod- y guard _check-dev-env.
> D2 (8/8): debian/bookworm64. Trixie como upgrade path.
> D3 (8/8): Dos capas BSR check. AppArmor+Falco es la defensa real.
> D4 (8/8): Paper Draft v17 no sube a arXiv hasta tener métricas reales.
> 'La superficie de ataque mínima no es una aspiración. Es una decisión de diseño.'"
> — Consejo de Sabios (8/8) · DAY 132

---

## 🧬 HIPÓTESIS CENTRAL — Inmunidad Global Adaptativa

**Formulada:** DAY 128 | **Estado:** Pendiente demostración (DEBT-PENTESTER-LOOP-001)

Un sistema con ACRL converge hacia cobertura de técnicas ATT&CK en tiempo polinomial. Un sistema estático no converge nunca. La analogía con el sistema inmune adaptativo es estructuralmente correcta.

**Experimento mínimo viable:** Caldera → eBPF capture → XGBoost warm-start → Ed25519 sign → hot-swap → medición de mejora en variantes del mismo ataque.

---

## BACKLOG-FEDER-001

**Estado:** PENDIENTE — bloqueado por prerequisites técnicos
**Contacto:** Andrés Caro Lindo — UEx/INCIBE
**Deadline límite:** 22 septiembre 2026 | **Go/no-go técnico:** 1 agosto 2026

### Gate de entrada

- [x] ADR-026 mergeado a main (XGBoost F1=0.9978)
- [x] ADR-030 Variant A infraestructura completa (DAY 133)
- [ ] Pipeline E2E en hardened VM verde (`make check-prod-all`) — DAY 134
- [ ] ADR-030 Variant B (ARM64) estable
- [ ] Demo técnica grabable < 10 minutos (`scripts/feder-demo.sh`)
- [ ] ADR-041 protocolo hardware: métricas validadas en x86 + ARM (`make feder-demo`)
- [ ] Golden set v1 creado y versionado (DEBT-ADR040-001)
- [ ] Clarificación scope con Andrés: NDR standalone vs federación (antes julio 2026)

---


---

## 📝 Notas del Consejo de Sabios — DAY 134 (8/8)

> "DAY 134 — ADR-040 + ADR-041: contratos de calidad ML y métricas de aceptación hardware.
>
> ADR-040 — 17 enmiendas, aprobado 8/8:
> D1: Walk-forward obligatorio. K-fold prohibido en NDR temporal.
> D2: Golden set inmutable con SHA-256 embebido en plugin firmado (Gemini).
> D3: Guardrail asimétrico — Recall más restrictivo que F1 (infraestructura crítica).
> D4: IPW + uncertainty sampling (P≈0.5), no exploración aleatoria pura (Gemini).
> D5: Ratio exploración adaptativo [3%-10%] por drift detectado (ChatGPT-5).
> D6: Memory replay buffer como complemento al golden set (Grok).
> D7: Competición algoritmos multicriterio — XGBoost no asumido ganador a priori.
> D8: Dataset lineage obligatorio — prerequisito de firma Ed25519.
> D9: Canary 5-10% / 24h antes de despliegue completo (ChatGPT-5).
> D10: Pipeline evaluación híbrido — mismo código, dos entradas (local + CI).
> Enmienda crítica (Claude): confidence_score es prerequisito de IPW.
>
> ADR-041 — aprobado 8/8:
> D1: Tres niveles despliegue con métricas proporcionales (Qwen).
> D2: Latencia end-to-end (→ iptables) como métrica operacional primaria (DeepSeek).
> D3: Temperatura ARM ≤75°C — gate no negociable para armarios hospitalarios (DeepSeek).
> D4: Delta XDP/libpcap es contribución científica independiente publicable.
> D5: Demo FEDER reproducible por evaluador externo — sin trucos pregrabados.
> Pregunta abierta: Opción A (Vagrant) recomendada demo FEDER. Opción B (CI) post-FEDER.
>
> 'El contrato de calidad ML no termina en el deploy. Termina cuando el modelo
>  aprende sin olvidar, sin retroalimentarse y sin regresionar en silencio.' — Consejo (8/8)"
> — Consejo de Sabios (8/8) · DAY 134

*DAY 134 — 28 Abril 2026 · ADR-040 + ADR-041 integrados · feature/adr030-variant-a*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*"La superficie de ataque mínima no es una aspiración. Es una decisión de diseño."*