# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_8_Models-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: make test-all VERDE](https://img.shields.io/badge/Tests-make_test--all_VERDE-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Integrity](https://img.shields.io/badge/Plugin_Integrity-ADR--025_Ed25519-brightgreen)](docs/adr/ADR-025-plugin-integrity-ed25519.md)
[![safe_path](https://img.shields.io/badge/safe__path-ADR--037_header--only-brightgreen)](contrib/safe-path/)
[![PHASE 4](https://img.shields.io/badge/PHASE_4-COMPLETADA-brightgreen)]()
[![AppArmor](https://img.shields.io/badge/AppArmor-6%2F6_enforce-brightgreen)]()
[![Falco](https://img.shields.io/badge/Falco-11_reglas_aRGus-brightgreen)]()
[![BSR](https://img.shields.io/badge/BSR-cap__bpf_ADR--039-brightgreen)]()
[![ADR-040](https://img.shields.io/badge/ADR--040-ML_Retraining_Contract-blue)](docs/adr/ADR-040-ml-plugin-retraining-contract.md)
[![ADR-041](https://img.shields.io/badge/ADR--041-FEDER_HW_Metrics-orange)](docs/adr/ADR-041-hardware-acceptance-metrics-feder.md)
[![Reproducible](https://img.shields.io/badge/Infra-make_bootstrap-brightgreen)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Prec%3D0.9945_In--Distribution-brightgreen)]()
[![Hardened](https://img.shields.io/badge/Security-v0.6.0--hardened__variant__a-brightgreen)]()
[![OOD Finding](https://img.shields.io/badge/OOD_Finding-Published_DAY_122-orange)]()
[![PRE-PRODUCTION](https://img.shields.io/badge/Status-PRE--PRODUCTION-orange)]()
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)

📜 Living contracts: [Protobuf schema](docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](docs/contracts/JSON%20contracts.md) · [RAG API](docs/contracts/Rag%20security%20commands.md)

---

✅ `main` is tagged `v0.6.0-hardened-variant-a`. Branch activa: `main` — ADR-030 Variant A mergeada (DAY 136) · ADR-040 + ADR-041 aprobados (DAY 134).
**PRE-PRODUCTION: do not deploy in hospitals until ACRL (DEBT-PENTESTER-LOOP-001) is complete.**

---

## Estado actual — DAY 136 (2026-04-29)

**Tag activo:** `v0.6.0-hardened-variant-a` | **Commit:** `737ba0d5` | **Branch activa:** `main`
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
**Paper:** arXiv:2604.04952 · Draft v18 enviado (Cornell procesando)
**ADR-040:** ML Plugin Retraining Contract — APROBADO v2 (Consejo 8/8, 17 enmiendas)
**ADR-041:** Hardware Acceptance Metrics FEDER — APROBADO (Consejo 8/8)

### Pipeline
- 6/6 componentes RUNNING — validado en VM destruida y reconstruida desde cero (REGLA EMECAS)
- `make test-all`: ALL TESTS COMPLETE
- `make hardened-full`: EMECAS HARDENED PASSED ✅
- `make check-prod-all`: PASSED sin warnings ✅

### Hitos DAY 136 🎉
- **feature/adr030-variant-a → main MERGEADO** — 179 ficheros, 19.443 inserciones.
- **Tag v0.6.0-hardened-variant-a** publicado en origin.
- **EMECAS HARDENED PASSED** — 6 binarios, 6 librerías, AppArmor 6/6, Falco 11 reglas, seeds sin WARNs.
- **docs/KNOWN-DEBTS-v0.6.md** — 6 deudas documentadas pre-merge.
- **hardened-full-with-seeds** — target Makefile FEDER ONLY añadido.
- **Consejo 8/8 DAY 136** — 3 nuevas deudas: Jenkins seeds, Vault criptográfico, compiler warnings.

### Hitos DAY 135
- **EMECAS hardened PASSED** — DEBT-EMECAS-HARDENED-001 CERRADO.
- **DEBT-PROD-APT-SOURCES-INTEGRITY-001 CERRADO** — FailureAction=poweroff, ExecStartPre corregido.
- **DEBT-SEEDS-DEPLOY-001 CERRADO** — prod-deploy-seeds, 6 seeds + plugin_signing.pk.
- **DEBT-CONFIDENCE-SCORE-001 CERRADO** — BENIGN=0.854557, RANSOMWARE=0.700000.
- **ADR-042 IRP DRAFT v2** — Consejo 8/8 x2 rondas adversariales, aprobado como arquitectura.
- **arXiv replace v15→v18 ENVIADO.**

### Deuda técnica abierta

| Deuda | Prioridad | Target |
|-------|-----------|--------|
| DEBT-IRP-NFTABLES-001 | 🔴 Alta | P0 pre-FEDER |
| DEBT-IRP-QUEUE-PROCESSOR-001 | 🔴 Alta | post-merge |
| DEBT-JENKINS-SEED-DISTRIBUTION-001 | 🔴 Alta | pre-FEDER |
| DEBT-CRYPTO-MATERIAL-STORAGE-001 | 🔴 Alta | pre-FEDER demo |
| DEBT-COMPILER-WARNINGS-CLEANUP-001 | 🔴 Alta | DAY 137+ rama dedicada |
| DEBT-SEEDS-SECURE-TRANSFER-001 | 🔴 Alta | post-FEDER |
| DEBT-SEEDS-LOCAL-GEN-001 | 🔴 Alta | post-FEDER |
| DEBT-SEEDS-BACKUP-001 | 🔴 Alta | post-FEDER |
| DEBT-KEY-SEPARATION-001 | 🟡 Media | post-FEDER |
| DEBT-PROD-APPARMOR-PORTS-001 | 🟢 Baja | post-JSON |
| DEBT-ADR040-001..012 | ⏳ | post-FEDER (ver BACKLOG.md) |
| DEBT-ADR041-001..006 | ⏳ | pre-FEDER (ver BACKLOG.md) |

### Próxima frontera — DAY 137
- **PASO 1** — `git checkout -b feature/variant-b-libpcap` (ADR-029 Variant B)
- **PASO 2** — `fix/compiler-warnings-cleanup-001` — ODR violations, Protobuf dual-copy
- **PASO 3** — `DEBT-IRP-NFTABLES-001` — implementar argus-network-isolate con nftables
- **PASO 4** — `DEBT-CRYPTO-MATERIAL-STORAGE-001` — prototipo HashiCorp Vault en Vagrant
- **DEBT-PENTESTER-LOOP-001** — ACRL: Caldera → eBPF capture → XGBoost retrain → Ed25519 sign → hot-swap

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa (`main`) | x86-debug, imagen Vagrant completa. Para investigación y desarrollo diario. |
| **aRGus-production** | 🟡 En construcción | x86-apparmor + arm64-apparmor. AppArmor enforce, cap_bpf, Falco, noexec. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ Diseño futuro | Apéndice científico. Kernel seL4, libpcap. Branch independiente. |

---

## 📄 Preprint

**arXiv:** [arXiv:2604.04952 \[cs.CR\]](https://arxiv.org/abs/2604.04952)
**Published:** 3 April 2026 · **Draft v18** (DAY 133 — pre-arXiv, pendiente tabla fuzzing) · MIT license
**Code:** https://github.com/alonsoir/argus

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions.

**Philosophy**: *Via Appia Quality* — Systems built like Roman roads, designed to endure.

> *"Un escudo que aprende de su propia sombra."*

---

## 📊 Validated Results

| Metric | Value | Notes |
|---|---|---|
| **F1-score (CTU-13 Neris)** | **0.9985** | Stable across 4 replay runs |
| **Precision** | **0.9969** | |
| **Recall** | **1.0000** | Zero missed attacks (FN=0) |
| **XGBoost Precision (CIC-IDS-2017 val)** | **0.9945** | In-distribution, threshold=0.8211 |
| **XGBoost Wednesday OOD** | **Documented impossibility** | Structural covariate shift — §8 paper |
| **Inference latency (XGBoost)** | **1.986 µs/sample** | Gate <2µs ✅ |
| **Inference latency (RF)** | **0.24–1.06 µs** | Per-class, embedded C++20 |
| **Throughput ceiling (virtualized)** | **~33–38 Mbps** | VirtualBox NIC limit, not pipeline |
| **Stress test** | **2,374,845 packets — 0 drops** | 100 Mbps requested, loop=3 |
| **RAM (full pipeline)** | **~1.28 GB** | Stable under load |
| **BSR — Dev VM** | **719 pkgs / 5.9 GB** | gcc, g++, clang, cmake present |
| **BSR — Hardened VM** | **304 pkgs / 1.3 GB** | NONE (check-prod-no-compiler: OK) ✅ |
| **AppArmor profiles** | **6/6 enforce** | cap_bpf (Linux ≥5.8), no cap_sys_admin |
| **Falco rules** | **11 aRGus-specific** | modern_ebpf driver |

---

## 🔒 DAY 133 Security Hardening — ADR-030 Variant A

### Build/Runtime Separation (BSR) — ADR-039

| Environment | Packages | Disk | Compilers |
|---|---|---|---|
| Dev VM | 719 | 5.9 GB | gcc, g++, clang, cmake |
| **Hardened VM** | **304** | **1.3 GB** | **NONE** ✅ |
| Minbase target† | ~100 | ~0.4 GB | NONE |

†DEBT-PROD-FS-MINIMIZATION-001. Vagrant base box floor: ~250 packages.

### Linux Capabilities — no SUID root (post-Consejo DAY 133)

| Component | Capabilities | Nota |
|---|---|---|
| sniffer | `cap_net_admin,cap_net_raw,cap_bpf,cap_ipc_lock` | cap_bpf reemplaza cap_sys_admin (Linux ≥5.8) |
| firewall-acl-agent | `cap_net_admin` | iptables/ipset |
| etcd-server | `cap_ipc_lock` (+ LimitMEMLOCK=16M systemd) | cap_net_bind_service eliminada (2379 > 1024) |
| ml-detector | none | argus no-root real |
| rag-ingester | none | argus no-root real |
| rag-security | none | argus no-root real |

### AppArmor — 6 profiles enforce

Un perfil por componente en `security/apparmor/`. Default-deny. `deny` explícitos mantenidos para claridad auditiva (decisión founder DAY 133).

### Falco — 10 aRGus-specific rules (modern_ebpf)

Reglas: unexpected writes, unexpected exec, shell spawn, binary modification (BSR), seed access by wrong process, raw socket from non-sniffer, config tampering, model/plugin replacement, AppArmor profile tampering.

**Estrategia:** AppArmor complain + Falco WARNING → 30 min stable → AppArmor enforce + Falco CRITICAL.

---

## 🔧 Prerequisites

### macOS

```bash
brew install --cask virtualbox
brew install --cask vagrant
xcode-select --install
```

### Linux (Debian/Ubuntu)

```bash
sudo apt-get install -y make virtualbox
wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install -y vagrant
```

> **Note:** All pipeline logic runs inside a Vagrant/VirtualBox VM. No C++ toolchain required on the host.

---

## 🚀 Quick Start

### Primera vez

```bash
git clone https://github.com/alonsoir/argus.git
cd argus
make up          # vagrant up — full provisioning ~20-30 min
make bootstrap   # all 8 steps in one command
```

### Workflow diario (REGLA EMECAS)

```bash
vagrant destroy -f && vagrant up && make bootstrap && make test-all
```

### Hardened VM (ADR-030 Variant A)

```bash
# EMECAS sagrado (reproducibilidad total — para demo FEDER y validación)
make hardened-full            # destroy → up → provision → build → deploy → check

# Workflow alternativo (iteración rápida durante desarrollo)
make hardened-up
make hardened-provision-all   # filesystem + AppArmor + Falco
make prod-full-x86            # build → sign → checksums → deploy
make check-prod-all           # 5 security gates
```

---

## 🗺️ Roadmap

### ✅ DONE — DAY 133 (27 Apr 2026) — ADR-030 Variant A infrastructure 🎉
- [x] Paper Draft v18 — §6.12 BSR métricas reales + §6.8 fuzzing reformulado (post-Consejo)
- [x] AppArmor 6 perfiles enforce — `security/apparmor/`
- [x] Linux Capabilities mínimas — `cap_bpf` reemplaza `cap_sys_admin` (Consejo 8/8)
- [x] Falco 10 reglas — `modern_ebpf` driver, estrategia 3 fases
- [x] Filesystem hardened — usuario `argus`, `/tmp` noexec, seeds 0400
- [x] Makefile prod-* targets — `prod-full-x86`, `check-prod-all`
- [x] Acta Consejo DAY 133 — convergencias + divergencias + decisiones

### ✅ DONE — DAY 134 (28 Apr 2026) — ADR-040 + ADR-041 🎉
- [x] ADR-040 ML Plugin Retraining Contract v2 — 7 reglas, 12 deudas, Consejo 8/8 (17 enmiendas)
- [x] ADR-041 Hardware Acceptance Metrics FEDER — 3 niveles, 10 métricas, Consejo 8/8
- [x] BACKLOG.md + README.md actualizados con ADR-040 + ADR-041

### ✅ DONE — DAY 132 (26 Apr 2026)
- [x] Paper Draft v17 · HARDWARE-REQUIREMENTS · vagrant/hardened-x86 skeleton · Prerequisites README

### ✅ DONE — DAY 124–130
- [x] ADR-037 safe_path · v0.5.2-hardened · CWE-78 cerrado · libFuzzer 2.4M runs · REGLA EMECAS

### ✅ DONE — DAY 134 (28 Apr 2026) — Pipeline E2E hardened + ADR-040/041 🎉
- [x] `make check-prod-all` PASSED — 5/5 gates verdes en hardened VM
- [x] DEBT-KERNEL-COMPAT-001 CERRADO — cap_bpf + XDP en kernel 6.1 ✅
- [x] DEBT-PAPER-FUZZING-METRICS-001 CERRADO — tabla §6.8 con datos reales ✅
- [x] Draft v18 completo — 42 páginas, listo para arXiv replace
- [x] ADR-040 ML Retraining Contract (8/8, 17 enmiendas) + ADR-041 FEDER HW Metrics (8/8)

### ✅ DONE — DAY 135-136: EMECAS hardened + merge + v0.6.0 🎉
- [x] make hardened-full PASSED desde VM destruida
- [x] DEBT-PROD-APT-SOURCES-INTEGRITY-001 CERRADO — FailureAction=poweroff
- [x] DEBT-CONFIDENCE-SCORE-001 CERRADO — variabilidad confirmada
- [x] arXiv replace v15 → v18 ENVIADO
- [x] feature/adr030-variant-a → main MERGEADO
- [x] Tag v0.6.0-hardened-variant-a publicado
- [x] docs/KNOWN-DEBTS-v0.6.md creado (6 deudas)
- [x] hardened-full-with-seeds target (FEDER ONLY)

### 🔜 NEXT — DAY 137: Variant B + compiler warnings + IRP + Vault

| Priority | Task |
|---|---|
| 🔴 P0 | EMECAS completo desde VM destruida |
| 🔴 P0 | `git checkout -b feature/variant-b-libpcap` — sniffer libpcap, Vagrantfile ARM64 |
| 🔴 P0 | DEBT-IRP-NFTABLES-001 — argus-network-isolate con nftables drop-all |
| 🔴 P0 | `fix/compiler-warnings-cleanup-001` — ODR violations UB (bloqueante certificación) |
| 🟡 P1 | DEBT-CRYPTO-MATERIAL-STORAGE-001 — prototipo HashiCorp Vault en Vagrant |
| 🟡 P1 | DEBT-JENKINS-SEED-DISTRIBUTION-001 — mecanismo mínimo viable CI seeds |
| 🟢 P2 | DEBT-IRP-QUEUE-PROCESSOR-001 — irp-queue límites + procesador systemd |

### 🔜 THEN — PHASE 5: Adversarial Capture-Retrain Loop

| Priority | Task |
|---|---|
| P0 | **DEBT-PENTESTER-LOOP-001** — MITRE Caldera → real flows → XGBoost retrain |
| P0 | **BACKLOG-FEDER-001** — clarificar scope con Andrés Caro Lindo |
| P1 | aRGus-production ARM64 |
| P2 | aRGus-seL4 research branch |

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Kimi** (Moonshot) · **Mistral**

Metodología: desacuerdo estructurado. Los problemas deben demostrarse con tests compilables o matemáticas antes de proponer soluciones. Documentado en §6 del preprint.

---

## 🗺️ Milestones

- ✅ DAY 111: **arXiv:2604.04952 PUBLICADO** 🎉
- ✅ DAY 113: **ADR-025 MERGED — v0.3.0-plugin-integrity** 🎉
- ✅ DAY 118: **PHASE 3 COMPLETADA — v0.4.0** 🎉
- ✅ DAY 122: **PHASE 4 COMPLETADA — v0.5.0-preproduction** 🎉
- ✅ DAY 124: **ADR-037 MERGED — v0.5.1-hardened** 🎉
- ✅ DAY 126: **v0.5.2-hardened — lstat() + prefix fijo** 🎉
- ✅ DAY 129: **CWE-78 CERRADO — execv() sin shell** 🎉
- ✅ DAY 130: **REGLA EMECAS · libFuzzer 2.4M runs** 🎉
- ✅ DAY 132: **Paper Draft v17 · HARDWARE-REQUIREMENTS · vagrant/hardened-x86 · Consejo 8/8** 🎉
- ✅ DAY 133: **ADR-030 Variant A — cap_bpf · AppArmor 6/6 · Falco 10 reglas · Paper v18** 🎉
- ✅ DAY 134: **ADR-040 ML Retraining Contract (8/8, 17 enmiendas) · ADR-041 FEDER HW Metrics (8/8)** 🎉
- ✅ DAY 134: **Pipeline E2E hardened · check-prod-all PASSED · Draft v18 completo · ADR-040+041** 🎉
- ✅ DAY 135: **make hardened-full EMECAS · apt-integrity · seeds deploy · confidence_score · arXiv v18** 🎉
- ✅ DAY 136: **v0.6.0-hardened-variant-a · merge main · Consejo 8/8 · KNOWN-DEBTS-v0.6.md** 🎉
- 🔜 DAY 137: **feature/variant-b-libpcap · fix/compiler-warnings · IRP nftables · Vault prototipo**

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*
## Hardened Deployment (ADR-030 Variant A)

Production-hardened VM with AppArmor, Falco, BSR gate, and apt-sources integrity.

### Prerequisites
```bash
# Dev VM must be running with pipeline built
make up && make bootstrap && make test-all
make vendor-download  # Verify Falco .deb checksum
```

### Full hardened deploy (gate pre-merge)
```bash
make hardened-full          # destroy → up → provision → build → deploy → check
```

### Daily iteration (fast, no destroy)
```bash
make hardened-redeploy      # build → deploy → check
make prod-deploy-seeds      # deploy seeds explicitly (D2 — never in EMECAS)
make check-prod-permissions # verify seeds + permissions
```

### Security gates (check-prod-all)
```bash
make check-prod-all         # 5/5 gates: BSR + AppArmor + cap_bpf + permissions + Falco
```

### APT Sources Integrity
- SHA-256 of apt sources captured at provisioning time
- Verified on every boot via systemd oneshot
- **`FailureAction=poweroff` — immediate, no grace period**
- A node with compromised apt sources is radioactive material.
  It does not restart. It is isolated, autopsied, and restored from scratch.

### Hardened VM SSH
```bash
cd vagrant/hardened-x86 && vagrant ssh -c '...'
```

