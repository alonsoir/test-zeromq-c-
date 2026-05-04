````# ML Defender (aRGus NDR)

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
[![Variant B](https://img.shields.io/badge/ADR--029-Variant_B_libpcap_pipeline-blue)]()
[![Reproducible](https://img.shields.io/badge/Infra-make_bootstrap-brightgreen)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Prec%3D0.9945_In--Distribution-brightgreen)]()
[![Hardened](https://img.shields.io/badge/Security-v0.6.0--hardened__variant__a-brightgreen)]()
[![PRE-PRODUCTION](https://img.shields.io/badge/Status-PRE--PRODUCTION-orange)]()
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)

📜 Living contracts: [Protobuf schema](docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](docs/contracts/JSON%20contracts.md) · [RAG API](docs/contracts/Rag%20security%20commands.md)

---

✅ `main` is tagged `v0.6.0-hardened-variant-a`. Branch activa: `feature/variant-b-libpcap` — ADR-029 Variant B pipeline completo (DAY 138).
**PRE-PRODUCTION: do not deploy in hospitals until ACRL (DEBT-PENTESTER-LOOP-001) is complete.**

---

## Estado actual — DAY 141 (2026-05-04)

**Tag activo:** `v0.6.0-hardened-variant-a` | **Branch activa:** `feature/variant-b-libpcap` @ `63a37d9d`
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
**Paper:** arXiv:2604.04952 · Draft v18 (Cornell procesando)
**FEDER deadline:** 22-Sep-2026 | **Go/no-go:** 1-Ago-2026

### Pipeline
- 6/6 componentes RUNNING — validado EMECAS DAY 138 ✅
- `make test-all`: ALL TESTS COMPLETE (9/9 sniffer, incluyendo 8 tests Variant B)
- `make sniffer && make sniffer-libpcap`: ambos ✅ sin warnings nuevos

### Hitos DAY 138 🎉
- **DEBT-CAPTURE-BACKEND-ISP-001 CERRADA** — commit `1a7f723a`. `CaptureBackend` a 5 métodos puros. Métodos eBPF en `EbpfBackend`. Consejo 5-2-1 → implementado.
- **DEBT-VARIANT-B-PCAP-IMPL-001 CERRADA** — commits `22df0099` + `da1badf7`. Pipeline completo `pcap_dispatch → proto → LZ4 → ChaCha20 → ZMQ`. Wire format idéntico a Variant A. 8/8 tests PASSED.
- **DEBT-VARIANT-B-CONFIG-001 REGISTRADA** — JSON propio pendiente. Campos multihilo hardcodeados en binario.
- **Consejo 8/8 DAY 138** — 7 preguntas, veredictos unánimes: ODR P0 bloqueante, dontwait correcto, nft -f transaccional, seL4 no diseñar ahora.

### Deuda técnica abierta

| Deuda | Prioridad | Target |
|-------|-----------|--------|
| DEBT-VARIANT-B-BUFFER-SIZE-001 | 🔴 P1 | pre-FEDER (pre-benchmark ARM64) |
| DEBT-VARIANT-B-MUTEX-001 | 🔴 P1 | pre-FEDER (Nivel 1 script) |

| DEBT-IRP-NFTABLES-001 | 🔴 Alta | pre-FEDER |
| DEBT-IRP-QUEUE-PROCESSOR-001 | 🔴 Alta | post-merge |
| DEBT-JENKINS-SEED-DISTRIBUTION-001 | 🔴 Alta | pre-FEDER |
| DEBT-CRYPTO-MATERIAL-STORAGE-001 | 🔴 Alta | pre-FEDER |
| DEBT-SEEDS-SECURE-TRANSFER-001 | 🔴 Alta | post-FEDER |

| DEBT-KEY-SEPARATION-001 | 🟡 Media | post-FEDER |
| DEBT-ADR040-001..012 | ⏳ | post-FEDER |
| DEBT-ADR041-001..006 | ⏳ | pre-FEDER |

### Próxima frontera — DAY 142
1. EMECAS obligatorio
2. `DEBT-IRP-NFTABLES-001` — sesión 1/3 (argus-network-isolate, nftables transaccional)
3. `DEBT-VARIANT-B-BUFFER-SIZE-001` — pcap_create()+pcap_set_buffer_size()
4. `DEBT-VARIANT-B-MUTEX-001` — script exclusión mutua Nivel 1


---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa (`main`) | x86-debug, imagen Vagrant completa. Para investigación y desarrollo diario. |
| **aRGus-production** | 🟡 En construcción | x86-apparmor + arm64-apparmor. AppArmor enforce, cap_bpf, Falco, noexec. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ Research track post-FEDER | Kernel seL4, libpcap. Reescritura completa. Branch independiente. |

---

## 📄 Preprint

**arXiv:** [arXiv:2604.04952 \[cs.CR\]](https://arxiv.org/abs/2604.04952)
**Published:** 3 April 2026 · **Draft v18** (Cornell procesando) · MIT license
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
| **Variant B tests** | **8/8 PASSED** | DAY 138 — unit/integ/stress/regression |

---

## 🔒 Security Hardening — ADR-030 Variant A

### Build/Runtime Separation (BSR) — ADR-039

| Environment | Packages | Disk | Compilers |
|---|---|---|---|
| Dev VM | 719 | 5.9 GB | gcc, g++, clang, cmake |
| **Hardened VM** | **304** | **1.3 GB** | **NONE** ✅ |

### Linux Capabilities — no SUID root

| Component | Capabilities |
|---|---|
| sniffer | `cap_net_admin,cap_net_raw,cap_bpf,cap_ipc_lock` |
| firewall-acl-agent | `cap_net_admin` |
| etcd-server | `cap_ipc_lock` (+ LimitMEMLOCK=16M) |
| ml-detector, rag-ingester, rag-security | none |

### AppArmor — 6 profiles enforce · Falco — 11 aRGus-specific rules

---

## 🔧 Prerequisites

### macOS

```bash
brew install --cask virtualbox
brew install --cask vagrant
xcode-select --install
```

> **Note:** `git clone --recurse-submodules` is required. `third_party/llama.cpp` is a git submodule. Cloning without this flag leaves it empty and `rag-security` builds without LLM support. Use `make submodule-init` to fix an existing clone.

### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y make
```

VirtualBox from official repo (apt may be outdated):
```bash
wget -q https://www.virtualbox.org/download/oracle_vbox_2016.asc -O- | sudo gpg --dearmor -o /usr/share/keyrings/oracle-virtualbox.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/oracle-virtualbox.gpg] https://download.virtualbox.org/virtualbox/debian $(lsb_release -cs) contrib" | sudo tee /etc/apt/sources.list.d/virtualbox.list
sudo apt-get update && sudo apt-get install -y virtualbox-7.0
```

Vagrant:
```bash
wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install -y vagrant
```

### Linux (RHEL/Fedora/CentOS)

```bash
sudo dnf install -y make
```

VirtualBox:
```bash
sudo dnf install -y kernel-devel kernel-headers dkms
sudo dnf config-manager --add-repo https://download.virtualbox.org/virtualbox/rpm/fedora/virtualbox.repo
sudo dnf install -y VirtualBox-7.0
```

Vagrant:
```bash
sudo dnf config-manager --add-repo https://rpm.releases.hashicorp.com/fedora/hashicorp.repo
sudo dnf install -y vagrant
```

> **Note:** `git clone --recurse-submodules` is required. `third_party/llama.cpp` is a git submodule. Cloning without this flag leaves it empty and `rag-security` builds without LLM support. Use `make submodule-init` to fix an existing clone.

> **Note (RHEL/CentOS):** VirtualBox requires Secure Boot to be disabled or the kernel module to be signed. On WSL2, VirtualBox is not supported — use a native Linux install.

### Windows 11 (best-effort, not officially supported)

> ⚠️ **aRGus NDR only produces Linux binaries** (x86-64 and ARM64). There are no Windows binaries and none are planned. The pipeline runs inside a Linux VM — Windows is only the host.

Prerequisites:
```powershell
winget install Git.Git
winget install Oracle.VirtualBox
winget install Hashicorp.Vagrant
```

Run all commands from **Git Bash** (not CMD or PowerShell — the Makefile requires bash syntax).

> ⚠️ **Hyper-V conflict:** Windows 11 enables Hyper-V by default for WSL2. VirtualBox 7.0+ has experimental Hyper-V support but with ~30% performance penalty. You must choose one of:
> - Disable Hyper-V (loses WSL2): `bcdedit /set hypervisorlaunchtype off` + reboot
> - Use VirtualBox 7.0+ in Hyper-V mode (slower, less stable)

**Not tested by the maintainer.** If you hit issues on Windows 11, please [open an issue](https://github.com/alonsoir/argus/issues) — we'll help with the resources we have.


---

## 🚀 Quick Start

> ⚠️ **Vagrant is required.** Native Linux bootstrap without Vagrant is not yet implemented ([DEBT-NATIVE-LINUX-BOOTSTRAP-001](docs/KNOWN-DEBTS-v0.6.md)). Running `make` directly on a bare Linux host will fail.

```bash
# STEP 1 — Clone with submodules (mandatory — llama.cpp is a git submodule)
git clone --recurse-submodules https://github.com/alonsoir/argus.git
cd argus

# Already cloned without --recurse-submodules? Fix it:
# make submodule-init
```

> 📦 **TinyLlama model** (`tinyllama-1.1b-chat-v1.0.Q4_0.gguf`, ~700MB) is downloaded
> automatically during `vagrant up`. It is gitignored and never committed to the repo.

```bash
# STEP 2 — Start VM and provision all dependencies (~20-30 min first time)
# Downloads TinyLlama, builds llama.cpp, installs FAISS/ONNX/XGBoost/libsodium
make up && make bootstrap
```

### Workflow diario (REGLA EMECAS)

```bash
vagrant destroy -f && vagrant up && make bootstrap && make test-all
```

### Build Profiles

| Profile | Flags | Cuándo usarlo |
|---------|-------|---------------|
| `debug` (**default**) | `-g -O0` | Desarrollo diario — símbolos de depuración, sin optimización |
| `production` | `-O3 -flto -march=native -DNDEBUG` | ODR verification, capacity benchmarks (FEDER), builds equivalentes a hardened VM |
| `tsan` | `-fsanitize=thread -g -O1` | Detección de race conditions en código multihilo (EbpfBackend, RingBufferConsumer) |
| `asan` | `-fsanitize=address,undefined -g -O1` | Detección de memory errors, buffer overflows, UB |

```bash
make all                        # debug (default)
make PROFILE=production all     # ODR check via LTO — equivalente a hardened VM
make PROFILE=tsan all           # ThreadSanitizer
make PROFILE=asan all           # AddressSanitizer + UBSan
```

> ⚠️ `PROFILE=production` activa `-flto` (Link Time Optimization), que fuerza verificación ODR cross-module en link time. Es el único profile que detecta ODR violations — **P0 bloqueante por decisión del Consejo DAY 138 (8/8)**. Tiempo de build estimado: 30–45 min en VM por la fase LTO del linker.

### Hardened VM (ADR-030 Variant A)

```bash
make hardened-full   # destroy → up → provision → build → deploy → check
```

---

## 🗺️ Roadmap

### ✅ DONE — DAY 138 (1 May 2026) — ADR-029 Variant B pipeline 🎉
- [x] DEBT-CAPTURE-BACKEND-ISP-001 CERRADA — `CaptureBackend` 5 métodos puros
- [x] DEBT-VARIANT-B-PCAP-IMPL-001 CERRADA — pipeline pcap → proto → LZ4 → ChaCha20 → ZMQ
- [x] Suite 8 tests Variant B — 8/8 PASSED en make test-all
- [x] `PcapCallbackData` — mecanismo callback sin friend/miembros públicos
- [x] Wire format idéntico a Variant A — ml-detector recibe ambos sin modificación
- [x] Consejo 8/8 DAY 138 — 7 veredictos, ODR P0 bloqueante confirmado

### ✅ DONE — DAY 137 (30 Apr 2026) — feature/variant-b-libpcap 🎉
- [x] EMECAS dev + EMECAS hardened PASSED
- [x] capture_backend.hpp · ebpf_backend.hpp/cpp · pcap_backend.hpp/cpp
- [x] main_libpcap.cpp — Variant B sin #ifdef
- [x] sniffer-libpcap compilable y arranca limpio

### ✅ DONE — DAY 135-136: v0.6.0 🎉
- [x] make hardened-full EMECAS PASSED
- [x] feature/adr030-variant-a → main MERGEADO
- [x] Tag v0.6.0-hardened-variant-a publicado
- [x] arXiv replace v15 → v18 ENVIADO

### ✅ DONE — DAY 133-134: ADR-030 + ADR-040 + ADR-041 🎉
- [x] AppArmor 6/6 enforce · Falco 10 reglas · cap_bpf · Paper v18
- [x] ADR-040 ML Retraining Contract (8/8, 17 enmiendas)
- [x] ADR-041 Hardware Acceptance Metrics FEDER (8/8)
- [x] Pipeline E2E hardened · check-prod-all PASSED

### 🔜 NEXT — DAY 139

| Priority | Task |
|---|---|
| 🔴 P0 BLOQUEANTE | `DEBT-COMPILER-WARNINGS-CLEANUP-001` — sub-tarea ODR (UB en C++20) |
| 🔴 P0 | `DEBT-VARIANT-B-CONFIG-001` — sniffer-libpcap.json propio + test e2e |
| 🔴 P0 | `DEBT-IRP-NFTABLES-001` — argus-network-isolate con nft -f transaccional |
| 🟡 P1 | `DEBT-CRYPTO-MATERIAL-STORAGE-001` — prototipo HashiCorp Vault |

### 🔜 THEN — PHASE 5: Adversarial Capture-Retrain Loop

- DEBT-PENTESTER-LOOP-001 — ACRL completo
- BACKLOG-FEDER-001 — presentación Andrés Caro Lindo
- aRGus-production ARM64
- aRGus-seL4 research branch (post-FEDER, equipo especializado)

---

## 🗺️ Milestones

- ✅ DAY 111: **arXiv:2604.04952 PUBLICADO** 🎉
- ✅ DAY 113: **ADR-025 MERGED — v0.3.0-plugin-integrity** 🎉
- ✅ DAY 118: **PHASE 3 COMPLETADA — v0.4.0** 🎉
- ✅ DAY 122: **PHASE 4 COMPLETADA — v0.5.0-preproduction** 🎉
- ✅ DAY 124: **ADR-037 MERGED — v0.5.1-hardened** 🎉
- ✅ DAY 129: **CWE-78 CERRADO — execv() sin shell** 🎉
- ✅ DAY 130: **REGLA EMECAS · libFuzzer 2.4M runs** 🎉
- ✅ DAY 133: **ADR-030 Variant A — cap_bpf · AppArmor 6/6 · Falco 10 reglas** 🎉
- ✅ DAY 134: **ADR-040 (8/8, 17 enmiendas) · ADR-041 FEDER HW Metrics (8/8)** 🎉
- ✅ DAY 136: **v0.6.0-hardened-variant-a · merge main** 🎉
- ✅ DAY 137: **feature/variant-b-libpcap · sniffer-libpcap compilable · KISS** 🎉
- ✅ DAY 138: **ISP cerrado · pipeline Variant B completo · 8/8 tests · Consejo 8/8** 🎉
- ✅ DAY 139: **192→67 warnings — Wreorder·OpenSSL·Wsign-conversion·Wconversion eliminados** 🎉
- ✅ DAY 140: **192→0 warnings · -Werror activo · ODR limpio con LTO · Jenkinsfile skeleton · THIRDPARTY-MIGRATIONS.md** 🎉
- ✅ DAY 141: **DEBT-VARIANT-B-CONFIG-001 · sniffer-libpcap.json · exclusión mutua · emails FEDER** 🎉
- 🔜 DAY 142: **DEBT-IRP-NFTABLES-001 sesión 1 · DEBT-VARIANT-B-BUFFER-SIZE-001**

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Kimi** (Moonshot) · **Mistral**

Metodología: desacuerdo estructurado. Documentado en §6 del preprint.

---

## Hardened Deployment (ADR-030 Variant A)

```bash
make hardened-full          # EMECAS sagrado — destroy → up → provision → build → deploy → check
make hardened-redeploy      # iteración rápida sin destroy
make prod-deploy-seeds      # deploy seeds explícito (nunca en EMECAS)
make check-prod-all         # 5/5 gates: BSR + AppArmor + cap_bpf + permissions + Falco
```

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*````