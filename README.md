# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_The_Council-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: make test-all VERDE](https://img.shields.io/badge/Tests-make_test--all_VERDE-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Integrity](https://img.shields.io/badge/Plugin_Integrity-ADR--025_Ed25519_MERGED-brightgreen)](docs/adr/ADR-025-plugin-integrity-ed25519.md)
[![PHASE 3](https://img.shields.io/badge/PHASE_3-COMPLETADO-brightgreen)]()
[![AppArmor](https://img.shields.io/badge/AppArmor-6%2F6_enforce-brightgreen)]()
[![Reproducible](https://img.shields.io/badge/Infra-Reproducible_from_vagrant_destroy-brightgreen)]()
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)
[![Docs](https://img.shields.io/badge/Docs-alonsoir.github.io%2Fargus-blue)](https://alonsoir.github.io/argus/)

📜 Living contracts: [Protobuf schema](docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](docs/contracts/JSON%20contracts.md) · [RAG API](docs/contracts/Rag%20security%20commands.md)

---

✅ `main` is tagged `v0.4.0-phase3-hardening` — PHASE 3 complete. Active branch: `feature/adr026-xgboost`.

---

## 📄 Preprint

**ML Defender (aRGus NDR)** is documented in a peer-reviewed preprint published on **arXiv cs.CR** (April 2026).

> *ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and Anomalous Traffic Detection in Resource-Constrained Organizations*
> — Alonso Isidoro Román

**arXiv:** [arXiv:2604.04952 \[cs.CR\]](https://arxiv.org/abs/2604.04952)
**DOI:** https://doi.org/10.48550/arXiv.2604.04952
**Published:** 3 April 2026 · Draft v15 · MIT license
**Code:** https://github.com/alonsoir/argus

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* — Systems built like Roman roads, designed to endure.

> ML Defender stops ransomware propagation. What comes next is detecting infiltration.

---

## 🛡️ Threat Model Scope

ML Defender is a **Network Detection and Response (NDR)** system. Its guiding principle is **network surveillance**: every component operates on network traffic — packet capture, flow-level feature extraction, ML classification, firewall response.

**Physical and removable-media vectors are explicitly out of scope by conscious design decision.** File system activity, USB-borne payloads, and removable storage are not monitored. This is an architectural boundary, not an oversight.

**Complementary mode with Wazuh:** for organizations requiring file integrity monitoring, ML Defender is designed to operate alongside battle-tested tools like [Wazuh](https://wazuh.com).

---

## 📊 Validated Results (DAY 119 — 16 April 2026)

| Metric | Value | Notes |
|---|---|---|
| **F1-score (CTU-13 Neris)** | **0.9985** | Stable across 4 replay runs |
| **Precision** | **0.9969** | |
| **Recall** | **1.0000** | Zero missed attacks (FN=0) |
| **Inference latency** | **0.24–1.06 μs** | Per-class, embedded C++20 |
| **Throughput ceiling (virtualized)** | **~33–38 Mbps** | VirtualBox NIC limit, not pipeline |
| **Stress test** | **2,374,845 packets — 0 drops** | 100 Mbps requested, loop=3 bigFlows |
| **RAM (full pipeline)** | **~1.28 GB** | Stable under load |
| **Pipeline components** | **6/6 RUNNING** | Reproducible from `vagrant destroy` |
| **Plugin integrity** | **ADR-025 MERGED — v0.3.0-plugin-integrity** | Ed25519 + TOCTOU-safe dlopen |
| **Plugin integ tests** | **6/6 PASSED incl. TEST-INTEG-SIGN** | DAY 119 — SIGN reparado |
| **CI gate** | **TEST-PROVISION-1 PASSED 8/8** | DAY 118 |
| **AppArmor** | **6/6 enforce** | 0 denials — DAY 118 |
| **Reproducibility** | **vagrant destroy → full rebuild validated** | DAY 119 |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       ML Defender Pipeline                       │
├──────────────────────────────────────────────────────────────────┤
│  Network Traffic                                                 │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  sniffer (C++20) │  eBPF/XDP zero-copy packet capture        │
│  │                  │  ShardedFlowManager (16 shards)           │
│  │                  │  Fast Detector (rule-based heuristics)    │
│  │                  │  plugin-loader PHASE 2c ✅ NORMAL         │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (ChaCha20-Poly1305 encrypted)                  │
│  ┌──────────────────┐                                            │
│  │  ml-detector     │  4× Embedded RandomForest classifiers     │
│  │  (C++20)         │  DDoS: 0.24 μs | Ransomware: 1.06 μs     │
│  │                  │  plugin-loader PHASE 2d ✅ post-inference  │
│  │                  │  XGBoost plugin (ADR-026 — in progress)   │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (encrypted)                                    │
│  ┌──────────────────┐                                            │
│  │  etcd-server     │  Component registration + JSON config     │
│  │  (C++20)         │  HMAC key management + seed distribution  │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │ firewall-acl     │  Autonomous blocking via ipset/iptables   │
│  │ agent (C++20)    │  plugin-loader PHASE 2a ✅ NORMAL         │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  FAISS + SQLite event ingestion           │
│  │  (C++20)         │  plugin-loader PHASE 2b ✅ READONLY       │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-security    │  TinyLlama natural language interface      │
│  │  (C++20+LLM)     │  Local inference — no cloud exfiltration  │
│  │                  │  plugin-loader PHASE 2e ✅ READONLY       │
│  └──────────════════╝                                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Security Properties

| Property | Status |
|---|---|
| ChaCha20-Poly1305 AEAD encryption | ✅ All inter-component transport |
| HKDF-SHA256 channel-scoped key derivation | ✅ Distinct tx/rx subkeys per channel |
| libsodium 1.0.19 (compiled from source) | ✅ Required for HKDF-SHA256 API |
| HMAC-SHA256 log integrity | ✅ All CSV logs |
| Autonomous blocking (ipset/iptables) | ✅ Millisecond response |
| Fail-closed design (std::terminate) | ✅ All 6 main() functions |
| Plugin integrity Ed25519 (ADR-025) | ✅ MERGED main — v0.3.0-plugin-integrity |
| Plugin signing key rotation | ✅ provision.sh check-plugins dev/prod modes |
| Dev plugins blocked from production | ✅ BUILD_DEV_PLUGINS=OFF + validate-prod-configs |
| systemd hardening (PHASE 3) | ✅ Restart=always, LD_PRELOAD=unset |
| CI gate TEST-PROVISION-1 (8/8 checks) | ✅ DAY 118 |
| AppArmor profiles (6 components) | ✅ 6/6 enforce — DAY 118 |
| Reproducible from vagrant destroy | ✅ Validated DAY 119 |
| explicit_bzero(seed) post-HKDF | ⏳ DEBT-CRYPTO-003a |
| mlock() derived keys | ⏳ DEBT-CRYPTO-003a |
| TPM measured boot (seed in hardware) | ⏳ ADR-033 post-PHASE 4 |
| ADR-032 Plugin Distribution Chain (HSM) | ⏳ YubiKey OpenPGP Ed25519 |

---

## 🚀 Quick Start

### First clone / full rebuild from scratch

> **Critical rule:** Always use `make <target>`. Never compile or install manually in the VM.
> The Vagrantfile and Makefile are the single source of truth for build order and dependencies.

```bash
git clone https://github.com/alonsoir/argus.git
cd argus

# 1. Start VM (provisions all system dependencies automatically)
make up
# Wait ~20-30 minutes for full provisioning
# Includes: libsodium 1.0.19, XGBoost 3.2.0, ONNX, FAISS, tmux, xxd...

# 2. Sync plugin signing pubkey (reads active keypair from VM → recompiles plugin-loader)
make sync-pubkey

# 3. Activate build profile symlinks
make set-build-profile       # default: debug

# 4. Install systemd units
make install-systemd-units

# 5. Build all components (libs → components, in correct dependency order)
make pipeline-build

# 6. Sign plugins with Ed25519 keypair (ADR-025)
make sign-plugins

# 7. Run CI gate (8/8 checks)
make test-provision-1

# 8. Start pipeline
make pipeline-start && make pipeline-status
# Expected: 6/6 RUNNING

# 9. Verify plugin integration
make plugin-integ-test
# Expected: 6/6 PASSED including TEST-INTEG-SIGN
```

> **Coming in DAY 120:** `make bootstrap` will automate all 9 steps with checkpoints.

### Daily workflow (VM already running)

```bash
make pipeline-stop
make pipeline-build 2>&1 | tail -5
make sign-plugins
make test-provision-1
make pipeline-start && make pipeline-status
make plugin-integ-test
```

### CI Gate

```bash
make test-all   # libs + components + TEST-PROVISION-1 (8/8) + TEST-INVARIANT-SEED + plugin-integ-test
```

### Key Rotation

```bash
# After vagrant destroy + up, the keypair rotates automatically.
# Always run sync-pubkey BEFORE sign-plugins:
make sync-pubkey    # reads active pubkey → updates CMakeLists.txt → recompiles plugin-loader
make sign-plugins   # re-signs all plugins with new keypair
```

---

## 🗺️ Roadmap

### ✅ DONE — DAY 119 (16 Apr 2026)
- [x] Full reproducibility from `vagrant destroy` validated ✅
- [x] 10 infrastructure gaps fixed in Vagrantfile + Makefile ✅
- [x] libsodium 1.0.19 added to Vagrantfile (before ONNX/FAISS/XGBoost) ✅
- [x] tmux + xxd added to base packages ✅
- [x] pipeline-build explicit lib dependencies ✅
- [x] install-systemd-units + set-build-profile Makefile targets ✅
- [x] plugin_xgboost API corrected (PluginResult + PluginConfig*) ✅
- [x] plugin_test_message + /usr/lib/ml-defender/plugins/ in Vagrantfile ✅
- [x] make sync-pubkey: robust pubkey sync after vagrant destroy ✅
- [x] 6/6 RUNNING + make test-all VERDE incl. TEST-INTEG-SIGN ✅

### ✅ DONE — DAY 118 (15 Apr 2026)
- [x] **PHASE 3 COMPLETADA — v0.4.0-phase3-hardening MERGEADO A MAIN** 🎉
- [x] AppArmor enforce 6/6 (0 denials) ✅
- [x] CHANGELOG-v0.4.0.md ✅
- [x] feature/adr026-xgboost opened ✅

### ✅ DONE — DAY 117–111 *(see git log)*

### 🔜 NEXT — DAY 120 (feature/adr026-xgboost)
- [ ] **DEBT-PUBKEY-RUNTIME-001**: move pubkey to runtime file — eliminate sync-pubkey
- [ ] **DEBT-BOOTSTRAP-001**: `make bootstrap` — 9 steps, checkpoints, idempotent
- [ ] **DEBT-INFRA-VERIFY-001/002**: `make check-system-deps` + `make post-up-verify`
- [ ] `vagrant destroy && vagrant up` full reproductibility test (second run)
- [ ] PASO 3: locate RF feature set → `docs/xgboost/features.md`
- [ ] PASO 4: `scripts/train_xgboost_baseline.py` — gate Precision≥0.99 + F1≥0.9985

### P3 — Post-PHASE 4
- [ ] DEBT-CRYPTO-003a: mlock() + explicit_bzero(seed) post-HKDF
- [ ] ADR-024 Noise_IKpsk3 implementation
- [ ] ADR-037 Snyk C++ Security Hardening
- [ ] ADR-026 XGBoost plugins Track 1 (Precision ≥ 0.99 gate médico)
- [ ] ADR-032 Fase A: manifest + multi-key loader
- [ ] ADR-033 TPM 2.0 Measured Boot
- [ ] ADR-029 variantes hardened: AppArmor+eBPF/XDP · AppArmor+libpcap · seL4+libpcap
- [ ] BARE-METAL stress test

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

Seven large language models serve as intellectual co-reviewers across all development phases:

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Parallel.ai**

Methodology: structured disagreement. Problems must be demonstrated with compilable tests or mathematics before fixes are proposed. Documented in the preprint §6.

---

## 🗺️ Milestones

- ✅ DAY 111: **arXiv:2604.04952 PUBLICADO** 🎉
- ✅ DAY 113: ADR-025 IMPLEMENTED · 11/11 tests
- ✅ DAY 114: **ADR-025 MERGED — v0.3.0-plugin-integrity** 🎉 · arXiv v15
- ✅ DAY 115: **PHASE 3 ítems 1-4 DONE** 🎉 · TEST-PROVISION-1 CI gate
- ✅ DAY 116: **PHASE 3 CORE COMPLETADO** 🎉 · --reset + AppArmor complain · INVARIANTE-SEED-001
- ✅ DAY 118: **PHASE 3 COMPLETADA — v0.4.0 MERGEADO** 🎉 · AppArmor 6/6 enforce · tag v0.4.0-phase3-hardening
- ✅ DAY 119: **Full reproducibility from vagrant destroy validated** 🎉 · 10 infra fixes · make test-all VERDE

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*