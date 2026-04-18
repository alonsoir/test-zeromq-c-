cat > /tmp/README.md << 'MDEOF'
# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_7_Models-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: make test-all VERDE](https://img.shields.io/badge/Tests-make_test--all_VERDE-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Integrity](https://img.shields.io/badge/Plugin_Integrity-ADR--025_Ed25519_MERGED-brightgreen)](docs/adr/ADR-025-plugin-integrity-ed25519.md)
[![PHASE 3](https://img.shields.io/badge/PHASE_3-COMPLETADO-brightgreen)]()
[![AppArmor](https://img.shields.io/badge/AppArmor-6%2F6_enforce-brightgreen)]()
[![Reproducible](https://img.shields.io/badge/Infra-make_bootstrap-brightgreen)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-F1%3D0.9978_CIC--IDS--2017-brightgreen)]()
[![Medical Gate](https://img.shields.io/badge/Medical_Gate-Precision≥0.99_DAY_122-orange)]()
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

## 📊 Validated Results (DAY 120 — 17 April 2026)

| Metric | Value | Notes |
|---|---|---|
| **F1-score (CTU-13 Neris)** | **0.9985** | Stable across 4 replay runs |
| **Precision** | **0.9969** | |
| **Recall** | **1.0000** | Zero missed attacks (FN=0) |
| **XGBoost F1 (CIC-IDS-2017)** | **0.9978** | vs RF baseline 0.9968 (+0.001) |
| **XGBoost Precision (CIC-IDS-2017)** | **0.9973** | vs RF baseline 0.9944 (+0.003) |
| **XGBoost ROC-AUC** | **1.0000** | 2.83M flows, 23 features |
| **Inference latency** | **0.24–1.06 μs** | Per-class, embedded C++20 |
| **Throughput ceiling (virtualized)** | **~33–38 Mbps** | VirtualBox NIC limit, not pipeline |
| **Stress test** | **2,374,845 packets — 0 drops** | 100 Mbps requested, loop=3 bigFlows |
| **RAM (full pipeline)** | **~1.28 GB** | Stable under load |
| **Pipeline components** | **6/6 RUNNING** | Reproducible from `make bootstrap` |
| **Plugin integrity** | **ADR-025 MERGED** | Ed25519 + TOCTOU-safe dlopen |
| **Plugin integ tests** | **6/6 PASSED incl. TEST-INTEG-SIGN** | DAY 120 |
| **CI gate** | **TEST-PROVISION-1 PASSED 8/8** | DAY 118 |
| **AppArmor** | **6/6 enforce** | 0 denials — DAY 118 |
| **Reproducibility** | **vagrant destroy × 2 → make bootstrap validated** | DAY 120 |
| **XGBoost gate médico** | **BENIGN<0.1 / ATTACK>0.999 (FTP-Patator real)** | DAY 121 |
| **XGBoost DDoS latency** | **0.15 µs/sample (20× faster than RF)** | DAY 121 |
| **XGBoost Ransomware latency** | **2.09 µs/sample (6× faster than RF)** | DAY 121 |

---

## 🏗️ Architecture

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
│  │                  │  XGBoost plugin ADR-026 ✅ F1=0.9978      │
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
│  └──────────────────┘                                            │
└──────────────────────────────────────────────────────────────────┘

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
| Pubkey plugin-loader from runtime file | ✅ No hardcoding — DAY 120 |
| make bootstrap idempotent | ✅ vagrant destroy × 2 validated — DAY 120 |
| XGBoost model signature Ed25519 | ✅ sign-models — DAY 120 |
| explicit_bzero(seed) post-HKDF | ⏳ DEBT-CRYPTO-003a |
| mlock() derived keys | ⏳ DEBT-CRYPTO-003a |
| Seed audit (never in CMake/build) | ⏳ DEBT-SEED-AUDIT-001 — DAY 121 |
| TPM measured boot (seed in hardware) | ⏳ ADR-033 post-PHASE 4 |
| ADR-032 Plugin Distribution Chain (HSM) | ⏳ YubiKey OpenPGP Ed25519 |

---

## 🚀 Quick Start

> **Critical rules:**
> - Always use `make <target>`. Never compile or install manually in the VM.
> - The Vagrantfile and Makefile are the single source of truth.
> - Complex shell logic → `tools/script.sh`, never inline in Makefile.

---

### 👶 First time — fresh clone

```bash
git clone https://github.com/alonsoir/argus.git
cd argus
git checkout feature/adr026-xgboost

# Start VM — provisions ALL system dependencies automatically
# (libsodium 1.0.19, XGBoost 3.2.0, ONNX, FAISS, tmux, xxd, libgomp...)
make up
# Wait ~20-30 minutes for full provisioning

# Bootstrap everything in one command
make bootstrap
# Expected output:
#   [1/8] post-up-verify     ✅
#   [2/8] check-system-deps  ✅
#   [3/8] set-build-profile  ✅ 6/6
#   [4/8] install-systemd    ✅ 6/6
#   [5/8] pipeline-build     ✅ (reads pubkey from runtime file automatically)
#   [6/8] sign-plugins       ✅ 2/2
#   [7/8] test-provision-1   ✅ 8/8
#   [8/8] pipeline-start     ✅ 6/6 RUNNING
#   ✅ Bootstrap completado — 6/6 RUNNING
```

---

### 🔄 Daily workflow — VM already running

```bash
# If VM is stopped:
make up
# Wait for VM to boot (~1-2 min, no reprovisioning)

# Then:
make pipeline-stop
make pipeline-build 2>&1 | tail -5
make sign-plugins
make sign-models
make test-provision-1
make pipeline-start && make pipeline-status
make plugin-integ-test
```

---

### 🔁 Full rebuild from scratch (vagrant destroy)

```bash
make destroy          # vagrant destroy -f
make up               # vagrant up — full reprovisioning ~20-30 min
make bootstrap        # all 8 steps in one command
make test-all         # full test suite verification
```

---

### ✅ CI Gate

```bash
make test-all
# Runs: libs + components + TEST-PROVISION-1 (8/8)
#       TEST-INVARIANT-SEED + plugin-integ-test (6/6 incl. TEST-INTEG-SIGN)
#       TEST-INTEG-XGBOOST-1
```

---

### 🔑 Key Rotation (after vagrant destroy + up)

The keypair rotates automatically during provisioning.
`pipeline-build` reads the new pubkey from the runtime file automatically.

```bash
make bootstrap        # handles everything, including new pubkey
# No manual sync-pubkey needed (DEBT-PUBKEY-RUNTIME-001 resolved DAY 120)
```

---

## 🗺️ Roadmap

### ✅ DONE — DAY 120 (17 Apr 2026)
- [x] **DEBT-PUBKEY-RUNTIME-001** — pubkey from runtime file, no hardcoding ✅
- [x] **DEBT-BOOTSTRAP-001** — `make bootstrap` 8 steps, idempotent ✅
- [x] **DEBT-INFRA-VERIFY-001/002** — `make check-system-deps` + `make post-up-verify` ✅
- [x] **Idempotency validated × 2** — vagrant destroy cycles ✅
- [x] **ADR-026 PASO 4a** — `docs/xgboost/features.md` — 23 features LEVEL1, CIC-IDS-2017 ✅
- [x] **ADR-026 PASO 4b** — `docs/xgboost/plugin-contract.md` — float32[23] contract ✅
- [x] **ADR-026 PASO 4c** — XGBoost trained: F1=0.9978, Precision=0.9973, AUC=1.0 ✅
- [x] **ADR-026 PASO 4d** — `make sign-models` — Ed25519 model signature ✅
- [x] **ADR-026 PASO 4e** — `TEST-INTEG-XGBOOST-1 PASSED` — real inference ✅
- [x] libgomp symlink in Vagrantfile ✅
- [x] plugin_test_message moved to pipeline-build deps ✅

### ✅ DONE — DAY 119 (16 Apr 2026)
- [x] Full reproducibility from `vagrant destroy` validated ✅
- [x] 10 infrastructure gaps fixed ✅
- [x] 6/6 RUNNING + make test-all VERDE incl. TEST-INTEG-SIGN ✅

### ✅ DONE — DAY 118 (15 Apr 2026)
- [x] **PHASE 3 COMPLETADA — v0.4.0-phase3-hardening MERGEADO A MAIN** 🎉
- [x] AppArmor enforce 6/6 (0 denials) ✅

### ✅ DONE — DAY 117–111 *(see git log)*

### ✅ DONE — DAY 121 (18 Apr 2026)
- [x] **fix(provision)**: circular dependency plugin_signing.pk → plugin-loader cmake ✅
- [x] **DEBT-SEED-AUDIT-001** — seed ChaCha20 never in CMake, runtime-only mlock()+explicit_bzero() ✅
- [x] **DEBT-XGBOOST-TEST-REAL-001** — real CIC-IDS-2017 FTP-Patator flows, medical gate PASSED ✅
- [x] **DEBT-XGBOOST-DDOS-001** — XGBoost DDoS F1=1.0 on synthetic DeepSeek (20× faster than RF) ✅
- [x] **DEBT-XGBOOST-RANSOMWARE-001** — XGBoost Ransomware F1=0.9932 (6× faster than RF) ✅
- [x] **vagrant destroy × 3** — final idempotency certification ✅
- [x] **PAPER-SECTION-4** — §4.1 real + §4.2 synthetic with explicit limitations ✅
- [x] **sign-models × 3** — Ed25519 64B signatures for all XGBoost models ✅
- [ ] 🔴 **DEBT-PRECISION-GATE-001** — Precision=0.9875 < 0.99 medical gate. BLOQUEANTE MERGE.

### 🔜 NEXT — DAY 122 (feature/adr026-xgboost)
- [ ] **DEBT-PRECISION-GATE-001** — Wednesday held-out test set (BLIND). Train=Tue+Thu+Fri, Val=20% train, Test=Wednesday once. Target: Precision≥0.99

### P3 — Post-PHASE 4
- [ ] DEBT-CRYPTO-003a: mlock() + explicit_bzero(seed) post-HKDF
- [ ] ADR-037 Snyk C++ Security Hardening
- [ ] ADR-026 full: 3 XGBoost plugins + latency table + paper §4
- [ ] ADR-024 Noise_IKpsk3 implementation
- [ ] ADR-032 Fase A: manifest + multi-key loader
- [ ] ADR-033 TPM 2.0 Measured Boot
- [ ] ADR-029 hardened variants: AppArmor+eBPF/XDP · AppArmor+libpcap · seL4+libpcap
- [ ] BARE-METAL stress test

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

Seven large language models serve as intellectual co-reviewers across all development phases:

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Kimi** (Moonshot) · **Mistral**

Methodology: structured disagreement. Problems must be demonstrated with compilable tests or mathematics before fixes are proposed. Documented in the preprint §6.

---

## 🗺️ Milestones

- ✅ DAY 111: **arXiv:2604.04952 PUBLICADO** 🎉
- ✅ DAY 113: ADR-025 IMPLEMENTED · 11/11 tests
- ✅ DAY 114: **ADR-025 MERGED — v0.3.0-plugin-integrity** 🎉 · arXiv v15
- ✅ DAY 115: **PHASE 3 ítems 1-4 DONE** 🎉
- ✅ DAY 116: **PHASE 3 CORE COMPLETADO** 🎉
- ✅ DAY 118: **PHASE 3 COMPLETADA — v0.4.0 MERGEADO** 🎉 · AppArmor 6/6 enforce
- ✅ DAY 119: **Full reproducibility from vagrant destroy validated** 🎉
- ✅ DAY 120: **make bootstrap + XGBoost F1=0.9978 + DEBT-PUBKEY-RUNTIME-001** 🎉
- ✅ DAY 121: **DEBTs bloqueantes cerrados + gate médico PASADO + idempotencia ×3 certificada** 🎉

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*
MDEOF
cp /tmp/README.md /Users/aironman/CLionProjects/test-zeromq-docker/README.md
echo "✅ README.md actualizado"