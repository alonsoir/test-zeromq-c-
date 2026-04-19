# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_7_Models-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: make test-all VERDE](https://img.shields.io/badge/Tests-make_test--all_VERDE-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Integrity](https://img.shields.io/badge/Plugin_Integrity-ADR--025_Ed25519_MERGED-brightgreen)](docs/adr/ADR-025-plugin-integrity-ed25519.md)
[![PHASE 4](https://img.shields.io/badge/PHASE_4-COMPLETADA-brightgreen)]()
[![AppArmor](https://img.shields.io/badge/AppArmor-6%2F6_enforce-brightgreen)]()
[![Reproducible](https://img.shields.io/badge/Infra-make_bootstrap-brightgreen)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Prec%3D0.9945_In--Distribution-brightgreen)]()
[![OOD Finding](https://img.shields.io/badge/OOD_Finding-Published_DAY_122-orange)]()
[![PRE-PRODUCTION](https://img.shields.io/badge/Status-PRE--PRODUCTION-orange)]()
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)

📜 Living contracts: [Protobuf schema](docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](docs/contracts/JSON%20contracts.md) · [RAG API](docs/contracts/Rag%20security%20commands.md)

---

✅ `main` is tagged `v0.5.0-preproduction` — PHASE 4 complete. **PRE-PRODUCTION: do not deploy in hospitals until ACRL (DEBT-PENTESTER-LOOP-001) is complete.**

---

## 📄 Preprint

**ML Defender (aRGus NDR)** is documented in a peer-reviewed preprint published on **arXiv cs.CR** (April 2026).

> *ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and Anomalous Traffic Detection in Resource-Constrained Organizations*
> — Alonso Isidoro Román

**arXiv:** [arXiv:2604.04952 \[cs.CR\]](https://arxiv.org/abs/2604.04952)  
**DOI:** https://doi.org/10.48550/arXiv.2604.04952  
**Published:** 3 April 2026 · **Draft v16** (updated 19 April 2026) · MIT license  
**Code:** https://github.com/alonsoir/argus

Draft v16 adds: XGBoost in-distribution evaluation (Prec=0.9945/Rec=0.9818), Wednesday OOD impossibility result, §10.13 structural bias in academic datasets, §11.18 Adversarial Capture-Retrain Loop (ACRL). Cites Sommer & Paxson 2010.

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* — Systems built like Roman roads, designed to endure.

> *"Un escudo que aprende de su propia sombra."*

---

## 🛡️ Threat Model Scope

ML Defender is a **Network Detection and Response (NDR)** system. Its guiding principle is **network surveillance**: every component operates on network traffic — packet capture, flow-level feature extraction, ML classification, firewall response.

**Physical and removable-media vectors are explicitly out of scope by conscious design decision.** Complementary mode with [Wazuh](https://wazuh.com) for file integrity monitoring.

---

## 📊 Validated Results (DAY 122 — 19 April 2026)

| Metric | Value | Notes |
|---|---|---|
| **F1-score (CTU-13 Neris)** | **0.9985** | Stable across 4 replay runs |
| **Precision** | **0.9969** | |
| **Recall** | **1.0000** | Zero missed attacks (FN=0) |
| **XGBoost Precision (CIC-IDS-2017 val)** | **0.9945** | In-distribution, threshold=0.8211 |
| **XGBoost Recall (CIC-IDS-2017 val)** | **0.9818** | In-distribution |
| **XGBoost F1 (CIC-IDS-2017 val)** | **0.9881** | Val-AUCPR=0.99846 |
| **XGBoost Wednesday OOD** | **Documented impossibility** | Structural covariate shift — see §8 paper |
| **Inference latency (XGBoost)** | **1.986 µs/sample** | Gate <2µs ✅ |
| **Inference latency (RF)** | **0.24–1.06 µs** | Per-class, embedded C++20 |
| **Throughput ceiling (virtualized)** | **~33–38 Mbps** | VirtualBox NIC limit, not pipeline |
| **Stress test** | **2,374,845 packets — 0 drops** | 100 Mbps requested, loop=3 |
| **RAM (full pipeline)** | **~1.28 GB** | Stable under load |
| **Pipeline components** | **6/6 RUNNING** | Reproducible from `make bootstrap` |
| **Plugin integrity** | **ADR-025 MERGED** | Ed25519 + TOCTOU-safe dlopen |
| **AppArmor** | **6/6 enforce** | 0 denials |
| **CI gate** | **TEST-PROVISION-1 8/8** | |

---

## 🔬 DAY 122 Scientific Finding

On DAY 122, a rigorous temporal holdout evaluation on CIC-IDS-2017 revealed a structural covariate shift: Wednesday contains exclusively application-layer DoS attacks (Hulk, GoldenEye, Slowloris) absent from all training days. **No threshold can simultaneously satisfy Precision≥0.99 and Recall≥0.95 on Wednesday data.** This is not an XGBoost failure — it is an empirical impossibility result caused by the dataset's day-specific attack segregation design.

**This finding corroborates Sommer & Paxson (2010)** and provides new quantitative evidence that static classifiers trained on academic benchmarks are structurally insufficient for production NDR.

**The architectural response** — the Adversarial Capture-Retrain Loop (ACRL) — is proposed in §11.18 of the paper. The XGBoost plugin was designed from day one to be hot-swappable and Ed25519-signed (ADR-025/026) for exactly this reason.

> *"No entrenamos con Wednesday porque Wednesday no existe en el entrenamiento. Entrenamos con Tuesday, y aprendemos a detectar Wednesday en producción."* — Kimi, Consejo de Sabios DAY 122

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
│  │                  │  XGBoost plugin ADR-026 ✅ Prec=0.9945    │
│  │                  │  [PRE-PROD: ACRL pending]                 │
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
```

---

## 🚀 Quick Start

> **Critical rules:**
> - Always use `make <target>`. Never compile or install manually in the VM.
> - The Vagrantfile and Makefile are the single source of truth.

### 👶 First time — fresh clone

```bash
git clone https://github.com/alonsoir/argus.git
cd argus
make up          # vagrant up — full provisioning ~20-30 min
make bootstrap   # all 8 steps in one command
```

### 🔄 Daily workflow

```bash
make up              # if VM stopped
make pipeline-stop
make pipeline-build
make sign-plugins && make sign-models
make pipeline-start && make pipeline-status
make test-all
```

### ✅ CI Gate

```bash
make test-all
# Runs: libs + components + TEST-PROVISION-1 (8/8)
#       TEST-INVARIANT-SEED + plugin-integ-test (6/6 incl. TEST-INTEG-SIGN)
#       TEST-INTEG-XGBOOST-1
```

---

## 🗺️ Roadmap

### ✅ DONE — DAY 122 (19 Apr 2026) — PHASE 4 COMPLETADA 🎉
- [x] **DEBT-PRECISION-GATE-001** — Closed with scientific finding. In-distribution: Prec=0.9945/Rec=0.9818 ✅
- [x] **Wednesday OOD impossibility result** — Documented, sealed (md5), permanent artifact ✅
- [x] **train_xgboost_level1_v2.py** — Temporal split + validation calibration + blind test ✅
- [x] **xgboost_cicids2017_v2.ubj.sig** — Ed25519 signed via sign-model.sh ✅
- [x] **Paper Draft v16** — §8 XGBoost + §10.13 + §11.18 ACRL + sommer2010 ✅
- [x] **feature/adr026-xgboost → main** — Tag: v0.5.0-preproduction ✅
- [x] **arXiv v16 submitted** ✅

### ✅ DONE — DAY 121 (18 Apr 2026)
- [x] DEBT-SEED-AUDIT-001 ✅ · DEBT-XGBOOST-TEST-REAL-001 ✅ (medical gate PASSED)
- [x] DEBT-XGBOOST-DDOS-001 ✅ (F1=1.0, 20× faster RF) · DEBT-XGBOOST-RANSOMWARE-001 ✅
- [x] vagrant destroy × 3 idempotency certification ✅

### ✅ DONE — DAY 120–118 *(see git log)*

### 🔜 NEXT — PHASE 5: Adversarial Capture-Retrain Loop

| Priority | Task |
|---|---|
| P0 | **DEBT-PENTESTER-LOOP-001** — MITRE Caldera Fase 1 → real adversarial flows → XGBoost retraining |
| P0 | **ADR-038** — ACRL formal design document |
| P1 | DEBT-CRYPTO-003a — mlock() + explicit_bzero() |
| P1 | ADR-037 Snyk C++ hardening |
| P2 | ADR-024 Noise_IKpsk3 · ADR-032 HSM · ADR-033 TPM |
| P3 | ADR-029 hardened variants · bare-metal stress test |

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

Seven large language models serve as intellectual co-reviewers:

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Kimi** (Moonshot) · **Mistral**

Methodology: structured disagreement. Problems must be demonstrated with compilable tests or mathematics before fixes are proposed. Documented in the preprint §6.

---

## 🗺️ Milestones

- ✅ DAY 111: **arXiv:2604.04952 PUBLICADO** 🎉
- ✅ DAY 114: **ADR-025 MERGED — v0.3.0-plugin-integrity** 🎉
- ✅ DAY 118: **PHASE 3 COMPLETADA — v0.4.0 MERGEADO** 🎉
- ✅ DAY 120: **make bootstrap + XGBoost F1=0.9978** 🎉
- ✅ DAY 121: **DEBTs bloqueantes cerrados + gate médico PASADO** 🎉
- ✅ DAY 122: **PHASE 4 COMPLETADA — v0.5.0-preproduction** 🎉 · Wednesday OOD finding · arXiv v16

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.* 