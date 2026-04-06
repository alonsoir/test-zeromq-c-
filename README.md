# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_The_Council-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: 25/25 + INTEG](https://img.shields.io/badge/Tests-25%2F25_%2B_INTEG-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Loader](https://img.shields.io/badge/Plugin_Loader-ADR--023_PHASE2b_COMPLETE-blue)](docs/adr/ADR-012%20plugin%20loader%20architecture.md)
[![ADR-028](https://img.shields.io/badge/ADR--028-RAG_Trust_Model_APPROVED-green)](docs/adr/ADR-028.md)
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-submitted_cs.CR-red)](https://arxiv.org/search/?searchtype=author&query=Roman%2C+Alonso+Isidoro)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)
[![Docs](https://img.shields.io/badge/Docs-alonsoir.github.io%2Fargus-blue)](https://alonsoir.github.io/argus/)

📜 Living contracts: [Protobuf schema](https://github.com/alonsoir/argus/blob/main/docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](https://github.com/alonsoir/argus/blob/main/docs/contracts/JSON%20contracts.md) · [RAG API](https://github.com/alonsoir/argus/blob/main/docs/contracts/Rag%20security%20commands.md)

---

## 📄 Preprint

ML Defender (aRGus NDR) is documented in a preprint submitted to **arXiv cs.CR** (April 2026).
The paper covers architecture, cryptographic transport, plugin system, evaluation results (F1=0.9985),
and the Consejo de Sabios / Test-Driven Hardening methodology.

**arXiv submission:** `submit/7438768` — pending moderation. Link will be updated here upon publication.
**Draft v12** ready for Replace submission once v1 is announced.
**Code:** https://github.com/alonsoir/argus
**TDH methodology:** https://github.com/alonsoir/test-driven-hardening

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* — Systems built like Roman roads, designed to endure.

> ML Defender stops ransomware propagation. What comes next is detecting infiltration.

---

## 🛡️ Threat Model Scope

ML Defender is a **Network Detection and Response (NDR)** system. Its guiding principle is **network surveillance**: every component operates on network traffic — packet capture, flow-level feature extraction, ML classification, firewall response.

**Physical and removable-media vectors are explicitly out of scope by conscious design decision.** File system activity, USB-borne payloads, and removable storage are not monitored. This is an architectural boundary, not an oversight. USB ports in the DMZ should be physically or firmware-disabled by the IT team; internal policy should prohibit removable media on monitored hosts (CIS Controls v8).

**Complementary mode with Wazuh:** for organizations requiring file integrity monitoring, ML Defender is designed to operate alongside battle-tested tools like [Wazuh](https://wazuh.com). The two systems are architecturally orthogonal — ML Defender defends the network perimeter; Wazuh defends the host state. Integration via raw TCP event streaming is on the roadmap (FEAT-INT-1).

---

## 📊 Validated Results (DAY 109 — 6 April 2026)

| Metric | Value | Notes |
|---|---|---|
| **F1-score (CTU-13 Neris)** | **0.9985** | Stable across 4 replay runs |
| **Precision** | **0.9969** | |
| **Recall** | **1.0000** | Zero missed attacks (FN=0) |
| **True Positives** | **646** | Malicious flows from host 147.32.84.165 |
| **False Positives** | **2** | VirtualBox multicast/broadcast artifacts — absent in bare-metal |
| **True Negatives** | **12,075** | |
| **FPR (ML, Neris evaluation)** | **0.0002%** | |
| **FPR (Fast Detector, bigFlows)** | **6.61%** | DEBT-FD-001, Path B thresholds |
| **FP reduction (Fast → ML)** | **~500×** | ML reduces production blocks to zero on bigFlows |
| **Inference latency** | **0.24–1.06 μs** | Per-class, embedded C++20 |
| **Throughput ceiling (virtualized)** | **~33–38 Mbps** | VirtualBox NIC limit, not pipeline |
| **Stress test** | **2,374,845 packets — 0 drops, 0 errors** | 100 Mbps requested, loop=3 bigFlows |
| **RAM (full pipeline)** | **~3.5 GB** | Including TinyLlama, 3-4 cores, stable under load |
| **Pipeline components** | **6/6 RUNNING** | Reproducible from `vagrant destroy` |
| **Test suite** | **25/25 suites + TEST-INTEG-4a 3/3 + TEST-INTEG-4b PASSED** | DAY 109 |
| **Plugin Loader** | **ADR-023 PHASE 2b COMPLETE** | rag-ingester READ-ONLY contract |

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
│  │                  │  28/40 features · plugin-loader PHASE 1b  │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (ChaCha20-Poly1305 encrypted)                  │
│  ┌──────────────────┐                                            │
│  │  ml-detector     │  4× Embedded RandomForest classifiers     │
│  │  (C++20)         │  DDoS: 0.24 μs | Ransomware: 1.06 μs     │
│  │                  │  Maximum Threat Wins · plugin-loader 1b   │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (encrypted)                                    │
│  ┌──────────────────┐                                            │
│  │  etcd-server     │  Component registration + JSON config     │
│  │  (C++20)         │  HMAC key management + crypto seeds       │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │ firewall-acl     │  Autonomous blocking via ipset/iptables   │
│  │ agent (C++20)    │  plugin-loader ADR-023 PHASE 2a ✅        │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  FAISS + SQLite event ingestion           │
│  │  (C++20)         │  plugin-loader ADR-023 PHASE 2b ✅        │
│  │                  │  READ-ONLY plugin contract (pre-FAISS)    │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-security    │  TinyLlama natural language interface      │
│  │  (C++20+LLM)     │  Local inference — no cloud exfiltration  │
│  │                  │  plugin-loader PHASE 1b ✅                 │
│  └──────────────────┘                                            │
└──────────────────────────────────────────────────────────────────┘
```

### Integration Philosophy

ML Defender is composable, not monolithic. All external integrations use the same transport stack: **raw TCP + Protocol Buffers + ChaCha20-Poly1305**. No HTTP, no Kafka, no WebSocket. This ensures resource footprint and cryptographic guarantees remain consistent with resource-constrained deployment targets.

**FEAT-INT-1 (planned):** Wazuh agents and NIST-aligned scanners on DMZ hosts emit events via raw TCP → protobuf → ZeroMQ → rag-ingester. Graph quality motivation: correlating ML Defender's network anomalies with Wazuh's file integrity events produces composite attack signatures that neither system observes independently.

---

## 🔐 Security Properties

| Property | Status |
|---|---|
| ChaCha20-Poly1305 AEAD encryption | ✅ All inter-component transport |
| HKDF-SHA256 channel-scoped key derivation | ✅ Distinct tx/rx subkeys per channel |
| libsodium 1.0.19 (compiled from source) | ✅ SHA-256 verified |
| HMAC-SHA256 log integrity | ✅ All CSV logs |
| Autonomous blocking (ipset/iptables) | ✅ Millisecond response |
| Fail-closed design (std::terminate) | ✅ All 6 main() functions |
| MLD_ALLOW_UNCRYPTED escape hatch | ✅ Dev-only, explicit FATAL[DEV] log |
| Plugin Loader PHASE 2a (firewall) | ✅ MessageContext D8-v2 CRC32 |
| Plugin Loader PHASE 2b (rag-ingester) | ✅ READ-ONLY contract, TEST-INTEG-4b PASSED |
| ADR-028 RAG Ingestion Trust Model | ✅ APROBADO — FAISS como TCB lógico, anti-poisoning |
| Validation Layer D4 (rate-limit+antidating) | ✅ Configurable JSON, MAX_DRIFT 300s default |
| Rollback lógico RAG (SQLite valid flag) | ✅ O(1), no reindexación FAISS |
| D8-light READ-ONLY exception | ✅ nullptr+0 legitimate contract |
| Plugin integrity via Ed25519 (ADR-025) | ❌ Approved, pending post-PHASE 2 |
| Dynamic group key agreement (ADR-024) | ❌ Design approved, post-PHASE 2 |
| provision.sh reproducible (destroy→6/6) | ✅ DAY 108 |
| rag-security/config auto-created | ✅ DAY 109 |

---

## 🗺️ Roadmap

### ✅ DONE — DAY 109
- [x] FIX-A: MLD_ALLOW_UNCRYPTED escape hatch (3 etcd_client.cpp adapters)
- [x] FIX-B: provision.sh rag-security config dir + symlink
- [x] PHASE 2b: rag-ingester plugin_process_message() READ-ONLY contract
- [x] D8-light: READ-ONLY exception (nullptr+0 legitimate)
- [x] TEST-INTEG-4b: PASSED (make plugin-integ-test covers 4a+4b)
- [x] Paper Draft v12: threat model scope + Integration Philosophy §4
- [x] ADR-028: RAG Ingestion Trust Model — APROBADO Consejo 5/5 (2 rondas)

### 🔜 NEXT — PHASE 2c/2d/2e
- [ ] PHASE 2c — sniffer + plugin_process_message() + TEST-INTEG-4c
- [ ] PHASE 2d — ml-detector + TEST-INTEG-4d
- [ ] PHASE 2e — rag-security (g_plugin_loader global) + TEST-INTEG-4e

### P1 — feature/plugin-crypto (active branch)
- [ ] ADR-025 — Plugin Integrity Verification (Ed25519 + TOCTOU-safe dlopen)
- [ ] ADR-028 — RAG Ingestion Trust Model (before write-capable plugins)
- [ ] ADR-024 — Dynamic Group Key Agreement (Noise_IKpsk3)

### P2 — Post-PHASE 2
- [ ] BARE-METAL stress test
- [ ] DEBT-FD-001 — Fast Detector → JSON thresholds
- [ ] TEST-PROVISION-1 — CI gate: vagrant destroy → 6/6 RUNNING

---

## 🚀 Quick Start

```bash
git clone https://github.com/alonsoir/argus.git
cd argus
make up
make all
make pipeline-start
make pipeline-status
```

### F1 Validation
```bash
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
make test-replay-neris
python3 scripts/calculate_f1_neris.py logs/lab/sniffer.log --total-events 19135
```

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

Seven large language models serve as intellectual co-reviewers across all development phases:

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Parallel.ai**

Methodology: structured disagreement. Problems must be demonstrated with compilable tests or mathematics before fixes are proposed. Documented in the preprint §5 (Consejo de Sabios).

---

## 🗺️ Milestones

- ✅ DAY 106: Paper Draft v11 + arXiv SUBMITTED (submit/7438768)
- ✅ DAY 107: MAC failure root cause resolved
- ✅ DAY 108: provision.sh reproducible · ADR-026/027 committed
- ✅ DAY 109: PHASE 2b CLOSED · D8-light READ-ONLY · TEST-INTEG-4b · Paper v12 · ADR-028 APROBADO

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*