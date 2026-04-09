# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_The_Council-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: 25/25 + INTEG](https://img.shields.io/badge/Tests-25%2F25_%2B_INTEG_4a_4b_4c_4e-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Loader](https://img.shields.io/badge/Plugin_Loader-ADR--023_PHASE2_COMPLETE_5%2F5-brightgreen)](docs/adr/ADR-012%20plugin%20loader%20architecture.md)
[![ADR-029](https://img.shields.io/badge/ADR--029-async--signal--safe_APPROVED-green)](docs/adr/ADR-029-rag-security-global-plugin-loader-async-signal-safe.md)
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)
[![Docs](https://img.shields.io/badge/Docs-alonsoir.github.io%2Fargus-blue)](https://alonsoir.github.io/argus/)

📜 Living contracts: [Protobuf schema](https://github.com/alonsoir/argus/blob/main/docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](https://github.com/alonsoir/argus/blob/main/docs/contracts/JSON%20contracts.md) · [RAG API](https://github.com/alonsoir/argus/blob/main/docs/contracts/Rag%20security%20commands.md)

---

⚠️ Active development branch: `feature/plugin-crypto`
For current state, see that branch. `main` is behind.

## 📄 Preprint

**ML Defender (aRGus NDR)** is documented in a peer-reviewed preprint published on **arXiv cs.CR** (April 2026).

> *ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and Anomalous Traffic Detection in Resource-Constrained Organizations*
> — Alonso Isidoro Román

**arXiv:** [arXiv:2604.04952 \[cs.CR\]](https://arxiv.org/abs/2604.04952)
**DOI:** https://doi.org/10.48550/arXiv.2604.04952
**Published:** 3 April 2026 · 28 pages · MIT license
**Code:** https://github.com/alonsoir/argus

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

## 📊 Validated Results (DAY 112 — 9 April 2026)

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
| **RAM (full pipeline)** | **~1.28 GB** | Stable under load |
| **Pipeline components** | **6/6 RUNNING** | Reproducible from `vagrant destroy` |
| **Plugin Loader** | **ADR-023 PHASE 2 COMPLETE (5/5)** | 2a+2b+2c+2d+2e — TEST-INTEG-4a+4b+4c+4e PASSED |
| **Test suite** | **25/25 + 4a 3/3 + 4b + 4c 3/3 + 4e 3/3** | DAY 112 |

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
│  │                  │  Maximum Threat Wins                      │
│  │                  │  plugin-loader PHASE 2d ✅ post-inference  │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (encrypted)                                    │
│  ┌──────────────────┐                                            │
│  │  etcd-server     │  Component registration + JSON config     │
│  │  (C++20)         │  HMAC key management + crypto seeds       │
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
│  │                  │  Anti-poisoning trust model (ADR-028)     │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-security    │  TinyLlama natural language interface      │
│  │  (C++20+LLM)     │  Local inference — no cloud exfiltration  │
│  │                  │  plugin-loader PHASE 2e ✅ READONLY (ADR-029) │
│  └──────────────────┘                                            │
└──────────────────────────────────────────────────────────────────┘
```

### Integration Philosophy

ML Defender is composable, not monolithic. All external integrations use the same transport stack: **raw TCP + Protocol Buffers + ChaCha20-Poly1305**. No HTTP, no Kafka, no WebSocket. Four reasons:

1. **Deterministic latency** (<10ms; no HTTP/Kafka jitter)
2. **Attack surface** (no HTTP parsers = no CVE surface; >90% reduction)
3. **No broker = no SPOF** (Kafka/Redis incompatible with $150–200 single-node target)
4. **Minimal footprint** (no librdkafka, no libcurl, no boost.asio)

**FEAT-INT-1 (planned):** Wazuh agents emit events via raw TCP → protobuf → ZeroMQ → rag-ingester.

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
| D8-pre bidireccional (FIX-C + FIX-D) | ✅ NORMAL+nullptr→terminate, 64KB hard limit |
| Plugin Loader PHASE 2a (firewall) | ✅ NORMAL contract, TEST-INTEG-4a PASSED |
| Plugin Loader PHASE 2b (rag-ingester) | ✅ READONLY contract, TEST-INTEG-4b PASSED |
| Plugin Loader PHASE 2c (sniffer) | ✅ NORMAL + payload real, TEST-INTEG-4c 3/3 |
| Plugin Loader PHASE 2d (ml-detector) | ✅ NORMAL post-inference, DAY 111 |
| Plugin Loader PHASE 2e (rag-security) | ✅ READONLY contract, TEST-INTEG-4e 3/3, DAY 112 |
| ADR-030 AppArmor-Hardened variant | ✅ BACKLOG — producción ARM64/x86, post-PHASE 3 |
| ADR-031 seL4/Genode research | ✅ BACKLOG — investigación pura, spike GO/NO-GO obligatorio |
| ADR-028 RAG Ingestion Trust Model | ✅ APROBADO — FAISS anti-poisoning |
| ADR-029 async-signal-safe pattern | ✅ Documented, g_plugin_loader global |
| Plugin integrity via Ed25519 (ADR-025) | ❌ Approved, pending post-PHASE 2 |
| Dynamic group key agreement (ADR-024) | ❌ Design approved, post-PHASE 2 |
| provision.sh reproducible (destroy→6/6) | ✅ DAY 108 |

---

## 🗺️ Roadmap

### ✅ DONE — DAY 111
- [x] FIX-C: D8-pre inverso — PLUGIN_MODE_NORMAL + nullptr → std::terminate()
- [x] FIX-D: MAX_PLUGIN_PAYLOAD_SIZE 64KB hard limit
- [x] TEST-INTEG-4c: 3/3 PASSED (NORMAL payload real, D8 VIOLATION, result_code error)
- [x] PHASE 2d: ml-detector invoke_all post-inferencia
- [x] ADR-029: g_plugin_loader + async-signal-safe (rag-security pattern)
- [x] PHASE 2e: rag-security READONLY contract + TEST-INTEG-4e 3/3 PASSED
- [x] ADR-030: AppArmor-Hardened variant (BACKLOG, post-PHASE 3)
- [x] ADR-031: seL4/Genode research variant (BACKLOG, spike obligatorio)
- [x] **arXiv:2604.04952 [cs.CR] PUBLICADO** 🎉

### 🔜 NEXT — DAY 112
- [ ] PHASE 2e: rag-security (ADR-029 D1-D5) + TEST-INTEG-4e
- [ ] arXiv Replace v13 (pending Consejo Q3-112)

### P1 — feature/plugin-crypto (active branch)
- [ ] ADR-025 — Plugin Integrity Verification (Ed25519 + TOCTOU-safe dlopen)
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

Methodology: structured disagreement. Problems must be demonstrated with compilable tests or mathematics before fixes are proposed. Documented in the preprint §5 (Consejo de Sabios / Test-Driven Hardening).

---

## 🗺️ Milestones

- ✅ DAY 106: Paper Draft v11 + arXiv SUBMITTED (submit/7438768)
- ✅ DAY 107: MAC failure root cause resolved
- ✅ DAY 108: provision.sh reproducible · ADR-026/027 committed
- ✅ DAY 109: PHASE 2b CLOSED · D8-light READ-ONLY · TEST-INTEG-4b · Paper v12 · ADR-028 APROBADO
- ✅ DAY 110: PluginMode + PHASE 2c CLOSED · TEST-INTEG-4b · Paper v13 · 6/6 RUNNING
- ✅ DAY 111: **arXiv:2604.04952 PUBLICADO** 🎉 · FIX-C/D · TEST-INTEG-4c · PHASE 2d · ADR-029

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*