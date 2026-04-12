# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_The_Council-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: 25/25 + INTEG](https://img.shields.io/badge/Tests-25%2F25_%2B_INTEG_4a_4b_4c_4d_4e-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Integrity](https://img.shields.io/badge/Plugin_Integrity-ADR--025_Ed25519_MERGED-brightgreen)](docs/adr/ADR-025-plugin-integrity-ed25519.md)
[![Plugin Loader](https://img.shields.io/badge/Plugin_Loader-ADR--023_PHASE2_COMPLETE_5%2F5-brightgreen)](docs/adr/ADR-012%20plugin%20loader%20architecture.md)
[![ADR-029](https://img.shields.io/badge/ADR--029-async--signal--safe_APPROVED-green)](docs/adr/ADR-029-rag-security-global-plugin-loader-async-signal-safe.md)
[![PHASE 3](https://img.shields.io/badge/PHASE_3-4%2F6_items_DONE-yellow)]()
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)
[![Docs](https://img.shields.io/badge/Docs-alonsoir.github.io%2Fargus-blue)](https://alonsoir.github.io/argus/)

📜 Living contracts: [Protobuf schema](https://github.com/alonsoir/argus/blob/main/docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](https://github.com/alonsoir/argus/blob/main/docs/contracts/JSON%20contracts.md) · [RAG API](https://github.com/alonsoir/argus/blob/main/docs/contracts/Rag%20security%20commands.md)

---

⚠️ Active development branch: `feature/phase3-hardening`
For current state, see that branch. `main` is tagged `v0.3.0-plugin-integrity`.

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

**Physical and removable-media vectors are explicitly out of scope by conscious design decision.** File system activity, USB-borne payloads, and removable storage are not monitored. This is an architectural boundary, not an oversight. USB ports in the DMZ should be physically or firmware-disabled by the IT team; internal policy should prohibit removable media on monitored hosts (CIS Controls v8).

**Complementary mode with Wazuh:** for organizations requiring file integrity monitoring, ML Defender is designed to operate alongside battle-tested tools like [Wazuh](https://wazuh.com). The two systems are architecturally orthogonal — ML Defender defends the network perimeter; Wazuh defends the host state. Integration via raw TCP event streaming is on the roadmap (FEAT-INT-1).

---

## 📊 Validated Results (DAY 115 — 12 April 2026)

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
| **Plugin Loader** | **ADR-023 PHASE 2 COMPLETE (5/5)** | 2a+2b+2c+2d+2e — 12/12 INTEG tests PASSED |
| **Plugin Integrity** | **ADR-025 MERGED — v0.3.0-plugin-integrity** | Ed25519 + TOCTOU-safe dlopen, 7/7 SIGN tests |
| **Test suite** | **25/25 + 4a 3/3 + 4b + 4c 3/3 + 4d 3/3 + 4e 3/3 + SIGN-1..7** | DAY 114 |
| **PHASE 3 CI gate** | **TEST-PROVISION-1 PASSED 5/5** | DAY 115 |

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
| Async-signal-safe handlers (DEBT-SIGNAL-001) | ✅ write(STDERR_FILENO), verified via objdump |
| std::atomic shutdown_called_ (DEBT-SIGNAL-002) | ✅ DAY 114 |
| D8-pre bidireccional (FIX-C + FIX-D) | ✅ NORMAL+nullptr→terminate, 64KB hard limit |
| Plugin Loader PHASE 2a–2e (all 6 components) | ✅ 12/12 INTEG tests PASSED |
| Plugin integrity Ed25519 (ADR-025) | ✅ MERGED main — v0.3.0-plugin-integrity |
| Plugin signing key rotation (DEBT-SIGN-AUTO) | ✅ provision.sh check-plugins dev/prod modes |
| Dev plugins blocked from production (DEBT-HELLO-001) | ✅ BUILD_DEV_PLUGINS=OFF + validate-prod-configs gate |
| systemd hardening (PHASE 3) | ✅ Restart=always, LD_PRELOAD=unset, min capabilities |
| CI gate TEST-PROVISION-1 (5 checks) | ✅ pipeline-start dependency — DAY 115 |
| ADR-028 RAG Ingestion Trust Model | ✅ FAISS anti-poisoning |
| ADR-024 Noise_IKpsk3 — OQs 5..8 closed | ✅ Design complete, implementation post-PHASE 3 |
| provision.sh --reset (key rotation) | ⏳ DEBT-ADR025-D11 — deadline 18 Apr 2026 |
| AppArmor profiles (6 components) | ⏳ PHASE 3 ítem 5 — complain→enforce |
| ADR-032 Plugin Distribution Chain (HSM) | ⏳ APROBADO — YubiKey OpenPGP Ed25519 |
| ADR-033 TPM Measured Boot | ⏳ PROPUESTO |
| Dynamic group key agreement (ADR-024 impl) | ⏳ Post-PHASE 3 |
| provision.sh reproducible (destroy→6/6) | ✅ DAY 108 |

---

## 🗺️ Roadmap

### ✅ DONE — DAY 115 (12 Apr 2026)
- [x] ADR-024 OQ-5..8 closed — Noise_IKpsk3 design complete, implementation unblocked
- [x] **PHASE 3 ítem 1:** 6 systemd units (Restart=always, LD_PRELOAD=unset, build-active profiles)
- [x] **PHASE 3 ítem 2:** DEBT-SIGN-AUTO — provision.sh check-plugins (dev sign / prod verify-only)
- [x] **PHASE 3 ítem 3:** DEBT-HELLO-001 — BUILD_DEV_PLUGINS=OFF + production JSONs cleaned (bug: 4 components had active:true)
- [x] **PHASE 3 ítem 4:** TEST-PROVISION-1 CI gate — 5 checks, pipeline-start dependency

### ✅ DONE — DAY 114 (11 Apr 2026)
- [x] ADR-025: Plugin Integrity Ed25519 + TOCTOU-safe dlopen — **MERGED main** 🎉
- [x] Tag: **v0.3.0-plugin-integrity**
- [x] TEST-INTEG-4d: ml-detector PHASE 2d, 3/3 PASSED
- [x] DEBT-SIGNAL-001/002: async-signal-safe handlers + atomic<bool>
- [x] arXiv Replace v15 submitted
- [x] ADR-032: Plugin Distribution Chain (YubiKey HSM) — APROBADO
- [x] PHASE 3 branch opened: `feature/phase3-hardening`

### ✅ DONE — DAY 111–113
- [x] FIX-C/D: D8-pre bidireccional + MAX_PLUGIN_PAYLOAD_SIZE
- [x] TEST-INTEG-4c/4e: 3/3 PASSED
- [x] PHASE 2d/2e: ml-detector + rag-security plugin integration
- [x] **arXiv:2604.04952 [cs.CR] PUBLICADO** 🎉

### 🔜 NEXT — PHASE 3 remaining (feature/phase3-hardening)
- [ ] **DEBT-ADR025-D11:** provision.sh --reset (key rotation without auto-signing) — **deadline 18 Apr**
- [ ] **TEST-PROVISION-1 checks 6+7:** file permissions + JSON/plugin consistency
- [ ] **AppArmor profiles:** 6 components — complain → audit → enforce

### P3 — Post-PHASE 3
- [ ] ADR-024 Noise_IKpsk3 implementation (OQs closed, ready to build)
- [ ] DEBT-TOOLS-001: synthetic injectors + PluginLoader integration
- [ ] Stress test CTU-13 Neris with real pipeline (F1=0.9985 with plugins active)
- [ ] ADR-032 Fase A: manifest JSON format + multi-key loader + revocation
- [ ] ADR-032 Fase B: YubiKey OpenPGP signing (hardware acquisition)
- [ ] ADR-030 activation: AppArmor enforcing + Raspberry Pi hardware
- [ ] etcd legacy refactoring (etcd = config distribution + heartbeat only)
- [ ] ADR-031 spike: seL4/Genode (2–3 weeks)
- [ ] BARE-METAL stress test

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

### CI Gate (PHASE 3)
```bash
make test-provision-1   # 5 checks: keys, plugin sigs, prod configs, symlinks, systemd units
make validate-prod-configs   # ensure no dev plugins in production JSON configs
```

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

Seven large language models serve as intellectual co-reviewers across all development phases:

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Parallel.ai**

Methodology: structured disagreement. Problems must be demonstrated with compilable tests or mathematics before fixes are proposed. Documented in the preprint §6 (Consejo de Sabios / Test-Driven Hardening).

---

## 🗺️ Milestones

- ✅ DAY 106: Paper Draft v11 + arXiv SUBMITTED
- ✅ DAY 107: MAC failure root cause resolved
- ✅ DAY 108: provision.sh reproducible · ADR-026/027 committed
- ✅ DAY 109: PHASE 2b CLOSED · TEST-INTEG-4b · Paper v12 · ADR-028 APROBADO
- ✅ DAY 110: PluginMode + PHASE 2c CLOSED · Paper v13 · 6/6 RUNNING
- ✅ DAY 111: **arXiv:2604.04952 PUBLICADO** 🎉 · FIX-C/D · PHASE 2d · ADR-029
- ✅ DAY 112: PHASE 2e CLOSED · TEST-INTEG-4e 3/3 · ADR-030/031 documented
- ✅ DAY 113: ADR-025 IMPLEMENTED · 11/11 tests · Paper v14
- ✅ DAY 114: **ADR-025 MERGED — v0.3.0-plugin-integrity** 🎉 · TEST-INTEG-4d · Signal safety · arXiv v15 · ADR-032 APROBADO
- ✅ DAY 115: **PHASE 3 ítems 1-4 DONE** 🎉 · ADR-024 OQs 5..8 closed · TEST-PROVISION-1 CI gate · DEBT-HELLO-001 (bug: 4×active:true fixed)

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*