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
[![PHASE 3](https://img.shields.io/badge/PHASE_3-COMPLETADO-brightgreen)]()
[![AppArmor](https://img.shields.io/badge/AppArmor-6%2F6_enforce-brightgreen)]()
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)
[![Docs](https://img.shields.io/badge/Docs-alonsoir.github.io%2Fargus-blue)](https://alonsoir.github.io/argus/)

📜 Living contracts: [Protobuf schema](https://github.com/alonsoir/argus/blob/main/docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](https://github.com/alonsoir/argus/blob/main/docs/contracts/JSON%20contracts.md) · [RAG API](https://github.com/alonsoir/argus/blob/main/docs/contracts/Rag%20security%20commands.md)

---

✅ `main` is tagged `v0.4.0-phase3-hardening` — PHASE 3 complete. Next: `feature/adr026-xgboost`.

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

**Complementary mode with Wazuh:** for organizations requiring file integrity monitoring, ML Defender is designed to operate alongside battle-tested tools like [Wazuh](https://wazuh.com). Integration via raw TCP event streaming is on the roadmap (FEAT-INT-1).

---

## 📊 Validated Results (DAY 116 — 13 April 2026)

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
| **Plugin integ tests** | **12/12 PASSED** | TEST-INTEG-4a/4b/4c/4d/4e + SIGN |
| **CI gate** | **TEST-PROVISION-1 PASSED 8/8** | DAY 118 |
| **Key rotation** | **provision.sh --reset VALIDATED** | TEST-RESET-1/2/3 PASSED — DAY 116 |
| **AppArmor** | **6/6 enforce** | 0 denials — DAY 118 |

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
| libsodium 1.0.19 (compiled from source) | ✅ SHA-256 verified |
| HMAC-SHA256 log integrity | ✅ All CSV logs |
| Autonomous blocking (ipset/iptables) | ✅ Millisecond response |
| Fail-closed design (std::terminate) | ✅ All 6 main() functions |
| Plugin integrity Ed25519 (ADR-025) | ✅ MERGED main — v0.3.0-plugin-integrity |
| Plugin signing key rotation | ✅ provision.sh check-plugins dev/prod modes |
| Dev plugins blocked from production | ✅ BUILD_DEV_PLUGINS=OFF + validate-prod-configs |
| systemd hardening (PHASE 3) | ✅ Restart=always, LD_PRELOAD=unset |
| CI gate TEST-PROVISION-1 (8/8 checks) | ✅ DAY 118 |
| Key rotation provision.sh --reset | ✅ seed_family compartido — DAY 116 |
| AppArmor profiles (6 components) | ✅ 6/6 enforce — DAY 118 |
| explicit_bzero(seed) post-HKDF | ⏳ DEBT-CRYPTO-003a |
| mlock() derived keys | ⏳ DEBT-CRYPTO-003a |
| TPM measured boot (seed in hardware) | ⏳ ADR-033 post-PHASE 4 |
| ADR-032 Plugin Distribution Chain (HSM) | ⏳ YubiKey OpenPGP Ed25519 |

---

## 🗺️ Roadmap

### ✅ DONE — DAY 118 (15 Apr 2026)
- [x] **PHASE 3 COMPLETADA — v0.4.0-phase3-hardening MERGEADO A MAIN** 🎉
- [x] AppArmor enforce 6/6 (sniffer enforce — 0 denials) ✅
- [x] TEST-APPARMOR-ENFORCE: make test-all verde · 6/6 aa-status enforce ✅
- [x] noclobber audit ficheros críticos — limpio ✅
- [x] CHANGELOG-v0.4.0.md creado ✅
- [x] git merge --no-ff + tag v0.4.0-phase3-hardening + push ✅

### ✅ DONE — DAY 118 (15 Apr 2026)
- [x] **PHASE 3 COMPLETADA — v0.4.0-phase3-hardening MERGEADO A MAIN** 🎉
- [x] AppArmor enforce 6/6 (sniffer enforce — 0 denials) ✅
- [x] TEST-APPARMOR-ENFORCE: make test-all verde · 6/6 aa-status enforce ✅
- [x] noclobber audit ficheros críticos — limpio ✅
- [x] CHANGELOG-v0.4.0.md creado ✅
- [x] git merge --no-ff + tag v0.4.0-phase3-hardening + push ✅

### ✅ DONE — DAY 117 (14 Apr 2026)
- [x] 12/13 DEBTs bloqueantes PHASE 3 cerrados
- [x] AppArmor enforce 5/6 (etcd-server, rag-security, rag-ingester, ml-detector, firewall) — 0 denials
- [x] tools/apparmor-promote.sh — rollback automático si denials
- [x] TEST-PROVISION-1 8/8 · make test-all CI gate completo
- [x] DEBT-RAG-BUILD-001 · DEBT-SEED-PERM-001 · REC-2 · backup policy · ADR-021 addendum · Recovery Contract
- [x] arXiv Draft v15 recibido de Cornell

### ✅ DONE — DAY 116 (13 Apr 2026)
- [x] **PHASE 3 ítem 5:** DEBT-ADR025-D11 — provision.sh --reset con seed_family compartido (TEST-RESET-1/2/3 PASSED)
- [x] **PHASE 3 ítem 6:** TEST-PROVISION-1 checks #6 (permisos) + #7 (consistencia JSONs) → 7/7
- [x] **PHASE 3 ítem 7:** AppArmor 6 perfiles en complain mode — 0 denials
- [x] Bug arquitectural crítico resuelto: seeds independientes → HKDF MAC fail (INVARIANTE-SEED-001 documentado en ADR-021 addendum)

### ✅ DONE — DAY 115 (12 Apr 2026)
- [x] ADR-024 OQ-5..8 closed
- [x] PHASE 3 ítems 1–4: systemd, DEBT-SIGN-AUTO, DEBT-HELLO-001, TEST-PROVISION-1 (5/5)

### ✅ DONE — DAY 114 (11 Apr 2026)
- [x] **ADR-025 MERGED — v0.3.0-plugin-integrity** 🎉
- [x] arXiv Replace v15 submitted

### 🔜 NEXT — feature/adr026-xgboost
- [ ] Abrir rama feature/adr026-xgboost
- [ ] XGBOOST-VALIDATION.md con gate: Precision ≥ 0.99 + F1 ≥ 0.9985
- [ ] Consejo review pre-merge

### P3 — Post-PHASE 3
- [ ] DEBT-CRYPTO-003a: mlock() + explicit_bzero(seed) post-HKDF derivation
- [ ] ADR-024 Noise_IKpsk3 implementation
- [ ] ADR-026 XGBoost plugins Track 1 (Precision ≥ 0.99 gate médico)
- [ ] ADR-032 Fase A: manifest + multi-key loader
- [ ] ADR-033 TPM 2.0 Measured Boot (seed_family en hardware)
- [ ] ADR-029 variantes hardened: AppArmor+eBPF/XDP · AppArmor+libpcap · seL4+libpcap
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

### CI Gate (PHASE 3)
```bash
make test-all   # CI gate completo: libs + components + TEST-PROVISION-1 (8/8) + TEST-INVARIANT-SEED + plugin-integ-test
```

### Key Rotation
```bash
# Rotate ALL keys (seeds + Ed25519 keypairs + plugin signing keypair)
sudo CI=true bash tools/provision.sh --reset   # dev only
# Then: update MLD_PLUGIN_PUBKEY_HEX in CMakeLists.txt → rebuild → sign → start
```

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
- ✅ DAY 118: **PHASE 3 COMPLETADA — v0.4.0 MERGEADO** 🎉 · AppArmor 6/6 enforce · CHANGELOG · tag v0.4.0-phase3-hardening
- ✅ DAY 118: **PHASE 3 COMPLETADA — v0.4.0 MERGEADO** 🎉 · AppArmor 6/6 enforce · CHANGELOG · tag v0.4.0-phase3-hardening
- ✅ DAY 117: **PHASE 3 DEBTs CERRADOS** 🎉 · AppArmor 5/6 enforce · make test-all CI gate · arXiv v15

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*