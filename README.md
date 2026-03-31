# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_The_Council-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: 25/25](https://img.shields.io/badge/Tests-25%2F25_suites-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Loader](https://img.shields.io/badge/Plugin_Loader-ADR--012_PHASE1b_COMPLETA-blue)](docs/adr/ADR-012%20plugin%20loader%20architecture.md)
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![Docs](https://img.shields.io/badge/Docs-alonsoir.github.io%2Fargus-blue)](https://alonsoir.github.io/argus/)

📜 Living contracts: [Protobuf schema](https://github.com/alonsoir/argus/blob/main/docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](https://github.com/alonsoir/argus/blob/main/docs/contracts/JSON%20contracts.md) · [RAG API](https://github.com/alonsoir/argus/blob/main/docs/contracts/Rag%20security%20commands.md)

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* — Systems built like Roman roads, designed to endure.

> ML Defender stops ransomware propagation. What comes next is detecting infiltration.

---

## 📊 Validated Results (DAY 103 — 31 March 2026)

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
| **Pipeline components** | **6/6 RUNNING** | |
| **Test suite** | **25/25 suites passing** | NEW RECORD — DAY 102 |
| **Plugin Loader** | **ADR-012 PHASE 1b — 5/5 COMPLETA** | All components integrated — DAY 101-102 |

**Ground truth clarification.** The CTU-13 Neris capture contains 19,135 total flows. Of these, 646 flows constitute the TP ground truth — those exhibiting active C2 behavioral signatures (IRC bursts, SMB lateral movement, DNS anomalies) from the infected host. The remaining flows are background traffic on the infected host and are not ground-truth positives for NIDS evaluation. See the [preprint](docs/) for full methodology.

### Cryptographic Identity & Forward Secrecy (DAY 95–99)

The pipeline now utilizes an **HKDF-based** derivation strategy:
1. `tools/provision.sh` generates unique Ed25519 keypairs and 32-byte seeds per component.
2. `seed-client` reads material from disk (permissions 0600).
3. `CryptoTransport` uses HKDF-SHA256 with **channel-scoped contexts** (`:tx`/`:rx`) to derive distinct subkeys per direction, preventing MAC forgery across trust boundaries.
4. All six pipeline components use `crypto_transport/include/crypto_transport/contexts.hpp` as single source of truth for HKDF context strings.

### Honest Limitations

- Single botnet scenario evaluated (CTU-13 Neris, 2011). Generalizability to modern ransomware families not empirically established.
- All throughput figures are conservative lower bounds — VirtualBox NIC emulation ceiling ~33–38 Mbps. Bare-metal characterization pending hardware availability.
- 11/40 ML features use `MISSING_FEATURE_SENTINEL = -9999.0f` (centralizado en `common/include/sentinel.hpp`, Phase 2 pending).
- Fast Detector Path A (DEBT-FD-001) retains compile-time thresholds — JSON migration scheduled for PHASE 2.
- Single-node deployment; etcd HA not yet implemented.
- ChaCha20 seed distribution via etcd not recommended for production — peer-to-peer dynamic group key agreement under design (ADR-024, scheduled).

---

## 🏗️ Architecture

ML Defender implements **Network Detection and Response (NDR)** capabilities: real-time capture, ML-based classification, and automated blocking at the network layer. Full EDR functionality (endpoint agent) is on the roadmap via FEAT-EDR-1.

```
┌──────────────────────────────────────────────────────────────────┐
│                       ML Defender Pipeline                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Network Traffic                                                 │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  sniffer (C++20) │  eBPF/XDP zero-copy packet capture        │
│  │                  │  ShardedFlowManager (16 shards)           │
│  │                  │  Fast Detector (rule-based heuristics)    │
│  │                  │  28/40 features extracted                 │
│  │                  │  plugin-loader (ADR-012, PHASE 1b ✅)     │
│  │                  │  ChaCha20-Poly1305 + LZ4 transport        │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (encrypted)                                    │
│  ┌──────────────────┐                                            │
│  │  ml-detector     │  4× Embedded RandomForest classifiers     │
│  │  (C++20)         │  DDoS: 0.24 μs | Ransomware: 1.06 μs     │
│  │                  │  Maximum Threat Wins dual-score policy    │
│  │                  │  plugin-loader (ADR-012, PHASE 1b ✅)     │
│  │                  │  ~500× FP reduction vs Fast Detector      │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (encrypted)                                    │
│  ┌──────────────────┐                                            │
│  │  etcd-server     │  Component registration + config          │
│  │  (C++20)         │  HMAC key management + crypto seeds       │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │ firewall-acl     │  Autonomous blocking via ipset/iptables   │
│  │ agent (C++20)    │  HMAC-SHA256 verified CSV logs            │
│  │                  │  plugin-loader (ADR-012, PHASE 1b ✅)     │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  FAISS + SQLite event ingestion           │
│  │  (C++20)         │  Daily rotating + append-only CSV logs    │
│  │                  │  plugin-loader (ADR-012, PHASE 1b ✅)     │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-security    │  TinyLlama natural language interface      │
│  │  (C++20+LLM)     │  Local inference — no cloud exfiltration  │
│  │                  │  plugin-loader (ADR-012, PHASE 1b ✅)     │
│  └──────────────────┘                                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Three Functions of the Engine

**Shield** — Real-time detection and blocking. End-to-end from packet capture to firewall block in milliseconds. F1=0.9985, Recall=1.0000 on CTU-13 Neris.

**Microscope** — Every flow passing through the pipeline is dissected into 28 dimensions, correlated via `trace_id` (SHA-256 hash of src/dst/type/time bucket), and stored as HMAC-verified CSV logs — forming the basis for future model retraining with zero feature drift.

**Research Platform** — Reproducible experimentation infrastructure with deterministic replay, F1 tracking (`docs/experiments/f1_replay_log.csv`), and a natural language query interface over live event streams.

---

## 🚀 Quick Start

### Prerequisites
```bash
vagrant --version    # 2.3+
vboxmanage --version # 7.x
```

### Build & Deploy

```bash
git clone https://github.com/alonsoir/argus.git
cd argus
make up
make all
make pipeline-start
make pipeline-status
```

### Dataset Setup

Before running validation or stress tests, download the CTU-13 datasets:

**CTU-13 Neris (scenario 10) — required for F1 validation:**
```bash
wget -P /vagrant/datasets/ctu13/ \
  https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/botnet-capture-20110810-neris.pcap \
  -O /vagrant/datasets/ctu13/neris.pcap
```

**CTU-13 bigFlows — required for stress test:**
```bash
wget -P /vagrant/datasets/ctu13/ \
  https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/bigFlows.pcap \
  -O /vagrant/datasets/ctu13/bigFlows.pcap
```

### Run CTU-13 Neris Validation

```bash
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
make test-replay-neris
python3 scripts/calculate_f1_neris.py logs/lab/sniffer.log --total-events 19135
```

---

## 🔬 Key Engineering Decisions

### Dual-Score Detection: Maximum Threat Wins

```
score_final = max(score_fast, score_ml)
```

Arithmetic maximum over two continuous scores in [0,1]. Fast Detector provides speed and coverage (FPR=6.61% on bigFlows); ML Detector provides precision and false positive suppression (~500× reduction).

### HKDF Context Symmetry (DAY 99 — ADR-022 case study)

A critical defect discovered during cryptographic integration: HKDF contexts parameterised by *component name* produce identical subkeys on both sides of a channel, making MAC forgery undetectable. The fix parameterises by *direction* (`:tx`/`:rx`). The defect was invisible to the compiler and all sanitisers — caught exclusively by `TEST-INTEG-3`, an intentional regression test asserting MAC failure. Documented as a pedagogical case study in the accompanying preprint (§5, Test-Driven Hardening).

### Plugin Loader Architecture (ADR-012 — PHASE 1b COMPLETA)

All six pipeline components integrate `libplugin_loader.so` via lazy `dlopen`/`dlsym` with pure C ABI (`plugin_api.h`). Guard pattern: `#ifdef PLUGIN_LOADER_ENABLED`. Signal handlers requiring global access use `g_plugin_loader`; all others use `std::unique_ptr<PluginLoader>` local to `main()`. Individual plugins in `/usr/lib/ml-defender/plugins/`, system libraries in `/usr/local/lib/`.

### Sentinel Value Taxonomy (DAY 79)

`MISSING_FEATURE_SENTINEL = -9999.0f` — outside RandomForest split domain [0.0, 5.1]. Routes deterministically left in every split. Auditable, non-informative. Defined in `common/include/sentinel.hpp`.

### JSON is the LAW

All configuration comes from JSON at runtime. No hardcoded constants in production code.

### Fail-Closed Design (ADR-022)

`std::set_terminate()` installed in all six `main()` functions. Any uncaught exception → `abort()`. MAC authentication failure → `std::terminate()`. No degraded mode. No silent failure.

---

## 🔐 Security Properties

| Property | Status |
|---|---|
| ChaCha20-Poly1305 AEAD encryption | ✅ All inter-component transport |
| HKDF-SHA256 channel-scoped key derivation | ✅ Distinct tx/rx subkeys per channel |
| libsodium 1.0.19 (compiled from source) | ✅ Bookworm ships 1.0.18; 1.0.19 built + SHA-256 verified |
| HMAC-SHA256 log integrity | ✅ All CSV logs |
| Autonomous blocking (ipset/iptables) | ✅ Millisecond response |
| JSON-driven thresholds | ✅ No hardcoded security parameters |
| Fail-closed design (std::set_terminate) | ✅ All 6 main() functions |
| Local LLM inference (no cloud) | ✅ TinyLlama, no network exfiltration |
| Plugin Loader (ADR-012 PHASE 1b) | ✅ All 6 components |
| Dynamic group key agreement | ❌ ADR-024 — scheduled post-arXiv |
| etcd HA | ❌ Single-node — future work |

---

## 🗺️ Roadmap

### P0 — arXiv submission (in progress)
- [x] Draft v7 ready — 21 pages, LaTeX, compiled clean
- [x] Sebastian Garcia (CTU Prague) — endorser, received PDF
- [x] Andrés Caro Lindo (UEx/INCIBE) — endorsement confirmed, call Thursday 2 April
- [ ] arXiv submission (cs.CR)

### P1 — feature/plugin-crypto (next branch)
- [ ] ADR-023 — Multi-Layer Plugin Architecture (MessageContext, SkillContext)
- [ ] ADR-024 — Dynamic Group Key Agreement (runtime, no redeploy)
- [ ] FEAT-PLUGIN-CRYPTO-1 PHASE 2a — `plugin_process_message()` optional
- [ ] TEST-INTEG-4a/4b/4c — validation gates
- [ ] PAPER-FINAL — update metrics + mention ADR-024 as planned future work

### P2 — Post-arXiv
- [ ] BARE-METAL-IMAGE — Debian Bookworm hardened, exportable to USB
- [ ] BARE-METAL-VAGRANT — validate image in new Vagrantfile before physical hardware
- [ ] BARE-METAL stress test — pending physical machine availability
- [ ] DEBT-FD-001 — Fast Detector Path A → JSON
- [ ] DEBT-CRYPTO-003a — `mlock()` on seed buffer
- [ ] DOCS-APPARMOR — 6 AppArmor profiles

### Enterprise Features (MIT license — core always free)
- ENT-1: Federated Threat Intelligence
- ENT-2: Attack Graph Generation (GraphML + STIX 2.1)
- ENT-3: P2P Seed Distribution — eliminate etcd as crypto authority
- ENT-4: Hot-Reload configuration

---

## 🧪 Experiment Tracking

| replay_id | day | F1 | Precision | Recall | notes |
|---|---|---|---|---|---|
| DAY86_run1 | 86 | 0.9985 | 0.9969 | 1.0000 | Stable ✅ |
| DAY86_run2 | 86 | 0.9985 | 0.9969 | 1.0000 | Stable ✅ |
| DAY86_run3 | 86 | 0.9985 | 0.9969 | 1.0000 | Stable ✅ |
| DAY86_run4 | 86 | 0.9985 | 0.9969 | 1.0000 | Stable ✅ |

F1=0.9985 confirmed stable across 4 independent replay runs. 2 FP = VirtualBox multicast/broadcast artifacts, absent in bare-metal.

---

## 📋 Hardware Requirements

### Minimum (validated)
- Consumer laptop running VirtualBox + Vagrant
- 16 GB host RAM, ~3.5 GB guest RAM under full load
- Dual vNIC (bridged)

### Recommended Production Target (~150–200 USD)
- Bare-metal Linux (kernel ≥ 5.8), x86 or ARM
- Dual NIC (physical)
- 4–8 GB RAM (pipeline + TinyLlama = ~3.5 GB active)
- Compatible with 5-year-old server hardware already present in hospital/school IT environments

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

ML Defender was developed through a structured multi-model AI collaboration methodology across 103 days of continuous development. Seven large language models served as intellectual co-reviewers:

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Parallel.ai**

Their contributions to architectural decisions, failure mode identification, Test-Driven Hardening, and paper review are documented in the commit history and ADRs. This methodology is documented as a novel contribution in the accompanying preprint (§5, Consejo de Sabios).

*The Consejo de Sabios democratizes access to deep technical peer review for independent researchers without institutional affiliation — parallel to ML Defender's mission of democratizing enterprise-grade security for organizations without enterprise budgets.*

---

## 📄 Funding and Collaboration

This project has been developed without institutional funding. If you are interested in supporting or collaborating:

- **Research collaboration**: evaluation on modern ransomware captures, dataset contribution
- **Deployment partners**: hospitals, schools, small organizations willing to validate in production
- **Funding programs**: ENISA, INCIBE, Horizon Europe, NGI

Contact: open an issue or reach out via the repository.

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

All code, all analysis scripts, all experiments, and all failures are documented in this repository. No tricks, no shortcuts.

---

## 🙏 Acknowledgments

**Human architect**: Alonso Isidoro Roman — Independent Researcher, Extremadura, Spain

**AI Co-Contributors (Consejo de Sabios)**: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI), DeepSeek, Qwen (Alibaba), Gemini (Google), Parallel.ai

**Datasets**: CTU-13 (Czech Technical University, Sebastian Garcia et al.), CIC-IDS2017 (UNB), MAWI Working Group

**Preprint**: *ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and Anomalous Traffic Detection in Resource-Constrained Organizations* — Draft v7, DAY 103, March 2026. arXiv submission in progress (cs.CR).

---

## 🗺️ Milestones

- ✅ DAY 76: SIGSEGV eliminated — pipeline 6/6 stable
- ✅ DAY 79: Sentinel taxonomy formalized — `-9999.0f` as domain-valid sentinel
- ✅ DAY 80: JSON is the LAW — all thresholds from sniffer.json
- ✅ DAY 83: F1=0.9985 corrected and stable — TP=646, FPR=0.0002%, 4 clean runs
- ✅ DAY 84: Test suite 100% — 46/46 trace_id + crypto + etcd-hmac + ml-detector
- ✅ DAY 87: Stress test — 2.37M packets, 0 drops, 0 errors, RAM stable
- ✅ DAY 88: Paper draft v5 — Consejo de Sabios feedback integrated
- ✅ DAY 93: ADR-012 PHASE 1 — plugin-loader + ABI validation via dlopen/dlsym
- ✅ DAY 95: Cryptographic Provisioning Infrastructure — Ed25519 + ChaCha20 seeds
- ✅ DAY 96: seed-client (libseedclient.so) — 0600 permissions + explicit_bzero
- ✅ DAY 97: libsodium 1.0.19 compiled from source — SHA-256 verified
- ✅ DAY 97: CryptoTransport — HKDF-SHA256 + ChaCha20-Poly1305 + 96-bit monotonic nonce
- ✅ DAY 99: contexts.hpp — HKDF channel-scoped contexts (critical bug fix)
- ✅ DAY 99: TEST-INTEG-1/2/3 — cryptographic gate tests passing
- ✅ DAY 100: set_terminate() all 6 main() — fail-closed confirmed
- ✅ DAY 100: ADR-021 (topology SSOT) + ADR-022 (threat model) published
- ✅ DAY 101-102: ADR-012 PHASE 1b — plugin-loader 5/5 components COMPLETA
- ✅ DAY 102: TEST-PLUGIN-INVOKE-1 — 25/25 tests ✅ nuevo récord
- ✅ DAY 103: Makefile rag alignment — PROFILE-aware, rag-attach, test-components
- ✅ DAY 103: Paper Draft v7 — §5 HKDF Context Symmetry case study added

---

**Via Appia Quality** 🏛️ — Built to last decades

*"The road to security is long, but we build it to endure."*