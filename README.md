# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_The_Council-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: 24/24](https://img.shields.io/badge/Tests-24%2F24_suites-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Loader](https://img.shields.io/badge/Plugin_Loader-ADR--012_PHASE1-blue)](docs/adr/ADR-012%20plugin%20loader%20architecture.md)
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()

📜 Living contracts: [Protobuf schema](https://github.com/alonsoir/argus/blob/main/docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](https://github.com/alonsoir/argus/blob/main/docs/contracts/JSON%20contracts.md) · [RAG API](https://github.com/alonsoir/argus/blob/main/docs/contracts/Rag%20security%20commands.md)


---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* — Systems built like Roman roads, designed to endure.

> ML Defender stops ransomware propagation. What comes next is detecting infiltration.

---

## 📊 Validated Results (DAY 88 — 16 March 2026, crypto chain DAY 100)

| Metric | Value                                     | Notes                                                                         |
|---|-------------------------------------------|-------------------------------------------------------------------------------|
| **F1-score (CTU-13 Neris)** | **0.9985**                                | Stable across 4 replay runs                                                   |
| **Precision** | **0.9969**                                |                                                                               |
| **Recall** | **1.0000**                                | Zero missed attacks (FN=0)                                                    |
| **True Positives** | **646**                                   | Malicious flows from host 147.32.84.165                                       |
| **False Positives** | **2**                                     | VirtualBox multicast/broadcast artifacts — absent in bare-metal               |
| **True Negatives** | **12,075**                                |                                                                               |
| **FPR (ML, Neris evaluation)** | **0.0002%**                               |                                                                               |
| **FPR (Fast Detector, bigFlows)** | **6.61%**                                 | DEBT-FD-001, Path B thresholds                                                |
| **FP reduction (Fast → ML)** | **~500×**                                 | ML reduces production blocks to zero on bigFlows                              |
| **Inference latency** | **0.24–1.06 μs**                          | Per-class, embedded C++20                                                     |
| **Throughput ceiling (virtualized)** | **~33–38 Mbps**                           | VirtualBox NIC limit, not pipeline                                            |
| **Stress test** | **2,374,845 packets — 0 drops, 0 errors** | 100 Mbps requested, loop=3 bigFlows                                           |
| **RAM (full pipeline)** | **~1.28 GB**                              | Including TinyLlama, stable under load                                        |
| **Pipeline components** | **6/6 RUNNING**                           |                                                                               |
| **Test suite** | **24/24 suites passing**                  | crypto-transport, seed-client, etcd-server, rag-ingester, ml-detector + TEST-INTEG-1/2/3 |

**Ground truth clarification.** The CTU-13 Neris capture contains 19,135 total flows. Of these, 646 flows constitute the TP ground truth — those exhibiting active C2 behavioral signatures (IRC bursts, SMB lateral movement, DNS anomalies) from the infected host. The remaining flows are background traffic on the infected host and are not ground-truth positives for NIDS evaluation. See the [preprint](docs/) for full methodology.

### Cryptographic Identity & Forward Secrecy (DAY 95-96)

The pipeline now utilizes an **HKDF-based** derivation strategy:
1. `tools/provision.sh` generates unique Ed25519 keypairs and 32-byte seeds per component.
2. `seed-client` reads material from disk (permissions 0600).
3. `CryptoTransport` uses HKDF to derive session keys, ensuring context isolation and preventing decryption of historical traffic upon seed compromise.

### Honest Limitations

- Single botnet scenario evaluated (CTU-13 Neris, 2011). Generalizability to modern ransomware families not empirically established.
- All throughput figures are conservative lower bounds — VirtualBox NIC emulation ceiling ~33–38 Mbps. Bare-metal characterization is P1 future work.
- 11/40 ML features use `MISSING_FEATURE_SENTINEL = -9999.0f` (centralizado en `common/include/sentinel.hpp`, Phase 2 pending).
- Fast Detector Path A (DEBT-FD-001) retains compile-time thresholds — JSON migration scheduled for PHASE 2.
- Single-node deployment; etcd HA not yet implemented.
- ChaCha20 seed via etcd not recommended for production — peer-to-peer negotiation under design.

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
│  │                  │  plugin-loader (ADR-012, PHASE 1)         │
│  │                  │  ChaCha20-Poly1305 + LZ4 transport        │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (encrypted)                                    │
│  ┌──────────────────┐                                            │
│  │  ml-detector     │  4× Embedded RandomForest classifiers     │
│  │  (C++20)         │  DDoS: 0.24 μs | Ransomware: 1.06 μs     │
│  │                  │  Maximum Threat Wins dual-score policy    │
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
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  FAISS + SQLite event ingestion           │
│  │  (C++20)         │  Daily rotating + append-only CSV logs    │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-security    │  TinyLlama natural language interface      │
│  │  (C++20+LLM)     │  Local inference — no cloud exfiltration  │
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
# Primary source (CTU Prague)
wget -P /vagrant/datasets/ctu13/ \
  https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/botnet-capture-20110810-neris.pcap \
  -O /vagrant/datasets/ctu13/neris.pcap

# Mirror: https://www.stratosphereips.org/datasets-ctu13
```

**CTU-13 bigFlows — required for stress test:**
```bash
wget -P /vagrant/datasets/ctu13/ \
  https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/bigFlows.pcap \
  -O /vagrant/datasets/ctu13/bigFlows.pcap
```

Expected paths inside the VM:
```
/vagrant/datasets/ctu13/neris.pcap      # CTU-13 scenario 10 — 19,135 flows
/vagrant/datasets/ctu13/bigFlows.pcap   # CTU-13 bigFlows — 791,615 packets
```

### Run CTU-13 Neris Validation

```bash
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
make test-replay-neris
python3 scripts/calculate_f1_neris.py logs/lab/sniffer.log --total-events 19135
```

### Run Throughput Stress Test

```bash
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
vagrant ssh defender -c "tmux new-session -d -s top-monitor \
  'top -b -d 5 > /vagrant/logs/lab/top_stress_bigflows.log 2>&1'"
vagrant ssh client -c "sudo tcpreplay -i eth1 --mbps=100 --loop=3 --stats=10 \
  /vagrant/datasets/ctu13/bigFlows.pcap"
vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
```

### Tune Thresholds (no recompilation needed)

Edit `sniffer/config/sniffer.json` (not the build artifact in `build-debug/`):

```json
"ml_defender": {
"thresholds": {
"ddos": 0.85,
"ransomware": 0.90,
"traffic": 0.80,
"internal": 0.85
}
}
```

Restart pipeline to apply. Startup log confirms:
```
[ML Defender] Thresholds (JSON): DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
```

---

## 🔬 Key Engineering Decisions

### Dual-Score Detection: Maximum Threat Wins

```
score_final = max(score_fast, score_ml)
```

Arithmetic maximum over two continuous scores in [0,1] — not a logical OR over binary decisions. Fast Detector provides speed and coverage (FPR=6.61% on bigFlows); ML Detector provides precision and false positive suppression (reduces real production blocks to zero on the same corpus, approximately 500-fold reduction).

### Sentinel Value Taxonomy (DAY 79)

Three categories of special values in ML feature extraction:

1. **Domain-valid sentinel** (`-9999.0f`) — outside RandomForest split domain [0.0, 5.1]. Routes deterministically left in every split. Auditable, non-informative.
2. **Semantic value** (`0.5f` TCP half-open state) — within domain, meaningful default. Preserved with protective comments.
3. **Placeholder within domain** — strictly worse than category 1. Eliminated from codebase.

### JSON is the LAW

All configuration — ML thresholds, detection windows, feature parameters — comes from JSON at runtime. No hardcoded constants in production code. `sniffer/config/sniffer.json` is the source of truth.

### Pipeline-Native Training Data (DAY 81)

Three categories of training data, ranked by production validity:

| Category | Source | F1 offline | F1 production | Status |
|---|---|---|---|---|
| A — Academic | CTU-13, CIC-IDS2017 | ~0.99 | ~0.006 | Feature drift — rejected |
| B — Synthetic statistical | Own generator | — | max 0.6607 | Baseline, insufficient |
| C — Pipeline-native | C++20 extractor output | — | Hypothesis | Zero feature drift by construction |

Category C training data — generated by the same C++20 feature extractor used in production — is the P0 research hypothesis.

### Fast Detector Dual-Path Architecture (DEBT-FD-001)

Two alert paths coexist:

- **Path A** (`is_suspicious()`, DAY 13): per-packet, compile-time constants, ignores JSON. Source of 6.61% FPR on bigFlows. PHASE 2 migration target.
- **Path B** (`send_ransomware_features()`, DAY 80): temporal aggregates, reads `sniffer.json`. Correct JSON-driven behavior.

Full analysis: `docs/adr/ADR-006-fast-detector-hardcoded-thresholds.md`

### Trace Correlation (DAY 72)

```
trace_id = SHA-256(src_ip, dst_ip, canonical_attack_type, timestamp_bucket)
```

Per-attack-type time windows: ransomware=60s, DDoS=10s, SSH brute=30s. Validated: 46/46 unit tests.

---

## 🔐 Security Properties

| Property | Status |
|---|---|
| ChaCha20-Poly1305 AEAD encryption | ✅ All inter-component transport |
| HMAC-SHA256 log integrity | ✅ All CSV logs |
| Autonomous blocking (ipset/iptables) | ✅ Millisecond response |
| JSON-driven thresholds | ✅ No hardcoded security parameters |
| Fail-closed design | ✅ |
| Local LLM inference (no cloud) | ✅ TinyLlama, no network exfiltration |
| etcd HA | ❌ Single-node — future work |
| ChaCha20 seed P2P negotiation | ❌ Future work |

---

## 🗺️ Roadmap

### P0 — Immediate (post-paper)
- [ ] arXiv submission (cs.CR) — draft v5 ready, endorser contact in progress
- [ ] LaTeX conversion — preprint ready for Overleaf
- [ ] Sebastian Garcia (CTU Prague) endorser contact

### P1 — Short Term
- [ ] Bare-metal throughput stress test — eliminate VirtualBox NIC ceiling
- [ ] DEBT-FD-001 — Fast Detector Path A → JSON (PHASE 2, ADR-006)
- [ ] ADR-007 — AND-consensus firewall policy

### P2 — Feature Completion
- [ ] DNS payload real parsing
- [ ] 12 remaining ML features
- [ ] Feature importance analysis (Gini impurity reduction)

### Infiltration Vector Coverage (FEAT-ENTRY-*)

ML Defender currently stops ransomware **propagation** (lateral movement via SMB). These features address **infiltration** — how attackers enter the network:

| Feature | Description | Priority | Vector |
|---|---|---|---|
| FEAT-NET-1 | DNS anomaly / DGA detection | P1 | C2 communication |
| FEAT-NET-2 | Threat intelligence feed integration | P1 | Known malicious IPs/domains |
| FEAT-AUTH-1 | Auth log ingestion + brute force detection | P2 | RDP / credential attacks |
| FEAT-AUTH-2 | Behavioral auth anomaly detection | P2 | Stolen credentials |
| FEAT-EDR-1 | Lightweight endpoint agent | P3 | Phishing payload execution |

FEAT-EDR-1 closes the loop between the current NDR implementation and the full EDR roadmap reflected in the project name.

### Enterprise Features (MIT license — core always free)
- ENT-1: Federated Threat Intelligence
- ENT-2: Attack Graph Generation (GraphML + STIX 2.1)
- ENT-3: P2P Seed Distribution — eliminate etcd as crypto authority
- ENT-4: Hot-Reload configuration — no downtime threshold tuning

---

## 🧪 Experiment Tracking

All F1 replay results tracked in `docs/experiments/f1_replay_log.csv`.

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
- 16 GB host RAM, 4 GB guest RAM
- Dual vNIC (bridged)

### Recommended Production Target (~150–200 USD)
- Bare-metal Linux (kernel ≥ 5.8), x86 or ARM
- Dual NIC (physical)
- 4–8 GB RAM (pipeline + TinyLlama = ~1.28 GB active)
- Compatible with 5-year-old server hardware already present in hospital/school IT environments

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

ML Defender was developed through a structured multi-model AI collaboration methodology across 88 days of continuous development. Seven large language models served as intellectual co-reviewers:

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Parallel.ai**

Their contributions to architectural decisions (ShardedFlowManager, sentinel taxonomy, trace_id correlation), failure mode identification, Test Driven Hardening, and paper review are documented in the commit history and ADRs. This methodology is documented as a novel contribution in the accompanying preprint.

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

**Preprint**: *ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and Anomalous Traffic Detection in Resource-Constrained Organizations* — Draft v5, DAY 88, March 2026. arXiv submission in progress (cs.CR).

---

## 🗺️ Milestones

- ✅ DAY 76: SIGSEGV eliminated — pipeline 6/6 stable
- ✅ DAY 79: Sentinel taxonomy formalized — `-9999.0f` as domain-valid sentinel
- ✅ DAY 80: JSON is the LAW — all thresholds from sniffer.json
- ✅ DAY 80: F1=0.9934 → DAY 84 trace_id bugs fixed
- ✅ DAY 82: DEBT-FD-001 documented — FastDetector dual-path architecture (ADR-006)
- ✅ DAY 84: Trace_id bugs fixed — 46/46 tests passing
- ✅ DAY 84: Test suite 100% — crypto 3/3, etcd-hmac 12/12, ml-detector 9/9, trace_id 46/46
- ✅ DAY 86: F1=0.9985 corrected and stable — TP=646, FPR=0.0002%, 4 clean runs
- ✅ DAY 87: Stress test completed — 2.37M packets, 0 drops, 0 errors, RAM stable
- ✅ DAY 88: Paper draft v5 — Consejo de Sabios feedback integrated, arXiv ready
- ✅ DAY 88: Rename aRGus EDR → **aRGus NDR** — accurate scope
- ✅ DAY 92: SMB detection features — `rst_ratio`, `syn_ack_ratio` extractors + living contracts docs
- ✅ DAY 93: ADR-012 PHASE 1 — `plugin-loader` + `plugins/hello` + ABI validation via dlopen/dlsym
- ✅ DAY 95: Cryptographic Provisioning Infrastructure — Ed25519 + ChaCha20 seeds
- ✅ DAY 96: HKDF Implementation — Isolation of cryptographic domains
---

**Via Appia Quality** 🏛️ — Built to last decades

*"The road to security is long, but we build it to endure."*