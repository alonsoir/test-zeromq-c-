# ML Defender (aRGus EDR) — Architecture

> **Version:** 6.0.0 — DAY 91 (19 marzo 2026)
> **Authors:** Alonso Isidoro Román, with the Consejo de Sabios (Claude, Grok, ChatGPT, DeepSeek, Qwen, Gemini, Parallel.ai)
> **License:** Open Source — see `LICENSE`
> **Status:** Active research system — arXiv preprint in preparation

---

## Table of Contents

1. [Vision and Motivation](#1-vision-and-motivation)
2. [Target Audience](#2-target-audience)
3. [System Overview](#3-system-overview)
4. [Pipeline Components](#4-pipeline-components)
5. [Data Flow](#5-data-flow)
6. [ASCII Pipeline Diagram](#6-ascii-pipeline-diagram)
7. [Technology Stack](#7-technology-stack)
8. [Architectural Decision Records](#8-architectural-decision-records)
9. [Deployment](#9-deployment)
10. [Security Design](#10-security-design)
11. [ML Subsystem](#11-ml-subsystem)
12. [Performance Envelope](#12-performance-envelope)
13. [Known Limitations](#13-known-limitations)
14. [Roadmap](#14-roadmap)

---

## 1. Vision and Motivation

ML Defender (internal codename **aRGus EDR**) is an open-source, network-level Intrusion Detection System (IDS) / Endpoint Detection and Response (EDR) system written in **C++20**, designed for **resource-constrained organizations**: hospitals, schools, municipal governments, and small businesses.

These organizations are the most frequent victims of ransomware (WannaCry, NotPetya, Conti, LockBit) and are simultaneously the least equipped to defend themselves. Commercial EDR solutions require enterprise licensing, dedicated security staff, and cloud connectivity — all of which are out of reach for a rural hospital or a primary school.

ML Defender is built on a core philosophy:

> **"Via Appia Quality"** — build for permanence. Systems that will still work in 10 years without a support contract.

The system runs entirely **on-premises**, with no cloud dependency, no telemetry, and no external API calls in its critical path. It is designed to be deployed by a single sysadmin with no security background, yet provide detection capabilities comparable to commercial tools.

---

## 2. Target Audience

**For this document:**
- External contributors evaluating the codebase
- Security researchers validating the detection methodology
- Potential adopters assessing deployment feasibility
- Academic reviewers of the companion arXiv preprint

**For the system itself:**
- IT staff at resource-constrained organizations
- Network administrators wanting passive, zero-impact monitoring
- SOC teams at small-to-medium organizations needing an affordable EDR layer

---

## 3. System Overview

ML Defender operates as a **passive network monitor** combined with an **active response layer**. It captures packets at the kernel level via eBPF/XDP, extracts statistical flow features, classifies traffic using an embedded Random Forest model (ONNX Runtime), and — upon detection of malicious flows — applies firewall rules automatically.

A secondary **Retrieval-Augmented Generation (RAG)** subsystem indexes historical detection events and makes them queryable via a confined local LLM, enabling forensic analysis without external connectivity.

### Core properties

| Property | Value |
|---|---|
| Language | C++20 |
| Detection latency | < 1 ms (Fast Detector, Path A) |
| ML inference latency | < 5 ms (ONNX Runtime, Path B) |
| Encryption | ChaCha20-Poly1305 (all inter-component messages) |
| Integrity | HMAC-SHA256 per message and per CSV row |
| Compression | LZ4 (inter-component transport) |
| Coordination | etcd v3 (service discovery, configuration) |
| Messaging | ZeroMQ (PUSH/PULL topology) |
| Test coverage | 31/31 CTest ✅ |

---

## 4. Pipeline Components

The system consists of **6 active components**, each running as an independent process. They communicate exclusively via encrypted ZeroMQ channels, with etcd as the service discovery authority.

---

### 4.1 `etcd-server`

**Role:** Configuration authority and service discovery broker.

All components register their ZeroMQ endpoint addresses with etcd on startup. No component has a hardcoded peer address — all addresses are resolved dynamically from etcd at connection time. This enables hot-reload of configuration and makes the system topology reconfigurable without recompilation.

**Key responsibilities:**
- Publish resource paths (ZeroMQ endpoints) for all other components
- Store and distribute the shared HMAC key (with rotation support — see ADR-004)
- Provide a single source of truth for `sniffer.json` configuration

**ADR reference:** ADR-005 (etcd client restoration), ADR-004 (key rotation with cooldown)

---

### 4.2 `sniffer`

**Role:** Kernel-space packet capture and flow feature extraction.

The sniffer attaches an **eBPF/XDP** program to the monitored network interface and captures packets at the earliest possible point in the Linux network stack — before the kernel's networking subsystem processes them. This gives ML Defender zero-copy, sub-microsecond access to raw packet data.

From raw packets, the sniffer assembles **network flows** (5-tuple: src IP, dst IP, src port, dst port, protocol) and computes a **127-column feature vector** per flow over a configurable time window (default: 10 seconds).

**Feature categories:**
- Packet rate, byte rate, flow duration
- TCP flag ratios (SYN, ACK, RST, FIN)
- Port diversity and destination IP diversity
- Payload entropy (approximated)
- Protocol distribution

**Output:** Protobuf-serialized `FlowRecord` messages, ChaCha20-Poly1305 encrypted, LZ4 compressed, published to `ml-detector` via ZeroMQ PUSH socket.

**Configuration:** `sniffer.json` (read from etcd — "JSON is the law" principle)

**ADR reference:** ADR-006 (Fast Detector hardcoded thresholds — DEBT-FD-001), ADR-009 (datagram correlation)

---

### 4.3 `ml-detector`

**Role:** Dual-path classification engine.

The ml-detector receives `FlowRecord` messages from the sniffer and applies a **two-path detection architecture**:

#### Path A — Fast Detector (`FastDetector::is_suspicious()`)
A handcrafted rule engine with compiled-in thresholds (currently hardcoded — see DEBT-FD-001). Operates in **< 1 ms** and acts as a first-pass filter. Flows flagged by Path A bypass the ML model and are forwarded directly for blocking.

> ⚠️ **DEBT-FD-001:** Path A currently ignores `sniffer.json` thresholds. This is a known architectural debt, documented in ADR-006, scheduled for resolution in PHASE2.

#### Path B — ML Detector (ONNX Runtime)
A **Random Forest** classifier, trained on CTU-13 Neris botnet traffic and compiled to ONNX format. Loaded at runtime via ONNX Runtime. Operates on the full 127-column feature vector and produces a **maliciousness probability score** (0.0–1.0).

**Validated performance (CTU-13 Neris, DAY 86 re-validation):**
| Metric | Value |
|---|---|
| F1 Score | 0.9985 |
| Recall | 1.0000 |
| False Negative Rate | 0.00% |
| False Positive Rate | 6.61% |
| True Positives | 646 |

**Scoring and alerting (ADR-007):**
- **OR logic** for alerts: Path A suspicious OR Path B score > threshold → alert generated
- **AND logic** for blocks: Path A suspicious AND Path B score > threshold → firewall rule applied
- This prevents ML score poisoning: an attacker cannot avoid blocking by manipulating features to fool only one path

**Sentinel value system:**
All missing or semantically undefined features use `MISSING_FEATURE_SENTINEL = -9999.0f` — a value mathematically unreachable in real traffic. This enables deterministic routing of incomplete flows without ambiguity. Semantic values (e.g., `0.5f` for TCP half-open states) are strictly separate from sentinel values.

**Output:** Alert messages and block commands to `firewall-acl-agent`; flow records (with HMAC-SHA256 per-row integrity) to daily-rotating CSV files consumed by `rag-ingester`.

**ADR reference:** ADR-007 (consensus scoring), ADR-003 (ML autonomous evolution), ADR-011 (log unification)

---

### 4.4 `firewall-acl-agent`

**Role:** Automated network response layer.

Receives block commands from `ml-detector` and applies **iptables** rules to drop traffic from flagged source IPs. Implements a grace period mechanism: rules are applied immediately on detection and removed after a configurable cooldown window.

**Design constraints:**
- Never blocks without a corresponding ml-detector command (no autonomous decisions)
- Maintains an audit log of all applied and removed rules
- Rule application is idempotent: applying the same block twice has no effect
- Human-in-the-loop: the Consejo de Sabios review process governs threshold changes

**ADR reference:** ADR-007 (AND-consensus requirement before blocking)

---

### 4.5 `rag-ingester`

**Role:** Historical event indexer and FAISS index builder.

Consumes two CSV data sources:
1. **ml-detector daily rotating CSV** — per-flow detection events with HMAC integrity
2. **firewall-acl-agent append-only CSV** — applied/removed rule audit log

For each batch of new records, `rag-ingester`:
1. Verifies HMAC-SHA256 integrity per row
2. Extracts text embeddings (via a lightweight embedding model)
3. Upserts vectors into **FAISS** indices (class-separated to avoid the curse of dimensionality)
4. Updates the **SQLite MetadataDB** (14-column schema including `trace_id`, `source_ip`, `dest_ip`, timestamps)

**FAISS anti-curse design:**
- Separate indices per traffic class (benign / malicious / unknown)
- PCA dimensionality reduction applied before indexing
- Temporal tiers: recent events in hot index, older events in cold index
- No cross-class contamination in similarity search

**ADR reference:** ADR-002 (multi-engine provenance intelligence), ADR-008 (flow graphs and forensic retraining)

---

### 4.6 `rag-security`

**Role:** Local RAG query interface — confined LLM for forensic analysis.

Provides a query interface over the FAISS indices built by `rag-ingester`. A security analyst can ask natural-language questions about historical detection events:

> *"Show me all flows from 192.168.1.45 in the last 24 hours"*
> *"Which source IPs generated the most RST packets this week?"*

The LLM (TinyLlama, running entirely locally) generates responses grounded in the retrieved FAISS context. The LLM has no internet access, no external API calls, and cannot modify system configuration — it is **strictly read-only and confined** (ADR-010).

**Skills system (ADR-010):**
The LLM operates via a structured skills registry. Each skill defines a permitted query pattern and the FAISS/SQLite operations it may execute. Skills outside the registry are rejected. This prevents prompt injection from attacker-controlled traffic payloads appearing in the indexed data.

**ADR reference:** ADR-010 (confined LLM skills), ADR-002 (situational intelligence)

---

## 5. Data Flow

```
[Network Interface]
        │
        ▼ eBPF/XDP (kernel space)
[sniffer]
   • 127-column FlowRecord extraction
   • 10s aggregation window
   • Protobuf serialization
        │ ZeroMQ PUSH
        │ ChaCha20-Poly1305 + LZ4
        ▼
[ml-detector]
   • Path A: FastDetector (< 1ms, hardcoded rules)
   • Path B: ONNX Runtime RandomForest (< 5ms)
   • Consensus scoring (ADR-007)
   • trace_id generation per flow
        │                    │
        │ BLOCK command       │ Daily CSV (HMAC per row)
        ▼                    ▼
[firewall-acl-agent]    [rag-ingester]
   • iptables rules        • FAISS index build
   • Audit CSV             • SQLite MetadataDB
                                │
                                ▼
                          [rag-security]
                          • FAISS similarity search
                          • TinyLlama (local, confined)
                          • Skills registry (ADR-010)

[etcd-server] ──────────────► all components
   • ZeroMQ endpoint registry
   • HMAC key distribution
   • sniffer.json configuration
```

---

## 6. ASCII Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML DEFENDER (aRGus EDR)                          │
│                      Network Intrusion Detection                         │
└─────────────────────────────────────────────────────────────────────────┘

         ┌──────────────┐
         │  etcd-server │  ◄── Configuration authority
         │   (port 2379)│      Service discovery
         └──────┬───────┘      HMAC key distribution
                │ service discovery (all components)
                │
 ┌──────────────┼──────────────────────────────────────────────┐
 │              │                                              │
 ▼              ▼              ▼              ▼                ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│ sniffer │ │ml-detect │ │ firewall │ │rag-ingest│ │ rag-security │
│         │ │          │ │-acl-agent│ │          │ │              │
│ eBPF/   │ │ Path A:  │ │          │ │ FAISS    │ │ TinyLlama    │
│ XDP     │─►FastDetect│─► iptables │ │ SQLite   │ │ (local only) │
│         │ │ Path B:  │ │ rules    │ │ MetadataDB│ │ Skills API  │
│ 127-col │ │ ONNX RF  │ │          │ │          │─► query iface │
│ features│ │          │ │          │ │          │ │              │
└─────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘
     │            │                         ▲
     │ ZeroMQ     │ ZeroMQ                  │
     │ PUSH       │ PUSH                    │
     │ ChaCha20   │ CSV                     │
     │ + LZ4      │ + HMAC                  │
     └────────────┘─────────────────────────┘

Legend:
  ─► ZeroMQ encrypted channel (ChaCha20-Poly1305 + LZ4)
  ──► etcd service discovery (gRPC)
  CSV files protected with HMAC-SHA256 per row
```

---

## 7. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | C++20 | Sub-microsecond latency; zero GC pauses; embedded target compatibility |
| Packet capture | eBPF/XDP | Kernel-bypass; zero-copy; earliest hook point in Linux network stack |
| Messaging | ZeroMQ 4.x | Battle-tested; PUSH/PULL topology; cross-platform |
| Serialization | Protocol Buffers (proto3) | Schema-enforced; compact binary; forward-compatible |
| Encryption | ChaCha20-Poly1305 (libsodium) | AEAD; constant-time; no hardware AES dependency |
| Integrity | HMAC-SHA256 | Per-message and per-CSV-row audit trail |
| Compression | LZ4 | Fastest real-time compression; negligible CPU cost |
| Coordination | etcd v3 | Distributed KV with watch semantics; gRPC transport |
| ML runtime | ONNX Runtime | Vendor-neutral; supports RandomForest, future neural models |
| Vector search | FAISS (Facebook AI) | Approximate nearest-neighbor at scale; CPU-only deployment |
| Metadata | SQLite 3 | Zero-config; embedded; ACID; 14-column MetadataDB schema |
| LLM | TinyLlama (local) | Fully offline; < 2GB RAM; sufficient for structured forensic queries |
| Build system | CMake + Conan 2 | Reproducible; cross-platform; dependency management |
| Test framework | CTest + custom harnesses | 31/31 passing; crypto, HMAC, ML, RAG, trace_id suites |
| VM / CI | Vagrant + VirtualBox | Multi-VM lab simulation; repeatable dataset replay |

---

## 8. Architectural Decision Records

All major architectural decisions are documented in `docs/adr/`. Key decisions relevant to contributors:

| ADR | Title | Status | Impact |
|---|---|---|---|
| ADR-001 | Deployment stack (Vagrant, Docker, bare-metal) | ACCEPTED | Deployment model |
| ADR-002 | Multi-engine provenance situational intelligence | ACCEPTED | RAG architecture |
| ADR-003 | ML autonomous evolution | ACCEPTED | Retraining pipeline |
| ADR-004 | HMAC key rotation with cooldown windows | ACCEPTED | Security — key management |
| ADR-005 | etcd client restoration | ACCEPTED | Resilience |
| ADR-006 | Fast Detector hardcoded thresholds (DEBT-FD-001) | ACCEPTED (debt) | Path A architecture |
| ADR-007 | AND-consensus scoring for firewall blocks | ACCEPTED | Detection logic |
| ADR-008 | Flow graphs and forensic retraining | ACCEPTED | ML evolution |
| ADR-009 | Suspicious datagram capture and correlation | ACCEPTED | Feature extraction |
| ADR-010 | Confined LLM with skills registry (rag-security) | ACCEPTED | LLM safety |
| ADR-011 | Log unification across 6 components | ACCEPTED | Observability |
| ADR-012 | Plugin loader architecture | PROPOSED | Extensibility |

> **For contributors:** Before implementing any feature that touches detection logic, scoring, or firewall rules, read ADR-007. Before modifying the RAG subsystem, read ADR-010. For all new ADRs, use the template in `docs/adr/`.

---

## 9. Deployment

### 9.1 Experimental — Vagrant Multi-VM Lab

Used for all dataset replay experiments, stress testing, and CI validation.

```bash
# Start the lab
cd /path/to/test-zeromq-docker
vagrant up

# SSH into the defender VM
vagrant ssh defender

# Run the full pipeline
cd /vagrant
./scripts/start_pipeline.sh

# Run tests
cd build && ctest --output-on-failure
```

**VM topology:**
- `defender` — runs all 6 pipeline components + etcd
- Traffic replay via `tcpreplay` against CTU-13 datasets

**Validated datasets:**
- CTU-13 Neris (botnet, DAY 79–87 baseline)
- CTU-13 bigFlows (mixed traffic, stress testing)

**Stress test results (DAY 87):**
- Tested at 10 / 25 / 50 / 100 Mbps via progressive tcpreplay escalation
- **Bottleneck confirmed:** VirtualBox NIC virtualization layer (~33–38 Mbps physical limit)
- **Pipeline itself:** 100% of deliverable traffic processed with zero errors at all tested rates
- Bare-metal testing is the next P1 milestone

### 9.2 Production — Bare-Metal Linux

Target environment for real deployments.

**Requirements:**
- Linux kernel ≥ 5.10 (eBPF/XDP support)
- Network interface with XDP driver support (common NICs: Intel i40e, Mellanox mlx5)
- ≥ 4 GB RAM (TinyLlama KV cache + FAISS indices)
- ≥ 2 CPU cores (pipeline is multi-threaded; ML inference is single-threaded)
- No internet connectivity required (fully air-gapped capable)

**Quick start:**
```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Configure
cp config/sniffer.json.example config/sniffer.json
# Edit sniffer.json: set interface, thresholds, etcd address

# Start etcd
etcd --data-dir /var/lib/etcd &

# Start pipeline components (order matters)
./etcd-server &
./rag-ingester &
./rag-security &
./ml-detector &
./firewall-acl-agent &
./sniffer --interface eth0 &
```

**Logs:** All components write structured logs to `/var/log/ml-defender/COMPONENT.log`
(In Vagrant lab: `/vagrant/logs/lab/COMPONENT.log`)

### 9.3 Configuration — `sniffer.json`

`sniffer.json` is the single configuration file for the entire pipeline. It is stored in etcd and distributed to all components at startup. **No component may hardcode a parameter that belongs in `sniffer.json`.** (This principle is currently violated by Path A — see DEBT-FD-001 / ADR-006.)

Key parameters:
```json
{
  "interface": "eth0",
  "flow_window_seconds": 10,
  "ml_score_threshold": 0.7,
  "fast_detector_rst_ratio_threshold": 0.5,
  "fast_detector_connection_rate_threshold": 100,
  "hmac_key_rotation_interval_seconds": 3600,
  "log_level": "INFO"
}
```

---

## 10. Security Design

### 10.1 Transport Security

All inter-component communication is encrypted with **ChaCha20-Poly1305** (AEAD). The authentication tag provides both integrity and authenticity — a modified message is detected and dropped before processing.

The shared encryption key is distributed via etcd. Key rotation is supported with a **cooldown window** and **grace period** (ADR-004) to prevent message drops during rotation.

### 10.2 Data Integrity

Every CSV row written by `ml-detector` and `firewall-acl-agent` carries an **HMAC-SHA256** computed with the current rotation key. `rag-ingester` verifies this HMAC before indexing any row. A tampered audit log is detected and rejected.

### 10.3 LLM Confinement (ADR-010)

The `rag-security` LLM is confined by a **skills registry**. Attackers who control traffic payloads that get indexed in FAISS cannot use those payloads to exfiltrate data or execute commands, because the LLM may only execute predefined skill operations.

### 10.4 Trace ID System

Every flow is assigned a **`trace_id`** at the sniffer level, derived from the 5-tuple and a timestamp bucket. This enables end-to-end correlation of a single flow across all 6 components without transmitting the full 5-tuple in every message.

The `trace_id_generator` handles edge cases:
- `fallback_applied` flag for flows with wildcard IPs (`0.0.0.0`) — these are valid network events, not errors
- Timestamp bucket alignment to prevent boundary artifacts

---

## 11. ML Subsystem

### 11.1 Model Architecture

The production classifier is a **Random Forest** with 100 trees, trained on CTU-13 Neris botnet captures. The model is:
1. Trained in Python (scikit-learn)
2. Exported to ONNX format
3. Loaded at runtime by `ml-detector` via ONNX Runtime (C++ API)
4. Inference: < 5 ms per flow on a single CPU core

### 11.2 Feature Engineering

The 127-column feature vector covers:
- **Flow-level statistics:** duration, packet count, byte count, inter-arrival times
- **TCP flag ratios:** SYN ratio, ACK ratio, RST ratio (`rst_ratio` — P1 backlog), FIN ratio
- **Connection quality:** `syn_ack_ratio` (P1 backlog), half-open connections
- **Destination diversity:** unique destination IPs, unique destination ports (`port_diversity_ratio` — P2)
- **Protocol signals:** dst port 445 ratio (SMB — P2), DNS query count (P3)
- **Temporal patterns:** connection rate, burst detection

### 11.3 Sentinel Value Taxonomy

| Value | Name | Meaning |
|---|---|---|
| `-9999.0f` | `MISSING_FEATURE_SENTINEL` | Feature not computable for this flow — deterministic routing |
| Semantic values | Various | Domain-specific defaults (e.g., `0.5f` for TCP half-open) |

The sentinel `-9999.0f` was chosen because it is **mathematically unreachable** in real traffic. It is embedded in all 40 float fields across 4 protobuf submessages at initialization (`init_embedded_sentinels()`), preventing Proto3's default-value-suppression bug (DAY 76–78).

### 11.4 Generalization Limitations

The current model is validated on CTU-13 Neris (IRC botnet C2 traffic). Generalization to other attack families:

| Attack Type | Estimated Recall | Notes |
|---|---|---|
| Neris IRC botnet | 1.0000 | Training distribution |
| Generic port scanning | ~0.85–0.95 | Similar flow statistics |
| WannaCry / NotPetya (SMB) | ~0.70–0.85 | Requires `rst_ratio`, `syn_ack_ratio` features + retraining |
| DNS-based C2 | Low | Layer 3/4 only — DPI required |

**Planned mitigations:** Synthetic SMB traffic generation (`docs/design/synthetic_data_wannacry_spec.md`), `rst_ratio` and `syn_ack_ratio` feature implementation (P1 backlog), retraining pipeline (ADR-008).

---

## 12. Performance Envelope

### Validated (Vagrant / VirtualBox)
| Metric | Value | Condition |
|---|---|---|
| Throughput (pipeline) | 100% at ≤ 38 Mbps | VirtualBox NIC limit |
| Throughput (physical) | ≥ 100 Mbps | Estimated — bare-metal pending |
| Detection latency (Path A) | < 1 ms | FastDetector rule engine |
| Detection latency (Path B) | < 5 ms | ONNX Runtime RandomForest |
| Crypto errors | 0 / 36,000 events | DAY 52–62 stress test |
| F1 Score | 0.9985 | CTU-13 Neris, DAY 86 |
| False Negative Rate | 0.00% | Zero missed malicious flows |

### Next milestone
- **Bare-metal stress test** — confirm ≥ 100 Mbps without VirtualBox NIC bottleneck
- Target hardware: standard x86_64 server, Intel NIC with XDP driver support

---

## 13. Known Limitations

1. **DEBT-FD-001 — Path A hardcoded thresholds:** `FastDetector::is_suspicious()` ignores `sniffer.json`. Thresholds are compile-time constants. Resolution scheduled for PHASE2 (ADR-006).

2. **Single-window aggregation:** Current flow aggregation uses a fixed 10-second window. NotPetya's lateral movement pattern requires longer windows (60s+). Mitigation: FEAT-WINDOW-2 (P2 backlog).

3. **No DPI (Deep Packet Inspection):** ML Defender operates on layer 3/4 headers and flow statistics only. DNS-based C2 traffic, encrypted C2, and WannaCry killswitch domain queries are not detectable without DPI. This is an explicit design choice (resource-constrained targets cannot afford DPI overhead).

4. **Dataset bias:** The model was trained on a dataset with 98% malicious traffic (CTU-13 Neris). Performance on balanced real-world traffic distributions is estimated but not yet validated at bare-metal scale.

5. **SMB ransomware generalization:** Without `rst_ratio` / `syn_ack_ratio` features and SMB synthetic retraining data, recall against WannaCry/NotPetya is estimated at 0.70–0.85. These features are in the P1 backlog.

6. **LLM quality:** TinyLlama is used for its offline, low-resource profile. Complex forensic reasoning that requires multi-hop inference may produce degraded results compared to larger models.

---

## 14. Roadmap

### PHASE 1 — Current (DAY 91)
- ✅ 6/6 pipeline components running end-to-end
- ✅ 31/31 tests passing
- ✅ F1 = 0.9985 on CTU-13 Neris
- ✅ Full arXiv paper draft (v4)
- 🔄 arXiv submission (awaiting endorser response)
- 🔄 Bare-metal stress test

### PHASE 2 — Features (DAY 91–120)
| ID | Feature | Priority |
|---|---|---|
| rst_ratio | RST packet ratio feature | **P1** |
| syn_ack_ratio | SYN/ACK ratio feature | **P1** |
| DEBT-FD-001 | Fix Path A to read sniffer.json | P1 |
| FEAT-NET-1 | DNS/DGA detection | P1 |
| FEAT-NET-2 | Threat intelligence feeds | P1 |
| port_diversity_ratio | Port diversity feature | P2 |
| new_dst_ip_rate | New destination IP rate | P2 |
| dst_port_445_ratio | SMB port ratio | P2 |
| FEAT-WINDOW-2 | 60s secondary aggregation window | P2 |
| dns_query_count | DNS query volume | P3 |
| smb_connection_burst | SMB burst detection | P3 |
| FEAT-AUTH-1 | Auth log monitoring | P2 |
| FEAT-AUTH-2 | Brute force detection | P2 |
| FEAT-EDR-1 | Lightweight endpoint agent | P3 |
| ADR-007 | AND-consensus firewall implementation | P1 |
| ENT-1 | Federated Threat Intelligence | Enterprise |
| ENT-2 | Attack Graph Generation | Enterprise |
| ENT-3 | P2P Seed Distribution (Protobuf) | Enterprise |
| ENT-4 | Hot-Reload JSON config | Enterprise |
| ADR-012 | Plugin loader architecture | P2 |

### Long-term — GAIA Vision
ML Defender is designed as the leaf node of a **hierarchical immune network** (GAIA):
- **Local RAG-clients** (this system) — per-organization, fully offline
- **Campus RAG-masters** — aggregate threat intelligence across local nodes
- **Global RAG-masters** — federated intelligence sharing across organizations

This vision informed the modular, service-discovery-based architecture from the beginning (DAY 1).

---

## Contributing

Please read `CONTRIBUTING.md` before submitting pull requests.

**Methodology:** ML Defender uses **Test Driven Hardening (TDH)** — every new feature is accompanied by:
1. A failing test that defines the expected behavior
2. The implementation
3. A passing test suite (31/31 minimum)
4. An ADR if the change affects system architecture

**Consejo de Sabios:** Major architectural decisions are reviewed by a panel of 7 AI models (Claude, Grok, ChatGPT, DeepSeek, Qwen, Gemini, Parallel.ai) before acceptance. See `docs/consejo/` for decision records.

**macOS development note:** Never use `sed -i` without `-e ''` on macOS (BSD sed incompatibility). Use Python3 inline scripts or edit inside the Vagrant VM.

---

*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic)*
*Consejo de Sabios — ML Defender (aRGus EDR)*
*DAY 91 — 19 marzo 2026*