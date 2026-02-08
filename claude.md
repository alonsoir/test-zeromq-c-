# ML Defender (aegisIDS) — CLAUDE.md

> Open-source Network Intrusion Detection & Prevention System  
> Designed to protect critical infrastructure (hospitals, schools, SMBs) from ransomware and DDoS attacks.  
> **Philosophy:** Via Appia Quality — built to last decades.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ML DEFENDER PIPELINE                            │
│                                                                        │
│  sniffer (eBPF/XDP)                                                    │
│  ├─ Packet capture via eBPF ring buffer                                │
│  ├─ Fast Detector (rule-based, Layer 1)                                │
│  ├─ 83+ feature extraction                                             │
│  └─ Output: Protobuf event → encrypt(ChaCha20) + compress(LZ4) → ZMQ   │
│                          │                                             │
│                     ZMQ:5571                                           │
│                          ▼                                             │
│  ml-detector (C++20 RandomForest, Layer 2)                             │
│  ├─ Decrypt + Decompress incoming events                               │
│  ├─ 4 embedded ONNX detectors:                                         │
│  │   ├─ DDoS detector                                                  │
│  │   ├─ Ransomware detector                                            │
│  │   ├─ Traffic classifier                                             │
│  │   └─ Internal anomaly detector                                      │
│  ├─ Dual-Score: max(fast_score, ml_score) — Maximum Threat Wins        │
│  ├─ RAG logger (plaintext JSONL, for FAISS indexing)                   │
│  └─ Output: Protobuf → encrypt + compress → ZMQ                        │
│                          │                                             │
│                     ZMQ:5572                                           │
│                          ▼                                             │
│  firewall-acl-agent (C++20)                                            │
│  ├─ Decrypt + Decompress incoming detections                           │
│  ├─ Batch processing (configurable threshold + timeout)                │
│  ├─ IPTables / IPSet management                                        │
│  └─ RAG logger (plaintext)                                             │
│                                                                        │
│  ────────────── Supporting Services ──────────────                     │
│                                                                        │
│  etcd-server         Distributed config, service registry, heartbeats  │
│  etcd-client         Shared library for component registration         │
│  crypto-transport    Unified ChaCha20-Poly1305 + LZ4 library           │
│  RAG + TinyLlama     Natural language security query system (caché L1) │
│  protobuf            Unified .proto definitions (network_security)     │
│  Rag-ingester        ingest data from ml-detector                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
/vagrant/                          (project root, Vagrant synced folder)
├── sniffer/                       eBPF/XDP packet capture + Fast Detector
├── ml-detector/                   C++20 ML inference engine (ONNX Runtime)
│   └── models/production/         Trained models (level1, level2, level3)
├── firewall-acl-agent/            IPTables/IPSet autonomous blocking
├── etcd-server/                   Configuration & service supervisor
├── etcd-client/                   Shared registration library
├── crypto-transport/              ChaCha20-Poly1305 + LZ4 (libcrypto_transport.so)
├── rag/                           RAG system + FAISS + TinyLlama
│   └── models/                    LLM models (tinyllama-1.1b-chat-v1.0.Q4_0.gguf)
├── protobuf/                      Shared .proto definitions + generate.sh
├── config/                        JSON configs for all components
├── tools/                         Utilities (synthetic_ml_output_injector, etc.)
├── logs/                          Runtime logs
│   ├── lab/                       Lab-mode logs
│   ├── rag/                       RAG artifacts and JSONL events
│   └── firewall-acl-agent/        Firewall detailed logs
├── ml-training/                   Training scripts and outputs
│   └── outputs/onnx/              Exported ONNX models
├── third_party/                   Vendored dependencies (llama.cpp, etc.)
├── Vagrantfile                    Lab environment (Debian Bookworm)
├── Makefile                       Root build orchestration
└── CLAUDE.md                      ← You are here
```

---

## Build System

All builds from project root via `make`. Components use CMake internally.

```bash
# Build order matters — dependencies first:
make proto-unified              # 1. Generate protobuf C++ files
make crypto-transport-build     # 2. Build shared crypto library (FIRST!)
make etcd-client-build          # 3. Build etcd-client (depends on crypto-transport)
make etcd-server-build          # 4. Build etcd-server
make sniffer                    # 5. Build eBPF/XDP sniffer
make detector                   # 6. Build ml-detector
make firewall                   # 7. Build firewall-acl-agent
make rag                        # 8. Build RAG system

# Debug builds:
make firewall PROFILE=debug
make tools PROFILE=debug        # Builds synthetic_ml_output_injector

# Verify all components link crypto-transport:
make verify-crypto-linkage      # All should show libcrypto_transport.so.1

# Run tests:
cd crypto-transport/build && ctest --output-on-failure   # 16/16 tests
cd etcd-client/build && ctest --output-on-failure        # 3/3 tests
```

### Vagrant Lab Environment

```bash
vagrant up defender              # Mode 1: Development (single VM)
vagrant up defender client       # Mode 2: Gateway testing (dual-NIC)
vagrant up                       # Mode 3: Full demo

# Running the pipeline:
make run-lab-dev                 # Start all components
make status-lab                  # Check component status
make kill-lab                    # Stop all
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Encryption | ChaCha20-Poly1305 (libsodium) | Authenticated encryption, faster than AES-GCM on ARM/embedded |
| Compression | LZ4 | ~5 GB/s throughput, intelligent skip for small payloads |
| IPC | ZeroMQ (5571, 5572) | Lock-free, zero-copy capable, language-agnostic |
| ML Runtime | ONNX Runtime | Cross-platform, C++ native, model portability |
| Config | etcd (custom server) | Distributed consensus, live reconfig, service registry |
| Packet Capture | eBPF/XDP | Kernel-bypass performance, programmable filtering |
| Serialization | Protobuf | Compact binary, schema evolution, generated C++ |
| RAG Store | FAISS (planned) | Billion-scale similarity search, CPU-optimized |
| LLM | TinyLlama 1.1B (Q4_0 GGUF) | Runs on consumer hardware, llama.cpp backend |

---

## Crypto Pipeline (CRITICAL)

**Single source of truth:** `crypto-transport/` library. ALL components link against `libcrypto_transport.so`.

```
Sender:   JSON → compress(LZ4) → encrypt(ChaCha20-Poly1305) → ZMQ
Receiver: ZMQ → decrypt → decompress → process

Compression format: [4-byte original_size header][compressed_data]
Encryption overhead: +40 bytes fixed (24-byte nonce + 16-byte MAC)
```

**Key management:** Encryption keys distributed via etcd-server. Components retrieve keys during registration. Zero hardcoded keys.

**RAG logs are ALWAYS plaintext** — FAISS/vector DB cannot index encrypted data.

---

## Dual-Score Detection Architecture

```
Fast Detector (sniffer, rule-based):
  external_ips_30s >= 15  → score = 0.70
  smb_diversity >= 10     → score = 0.70
  dns_entropy > 0.95      → score = 0.70

ML Detector (ml-detector, RandomForest):
  4 embedded models → combined ml_score

Final: threat_score = max(fast_score, ml_score)   ← "Maximum Threat Wins"
```

Both scores preserved in events for RAG auditability. Divergence between detectors is **by design** — they measure different threat dimensions.

---

## Performance Targets

- Detection latency: < 1 µs (sub-microsecond)
- Crypto pipeline: < 3 µs per operation
- LZ4 compression ratio: ~50% on JSON configs
- Sustained operation: 10+ hours continuous, thousands of events
- Memory footprint: < 500 MB
- Batch IPSet operations: > 10K IPs/sec

---

## Development Phase & Current State

**Phase:** Phase 1 complete. Stress testing & hardening in progress.

**Validated:**
- End-to-end pipeline: eBPF capture → ML inference → firewall blocking ✅
- Crypto pipeline: 0 crypto/decompression errors across stress tests ✅
- etcd service registry + heartbeats (30s interval, 90s timeout) ✅
- 97.6% detection accuracy on CTU-13 Neris botnet (492K events) ✅
- 10+ hour continuous operation ✅

**Known Issues / Active Work (as of Day 52+):**
- `firewall-acl-agent`: Hardcoded ipset name (`ml_defender_blacklist`) — should read from config
- `firewall-acl-agent`: Logger path hardcoded vs config path
- `firewall-acl-agent`: Needs async queue + worker pool architecture for high throughput in a multithreaded enviroment.
- FAISS integration for semantic search: planned next phase

---

## Anti-Patterns & Lessons Learned

**DO NOT:**
- Hardcode encryption keys, ipset names, or log paths — everything config-driven
- Encrypt RAG logs — FAISS cannot index encrypted content
- Assume `etcd-client` decompression works without 4-byte header extraction (Day 52 bug)
- Skip stress testing on any component — firewall-acl-agent bugs only surfaced under load
- Trust academic cybersecurity datasets blindly — validate ground truth independently

**DO:**
- Run `make verify-crypto-linkage` after any build change
- Start etcd-server FIRST before any component
- Use `synthetic_ml_output_injector` for integration testing
- Preserve both fast_score and ml_score in all events
- Document what works AND what doesn't (Scientific Honesty)

---

## Via Appia Quality Principles

1. **Funciona > Perfecto** — Working code first, optimization later
2. **Seguridad en Mente** — Security baked in, not bolted on
3. **Zero Hardcoding** — Config-driven, not magic numbers
4. **Scientific Honesty** — Document what works AND what doesn't
5. **La Rueda es Redonda** — Use standards (systemd, etcd, protobuf)

---

## Collaboration Model

This project is developed collaboratively between a human architect (Alonso Isidoro Roman) and multiple AI systems, explicitly credited as co-authors:

- **Claude (Anthropic):** Principal developer — daily implementation, security patterns, debugging
- **Gemini:** Strategic architect — large refactors, paper planning
- **DeepSeek, Grok, ChatGPT, Qwen:** "Consejo de Sabios" (Council of Wise Ones) — peer review

AI contributions are credited transparently in commits and documentation.

---

## Quick Reference: Running a Stress Test

```bash
# Terminal 1: etcd-server
cd etcd-server/build && ./etcd-server --config config/etcd-server.json

# Terminal 2: firewall-acl-agent
cd firewall-acl-agent/build-debug && sudo ./firewall-acl-agent --config /vagrant/config/firewall.json --verbose

# Terminal 3: Injector (moderate load)
cd tools/build-debug && ./synthetic_ml_output_injector 1000 100

# Terminal 4: Monitor
tail -f /vagrant/logs/firewall-acl-agent/firewall_detailed.log
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| libsodium | >= 1.0.18 | ChaCha20-Poly1305 encryption |
| liblz4 | >= 1.9.0 | LZ4 compression |
| libzmq | >= 4.3 | Inter-component messaging |
| protobuf | >= 3.x | Binary serialization |
| ONNX Runtime | >= 1.x | ML model inference |
| libbpf | latest | eBPF/XDP packet capture |
| llama.cpp | vendored | TinyLlama inference |
| FAISS | planned | Vector similarity search |

**Build requirements:** C++20, CMake >= 3.16, GCC/Clang with C++20 support.

---

*Author: Alonso Isidoro Roman + Claude (Anthropic)*  
*Via Appia Quality — Built to last decades 🏛️*