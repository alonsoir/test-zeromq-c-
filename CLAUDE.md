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
│  firewall-acl-agent (C++20) ✅ Day 52 Production-Ready                 │
│  ├─ Decrypt + Decompress incoming detections                           │
│  ├─ Batch processing (configurable threshold + timeout)                │
│  ├─ IPTables / IPSet management                                        │
│  ├─ Config-driven (JSON is law, zero hardcoding)                       │
│  ├─ Graceful degradation (tested 364 events/sec)                       │
│  └─ RAG logger (plaintext, ready for ingestion)                        │
│                                                                        │
│  ────────────── Supporting Services ──────────────                     │
│                                                                        │
│  etcd-server         Distributed config, service registry, heartbeats  │
│  etcd-client         Shared library for component registration         │
│  crypto-transport    Unified ChaCha20-Poly1305 + LZ4 library           │
│  RAG + TinyLlama     Natural language security query system (caché L1) │
│  protobuf            Unified .proto definitions (network_security)     │
│  rag-ingester        Ingest data from ml-detector (+ firewall planned) │
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
│   ├── BACKLOG.md                 Product roadmap (Day 52+)
│   └── config/firewall.json       Config-driven setup (no hardcoding)
├── etcd-server/                   Configuration & service supervisor
├── etcd-client/                   Shared registration library
├── crypto-transport/              ChaCha20-Poly1305 + LZ4 (libcrypto_transport.so)
├── rag/                           RAG system + FAISS + TinyLlama
│   ├── BACKLOG.md                 Query engine roadmap
│   └── models/                    LLM models (tinyllama-1.1b-chat-v1.0.Q4_0.gguf)
├── rag-ingester/                  Log parser & vector DB ingestion
│   └── BACKLOG.md                 Ingestion roadmap (firewall logs planned)
├── protobuf/                      Shared .proto definitions + generate.sh
├── config/                        JSON configs for all components
├── tools/                         Utilities (synthetic_ml_output_injector, etc.)
├── logs/                          Runtime logs
│   ├── lab/                       Lab-mode logs (firewall-agent.log, etc.)
│   ├── rag/                       RAG artifacts and JSONL events
│   └── firewall-acl-agent/        Legacy path (deprecated Day 52)
├── ml-training/                   Training scripts and outputs
│   └── outputs/onnx/              Exported ONNX models
├── third_party/                   Vendored dependencies (llama.cpp, etc.)
├── docs/                          Project documentation
│   └── day52_continuity_prompt.md Latest session summary
├── Vagrantfile                    Lab environment (Debian Bookworm)
├── Makefile                       Root build orchestration
├── README.md                      Project overview & quickstart
└── CLAUDE.md                      ← You are here (Technical Reference)
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

**Day 52 Validation:** 36,000 events stress tested with 0 crypto errors, 0 decompression errors. Pipeline is production-ready.

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

## Performance Targets & Achievements (Day 52)

**Targets:**
- Detection latency: < 1 µs (sub-microsecond)
- Crypto pipeline: < 3 µs per operation
- LZ4 compression ratio: ~50% on JSON configs
- Sustained operation: 10+ hours continuous, thousands of events
- Memory footprint: < 500 MB
- Batch IPSet operations: > 10K IPs/sec (planned async queue)

**Achieved (Day 52 Stress Testing):**
- ✅ Detection latency: 0.24 µs - 1.06 µs (4 models)
- ✅ Crypto pipeline: 0 errors @ 36,000 events
- ✅ Sustained operation: 10+ hours, 364 events/sec peak
- ✅ Memory: 127 MB RSS under extreme load
- ✅ CPU: 54% max @ 364 events/sec
- ✅ Graceful degradation: No crashes when IPSet capacity exceeded

**Stress Test Results:**

| Test | Events | Target Rate | Actual Rate | Duration | Result |
|------|--------|-------------|-------------|----------|--------|
| 1    | 1,000  | 50/sec      | 42.6/sec    | 23.5s    | ✅ PASS |
| 2    | 5,000  | 100/sec     | 94.9/sec    | 52.7s    | ✅ PASS |
| 3    | 10,000 | 200/sec     | 176.1/sec   | 56.8s    | ✅ PASS |
| 4    | 20,000 | 500/sec     | 364.9/sec   | 54.8s    | ✅ PASS |

**Total:** 36,000 events, 0 crypto errors, 0 parse errors, 0 crashes.

---

## Development Phase & Current State (Day 52 — Feb 8, 2026)

**Phase:** Phase 1 complete ✅. Hardening & capacity optimization in progress.

**Validated:**
- ✅ End-to-end pipeline: eBPF capture → ML inference → firewall blocking
- ✅ Crypto pipeline: 0 crypto/decompression errors (36K events stress tested)
- ✅ etcd service registry + heartbeats (30s interval, 90s timeout)
- ✅ 97.6% detection accuracy on CTU-13 Neris botnet (492K events)
- ✅ 10+ hour continuous operation
- ✅ Config-driven architecture (no hardcoded values)
- ✅ IPSet verification on startup
- ✅ Graceful degradation under capacity limits

**Known Issues / Active Work:**

**RESOLVED (Day 52):**
- ~~`firewall-acl-agent`: Hardcoded ipset name~~ ✅ Now reads from `config.ipsets` map
- ~~`firewall-acl-agent`: Logger path hardcoded~~ ✅ Now reads from `config.logging.file`
- ~~`firewall-acl-agent`: Duplicate logging config~~ ✅ Single source of truth

**REMAINING:**
- `firewall-acl-agent`: Needs async queue + worker pool for 1K+ events/sec sustained (Backlog P1.2)
- `firewall-acl-agent`: Multi-tier storage (IPSet → SQLite → Parquet) for unlimited capacity (Backlog P1.1)
- `firewall-acl-agent`: Capacity monitoring + auto-eviction (Backlog P1.3)
- `rag-ingester`: Firewall log parser (ground truth blocking data for RAG, Backlog P1.1)
- `rag`: Cross-component queries (detection ↔ block linking, Backlog P1.1)
- FAISS integration for semantic search: Backlog P2.2

**NEXT (Day 53 - Log Security):**
- **CRITICAL**: HMAC-based log integrity for RAG logs (prevent log poisoning attacks)
- Phase 1: Audit ml-detector + rag-ingester current state
- Phase 2: Implement HMAC in firewall-acl-agent → rag-ingester → rag
- Phase 3: Implement HMAC in ml-detector → rag-ingester → rag

**See detailed backlogs:**
- [firewall-acl-agent/BACKLOG.md](firewall-acl-agent/BACKLOG.md)
- [rag-ingester/BACKLOG.md](rag-ingester/BACKLOG.md)
- [rag/BACKLOG.md](rag/BACKLOG.md)

---

## RAG Integration Architecture (Day 52 Discovery)

**Critical Insight:** RAG needs BOTH ml-detector AND firewall-acl-agent logs.

```
ml-detector logs:
  ✅ What was detected (IP, confidence, attack type)
  ❌ Whether it was actually blocked
  
firewall-acl-agent logs:
  ✅ What was blocked (IP, timestamp, duration)
  ✅ Packets/bytes dropped (impact measurement)
  ✅ Eviction events (capacity management)
  ❌ Why it was detected (no ML features)
```

**Together they provide:**
- Detection efficacy: "X% of detections resulted in blocks"
- Forensic timeline: "IP 1.2.3.4 detected at T1, blocked at T2, dropped N packets"
- False positive analysis: "Internal IPs blocked with low confidence"
- ML retraining data: Ground truth labels for model improvement

**Planned:** rag-ingester P1.1 will parse firewall logs and cross-reference with ml-detector detections.

---

## RAG Log Security (Day 53 Plan)

**Problem**: Log poisoning attacks against RAG systems.

**Threat Model**:
```
Attacker with filesystem access can:
  1. Inject malicious content into logs → contaminate RAG
  2. Prompt injection via logs → manipulate LLM responses
  3. Poison ML retraining data → degrade detection accuracy
  4. Hide malicious activity with noise
```

**Why NOT encrypt logs with ChaCha20?**
- FAISS cannot index encrypted content (needs plaintext)
- Encryption provides confidentiality, NOT integrity
- Attacker with leaked key can create valid ciphertexts
- No detection of tampering or injected lines

**Solution: HMAC-based Log Integrity**

```
Component writes:
  log_line = "IP 1.2.3.4 blocked at 12:34:56"
  hmac = HMAC-SHA256(log_line, secret_key)
  write_to_file(f"{log_line}|HMAC:{hmac}")

rag-ingester validates:
  read_line → split(message, hmac)
  expected_hmac = HMAC-SHA256(message, secret_key)
  if hmac != expected_hmac:
      ALERT: Log tampering detected
      REJECT line (do not ingest to RAG)
  else:
      parse and ingest
```

**Benefits**:
- ✅ Detects tampering (modified lines)
- ✅ Detects injection (added lines without valid HMAC)
- ✅ FAISS can still index (plaintext + MAC)
- ✅ Faster than encryption (2μs vs 10μs)
- ✅ Logs remain human-readable
- ✅ Auditability preserved

**Implementation Plan (Day 53)**:

Phase 1: Audit existing state
- Review ml-detector RAG logger implementation
- Review rag-ingester parsing logic
- Document current vulnerabilities

Phase 2: firewall-acl-agent + rag-ingester
- Add HMAC key management to etcd-server
- Implement SecureLogger in firewall-acl-agent
- Implement HMAC validation in rag-ingester
- Add tampering alerts (Slack/email)

Phase 3: ml-detector + rag-ingester
- Add HMAC to existing RAG logger
- Update rag-ingester ML detector parser
- Validate end-to-end integrity

**Security Guarantees (after Day 53)**:
- ✅ No log injection without valid HMAC
- ✅ No log modification without detection
- ✅ Tampering triggers immediate alerts
- ✅ RAG contains only validated logs
- ✅ ML retraining data integrity verified

**Key Management**:
```
etcd-server generates:
  /secrets/ml-detector/log_hmac_key (32 bytes)
  /secrets/firewall/log_hmac_key (32 bytes)

Components retrieve on startup:
  HMAC key from etcd (read-only access)
  Weekly key rotation (planned)
```

**Defense in Depth**:
- File permissions: 0400 (read-only after write)
- Separate user for rag-ingester (no root)
- HMAC validation before parsing
- Content sanitization (XSS, SQL injection)
- Alerting on suspicious patterns

---

## Anti-Patterns & Lessons Learned

**DO NOT:**
- Hardcode encryption keys, ipset names, or log paths — everything config-driven ✅ (Day 52 fixed)
- Encrypt RAG logs with ChaCha20 — FAISS cannot index ciphertext, use HMAC for integrity
- Assume `etcd-client` decompression works without 4-byte header extraction (Day 52 bug)
- Skip stress testing on any component — capacity issues only surface under load
- Trust academic cybersecurity datasets blindly — validate ground truth independently
- Ignore capacity planning — IPSet has finite limits, need eviction strategy
- Write RAG logs without integrity protection — vulnerable to log poisoning attacks (Day 53)

**DO:**
- Run `make verify-crypto-linkage` after any build change
- Start etcd-server FIRST before any component
- Use `synthetic_ml_output_injector` for integration testing
- Preserve both fast_score and ml_score in all events
- Document what works AND what doesn't (Scientific Honesty)
- Test graceful degradation — systems should degrade, not crash
- Use single source of truth for config (JSON is law)
- Protect RAG logs with HMAC — detect tampering before ingestion (Day 53)

---

## Via Appia Quality Principles

1. **Funciona > Perfecto** — Working code first, optimization later
2. **Seguridad en Mente** — Security baked in, not bolted on
3. **Zero Hardcoding** — Config-driven, not magic numbers ✅ (Day 52 enforced)
4. **Scientific Honesty** — Document what works AND what doesn't
5. **La Rueda es Redonda** — Use standards (systemd, etcd, protobuf)
6. **Graceful Degradation** — Degrade under stress, never crash ✅ (Day 52 validated)
7. **El JSON es la Ley** — Configuration file is single source of truth ✅ (Day 52 principle)

---

## Collaboration Model

This project is developed collaboratively between a human architect (Alonso Isidoro Roman) and multiple AI systems, explicitly credited as co-authors:

- **Claude (Anthropic):** Principal developer — daily implementation, security patterns, debugging
- **Gemini:** Strategic architect — large refactors, paper planning
- **DeepSeek, Grok, ChatGPT, Qwen:** "Consejo de Sabios" (Council of Wise Ones) — peer review

AI contributions are credited transparently in commits and documentation.

---

## Quick Reference: Running a Stress Test (Day 52 Validated)

```bash
# Terminal 1: etcd-server
cd /vagrant/etcd-server/build
sudo ./etcd_server

# Terminal 2: firewall-acl-agent
cd /vagrant/firewall-acl-agent/build
sudo ./firewall-acl-agent -c ../config/firewall.json

# Terminal 3: Injector (moderate load)
cd /vagrant/tools/build
./synthetic_ml_output_injector 1000 50

# Terminal 4: Monitor
tail -f /vagrant/logs/lab/firewall-agent.log

# Verify IPSet
sudo ipset list ml_defender_blacklist_test | head -20

# Check for errors
grep -E "crypto_errors|ipset_failures" /vagrant/logs/lab/firewall-agent.log | tail -5
```

**Expected Results (Day 52 Validated):**
```
crypto_errors: 0              ← Perfect encryption/decryption
decompression_errors: 0       ← Perfect LZ4 pipeline
protobuf_parse_errors: 0      ← Perfect message parsing
ipset_successes: > 0          ← IPs successfully blocked
```

---

## Capacity Planning (Day 52 Lessons)

**IPSet Limits:**
```
Current config: max_elements: 1000, timeout: 3600 (1 hour)
Recommended dev: max_elements: 50000, timeout: 300 (5 min)
Recommended prod: max_elements: 500000, timeout: 900 (15 min)
```

**Formula:**
```
Required capacity = (arrival_rate × timeout) + safety_margin

Example @ 364 IPs/sec:
  With timeout=3600: Need ~1.3M IPs
  With timeout=300:  Need ~109K IPs
```

**When capacity exceeded:**
- System logs `ipset_failures` (not crashes)
- Queue backs up (bounded to prevent OOM)
- Graceful degradation (older entries remain blocked)
- Need: Multi-tier storage (IPSet → SQLite → Parquet) — Backlog P1.1

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
| ipset | >= 7.x | Kernel IP set management |
| iptables | >= 1.8 | Netfilter rule management |

**Build requirements:** C++20, CMake >= 3.16, GCC/Clang with C++20 support.

---

## Production Readiness Checklist (Day 52)

**✅ Ready:**
- Crypto pipeline (0 errors @ 36K events)
- Config-driven architecture (no hardcoding)
- IPSet/IPTables integration
- etcd service registration
- Graceful degradation
- Observability (detailed logging)

**⚠️ Needs Tuning:**
- IPSet capacity adjustment for expected load
- Multi-tier storage (SQLite persistence)
- Async queue + worker pool (1K+ IPs/sec)
- Capacity monitoring + alerting

**📋 Nice to Have:**
- Prometheus metrics exporter
- Grafana dashboards
- Health check endpoints (K8s)
- Runtime config updates via etcd
- RAG integration (firewall logs)

---

*Author: Alonso Isidoro Roman + Claude (Anthropic)*  
*Last Updated: Day 52 — February 8, 2026*  
*Status: Production-ready core, capacity optimization + log security in progress*  
*Next: Day 53 — HMAC-based log integrity (prevent log poisoning)*  
*Via Appia Quality — Built to last decades 🏛️*