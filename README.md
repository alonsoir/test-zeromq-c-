# ML Defender (aRGus NDR)

**Open-source, embedded-ML network detection and response system protecting critical infrastructure from ransomware and DDoS attacks.**

[![Via Appia Quality](https://img.shields.io/badge/Via_Appia-Quality-gold)](https://en.wikipedia.org/wiki/Appian_Way)
[![Council of Wise Ones](https://img.shields.io/badge/Architecture-Reviewed_by_8_Models-blueviolet)](#-consejo-de-sabios--multi-model-peer-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![F1=0.9985 Validated](https://img.shields.io/badge/Status-F1%3D0.9985_Validated-brightgreen)]()
[![Tests: make test-all VERDE](https://img.shields.io/badge/Tests-make_test--all_VERDE-brightgreen)]()
[![Pipeline: 6/6](https://img.shields.io/badge/Pipeline-6%2F6_RUNNING-brightgreen)]()
[![Plugin Integrity](https://img.shields.io/badge/Plugin_Integrity-ADR--025_Ed25519-brightgreen)](docs/adr/ADR-025-plugin-integrity-ed25519.md)
[![safe_path](https://img.shields.io/badge/safe__path-ADR--037_header--only-brightgreen)](contrib/safe-path/)
[![PHASE 4](https://img.shields.io/badge/PHASE_4-COMPLETADA-brightgreen)]()
[![AppArmor](https://img.shields.io/badge/AppArmor-6%2F6_enforce-brightgreen)]()
[![Reproducible](https://img.shields.io/badge/Infra-make_bootstrap-brightgreen)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Prec%3D0.9945_In--Distribution-brightgreen)]()
[![Hardened](https://img.shields.io/badge/Security-v0.5.2--hardened-brightgreen)]()
[![OOD Finding](https://img.shields.io/badge/OOD_Finding-Published_DAY_122-orange)]()
[![PRE-PRODUCTION](https://img.shields.io/badge/Status-PRE--PRODUCTION-orange)]()
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)

📜 Living contracts: [Protobuf schema](docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](docs/contracts/JSON%20contracts.md) · [RAG API](docs/contracts/Rag%20security%20commands.md)

---

✅ `main` is tagged `v0.5.2-hardened` — DAY 128 documentation + hardening complete. 4 debts closed, 5 property tests GREEN, Snyk 18 findings triaged.
**PRE-PRODUCTION: do not deploy in hospitals until ACRL (DEBT-PENTESTER-LOOP-001) is complete.**

---

## Estado actual — DAY 128 (2026-04-24)

**Tag activo:** `v0.5.2-hardened` | **Branch activa:** `main` (limpio)

### Pipeline
- 6/6 componentes RUNNING — validado en VM destruida y reconstruida desde cero
- `make test-all`: ALL TESTS COMPLETE
- TEST-PROVISION-1: 8/8 OK

### Hitos DAY 128
- **VM nueva desde cero** — vagrant destroy + up + bootstrap. Pipeline 6/6 RUNNING.
- **Hallazgo técnico:** `resolve_seed()` enforza `0400` con `std::terminate()`. Componentes con seeds deben arrancar con `sudo`. Documentado como patrón permanente.
- **4 deudas cerradas:** DEBT-SAFE-PATH-TAXONOMY-DOC-001, DEBT-PROPERTY-TESTING-PATTERN-001, DEBT-PROVISION-PORTABILITY-001, DEBT-SNYK-WEB-VERIFICATION-001
- **5 property tests GREEN** en `contrib/safe-path/tests/test_safe_path_property.cpp` — integrados en `make test-libs`
- **18 Snyk findings triados** — 1 HIGH nuevo: DEBT-IPTABLES-INJECTION-001 (CWE-78, DAY 129)
- **Consejo 8/8 DAY 128:** execve() sin shell para iptables (unánime), NDR standalone para FEDER (unánime), cleanup EtcdClient antes de ADR-024 (5/3)

### Deuda técnica abierta
Ver [docs/BACKLOG.md](docs/BACKLOG.md) para detalle completo.

| Deuda | Prioridad | Target |
|-------|-----------|--------|
| DEBT-IPTABLES-INJECTION-001 | 🔴 BLOQUEANTE | DAY 129 |
| DEBT-FEDER-SCOPE-DOC-001 | 🟡 Media | DAY 129 |
| DEBT-FIREWALL-CONFIG-PATH-001 | 🔍 Verificar | DAY 129 |
| DEBT-ETCDCLIENT-LEGACY-SEED-001 | ⏳ pre-P2P | feature/etcdclient-p2p-cleanup |

### Próxima frontera (post-deuda)
- **DEBT-PENTESTER-LOOP-001** — ACRL: Caldera → eBPF capture → XGBoost retrain → Ed25519 sign → hot-swap

### ⚠️ NO desplegar en producción hasta
- DEBT-PENTESTER-LOOP-001 completado (datos reales ACRL)
- ADR-036 (Formal Verification Baseline)

---

## 🏗️ Tres variantes del pipeline

| Variante | Estado | Descripción |
|----------|--------|-------------|
| **aRGus-dev** | ✅ Activa (`main`) | x86-debug, imagen Vagrant completa, build-debug. Para investigación y desarrollo diario. |
| **aRGus-production** | 🟡 Pendiente | x86-apparmor + arm64-apparmor. Imágenes Debian optimizadas. Para hospitales, escuelas, municipios. |
| **aRGus-seL4** | ⏳ Diseño futuro | Apéndice científico. Kernel seL4, libpcap (no eBPF/XDP), sniffer monohilo. Branch independiente. |

---

## 📄 Preprint

**ML Defender (aRGus NDR)** is documented in a peer-reviewed preprint published on **arXiv cs.CR** (April 2026).

> *ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and Anomalous Traffic Detection in Resource-Constrained Organizations*
> — Alonso Isidoro Román

**arXiv:** [arXiv:2604.04952 \[cs.CR\]](https://arxiv.org/abs/2604.04952)
**DOI:** https://doi.org/10.48550/arXiv.2604.04952
**Published:** 3 April 2026 · **Draft v16** (updated 19 April 2026) · MIT license
**Code:** https://github.com/alonsoir/argus

---

## 🎯 Mission

Democratize enterprise-grade cybersecurity for hospitals, schools, and small organizations that cannot afford commercial solutions. Built to last decades with scientific honesty and methodical development.

**Philosophy**: *Via Appia Quality* — Systems built like Roman roads, designed to endure.

> *"Un escudo que aprende de su propia sombra."*

---

## 🛡️ Threat Model Scope

ML Defender is a **Network Detection and Response (NDR)** system. Its guiding principle is **network surveillance**: every component operates on network traffic.

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
| **Path traversal prevention** | **ADR-037 MERGED** | `safe_path` header-only — 3 primitivas + 16+ RED→GREEN tests |
| **Dev/prod parity** | **DAY 127 MERGED** | `resolve_config()` — symlinks legítimos en prefix confiable |
| **CI gate** | **TEST-PROVISION-1 8/8** | |

---

## 🔬 DAY 122 Scientific Finding

On DAY 122, a rigorous temporal holdout evaluation on CIC-IDS-2017 revealed a structural covariate shift: Wednesday contains exclusively application-layer DoS attacks (Hulk, GoldenEye, Slowloris) absent from all training days. **No threshold can simultaneously satisfy Precision≥0.99 and Recall≥0.95 on Wednesday data.** This is not an XGBoost failure — it is an empirical impossibility result caused by the dataset's day-specific attack segregation design.

**This finding corroborates Sommer & Paxson (2010)** and provides new quantitative evidence that static classifiers trained on academic benchmarks are structurally insufficient for production NDR.

**The architectural response** — the Adversarial Capture-Retrain Loop (ACRL) — is proposed in §11.18 of the paper.

---

## 🔒 DAY 124-127 Security Hardening

### ADR-037 — safe_path (DAY 124)

`contrib/safe-path/` is a zero-dependency C++20 header-only library that prevents path traversal attacks across all production components. Three active primitives with distinct security semantics:

```cpp
// General paths — prefix verified post-canonical resolution
const auto safe = argus::safe_path::resolve(path, "/etc/ml-defender/");

// Cryptographic seed material — lstat() PRE-resolution, symlinks strictly rejected
// (fs::is_symlink(resolved) arrives too late — weakly_canonical() already resolved it)
const int fd = argus::safe_path::resolve_seed(seed_path, keys_dir_);

// Config files with legitimate symlinks — lexically_normal() verifies prefix
// BEFORE following symlinks (enables /etc/ml-defender/ → /vagrant/ dev/prod parity)
const auto cfg = argus::safe_path::resolve_config(config_path, "/etc/ml-defender/");
```

**Taxonomy (Consejo 8/8 · DAY 127):**

| Primitive | Use case | Symlinks | Verification |
|-----------|----------|----------|-------------|
| `resolve()` | General paths | Allowed post-check | `weakly_canonical()` post-resolution |
| `resolve_seed()` | Crypto material | ❌ Strictly rejected | `lstat()` pre-resolution |
| `resolve_config()` | Config files | ✅ Allowed in prefix | `lexically_normal()` pre-resolution |
| `resolve_model()` | ML models (future) | TBD | Ed25519 signature verify — backlog ADR-038 |

### Test-Driven Hardening — Property Testing (DAY 125-127)

DAY 125-127 validated key TDH principles through empirical evidence:

```cpp
// memory_utils.hpp — header-only, independently testable
[[nodiscard]] inline double compute_memory_mb(long pages, long page_size) noexcept {
    return (static_cast<double>(pages) * static_cast<double>(page_size)) / (1024.0 * 1024.0);
}
// Note: double chosen over int64_t — LONG_MAX/4096 * 8192 overflows int64_t.
// Property test PropertyNeverNegative caught this latent bug in the int64_t version.
```

**Testing hierarchy (Consejo 8/8 · DAY 127):**

| Layer | What it verifies | When |
|-------|-----------------|------|
| Unit tests | Specific known cases (RED→GREEN) | Every security fix |
| Property tests | Mathematical invariants | Every security fix |
| Fuzzing (libFuzzer) | Parsers and external interfaces | Post-property-testing |
| Mutation testing | Test suite quality | Pre-major-release |

**Permanent rules (Council 8/8):**
- Every security fix must include: (1) unit test RED→GREEN, (2) property test for invariants, (3) integration test in real component.
- Every new file-handling surface must be classified with PathPolicy before implementation.

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
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (ChaCha20-Poly1305 encrypted)                  │
│  ┌──────────────────┐                                            │
│  │  ml-detector     │  4× Embedded RandomForest classifiers     │
│  │  (C++20)         │  XGBoost plugin ADR-026 ✅ Prec=0.9945    │
│  └──────────────────┘                                            │
│         ↓  ZeroMQ (encrypted)                                    │
│  ┌──────────────────┐                                            │
│  │  etcd-server     │  Component registration + seed distrib.   │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │ firewall-acl     │  Autonomous blocking via ipset/iptables   │
│  │ agent (C++20)    │  safe_path::resolve_config() DAY 127 ✅   │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  FAISS + SQLite event ingestion           │
│  │  (C++20)         │  safe_path::resolve_config() DAY 127 ✅   │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-security    │  TinyLlama natural language interface      │
│  │  (C++20+LLM)     │  Local inference — no cloud exfiltration  │
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
make up
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
```

---

## 🗺️ Roadmap

### ✅ DONE — DAY 127 (23 Apr 2026) — DEBT-DEV-PROD-SYMLINK-001 🎉
- [x] **resolve_config()** ✅ — nueva primitiva safe_path para configs con symlinks legítimos
- [x] **Makefile paths absolutos** ✅ — fin de paths relativos en arranque de componentes
- [x] **Consejo 8/8** ✅ — taxonomía safe_path formalizada + pregunta crítica FEDER

### ✅ DONE — DAY 126 (23 Apr 2026) — v0.5.2-hardened 🎉
- [x] **DEBT-SAFE-PATH-SEED-SYMLINK-001** ✅ — lstat() pre-resolution, 11/11 tests
- [x] **DEBT-CONFIG-PARSER-FIXED-PREFIX-001** ✅ — prefix fijo, 4/4 + 3/3 tests
- [x] **DEBT-PRODUCTION-TESTS-REMAINING-001** ✅ — seed-client + firewall 3/3 + 3/3
- [x] **DEBT-MEMORY-UTILS-BOUNDS-001** ✅ — MAX_REALISTIC_MEMORY_MB, 5/5 tests
- [x] **Tag: v0.5.2-hardened** ✅

### ✅ DONE — DAY 125 (22 Apr 2026) — DEBT CLOSURE 🎉
- [x] **DEBT-GITIGNORE-TEST-SOURCES-001** ✅
- [x] **DEBT-INTEGER-OVERFLOW-TEST-001** ✅ — property test caught latent bug in own fix
- [x] **DEBT-SAFE-PATH-TEST-RELATIVE-001** ✅
- [x] **DEBT-SAFE-PATH-TEST-PRODUCTION-001** ✅ (rag-ingester)
- [x] **DEBT-CRYPTO-TRANSPORT-CTEST-001** ✅

### ✅ DONE — DAY 124 (21 Apr 2026) — ADR-037 HARDENING 🎉
- [x] **ADR-037** — `contrib/safe-path/` header-only C++20 · 9 RED→GREEN tests ✅
- [x] **Tag: v0.5.1-hardened** ✅

### ✅ DONE — DAY 128 (24 Apr 2026) — Documentation + Hardening 🎉
- [x] **DEBT-SAFE-PATH-TAXONOMY-DOC-001** ✅ — docs/SECURITY-PATH-PRIMITIVES.md
- [x] **DEBT-PROPERTY-TESTING-PATTERN-001** ✅ — 5 property tests GREEN en safe_path
- [x] **DEBT-PROVISION-PORTABILITY-001** ✅ — ARGUS_SERVICE_USER + sudo para seeds
- [x] **DEBT-SNYK-WEB-VERIFICATION-001** ✅ — 18 findings triados, DEBT-IPTABLES-INJECTION-001 identificado
- [x] **Hallazgo:** resolve_seed() enforza 0400 con std::terminate() — sudo es el mecanismo correcto
- [x] **Consejo 8/8** ✅ — 5 decisiones vinculantes documentadas

### 🔜 NEXT — DAY 129: CWE-78 Fix + EtcdClient Cleanup

| Priority | Task |
|---|---|
| 🔴 P0 BLOQUEANTE | DEBT-IPTABLES-INJECTION-001 — execve() sin shell en IPTablesWrapper |
| 🟡 P1 | DEBT-ETCDCLIENT-LEGACY-SEED-001 — cleanup pre-P2P, [[deprecated]] + eliminar |
| 🟡 P1 | DEBT-FEDER-SCOPE-DOC-001 — docs/FEDER-SCOPE.md |
| 🟡 P1 | Property test compute_memory_mb (F17) RED→GREEN |
| 🟡 P2 | Property test HKDF key derivation |

### 🔜 THEN — PHASE 5: Adversarial Capture-Retrain Loop

| Priority | Task |
|---|---|
| P0 | **DEBT-PENTESTER-LOOP-001** — MITRE Caldera → real adversarial flows → XGBoost retraining |
| P0 | **ADR-038** — ACRL formal design |
| P0 | **BACKLOG-FEDER-001** — clarificar scope con Andrés Caro Lindo (NDR standalone vs federación) |
| P1 | aRGus-production images (x86 + ARM64 apparmor) |
| P2 | aRGus-seL4 research branch |

---

## 🧠 Consejo de Sabios — Multi-Model Peer Review

Eight large language models serve as intellectual co-reviewers:

**Claude** (Anthropic) · **Grok** (xAI) · **ChatGPT** (OpenAI) · **DeepSeek** · **Qwen** (Alibaba) · **Gemini** (Google) · **Kimi** (Moonshot) · **Mistral**

Methodology: structured disagreement. Problems must be demonstrated with compilable tests or mathematics before fixes are proposed. Documented in the preprint §6.

---

## 🗺️ Milestones

- ✅ DAY 111: **arXiv:2604.04952 PUBLICADO** 🎉
- ✅ DAY 113: **ADR-025 MERGED — v0.3.0-plugin-integrity** 🎉
- ✅ DAY 118: **PHASE 3 COMPLETADA — v0.4.0** 🎉
- ✅ DAY 120: **make bootstrap + XGBoost F1=0.9978** 🎉
- ✅ DAY 122: **PHASE 4 COMPLETADA — v0.5.0-preproduction** 🎉
- ✅ DAY 124: **ADR-037 MERGED — v0.5.1-hardened** 🎉
- ✅ DAY 125: **5 debts closed · property testing validates TDH · 47 test sources recovered** 🎉
- ✅ DAY 126: **4 debts closed · lstat() pre-resolution · fixed prefix · v0.5.2-hardened** 🎉
- ✅ DAY 127: **resolve_config() · dev/prod parity · Consejo 8/8 taxonomía safe_path** 🎉
- ✅ DAY 128: **VM nueva 6/6 · 4 deudas cerradas · 5 property tests · Snyk 18 findings · Consejo 8/8** 🎉
- 🔜 DAY 129: **DEBT-IPTABLES-INJECTION-001 (CWE-78) · EtcdClient cleanup · FEDER scope doc**

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*