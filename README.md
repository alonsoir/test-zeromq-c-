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
[![Hardened](https://img.shields.io/badge/Security-v0.5.1--hardened-brightgreen)]()
[![OOD Finding](https://img.shields.io/badge/OOD_Finding-Published_DAY_122-orange)]()
[![PRE-PRODUCTION](https://img.shields.io/badge/Status-PRE--PRODUCTION-orange)]()
[![Crypto](https://img.shields.io/badge/Crypto-HKDF_SHA256+ChaCha20_Poly1305-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2604.04952_cs.CR-red)](https://arxiv.org/abs/2604.04952)
[![TDH](https://img.shields.io/badge/Methodology-Test_Driven_Hardening-purple)](https://github.com/alonsoir/test-driven-hardening)

📜 Living contracts: [Protobuf schema](docs/contracts/Protobuf%20contracts.md) · [Pipeline configs](docs/contracts/JSON%20contracts.md) · [RAG API](docs/contracts/Rag%20security%20commands.md)

---

✅ `main` is tagged `v0.5.1-hardened` — PHASE 4 complete + ADR-037 Snyk hardening merged.
**PRE-PRODUCTION: do not deploy in hospitals until debt closure (DAY 126-128) and ACRL (DEBT-PENTESTER-LOOP-001) are complete.**

---

## Estado actual — DAY 125 (2026-04-22)

**Tag activo:** `v0.5.1-hardened` | **Branch activa:** `fix/day125-debt-closure` (pending merge → v0.5.2)

### Pipeline
- 6/6 componentes RUNNING
- `make test-all`: ALL TESTS PASSED (desde VM fría: `vagrant halt → make up → make bootstrap → make test-all`)
- TEST-PROVISION-1: 8/8 OK

### Hitos DAY 125
- **5 deudas cerradas:** GITIGNORE + INTEGER-OVERFLOW + SAFE-PATH-RELATIVE + SAFE-PATH-PRODUCTION(rag-ingester) + CRYPTO-TRANSPORT-CTEST
- **47 fuentes de test versionadas** (descubiertas por fix de `.gitignore`)
- **Hallazgo metodológico clave:** property test `PropertyNeverNegative` encontró bug latente en fix F17 que unit test no cubría (`int64_t` overflow → fix correcto: aritmética `double` directa)
- **3 nuevas deudas bloqueantes identificadas** por Consejo 8/8 → cierre DAY 126 antes de merge a main

### Deuda técnica abierta
Ver [docs/BACKLOG.md](docs/BACKLOG.md) para detalle completo.

| Deuda | Prioridad | Target | Bloquea |
|-------|-----------|--------|---------|
| DEBT-SAFE-PATH-SEED-SYMLINK-001 | 🔴 Crítica | DAY 126 | merge a main |
| DEBT-CONFIG-PARSER-FIXED-PREFIX-001 | 🔴 Alta | DAY 126 | merge a main |
| DEBT-PRODUCTION-TESTS-REMAINING-001 | 🔴 Alta | DAY 126 | ADR-038 |
| DEBT-MEMORY-UTILS-BOUNDS-001 | 🟡 Media | DAY 126 | merge a main |
| DEBT-SNYK-WEB-VERIFICATION-001 | 🟡 Media | DAY 126 | científico |
| DEBT-PROPERTY-TESTING-RAPIDCHECK-001 | 🟢 Media | DAY 127 | — |
| DEBT-DEV-PROD-SYMLINK-001 | 🟢 Media | DAY 127 | — |
| DEBT-PROVISION-PORTABILITY-001 | 🟢 Media | DAY 128 | — |

### Próxima frontera (post-deuda)
- **DEBT-PENTESTER-LOOP-001** — ACRL: Caldera → eBPF capture → XGBoost retrain → Ed25519 sign → hot-swap

### ⚠️ NO desplegar en producción hasta
- Deuda técnica DAY 124-126 cerrada completamente
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
| **Path traversal prevention** | **ADR-037 MERGED** | `safe_path` header-only, 9 RED→GREEN tests |
| **CI gate** | **TEST-PROVISION-1 8/8** | |

---

## 🔬 DAY 122 Scientific Finding

On DAY 122, a rigorous temporal holdout evaluation on CIC-IDS-2017 revealed a structural covariate shift: Wednesday contains exclusively application-layer DoS attacks (Hulk, GoldenEye, Slowloris) absent from all training days. **No threshold can simultaneously satisfy Precision≥0.99 and Recall≥0.95 on Wednesday data.** This is not an XGBoost failure — it is an empirical impossibility result caused by the dataset's day-specific attack segregation design.

**This finding corroborates Sommer & Paxson (2010)** and provides new quantitative evidence that static classifiers trained on academic benchmarks are structurally insufficient for production NDR.

**The architectural response** — the Adversarial Capture-Retrain Loop (ACRL) — is proposed in §11.18 of the paper.

---

## 🔒 DAY 124-125 Security Hardening

### ADR-037 — safe_path (DAY 124)

`contrib/safe-path/` is a zero-dependency C++20 header-only library that prevents path traversal attacks across all production components:

```cpp
// General config files
const auto safe = argus::safe_path::resolve(config_path, "/etc/ml-defender/");

// Cryptographic seed material (O_NOFOLLOW + 0400 check + symlink rejection)
const int fd = argus::safe_path::resolve_seed(seed_path, keys_dir_);
```

### Test-Driven Hardening — Property Testing (DAY 125)

DAY 125 validated a key TDH principle: property tests catch bugs that synthetic unit tests miss.

```cpp
// memory_utils.hpp — extracted from zmq_handler for independent testability
[[nodiscard]] inline double compute_memory_mb(long pages, long page_size) noexcept {
    return (static_cast<double>(pages) * static_cast<double>(page_size)) / (1024.0 * 1024.0);
}
// Note: double arithmetic chosen over int64_t — LONG_MAX/4096 * 8192 overflows int64_t
// Property test PropertyNeverNegative caught this latent bug in the int64_t version.
```

**Permanent rule (Council 8/8 · DAY 125):** Every security fix must include: (1) synthetic unit test, (2) property test for invariants, (3) integration test in the real component.

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
│  │ agent (C++20)    │  safe_path::resolve() ADR-037 ✅          │
│  └──────────────────┘                                            │
│         ↓                                                        │
│  ┌──────────────────┐                                            │
│  │  rag-ingester    │  FAISS + SQLite event ingestion           │
│  │  (C++20)         │  safe_path::resolve() ADR-037 ✅          │
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

### ✅ DONE — DAY 125 (22 Apr 2026) — DEBT CLOSURE (partial) 🎉
- [x] **DEBT-GITIGNORE-TEST-SOURCES-001** ✅ — 47 fuentes de test versionadas
- [x] **DEBT-INTEGER-OVERFLOW-TEST-001** ✅ — `memory_utils.hpp` + 4 property tests RED→GREEN
- [x] **DEBT-SAFE-PATH-TEST-RELATIVE-001** ✅ — Test 10 en `safe_path`
- [x] **DEBT-SAFE-PATH-TEST-PRODUCTION-001** ✅ (rag-ingester) — `test_config_parser_traversal`
- [x] **DEBT-CRYPTO-TRANSPORT-CTEST-001** ✅ — permisos `0400` en test fixtures

### ✅ DONE — DAY 124 (21 Apr 2026) — ADR-037 HARDENING 🎉
- [x] **ADR-037** — `contrib/safe-path/` header-only C++20 · 9 RED→GREEN tests ✅
- [x] **Tag: `v0.5.1-hardened`** ✅

### 🔜 NEXT — DAY 126: Debt closure (bloqueante para merge → v0.5.2)

| Priority | Task | Why |
|---|---|---|
| 🔴 P0 | DEBT-SAFE-PATH-SEED-SYMLINK-001 | Symlinks en seeds = TOCTOU attack vector |
| 🔴 P0 | DEBT-CONFIG-PARSER-FIXED-PREFIX-001 | Prefix derivado del input = bypass |
| 🔴 P0 | DEBT-PRODUCTION-TESTS-REMAINING-001 | seed-client + firewall-acl-agent |
| 🟡 P1 | DEBT-MEMORY-UTILS-BOUNDS-001 | Guard realista en compute_memory_mb |
| 🟡 P1 | DEBT-SNYK-WEB-VERIFICATION-001 | Cierre científico ADR-037 |

### 🔜 DAY 127-128: Cleanup + Property Testing

| Priority | Task |
|---|---|
| 🟢 P2 | DEBT-PROPERTY-TESTING-RAPIDCHECK-001 — rapidcheck submodule |
| 🟢 P2 | DEBT-DEV-PROD-SYMLINK-001 — symlinks en Vagrantfile |
| 🟢 P2 | DEBT-PROVISION-PORTABILITY-001 — ARGUS_SERVICE_USER |

### 🔜 THEN — PHASE 5: Adversarial Capture-Retrain Loop

| Priority | Task |
|---|---|
| P0 | **DEBT-PENTESTER-LOOP-001** — MITRE Caldera → real adversarial flows → XGBoost retraining |
| P0 | **ADR-038** — ACRL formal design |
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
- 🔜 DAY 126: **v0.5.2 target — seed symlink + config prefix + remaining component tests**

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

**Via Appia Quality** 🏛️ — *Built to last decades.*