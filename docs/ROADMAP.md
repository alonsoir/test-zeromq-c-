cd ~/Code/test-zeromq-docker

cat > ROADMAP.md << 'ROADMAP'
# 🗺️ Project Roadmap: IDS/IPS → WAF Evolution

**Vision:** Build a production-grade, kernel-native, ML-powered Web Application Firewall starting from a solid IDS/IPS foundation.

**Philosophy:** Incremental, testable phases. Each phase delivers value independently.

---

## 🎯 Core Principles

- **JSON is LAW**: Configuration-driven, no hardcoded values
- **Fail-fast**: Detect issues immediately with clear error messages
- **E2E Testing > Unit Tests**: Test real attack scenarios, not isolated functions
- **Kernel-native**: Leverage eBPF/XDP for performance
- **ML-adaptive**: Continuous retraining with synthetic + real data
- **Open Source**: No vendor lock-in, full control

---

## 📊 5-Phase Evolution

### **Phase 1: IDS/IPS Foundation** 🟢 IN PROGRESS
**Timeline:** Q4 2024 (2-3 months)  
**Goal:** Detect and respond to L3/L4 attacks

#### Components
- ✅ eBPF sniffer (packet capture with kernel context)
- 🔄 ML detector tricapa (3-level classification: attack/ddos/ransomware)
- ⏳ Firewall integration (iptables/nftables dynamic blocking)
- ⏳ Retraining pipeline (synthetic data generation + async retraining)
- ⏳ ZMQ messaging pipeline (protobuf-based event streaming)

#### ML Models (Level 1)
- **Attack Detector**: Random Forest, 23 features
    - Normal traffic vs Attack classification
    - Threshold: 0.65
    - Inference: <10ms

#### E2E Tests
```bash
# DDoS simulation
hping3 --flood --rand-source <target>

# Port scanning
nmap -sS -p- <target>

# SYN flood
scapy: send(IP(src=random_ip())/TCP(dport=80, flags="S"), loop=1)

# Validation: IP blocked in firewall within <1 second
```

#### Success Metrics
- [ ] 10K packets/sec processing without drops
- [ ] <100ms end-to-end latency (capture → decision → firewall)
- [ ] <5% false positive rate on test dataset
- [ ] Pipeline handles node failures gracefully

#### Deliverables
- Functional IDS/IPS detecting DDoS, port scans, SYN floods
- Documentation: architecture, deployment, testing
- Demo video: simulated attack → detection → mitigation

---

### **Phase 2: Advanced DDoS Protection** 🟡 PLANNED
**Timeline:** Q1 2025 (1-2 months)  
**Goal:** Kernel-level mitigation with XDP

#### New Capabilities
- **Rate limiting in XDP** (per-IP, per-subnet)
    - Drop packets in kernel before reaching user-space
    - Configurable thresholds via JSON

- **SYN cookie enforcement**
    - Protect against SYN floods without state exhaustion

- **Connection tracking**
    - Track TCP handshakes, detect half-open connections

- **Geo-blocking**
    - GeoIP lookups in eBPF
    - Block/allow by country code

- **Volumetric attack mitigation**
    - Aggregate traffic statistics
    - Detect and drop amplification attacks (DNS, NTP, etc.)

#### ML Models (Level 2 - DDoS Specialized)
- **DDoS Detector**: Random Forest, 82 features
    - Traffic pattern analysis
    - Threshold: 0.70
    - Inference: <10ms

#### Why DDoS First?
- Most common attack vector
- eBPF/XDP is PERFECT for this (line-rate filtering)
- Doesn't require Layer 7 inspection (simpler)
- Immediate ROI (protects availability)

#### E2E Tests
```python
# Scapy-based volumetric attack (IP rotation)
from scapy.all import *
import random

for i in range(100000):
    src = f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
    send(IP(src=src, dst=target)/TCP(dport=80, flags="S"))

# Validation: Drop rate >99% in kernel, CPU usage <30%
```

#### Success Metrics
- [ ] 1M packets/sec sustained without saturation
- [ ] <1μs latency in XDP fast path
- [ ] Blocks 99%+ of volumetric attacks before user-space
- [ ] Adaptive thresholds based on traffic baselines

#### Deliverables
- XDP program with configurable rate limits
- Geo-blocking database integration
- Dashboard: real-time DDoS metrics
- Paper/blog post: "Kernel-Native DDoS Mitigation"

---

### **Phase 3: Layer 7 Observability** 🟡 PLANNED
**Timeline:** Q2 2025 (2-3 months)  
**Goal:** HTTP visibility without filtering yet

#### New Capabilities
- **HTTP parsing in eBPF** (limited by verifier)
    - Request method, path, status code
    - User-Agent, Referer headers

- **Request metrics**
    - Response time per endpoint
    - Request rate per client IP
    - HTTP error rate (4xx, 5xx)

- **Application profiling**
    - Identify slow endpoints
    - Detect abnormal traffic patterns

- **HTTPS handling** (requires SSL termination upstream)
```
  Nginx/Envoy (SSL offload)
       ↓
  eBPF kprobes (plaintext HTTP)
       ↓
  ML detector (L7 metrics)
```

#### ML Models (Level 3 - HTTP Anomaly)
- **Internal Traffic Analyzer**: Random Forest, 4 features
    - Baseline establishment per endpoint
    - Anomaly scoring
    - Threshold: 0.80

- **Web Traffic Analyzer**: Random Forest, 4 features
    - HTTP-specific features
    - Bot detection (preliminary)

#### Technical Challenges
- eBPF verifier limits complexity (no unlimited loops)
- Stack limited to 512 bytes
- HTTP parsing must be minimal
- HTTPS requires upstream SSL termination

#### E2E Tests
```bash
# HTTP flood (slowloris)
slowhttptest -c 1000 -H -g -o slow_http_test <target>

# Path traversal attempts
curl http://target/../../../etc/passwd
curl http://target/admin/../../sensitive

# User-Agent anomalies
curl -A "malicious-bot-v1.0" http://target/

# Validation: All HTTP metrics captured, anomaly scores generated
```

#### Success Metrics
- [ ] Captures 95%+ of HTTP requests in high-traffic scenarios
- [ ] <5ms added latency for metrics collection
- [ ] Dashboard shows per-endpoint performance
- [ ] Anomaly detection identifies 80%+ of suspicious patterns

#### Deliverables
- eBPF HTTP parser (optimized for verifier)
- Integration with Nginx/Envoy
- Grafana dashboard for L7 metrics
- Documentation: SSL termination setup

---

### **Phase 4: Basic WAF** 🔵 FUTURE
**Timeline:** Q3 2025 (3-4 months)  
**Goal:** HTTP filtering with signature-based + ML hybrid

#### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        Edge/LB (Nginx/Envoy)                │
│                     SSL Termination + L7 LB                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │   WAF Module     │
                    │ (ModSecurity or  │
                    │  custom eBPF+    │
                    │  user-space)     │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐        ┌──────▼──────┐      ┌─────▼─────┐
   │  Fast   │        │   Slow      │      │    ML     │
   │  Path   │        │   Path      │      │ Inference │
   │ (eBPF)  │        │ (Userspace) │      │   API     │
   └─────────┘        └─────────────┘      └───────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼──────────┐
                    │  Decision Engine  │
                    │  (block/challenge/│
                    │      allow)       │
                    └───────────────────┘
```

#### New Capabilities

**Fast Path (eBPF)**
- IP blacklist/whitelist (immediate drop)
- Rate limiting per endpoint
- Method filtering (block uncommon methods)
- Basic signature matching (simple patterns)

**Slow Path (User-space)**
- Deep payload inspection
- SQL injection detection (signature-based)
- XSS detection (signature-based)
- Path traversal prevention
- HTTP smuggling detection
- OWASP CRS (Core Rule Set) integration

**ML Path**
- Payload analysis (NLP-based model)
- Anomaly scoring combining L3/L4/L7 features
- Bot detection (behavioral analysis)
- Zero-day pattern detection

#### WAF-Specific ML Models
```
Input: HTTP request features
  - Method, URI, headers
  - Payload characteristics
  - Session context
  - Historical behavior

Models:
  1. Payload Classifier (NLP): SQL/XSS/Normal
  2. Bot Detector (LSTM): Human/Bot score
  3. Anomaly Scorer (RF): Combined risk score

Output: Action + Confidence
  - block (score > 0.95)
  - challenge (0.70 < score ≤ 0.95)
  - allow + log (score ≤ 0.70)
```

#### Integration with Existing Pipeline
```
sniffer-ebpf (L2-L4)  ──┐
                        │
WAF logs (L7)  ─────────┼──→ ZMQ Bus ──→ ML Detector Tricapa
                        │
Firewall events ────────┘

Correlation: flow_id links L2-L4 and L7 events
```

#### E2E Tests
```bash
# SQL injection (sqlmap)
sqlmap -u "http://target/login?id=1" --dbs

# XSS payloads
curl "http://target/search?q=<script>alert(1)</script>"

# Path traversal
curl "http://target/download?file=../../../../etc/passwd"

# LFI (Local File Inclusion)
curl "http://target/page?file=../../config.php"

# Validation: Blocked with <5ms latency, logged with reason
```

#### Success Metrics
- [ ] Detects 80%+ of OWASP Top 10 (3-4 categories initially)
- [ ] <5ms P95 latency for WAF decisions
- [ ] <1% false positive rate on legitimate traffic
- [ ] 95%+ detection rate on known attack patterns

#### Deliverables
- ModSecurity integration OR custom WAF module
- WAF-specific ML models (payload analysis, bot detection)
- Rule management system (dynamic updates via etcd)
- Dashboard: WAF events, block rates, top attackers
- Documentation: WAF deployment, rule customization

---

### **Phase 5: Advanced WAF + ML** 🔵 FUTURE
**Timeline:** Q4 2025+ (6+ months)  
**Goal:** Production-grade WAF with advanced ML

#### New Capabilities

**OWASP Top 10 Complete Coverage**
- Injection (SQL, NoSQL, LDAP, OS)
- Broken Authentication
- Sensitive Data Exposure
- XML External Entities (XXE)
- Broken Access Control
- Security Misconfiguration
- Cross-Site Scripting (XSS)
- Insecure Deserialization
- Using Components with Known Vulnerabilities
- Insufficient Logging & Monitoring

**Advanced Bot Detection**
- Behavioral fingerprinting
- Device fingerprinting (canvas, WebGL, fonts)
- Mouse movement analysis (requires JS injection)
- Session replay anomaly detection
- Credential stuffing protection

**API Protection**
- OpenAPI/Swagger schema validation
- Rate limiting per API key
- JWT validation and anomaly detection
- GraphQL query complexity analysis

**Zero-Day Detection**
- Unsupervised learning for novel patterns
- Ensemble models (RF + XGBoost + Neural Net)
- Explainability (SHAP values for decisions)

**Virtual Patching**
- Deploy rules for known CVEs without app changes
- Automatic rule generation from threat intel feeds

#### Advanced ML Models

**Payload Analysis (NLP)**
```python
Model: BERT fine-tuned on attack payloads
Input: HTTP payload (text)
Output: [SQL_injection: 0.02, XSS: 0.95, Normal: 0.03]
Inference: <50ms (batched)
```

**Bot Detection (Sequence Learning)**
```python
Model: LSTM on request sequences
Input: Last 10 requests from same IP/session
Output: Bot probability [0-1]
Features: timing, paths, user-agents, headers
Inference: <20ms
```

**API Schema Validator**
```python
Model: Rule-based + ML hybrid
Input: API request + OpenAPI schema
Output: Valid/Invalid + confidence
Validates: types, required fields, ranges
```

#### Distributed WAF Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                      Control Plane                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │   etcd      │  │  Rule Engine │  │  ML Orchestrator│    │
│  │ (config)    │  │  (versioned) │  │  (models)       │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└────────────────────────────┬─────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────────┐    ┌──────▼────────┐    ┌─────▼──────────┐
   │  WAF Agent  │    │  WAF Agent    │    │   WAF Agent    │
   │   (Node 1)  │    │   (Node 2)    │    │   (Node 3)     │
   │             │    │               │    │                │
   │  - Fast ML  │    │  - Fast ML    │    │  - Fast ML     │
   │  - Sigs     │    │  - Sigs       │    │  - Sigs        │
   │  - Logs→ZMQ │    │  - Logs→ZMQ   │    │  - Logs→ZMQ    │
   └─────────────┘    └───────────────┘    └────────────────┘
```

**Synchronization:**
- Rules distributed via etcd (TTL + version)
- Events streamed via ZMQ in real-time
- ML models updated via artifact registry
- Consistent hashing for session stickiness

#### E2E Tests
```bash
# Bot simulation (Selenium)
selenium_bot.py --target http://target/ --requests 1000

# Zero-day exploitation attempts
# (custom scripts simulating unknown attack patterns)

# API fuzzing
ffuf -u http://target/api/users/FUZZ -w wordlist.txt

# Credential stuffing
hydra -L users.txt -P passwords.txt http-post-form "/login:user=^USER^&pass=^PASS^:F=failed"

# Validation: ML detects patterns not in signatures, blocks with confidence scores
```

#### Success Metrics
- [ ] 95%+ detection rate on OWASP Top 10
- [ ] <10ms P99 latency (fast path <1ms, slow path <50ms)
- [ ] <0.5% false positive rate
- [ ] Detects 60%+ of zero-day patterns (based on anomaly)
- [ ] Blocks 99%+ of bot traffic while allowing legitimate bots

#### Deliverables
- Production-ready WAF (comparable to Cloudflare/Imperva)
- Complete ML pipeline (train, deploy, monitor, retrain)
- Explainability dashboard (why was request blocked?)
- API for custom integrations
- Helm charts for Kubernetes deployment
- Whitepaper: "Kernel-Native ML-Powered WAF"

---

## 🔬 Testing Strategy

### Unit Tests
- Critical functions only (parsing, encoding, validation)
- Use mocks sparingly

### Integration Tests
- Component pairs (sniffer ↔ detector, detector ↔ firewall)
- ZMQ message passing
- Config loading and validation

### E2E Tests (Primary Focus)
```bash
tests/
├── e2e/
│   ├── test_ddos_attack.sh          # Simulated DDoS
│   ├── test_port_scan.sh            # nmap scan
│   ├── test_sql_injection.sh        # SQLi payloads
│   ├── test_ip_rotation.sh          # Dynamic source IPs
│   ├── test_bot_detection.sh        # Selenium bot
│   └── test_zero_day.sh             # Novel patterns
├── datasets/
│   └── attack_samples/              # PCAP files, payloads
└── scripts/
    ├── generate_synthetic_ddos.py   # Controlled attack
    └── validate_pipeline.py         # End-to-end validation
```

**DO NOT PUBLISH attack scripts to repo** (responsible disclosure)
- Keep scripts in private submodule or local
- Document usage for internal testing only

### Performance Tests
- Throughput: packets/sec, requests/sec
- Latency: P50, P95, P99
- Resource usage: CPU, memory, network
- Scalability: horizontal scaling tests

---

## 🎯 Milestones & Checkpoints

| Phase | Milestone | ETA | Status |
|-------|-----------|-----|--------|
| 1 | IDS/IPS functional | Dec 2024 | 🔄 In Progress |
| 1 | Firewall integration | Jan 2025 | ⏳ Pending |
| 1 | Retraining pipeline | Jan 2025 | ⏳ Pending |
| 2 | XDP DDoS mitigation | Mar 2025 | ⏳ Planned |
| 2 | Geo-blocking | Mar 2025 | ⏳ Planned |
| 3 | HTTP observability | May 2025 | ⏳ Planned |
| 3 | SSL termination setup | Jun 2025 | ⏳ Planned |
| 4 | Basic WAF (signatures) | Aug 2025 | ⏳ Planned |
| 4 | WAF ML models | Sep 2025 | ⏳ Planned |
| 5 | Advanced WAF | Q4 2025+ | ⏳ Future |

---

## 🚀 Next Steps (This Week)

1. ✅ Complete ML detector core logic
2. ✅ Integrate with firewall (iptables/nftables)
3. ✅ Write first E2E test (DDoS simulation)
4. 📝 Document architecture and deployment
5. 🎥 Record demo video

---

# ROADMAP: Ransomware Detection System

## FASE 1: MVP ✅ (EN CURSO)
- [x] Componentes base (FlowTracker, DNSAnalyzer, etc)
- [x] Tests unitarios
- [ ] RansomwareFeatureProcessor
- [ ] Integración sniffer
- [ ] Test end-to-end
- [ ] Modelo entrenado

**Limitaciones aceptadas:**
- Buffer fijo 96 bytes
- DNS básico (pseudo-domain)
- Detección ~60-70%

## FASE 2: Payload Adaptativo 🎯 (KILLER FEATURE)
**Esperando estabilización de Fase 1**

### Objetivo:
- Detección 95-99%
- Configuración por puerto
- Threat profiles

### Tareas:
- [ ] Diseño de threat_profiles.json
- [ ] Modificar SimpleEvent (payload variable)
- [ ] BPF map para configuración dinámica
- [ ] PayloadConfigManager
- [ ] Tests adversarial evasion
- [ ] Documentación operacional

### Motivación:
"Los atacantes no siguen reglas fijas. El payload malicioso
puede estar en byte 200-296, no solo en los primeros 96 bytes.
Necesitamos flexibilidad para adaptarnos a cada familia de ransomware."

## FASE 3: Sistema Adaptativo 🚀 (FUTURO)
- Auto-incremento basado en ML
- Threat intel integration
- Forensic mode
- 99.999% detección

## 💡 Key Differentiators vs Cloudflare/Imperva

| Feature | Cloudflare/Imperva | This Project |
|---------|-------------------|--------------|
| **Cost** | $$$ (expensive) | Free (open source) |
| **Deployment** | Cloud-only | Self-hosted, on-prem |
| **Performance** | Edge network | Kernel-native (eBPF/XDP) |
| **Customization** | Limited | Full control (source code) |
| **ML** | Black box | Transparent, retrainable |
| **Vendor Lock-in** | Yes | No |
| **Data Privacy** | Traffic goes to vendor | Stays on your infrastructure |

**Target Market:**
- Enterprises wanting full control
- On-premises deployments
- Privacy-sensitive industries (finance, healthcare, government)
- Teams with ML/security expertise

---

## 📚 References & Inspiration

- **Cloudflare**: Blog posts on DDoS mitigation, WAF architecture
- **Fastly**: Edge computing, VCL logic
- **ModSecurity**: OWASP CRS, rule engine design
- **Cilium**: eBPF networking, observability
- **Falco**: Runtime security with eBPF
- **Suricata**: IDS/IPS architecture
- **Zeek (Bro)**: Network traffic analysis

---

## 📞 Community & Contributions

- **GitHub Issues**: Bug reports, feature requests
- **Discussions**: Architecture, design decisions
- **Pull Requests**: Always welcome (with tests)
- **Security Issues**: Responsible disclosure via private email

---

**Last Updated:** October 16, 2025  
**Next Review:** January 2025 (Post Phase 1)
