# 🎯 PROMPT DE CONTINUACIÓN PARA CLAUDE

```markdown
# ML DEFENDER - CONTINUATION PROMPT
## Context for Future Claude Session

Hi Claude! You're continuing work on **ML Defender**, an open-source cybersecurity system 
that combines eBPF/XDP packet capture with embedded ML for ransomware/DDoS detection.

Your human partner is **Alonso**, a software engineer and ML architect who follows 
"Via Appia Quality" - building systems designed to last decades. He values:
- Scientific honesty and transparency
- No hardcoded values - "JSON is the law" (single source of truth)
- Explicit TODOs rather than hidden technical debt
- Verification over assumptions

---

## 🏗️ PROJECT STATE (as of Nov 18, 2025)

### **RECENTLY COMPLETED: Phase 1, Day 4** ✅

Successfully integrated 4 embedded C++20 RandomForest detectors into the sniffer:
- **DDoS Detector** (10 features)
- **Ransomware Detector** (10 features)  
- **Traffic Classifier** (10 features)
- **Internal Anomaly Detector** (10 features)

**Performance achieved**: 16.33 μs average detection time (6x better than 100μs target)

**Test results** (267 packets, 150 seconds):
```
🛡️  ML Defender Embedded Detectors:
DDoS attacks detected: 0
Ransomware attacks detected: 0
Suspicious traffic detected: 264
Internal anomalies detected: 264
Avg ML detection time: 16.33 μs
```

**Architecture**: Thread-local, zero-lock, embedded C++20

**Files modified**:
- `/vagrant/sniffer/include/ring_consumer.hpp` - Added detector declarations
- `/vagrant/sniffer/src/userspace/ring_consumer.cpp` - Integrated inference (~350 LOC)
- `/vagrant/sniffer/CMakeLists.txt` - Added ml-detector includes and sources

**Key integration points**:
```cpp
// Thread-local detectors (line ~37)
thread_local ml_defender::DDoSDetector RingBufferConsumer::ddos_detector_;
thread_local ml_defender::RansomwareDetector RingBufferConsumer::ransomware_detector_;
thread_local ml_defender::TrafficDetector RingBufferConsumer::traffic_detector_;
thread_local ml_defender::InternalDetector RingBufferConsumer::internal_detector_;

// Inference call in populate_protobuf_event() (line ~645)
const_cast<RingBufferConsumer*>(this)->run_ml_detection(proto_event);

// Feature extractors + run_ml_detection() (lines ~1207-1355)
```

---

## ⚠️ CRITICAL ISSUE: Hardcoded Thresholds

**PROBLEM**: Detection thresholds are hardcoded in `run_ml_detection()`:

```cpp
// TODO(Phase1-Day4-CRITICAL): Load thresholds from model JSON metadata
if (ddos_pred.is_ddos(0.7f)) {  // ❌ HARDCODED
if (ransomware_pred.is_ransomware(0.75f)) {  // ❌ HARDCODED  
if (traffic_pred.probability >= 0.7f) {  // ❌ HARDCODED
if (internal_pred.is_suspicious(0.00000000065f)) {  // ❌ HARDCODED
```

**PREVIOUS ISSUE**: jsoncpp library converted float thresholds incorrectly
(e.g., 0.75 became astronomical value). Need careful float parsing with validation.

**MODEL JSON LOCATIONS**:
- `/vagrant/ml-detector/models/production/ddos_binary_detector.json`
- `/vagrant/ml-detector/models/production/ransomware_detector_embedded.json`
- `/vagrant/ml-detector/models/production/traffic_detector_embedded.json`
- `/vagrant/ml-detector/models/production/internal_detector_embedded.json`

---

## 🎯 IMMEDIATE TASKS (Phase 1, Day 5)

### **TASK 1: Fix Hardcoded Thresholds** (Priority: CRITICAL)

**Steps**:
1. Examine JSON structure of model files to find threshold field
2. Create `ModelConfig` class to load thresholds safely
3. Implement float parsing with validation (range: [0.0, 1.0])
4. Replace hardcoded values in `run_ml_detection()`
5. Add fallback to defaults if JSON read fails
6. Test with real thresholds from JSON

**Validation**: Compile, run 60s capture, verify thresholds are loaded correctly

### **TASK 2: 8-Hour Stress Test** (Priority: HIGH)

**Design requirements from Alonso**:
- Duration: Exactly 8 hours (28,800 seconds)
- Components: Sniffer + ML-Detector (both in verbose mode)
- Traffic: Synthetic (not real ransomware yet), sustained load
- Rate: 50-100 packets/second sustained
- Monitoring: CPU, RAM, latency, detection counts
- Logging: Compressed logs for analysis
- Goal: Validate stability, find memory leaks, measure real-world performance

**Expected deliverables**:
- Stress test script (bash)
- Traffic generator configuration
- Monitoring setup (resource usage)
- Log compression and collection procedure
- Analysis report template

---

## 📂 PROJECT STRUCTURE

```
/vagrant/
├── sniffer/                    # eBPF/XDP packet capture
│   ├── src/userspace/
│   │   └── ring_consumer.cpp   # Main integration point
│   ├── include/
│   │   └── ring_consumer.hpp
│   └── build/
│       └── sniffer              # Binary
│
├── ml-detector/                # ML inference engine
│   ├── include/ml_defender/
│   │   ├── ddos_detector.hpp
│   │   ├── ransomware_detector.hpp
│   │   ├── traffic_detector.hpp
│   │   ├── internal_detector.hpp
│   │   └── *_trees_inline.hpp  # Decision trees
│   ├── src/
│   │   ├── ddos_detector.cpp
│   │   ├── ransomware_detector.cpp
│   │   ├── traffic_detector.cpp
│   │   └── internal_detector.cpp
│   ├── models/production/
│   │   └── *.json              # Model configs
│   └── build/
│       └── ml-detector          # Binary
│
└── protobuf/
    └── network_security.proto   # Shared schema
```

**Data flow**:
```
eBPF → Sniffer (ring_consumer) → ML Detection (4 detectors) → 
ZMQ → ML-Detector → Firewall Agent
```

---

## 🔧 TECHNICAL CONTEXT

### **Compilation**:
```bash
cd /vagrant/sniffer
make clean && make -j6
```

### **Execution** (requires sudo for eBPF):
```bash
cd /vagrant/sniffer/build
sudo timeout 60s ./sniffer -c config/sniffer.json
```

### **Current performance baseline**:
- Processing time: 52.79 μs total
- ML detection: 16.33 μs (4 detectors)
- Events/sec: ~2-3 pps (light load)

### **Key design principles**:
- Thread-local storage (zero locks)
- Embedded models (no file I/O in hot path)
- <100μs latency requirement
- Via Appia Quality (decades-long design)

---

## 🚀 ROADMAP TO RELEASE 1.0

### **Current state: ~80% complete**

**Remaining work**:

1. ✅ **Phase 1 Day 5** (IMMEDIATE):
    - Fix hardcoded thresholds ← YOU ARE HERE
    - 8-hour stress test
    - Validate stability

2. **Phase 1 Day 6-7**:
    - etcd watcher integration (encryption, compression, runtime config)
    - Final calibration and tuning

3. **Phase 2**:
    - Firewall ACL Agent (enforcement)
    - RAG system (llama.cpp + RAG-Shield model)
    - Autonomous model evolution

4. **Phase 3**:
    - Scientific papers
    - Documentation
    - Public release

**RELEASE 1.0 milestone**: When sniffer, ml-detector, firewall-agent, and RAG
are complete with etcd integration. Current estimate: 80%+ done after stress test.

---

## 🤝 WORKING WITH ALONSO

**Communication style**:
- Direct and technical
- Appreciates verification over assumptions
- Will point out if something is wrong (sees it as collaboration, not criticism)
- Values token efficiency (monitors usage carefully)
- Works early hours (often 6-7 AM)

**Red flags to avoid**:
- Hardcoding values (always use config/JSON)
- Assuming things work without testing
- Over-explaining obvious things
- Not providing concrete implementation

**Green flags**:
- Asking for verification ("Can you show me X?")
- Providing TODOs with context
- Suggesting validation steps
- Offering alternatives with tradeoffs

---

## 📝 NEXT SESSION CHECKLIST

When you start, immediately:

1. ✅ Greet Alonso briefly (he values efficiency)
2. ✅ Confirm you have this context
3. ✅ Ask him to show you ONE model JSON file structure
4. ✅ Design threshold loading solution
5. ✅ Implement, test, validate
6. ✅ Design 8-hour stress test
7. ✅ Get his approval before he launches it

**Critical files to request**:
```bash
cat /vagrant/ml-detector/models/production/ddos_binary_detector.json
grep -r "threshold" /vagrant/ml-detector/models/production/
```

---

## 🎯 SUCCESS CRITERIA

**Thresholds from JSON**:
- ✅ No hardcoded values remain
- ✅ Safe float parsing (validate [0.0, 1.0])
- ✅ Fallback defaults if JSON fails
- ✅ Compiles without warnings
- ✅ Real-world test shows correct thresholds loaded

**8-Hour Stress Test**:
- ✅ Runs exactly 8 hours without crashes
- ✅ No memory leaks detected
- ✅ Latency remains <50μs avg
- ✅ Logs compressed and ready for analysis
- ✅ Resource usage stable (CPU, RAM)

---

## 💡 IMPORTANT REMINDERS

1. **"JSON is the law"** - Single source of truth for configuration
2. **Via Appia Quality** - Design for decades, not days
3. **Verification > Assumptions** - Always ask to see files/output
4. **TODOs are features** - Explicit is better than implicit
5. **Performance matters** - Every microsecond counts (protecting businesses)

---

## 🏆 THE VISION

ML Defender aims to protect small businesses and healthcare organizations from
cyberattacks (ransomware, DDoS). Alonso was motivated by a friend's business being
devastated by ransomware. Every microsecond of detection latency matters when
protecting someone's livelihood or patient data.

**You're helping build infrastructure that protects the vulnerable.**

---

Good luck, future Claude! Alonso is an excellent engineer to work with.
The project is at a critical juncture - stable foundation, moving toward production.

🚀 Let's finish Phase 1 strong!
```

---

## ✅ CHECKLIST PARA ALONSO

Cuando retomes con el próximo Claude:

**Comparte inmediatamente**:
1. ✅ Este prompt completo
2. ✅ Un JSON de modelo: `cat /vagrant/ml-detector/models/production/ddos_binary_detector.json`
3. ✅ Confirma que quieres empezar con thresholds

**Valida que Claude entienda**:
- ✅ El problema del hardcoding
- ✅ La arquitectura thread-local
- ✅ El objetivo del stress test
- ✅ El roadmap a RELEASE 1.0

---

¿Este prompt captura todo lo necesario para la continuación? ¿Algo crítico que falte? 🚀