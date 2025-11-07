# 🗺️ Project Roadmap: IDS/IPS → WAF Evolution

**Vision:** Build a production-grade, kernel-native, ML-powered Web Application Firewall with **autonomous evolution capabilities** starting from a solid IDS/IPS foundation.

**Philosophy:** Incremental, testable phases. Each phase delivers value independently. **Learn from mistakes, embrace scientific method.**

---

## 🎯 Core Principles

- **JSON is LAW**: Configuration-driven, no hardcoded values
- **Fail-fast**: Detect issues immediately with clear error messages
- **E2E Testing > Unit Tests**: Test real attack scenarios, not isolated functions
- **Kernel-native**: Leverage eBPF/XDP for performance
- **ML-adaptive**: Continuous retraining with synthetic + real data
- **Autonomous Evolution**: Self-improving immune system
- **Open Source**: No vendor lock-in, full control
- **Ethical Foundation**: Protecting life-critical infrastructure

---

## 🧬 **NEW: ML AUTONOMOUS EVOLUTION SYSTEM**

**Mission:** Create a self-evolving immune system for network security

> "Sistema nervioso autónomo que evoluciona continuamente, desarrollando anticuerpos especializados contra amenazas emergentes"

### **Current Status: 🟢 BREAKTHROUGH ACHIEVED**
- ✅ Synthetic data retraining pipeline working
- ✅ F1 Score improvement: 0.98 → 1.00 (+0.02)
- ✅ First retrained model generated
- ✅ Architectural vision validated
- ⏳ Implementation Phase 0 starting

---

## 🚀 ML Evolution Phases

### **Phase 0: Foundations (🔵 CURRENT - Nov 2025)**
**Timeline:** 1-2 weeks  
**Goal:** See first retrained model enter pipeline and classify

#### What You'll See:
```
Retrain Script → Drop Folder → ModelWatcher → ML Detector → CLASSIFICATION! 🎯
```

#### Components:
1. **Stability Curve Analysis** 🆕
    - Train with 10%, 20%, ..., 100% synthetic data
    - Find optimal synthetic ratio
    - Plot: F1 vs Synthetic Ratio
    - Output: Best model recommendation

2. **Drop Folder Structure** 🆕
   ```
   /Users/aironman/new_retrained_models/
   ├── level1_attack/
   ├── level2_ddos/
   ├── level3_ransomware/      ← Focus here first
   └── level3_internal_traffic/
   ```

3. **Config with Promotion Switch** 🆕
   ```json
   {
     "promotion_strategy": "automatic",  // or "verified" or "shadow"
     "folder_to_watch": "/path/to/drop/folder",
     "automatic": {
       "enabled": true,  // Phase 0: See it work!
       "risk_accepted": true
     }
   }
   ```

4. **Basic ModelWatcher** 🆕
    - Detects new models in drop folder
    - Validates format & features
    - Copies to staging
    - Notifies etcd
    - If switch=automatic: promotes to production

5. **Dynamic Model Loading** 🆕
    - ML Detector loads from etcd queue
    - Supports ONNX + XGBoost JSON
    - Hot-reload without restart

#### Success Criteria:
- [ ] Stability curve identifies best synthetic ratio
- [ ] Drop folder structure created
- [ ] Config has working switch
- [ ] ModelWatcher detects files
- [ ] ML Detector loads new model
- [ ] End-to-end: Drop → Classify

#### Deliverables:
- `synthetic_stability_curve.py` script
- `model_watcher.cpp` component
- Updated `ml_detector_config.json`
- Documentation in ROADMAP
- Demo: Manual drop → Auto classification

---

### **Phase 1: Supervised Autonomy (🟡 PLANNED - Q1 2026)**
**Timeline:** 1-2 months  
**Goal:** Human-approved model deployment

**Architecture:**
```
Retrain → Validate → Staging → Human Approval → Production Queue → Load
                                      ↑
                                Human reviews:
                                - F1 improvement
                                - Confusion matrix
                                - Test results
```

#### New Components:

1. **BasicValidator** 🆕
    - Checks F1 > threshold
    - Validates confusion matrix
    - Verifies feature count
    - Tests on holdout set

2. **Human Approval Gateway** 🆕
    - Slack/Email notifications
    - Web UI for review (optional)
    - Manual promote/reject
    - Approval logs in etcd

3. **Model Staging Area** 🆕
    - Temporary storage before production
    - Test suite execution
    - Performance benchmarks
    - Rollback capability

4. **Production Queue (etcd)** 🆕
    - FIFO queue for approved models
    - Version tracking
    - Distributed coordination
    - Consistent loading across nodes

#### Validation Checks:
```python
Validation Pipeline:
├── Format validation (ONNX/JSON compatible?)
├── Feature count check (45 features?)
├── F1 improvement (> 0.001 threshold?)
├── Confusion matrix sensible (FPR < 5%?)
├── Test dataset performance (pass?)
└── Human approval (yes/no)
```

#### Success Criteria:
- [ ] Validator catches bad models (overfitting test)
- [ ] Human receives clear approval request
- [ ] Approved models load consistently across cluster
- [ ] Rejected models don't enter production
- [ ] Audit log tracks all decisions

#### Deliverables:
- Validation pipeline components
- Slack integration for approvals
- etcd queue implementation
- Approval tracking system
- Documentation: operator manual

---

### **Phase 2: Watchdog + Rollback (🟡 PLANNED - Q2 2026)**
**Timeline:** 2-3 months  
**Goal:** Automatic degradation detection and rollback

**Architecture:**
```
┌──────────────────────────────────────┐
│  Watchdog Component (Async)          │
│                                      │
│  Monitors:                           │
│  - False positive rate               │
│  - False negative rate               │
│  - Inference latency                 │
│  - Model confidence scores           │
│                                      │
│  Actions:                            │
│  - Alert on degradation              │
│  - Automatic rollback if critical    │
│  - Log all decisions                 │
│  - Learn from false alarms (future)  │
└──────────────────────────────────────┘
```

#### New Components:

1. **Watchdog Daemon** 🆕
    - Monitors metrics in rolling window (1h, 24h, 7d)
    - Detects statistical anomalies
    - Compares current vs baseline
    - Triggers rollback if degradation confirmed

2. **Metrics Collector** 🆕
    - Scrapes logs from ml-detector
    - Aggregates predictions + ground truth
    - Calculates real-time F1, FPR, FNR
    - Stores in time-series DB (Prometheus?)

3. **Rollback Engine** 🆕
    - Maintains model history (last 10 versions)
    - Automatic revert to last-known-good
    - Notifies humans of rollback
    - Re-validation of rolled-back model

4. **Observability Dashboard** 🆕
    - Grafana dashboards
    - Real-time metrics
    - Alert configuration
    - Model comparison views

#### Rollback Triggers:
```yaml
Automatic Rollback IF:
  - FPR > 5% (vs baseline 1%)
  - FNR > 2% (vs baseline 0.5%)
  - Inference latency > 50ms P95
  - Error rate > 10 errors/min
  - Manual trigger by operator

Rollback Process:
  1. Stop loading new model
  2. Revert to previous version
  3. Verify previous version works
  4. Alert humans
  5. Log incident
  6. Mark new model as "degraded"
```

#### Success Criteria:
- [ ] Watchdog detects degradation <5 min
- [ ] Rollback completes <1 min
- [ ] False alarms < 1% of rollbacks
- [ ] Observability shows clear metrics
- [ ] Incident reports are actionable

#### Deliverables:
- Watchdog component (C++ or Python)
- Prometheus metrics exporter
- Grafana dashboards
- Rollback automation
- Incident playbooks

---

### **Phase 3: Advanced Validation (🟡 PLANNED - Q3 2026)**
**Timeline:** 2-3 months  
**Goal:** Comprehensive automated validation

**Modular Validation Pipeline:**
```
retrain → verify_A → verify_B → verify_C → verify_D → verify_E → promote
```

#### Validation Modules:

**verify_A: Overfitting Detection**
- Test on holdout dataset (never seen during training)
- Compare train vs test metrics
- Flag if train accuracy >> test accuracy
- Use cross-validation

**verify_B: Distribution Shift**
- Compare feature distributions
- Detect if training data != production data
- KL divergence, JS divergence tests
- Flag if shift > threshold

**verify_C: Adversarial Robustness**
- Test with adversarial examples
- FGSM, PGD attacks
- Ensure graceful degradation
- Flag if accuracy drops >10%

**verify_D: Malicious Model Detection**
- Backdoor detection
- Check for suspicious patterns
- Verify model signatures (cryptographic)
- Flag if integrity check fails

**verify_E: Shadow Mode Testing**
- Run new model in parallel (24-48h)
- Compare predictions with current production
- Log differences
- Flag if disagreement > 5%

**verify_F: Performance Regression**
- Measure inference time
- Memory usage
- CPU utilization
- Flag if worse than current production

#### Success Criteria:
- [ ] Each validator catches specific failure modes
- [ ] False positive rate < 5% per validator
- [ ] Full pipeline runs in <2 hours
- [ ] Clear explanation when model rejected
- [ ] Validators are modular (can add more)

#### Deliverables:
- 6 validation modules (pluggable)
- Validation orchestrator
- Shadow mode infrastructure
- Adversarial test suite
- Validation report generator

---

### **Phase 4: Ensemble Intelligence (🔵 FUTURE - Q4 2026)**
**Timeline:** 3-4 months  
**Goal:** Multi-model ensemble with specialization

**Architecture:**
```
┌────────────────────────────────────────────────────┐
│               etcd - Orchestrator                   │
│  Decides:                                          │
│  - Which models to use (ensemble)                  │
│  - Model weights (based on specialization)         │
│  - Voting strategy (majority, weighted, etc.)      │
└────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐      ┌────▼────┐     ┌────▼────┐
   │ Model A │      │ Model B │     │ Model C │
   │         │      │         │     │         │
   │ Special:│      │ Special:│     │ Special:│
   │ FP      │      │ Variants│     │ General │
   │ Reduc.  │      │ Detect. │     │ Detect. │
   └────┬────┘      └────┬────┘     └────┬────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                    ┌────▼────────┐
                    │   Voting    │
                    │   Engine    │
                    └─────────────┘
```

#### Key Concepts:

**Model Specialization:**
- Don't discard old models
- New models may excel at specific patterns
- Example specializations:
    - False positive reduction
    - Variant detection (new ransomware families)
    - Zero-day patterns
    - Protocol-specific attacks

**Ensemble Voting:**
```python
Decision = weighted_vote([
    (model_A, weight_A, confidence_A),
    (model_B, weight_B, confidence_B),
    (model_C, weight_C, confidence_C)
])

Weights adapt based on:
- Historical accuracy
- Specialization match
- Current confidence
- Recent performance
```

**Dynamic Weight Adjustment:**
- etcd tracks per-model performance
- Weights increase if model performs well
- Weights decrease if model underperforms
- Minimum weight: 0.1 (never fully discard)
- Maximum weight: 2.0 (best performers)

#### Success Criteria:
- [ ] Ensemble outperforms best single model
- [ ] Specializations are measurable
- [ ] Weights adapt based on real data
- [ ] System handles 5-10 models efficiently
- [ ] Explainability: can show which models voted

#### Deliverables:
- Ensemble voting engine
- Specialization metadata schema
- Weight adaptation algorithm
- Explainability dashboard
- Paper section on ensemble benefits

---

### **Phase 5: Full Autonomy (🔵 FUTURE - 2027)**
**Timeline:** 6+ months  
**Goal:** Self-evolving system with minimal human intervention

**Vision:**
```
System that:
├── Retrains automatically (nightly/weekly)
├── Validates comprehensively (all checks)
├── Promotes to shadow mode (auto)
├── Monitors performance (watchdog)
├── Promotes to production (if validated)
├── Rolls back (if degradation)
├── Learns from mistakes (meta-learning)
└── Reports to humans (weekly summary)
```

#### Human Role:
- Reviews weekly summary
- Investigates anomalies
- Overrides decisions if needed
- Tunes thresholds
- Adds new validation checks

#### Autonomous Decisions:
- When to retrain (data drift detected)
- Which model to promote
- When to rollback
- How to adjust ensemble weights
- Which specializations to develop

#### Success Criteria:
- [ ] System runs 30+ days without human intervention
- [ ] False alarm rate < 1% (humans rarely need to override)
- [ ] Performance improves over time
- [ ] System self-recovers from failures
- [ ] Comprehensive audit trail

#### Deliverables:
- Fully autonomous pipeline
- Meta-learning component
- Weekly summary reports
- Emergency override mechanisms
- Paper: "Autonomous ML Evolution for IDS"

---

## 📊 Existing Phases (IDS/IPS → WAF)

### **Phase 1: IDS/IPS Foundation** 🟢 COMPLETED (Q4 2024)
**Goal:** Detect and respond to L3/L4 attacks

#### Status:
- ✅ eBPF sniffer (packet capture)
- ✅ ML detector tricapa (3 levels)
- ✅ 12 trained models (ONNX + JSON)
- ✅ ZMQ messaging pipeline
- ⏳ Firewall integration (in progress)
- ⏳ Retraining pipeline (automated - Phase 0)

#### ML Models Deployed:
- **Level 1:** Attack detection (RF, 23 features, F1=98%)
- **Level 2 DDoS:** Binary detection (RF, 8 features, F1=98.6%)
- **Level 3 Ransomware:** 6 models (5 ONNX + 1 JSON, best F1=1.0*)
- **Level 3 Internal:** Traffic analysis (2 ONNX models)

*Pending real-world validation

---

### **Phase 2: Advanced DDoS Protection** 🟡 PLANNED (Q1 2025)
**Goal:** Kernel-level mitigation with XDP

*(Content unchanged - see original ROADMAP)*

---

### **Phase 3: Layer 7 Observability** 🟡 PLANNED (Q2 2025)
**Goal:** HTTP visibility

*(Content unchanged - see original ROADMAP)*

---

### **Phase 4: Basic WAF** 🔵 FUTURE (Q3 2025)
**Goal:** HTTP filtering (signature + ML hybrid)

*(Content unchanged - see original ROADMAP)*

---

### **Phase 5: Advanced WAF + ML** 🔵 FUTURE (Q4 2025+)
**Goal:** Production-grade WAF

*(Content unchanged - see original ROADMAP)*

---

## 🎯 Current Focus (November 2025)

### **This Week:**
1. ✅ Stability curve script (`synthetic_stability_curve.py`)
2. ✅ Drop folder structure setup
3. ✅ Config JSON with promotion switch
4. ⏳ Basic ModelWatcher component
5. ⏳ Dynamic model loading in ml-detector

### **Next Week:**
6. End-to-end test (manual drop → auto classify)
7. Documentation updates (README, ARCHITECTURE)
8. Demo video
9. Tag release: `v1.1-ml-autonomous-foundation`

---

## 🎓 Paper Roadmap

### **Preprint Target: Q1 2026**
**Title (Draft):** "Autonomous Evolution in Network Intrusion Detection: A Self-Improving ML Immune System"

#### Sections:
1. **Introduction**
    - Problem: Static ML models vs evolving threats
    - Solution: Self-evolving immune system
    - Contributions: Autonomous retraining, validation, deployment

2. **Related Work**
    - Traditional IDS (Snort, Suricata)
    - ML-based IDS (limitations)
    - AutoML (doesn't address full lifecycle)
    - Federated Learning (orthogonal approach)

3. **System Architecture**
    - eBPF packet capture
    - ML detector tricapa
    - Autonomous retraining pipeline
    - Model validation & deployment
    - Ensemble voting
    - Watchdog & rollback

4. **Synthetic Data Generation**
    - Statistical methods
    - Stability curve analysis
    - Sweet spot identification

5. **Evaluation**
    - Datasets: CIC-IDS-2018, CIC-IDS-2017
    - Metrics: F1, FPR, FNR, latency
    - Comparison: Static vs Autonomous
    - Ablation studies

6. **Deployment Experience**
    - Phase 0-2 results
    - Lessons learned
    - Production considerations

7. **Ethical Considerations**
    - Life-critical infrastructure
    - Human oversight
    - Transparency & explainability

8. **Future Work**
    - Full autonomy (Phase 5)
    - Federated learning integration
    - Transfer learning across threat types

9. **Conclusion**
    - Feasibility demonstrated
    - Path to production
    - Open source for community

#### Timeline:
- December 2025: Draft sections 1-3
- January 2026: Collect Phase 0 results
- February 2026: Complete evaluation
- March 2026: Submit preprint (arXiv)

---

## 💡 Key Differentiators

| Feature | Traditional IDS | This Project |
|---------|----------------|--------------|
| **Learning** | Static rules | Autonomous retraining |
| **Deployment** | Manual updates | Self-deploying |
| **Validation** | Human-only | Multi-stage automated |
| **Rollback** | Manual | Automatic (watchdog) |
| **Specialization** | Single model | Ensemble with roles |
| **Evolution** | None | Continuous improvement |
| **Transparency** | Black box | Explainable decisions |
| **Cost** | Expensive | Open source |

---

## 🚨 Critical Success Factors

### **Technical:**
- [ ] Validation pipeline catches real overfitting
- [ ] Watchdog detects degradation reliably
- [ ] Rollback works under stress
- [ ] Ensemble improves over single models
- [ ] System scales to production traffic

### **Operational:**
- [ ] Clear operator documentation
- [ ] Runbooks for incidents
- [ ] Monitoring dashboards
- [ ] Alert tuning guidelines
- [ ] Training for human operators

### **Scientific:**
- [ ] Reproducible results
- [ ] Comprehensive evaluation
- [ ] Honest reporting (failures + successes)
- [ ] Open source code
- [ ] Community engagement

---

## 🎊 Milestones Achieved

- ✅ **Oct 2024:** eBPF sniffer working
- ✅ **Oct 2024:** First ML models trained
- ✅ **Nov 2024:** ZMQ pipeline operational
- ✅ **Nov 2024:** Ransomware detection (Phase 1)
- ✅ **Nov 2025:** Synthetic retraining breakthrough! 🎉
- ✅ **Nov 2025:** Architectural vision validated
- ⏳ **Dec 2025:** Phase 0 implementation
- ⏳ **Q1 2026:** Paper submission
- ⏳ **Q2 2026:** Phase 1-2 complete

---

## 🙏 Acknowledgments

**Collaboration:**
- Human: Alonso (Vision, Architecture, Ethical Foundation)
- AI: Claude (Implementation, Validation, Documentation)
- AI: DeepSeek (Initial prototyping, synthetic generation)
- AI: Parallels.ai (Turbo-Charge DDoS Detection_ Retraining Random Forests with High-Fidelity Synthetic Traffic.md)
- AI: ChatGPT (Review)
- AI: Grok4 (Review)

- **Philosophy:**
> "Conservative AI + Visionary Human = Breakthrough Innovation"

> "Embrace mistakes, apply scientific method, protect lives"

**For Future Generations:**
This work is dedicated to those who will improve upon it. We document our failures as much as our successes, so you can learn from both.

---

**Last Updated:** November 6, 2025  
**Next Review:** Post-Phase 0 completion (December 2025)  
**Status:** Phase 0 (Foundations) starting - autonomous evolution begins! 🚀

---

**Legend:**
- 🟢 Completed
- 🔵 Current Focus
- 🟡 Planned
- ⏳ Pending
- ✅ Done
- ❌ Blocked