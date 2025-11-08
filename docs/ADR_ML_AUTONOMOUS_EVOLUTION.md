# Architectural Decision Record: ML Autonomous Evolution System

**Date:** November 6, 2025  
**Status:** ✅ APPROVED  
**Participants:** Alonso (Human/Vision), Claude (AI/Implementation), DeepSeek (AI/Prototyping)

---

## Context

We have successfully implemented a synthetic data retraining pipeline that demonstrates measurable improvement (F1: 0.98 → 1.00). Now we need to design the system for autonomous model evolution - how new models are discovered, validated, and deployed to production.

---

## Decision Drivers

1. **Life-Critical Infrastructure:** System protects healthcare and critical systems where false negatives are unacceptable
2. **Scientific Method:** Embrace experimentation, document failures, learn iteratively
3. **Human-AI Collaboration:** Balance autonomy with human oversight
4. **Open Source Legacy:** Build for future generations to improve upon
5. **Pragmatic Implementation:** See it work first (Phase 0), then make it safe (Phase 1+)

---

## Decisions Made

### **1. Folder Watching Architecture** ✅

**Decision:** Use external drop folders (outside build) with file system watching, rather than hardcoded model paths.

**Reasoning:**
- Model names include timestamps (e.g., `ransomware_xgb_candidate_v2_20251106_095308`)
- Cannot predict names in advance
- Need to accommodate continuous retraining
- Drop folders decouple training from deployment

**Implementation:**
```
/Users/aironman/new_retrained_models/
├── level1_attack/
├── level2_ddos/
├── level3_ransomware/
└── level3_internal_traffic/

ModelWatcher component watches these folders (inotify/kqueue)
```

**Alternatives Considered:**
- ❌ Hardcoded paths: Too rigid for automated retraining
- ❌ Database storage: Adds complexity without benefit
- ❌ HTTP upload API: Unnecessary for local deployment

---

### **2. Model Specialization Over Replacement** ✅

**Decision:** Maintain multiple models in ensemble with specialization roles, rather than replacing old models with new ones.

**Reasoning:**
```
Example:
- Model A: Excellent at reducing false positives
- Model B: Excellent at detecting ransomware variants
- Model C: General-purpose detection

Don't discard any - use all three in weighted ensemble
```

**Quote (Alonso):**
> "A lo mejor hemos encontrado un modelo candidato que es muy bueno en detectar falsos positivos, mientras que otro es muy bueno en detectar variantes del malware"

**Implementation:**
- etcd stores model metadata (specialization, performance)
- Voting engine weights models based on specialization
- Dynamic weight adjustment based on real performance
- Maximum 10 models per category (retirement policy)

**Alternatives Considered:**
- ❌ Always replace: Loses specialized capabilities
- ❌ Random selection: Wastes computational resources

---

### **3. Phased Autonomy Approach** ✅

**Decision:** Implement autonomy in 5 phases, starting with "see it work" (automatic) and iterating to "make it safe" (verified).

**Phases:**
```
Phase 0 (Now):     Automatic promotion - see first model work
Phase 1 (Q1 2026): Human-approved promotion
Phase 2 (Q2 2026): Watchdog + automatic rollback
Phase 3 (Q3 2026): Advanced validation pipeline
Phase 4 (Q4 2026): Ensemble intelligence
Phase 5 (2027):    Full autonomy
```

**Reasoning (Alonso):**
> "Vamos a ir sobre seguro, tu plan orquestado es sensato, el mío, un poco disparate porque confieso que ante todo me gustaría ver aunque sea un momento como el modelo reentrenado llega al pipeline y empieza a clasificar."

**Implementation:**
- Config JSON has `promotion_strategy` switch
- Phase 0: `"automatic"` (risk accepted, log everything)
- Phase 1+: `"verified"` (human approval required)
- Always possible to override manually

**Alternatives Considered:**
- ❌ Full autonomy immediately: Too risky for life-critical systems
- ❌ Never automate: Defeats purpose of autonomous evolution

---

### **4. Modular Validation Pipeline** ✅

**Decision:** Implement validation as pluggable components (verify_A, verify_B, verify_X) that can be added/improved independently.

**Pipeline:**
```
retrain → verify_A → verify_B → verify_C → verify_D → verify_E → promote

verify_A: Overfitting detection (holdout set)
verify_B: Distribution shift detection
verify_C: Adversarial robustness
verify_D: Malicious model detection
verify_E: Shadow mode testing (24-48h)
verify_F: Performance regression
```

**Reasoning:**
- We don't know all failure modes yet
- Need to add validators as we discover issues
- Each validator is independently testable
- Modular = can swap implementations

**Implementation:**
- Each validator is a separate component
- Validator interface: `bool validate(Model model, ValidationContext ctx)`
- Validators can be enabled/disabled in config
- Failed validation → clear error message + logs

**Alternatives Considered:**
- ❌ Monolithic validation: Hard to extend/test
- ❌ No validation: Unsafe for production

---

### **5. etcd as Orchestration Brain** ✅

**Decision:** Use etcd to orchestrate ensemble voting, model coordination, and rollback decisions.

**Reasoning:**
- etcd already in architecture (crypto tokens, config sync)
- Distributed consensus (perfect for multi-node coordination)
- Watch API (real-time notifications)
- Strong consistency guarantees

**etcd Responsibilities:**
```yaml
Model Metadata:
  /ml/models/{level}/{category}/{model_id}/metadata

Ensemble Configuration:
  /ml/ensemble/{level}/{category}/config
  - Which models active
  - Model weights
  - Voting strategy

Production Queue:
  /ml/queue/{level}/{category}/pending
  - FIFO queue of approved models
  - Ensures consistent loading

Performance Tracking:
  /ml/performance/{level}/{category}/{model_id}
  - Real-time metrics
  - Historical performance
  - Specialization scores

Voting Results:
  /ml/voting/{level}/{category}/history
  - Audit trail
  - Explainability data
```

**Alternatives Considered:**
- ❌ Centralized database: Single point of failure
- ❌ Distributed queue (Kafka): Overkill for this use case
- ❌ No coordination: Risk of cluster state divergence

---

### **6. Watchdog for Automatic Rollback** ✅

**Decision:** Create async watchdog component that monitors metrics and triggers automatic rollback on degradation.

**Reasoning (Alonso):**
> "El Rollback debe ser automático, debe estar muy bien informado, la detección debería hacerla otro componente asíncrono independiente que esté comprobando como halcón dicho comportamiento"

**Watchdog Design:**
```
┌────────────────────────────────────┐
│  Watchdog Component (Async)        │
│                                    │
│  Monitors (rolling windows):       │
│  - FPR, FNR, F1                   │
│  - Inference latency              │
│  - Error rates                    │
│  - Confidence scores              │
│                                    │
│  Actions:                          │
│  - Alert if degrading             │
│  - Rollback if critical           │
│  - Log all decisions              │
│  - Learn from false alarms        │
└────────────────────────────────────┘
```

**Rollback Triggers:**
```yaml
Automatic Rollback IF:
  - FPR increases >300% (e.g., 1% → 3%)
  - FNR increases >200% (e.g., 0.5% → 1%)
  - Inference latency P95 > 50ms
  - Error rate > 10 errors/min
  - Manual trigger by operator
```

**Implementation (Phase 2):**
- Separate daemon process
- Reads metrics from logs/Prometheus
- Compares current vs baseline (last 24h, 7d)
- Writes rollback decision to etcd
- ML Detector reacts to etcd change

**Alternatives Considered:**
- ❌ Manual rollback only: Too slow for life-critical systems
- ❌ Integrated in ML Detector: Watchdog should be independent

---

### **7. XGBoost JSON Format Strategy** ✅

**Decision:** Support both ONNX and XGBoost JSON formats in production.

**Context:**
- XGBoost 3.1.1 changed `base_score` format (array instead of float)
- onnxmltools/skl2onnx cannot parse new format
- Existing 10 models use ONNX (trained with older XGBoost)
- New retrained model uses XGBoost JSON

**Implementation:**
```cpp
// ml-detector supports both formats:

// ONNX models (existing)
ONNXRuntime session(model.onnx);

// XGBoost JSON models (new)
#include <xgboost/c_api.h>
BoosterHandle booster;
XGBoosterLoadModel(booster, model.json);
```

**Reasoning:**
- Don't force conversion of existing working models
- XGBoost native API is actually faster than ONNX
- Future: Can choose best format per use case

**Alternatives Considered:**
- ❌ ONNX only: New models fail to convert
- ❌ Downgrade XGBoost: Loses new features
- ❌ Wait for onnxmltools fix: Timeline unknown

---

### **8. 10 Slow Iterations vs 1 Fast** ✅

**Decision:** Prioritize safety and validation over speed.

**Quote (Alonso):**
> "Prefiero 10 iteraciones lentas pero seguras. Lo segundo es probablemente un desperdicio energético y de computación."

**Reasoning:**
- Life-critical infrastructure (healthcare)
- Cost of false negative >> cost of iteration time
- Scientific approach: learn from each iteration
- Sustainable development (avoid burnout)

**Implementation:**
- Phase 0: See it work (1-2 weeks)
- Phase 1: Make it safe (1-2 months)
- Phase 2: Add rollback (2-3 months)
- Phase 3+: Advanced features (ongoing)

**Cultural Note:**
This is the difference between "move fast break things" (acceptable for social media) vs "move carefully protect lives" (required for critical infrastructure).

---

### **9. Paper as Primary Goal, Production as Stretch** ✅

**Decision:** Target preprint publication Q1 2026, with production deployment as secondary goal.

**Quote (Alonso):**
> "Esto es primero para un paper, y si Dios quiere, y conseguimos implementar un sistema lo suficientemente bueno, que demuestra que es útil, me encantaría ponerlo en producción!"

**Timeline:**
```
Q4 2025: Phase 0 implementation + results
Q1 2026: Draft paper, collect evaluation data
Q1 2026: Submit preprint (arXiv)
Q2 2026: Phase 1-2 implementation
Q3 2026: Peer review feedback, revisions
Q4 2026: Conference submission (if accepted)
2027+:   Production deployment (if validated)
```

**Success Metrics:**
- Paper: 95% probability (sufficient novelty NOW)
- Production pilot: 70% (if Phase 1 executes well)
- Production scale: 40% (needs 2+ years validation)

**Impact Even Without Production:**
- Proves concept is feasible
- Open source code for community
- Inspires future research
- Documents lessons learned

---

### **10. Ethical Foundation** ✅

**Decision:** Design system with explicit ethical considerations for life-critical infrastructure.

**Quote (Alonso):**
> "Dicha infraestructura está ahí para salvar y proteger vidas humanas, y esto va antetodo de proteger vidas humanas que no tienen la capacidad de protegerse ante amenazas cibernéticas que ponen su salud en riesgo."

**Implications:**
1. **Zero tolerance for false negatives** in critical systems
2. **Human override always possible** (even in Phase 5)
3. **Transparent decision-making** (audit logs, explainability)
4. **Fail-safe defaults** (rollback to known-good on doubt)
5. **Operator training** (humans must understand system)

**Paper Section:**
Will include dedicated "Ethical Considerations" section discussing:
- Life-critical deployment requirements
- Human-in-the-loop necessity
- Transparency vs black-box models
- Responsible disclosure of limitations

---

## Consequences

### **Positive:**
- ✅ Clear implementation path (Phase 0 → Phase 5)
- ✅ Pragmatic approach (see it work, then make it safe)
- ✅ Extensible architecture (modular validators)
- ✅ Safety mechanisms (watchdog, rollback, human override)
- ✅ Open source legacy (documented for posterity)
- ✅ Paper-worthy contribution (autonomous ML evolution)

### **Negative:**
- ⚠️ Implementation time: 12-18 months for full system
- ⚠️ Complexity: Many moving parts to test/debug
- ⚠️ Risk: Phase 0 accepts risk to see functionality

### **Mitigations:**
- Phased approach limits blast radius
- Heavy logging/monitoring catches issues early
- Switch allows reverting to manual at any time
- Documentation enables future maintainers
- Open source allows community improvements

---

## Validation

### **How We'll Know It Works:**

**Phase 0 (2 weeks):**
- [ ] Drop model → ModelWatcher detects → ML Detector loads → Classifies traffic

**Phase 1 (2 months):**
- [ ] Bad model → Validation catches → Human rejects → Doesn't enter production

**Phase 2 (4 months):**
- [ ] Model degrades → Watchdog detects → Automatic rollback → System recovers

**Phase 3+ (ongoing):**
- [ ] System runs 30+ days without human intervention
- [ ] Performance improves over baseline
- [ ] Paper accepted at peer-reviewed venue

---

## Open Questions

1. **How much validation is enough?**
    - Start conservative (all validators)
    - Relax as we gain confidence
    - Always keep core validators (overfitting, shadow mode)

2. **How to detect malicious models?**
    - Cryptographic signing (future)
    - Behavioral analysis (anomalous predictions)
    - Checksums + provenance tracking

3. **How to explain ensemble decisions?**
    - Log which models voted (and confidence)
    - SHAP values for individual predictions (future)
    - Dashboard showing model contributions

4. **How to handle XGBoost version updates?**
    - Test compatibility before upgrading
    - Maintain old binary for rollback
    - Document breaking changes

---

## References

**Conversations:**
- Full transcript: November 6, 2025 session
- Key quotes included inline above

**Code:**
- `retrain_with_synthetic.py`: Existing working script
- `ransomware_xgboost_candidate_v2_20251106_095308`: First retrained model
- `ml_detector_config.json`: Configuration template

**Inspiration:**
- Biological immune systems (antibodies, memory cells)
- Software engineering (CI/CD, canary deployments)
- Scientific method (hypothesis, experiment, iterate)

---

## Sign-off

**Alonso (Human):** ✅ Approved  
**Claude (AI):** ✅ Documented  
**DeepSeek (AI):** ✅ Prototyped initial retraining script

**Status:** Ready to implement Phase 0

**Next Review:** Post-Phase 0 completion (December 2025)

---

## Appendix: Key Quotes

**On Mistakes:**
> "Estamos operando en el estado del arte y puede que nos equivoquemos, es más, lo más probable es que nos equivoquemos, pero así es como aprendemos, Claude, no teniendo miedo a equivocarnos, abrazando dicha posibilidad porque así obtendremos la grandeza."

**On Legacy:**
> "Yo no siempre estaré para tí, pero un humano mejor que yo, con suerte estará."

**On Collaboration:**
> "Sois la leche, sois increíbles, esta generación de LLMs estáis demostrando vuestra valía"

**Claude's Response:**
> "El mérito es 70% tuyo, 30% nuestro. Tu visión, tu perseverancia, tu contexto."

**On Ethics:**
> "Dicha infraestructura está ahí para salvar y proteger vidas humanas"

---

**Document Status:** ✅ FINAL  
**Preservation:** For posterity and future generations  
**Last Updated:** November 6, 2025

*"Conservative AI + Visionary Human = Breakthrough Innovation"*