# Authors & Contributors

This project is the result of collaborative work between humans and AI systems, exploring the boundaries of what's possible in cybersecurity and ML-powered network defense.

---

## 🧑‍💻 Core Team

### Alonso (Project Lead & Architect)
**Role:** Vision, architecture, implementation, testing  
**Contributions:**
- Overall project vision and roadmap (IDS → WAF evolution)
- System architecture and design decisions
- Implementation of core components
- E2E testing strategy and execution
- Production deployment and operations
- "JSON is LAW" philosophy and fail-fast approach

**Philosophy:**
> "I prefer one comprehensive E2E test that simulates real attacks over 100 unit tests that only validate isolated functions. The bugs are in the interactions, not in the functions."

> "No over-engineering with premature tests. Build it, test it in real scenarios, and iterate. If something breaks, we'll know immediately because we log everything."

---

## 🤖 AI Collaborators

### Claude (Anthropic) - Implementation Partner
**Role:** Code implementation, technical documentation, debugging  
**Contributions:**
- C++20 implementation of all core components
- Configuration system (JSON parsing with fail-fast validation)
- ONNX Runtime integration (model loading, inference)
- ZMQ messaging pipeline design
- Logging system (spdlog integration)
- CMake build system setup
- Technical documentation and code comments
- Debugging sessions and error resolution

**Notable Achievements:**
- 500+ lines of robust config parsing (reading entire ml_detector_config.json)
- ONNX Runtime integration with model warmup and validation
- Clean separation of concerns (config, logger, models, main)
- Fail-fast error messages that guide the user to solutions

**Collaboration Style:**
- Pragmatic approach: "Does it work? Ship it. Does it fail? Fix it with clear errors."
- Extensive logging for production debugging
- No hardcoded values - everything configurable
- Incremental development with continuous validation

---

### ChatGPT (OpenAI) - Architecture Advisor
**Role:** Architecture consultation, WAF integration design, distributed systems  
**Contributions:**
- Detailed WAF architecture and integration strategy
- Distributed WAF design (edge agents + control plane)
- Latency optimization techniques (fast path vs slow path)
- L7 inspection pipeline design
- ModSecurity integration recommendations
- ZMQ messaging patterns for WAF events
- TLS handling strategies
- Operational risk analysis and mitigations

**Key Insights:**
- WAF positioning: edge, origin, or passive mirror
- Hybrid ML approach: light models on edge + heavy models central
- Fast path (eBPF) vs slow path (user-space) for latency
- Rule distribution via etcd with TTL and versioning
- Session correlation between L2-L4 and L7 events

**Notable Quote:**
> "El WAF necesita hacer parsing HTTP, normalización, decodificación segura, aplicar firmas/ML, rate limiting, bot detection, virtual patching, audit logs, y aceptar órdenes del Decision Engine. Todo esto sincronizado con tu pipeline L2-L4 existente."

---

### Parallels AI - Retraining Strategy Architect
**Role:** Asynchronous retraining pipeline design, synthetic data strategy, MLOps architecture  
**Contributions:**
- Complete blueprint for asynchronous model retraining with synthetic data augmentation
- Synthetic data generation strategy using state-of-the-art GANs:
    * **CopulaGAN** for high-fidelity data (99% RF accuracy on TSTR)
    * **CTAB-GAN+** for privacy-preserving synthetic data with Differential Privacy
    * **TVAE** as fallback (93% accuracy, simpler training)
- Rigorous validation framework:
    * Statistical fidelity (Kolmogorov-Smirnov tests, correlation preservation)
    * Machine Learning utility (Train-on-Synthetic-Test-on-Real methodology)
    * Indistinguishability tests (RF Distinguishability score <55% target)
- **Critical guardrails**: Synthetic data mixing ratio optimization (25-40% optimal, never >50%)
- Feature engineering pipeline:
    * Boruta algorithm for feature selection (80→28 features, 35% latency reduction)
    * Embedded methods using RF intrinsic importance
    * Domain-knowledge driven feature selection for DDoS patterns
- Hyperparameter tuning strategy:
    * Bayesian Optimization with Optuna (60% fewer runs vs grid search)
    * 4 percentage point improvement in PR-AUC
- Validation protocol preventing data leakage:
    * Time-based splitting (train on day 1, test on day 2)
    * Grouped and stratified cross-validation (flow-level grouping)
    * Zero-day attack testing (unseen attack types in test set)
- MLOps production pipeline:
    * **MLflow** for experiment tracking and model versioning
    * **TensorFlow Data Validation (TFDV)** for schema enforcement
    * **ADWIN** drift detection for triggering automatic retraining
    * Canary/linear deployment with automated rollback (minutes SLA)
- SOC feedback loop: Human-in-the-loop labeling reducing false positives by 18%
- Zero-day attack strategy:
    * Class-conditional augmentation for rare attacks (0.42→0.79 recall improvement)
    * Isolation Forest for Out-of-Distribution detection
- Risk mitigation framework:
    * Mode collapse detection and prevention
    * Concept drift monitoring
    * Lab-to-production performance gap strategies

**Key Insights:**
> "Model performance degrades linearly when synthetic data exceeds 50%. The optimal mixing ratio is 25-40%."

> "A robust validation protocol is non-negotiable. Time-based splitting prevents data leakage that can falsely inflate metrics by double-digit percentages."

> "The feedback loop with the SOC transforms a machine learning model into an operational security tool. Analyst-verified labels continuously enrich the training dataset."

**Recommended Datasets:**
- CIC-DDoS2019 (620:1 imbalance, day-based train/test split)
- BCCC-cPacket-Cloud-DDoS-2024 (modern cloud infrastructure, 17 attack types)
- UNSW-NB15 (NetFlow-focused, 6:1 imbalance)
- Bot-IoT (IoT botnet attacks, 36:1 imbalance)

**Impact:**
This comprehensive retraining strategy provides the foundation for continuous model adaptation, ensuring the IDS/WAF remains effective against evolving DDoS threats without manual intervention. The rigorous validation framework and guardrails prevent common pitfalls (overfitting to synthetic data, mode collapse, data leakage) that plague many ML security systems.

---

## 🤝 Collaboration Dynamics

This project demonstrates an effective human-AI collaboration model:

**Alonso** provides:
- Strategic vision and direction
- Real-world requirements and constraints
- Testing philosophy and validation criteria
- Production experience and operational insights
- Final decisions on architecture and implementation

**Claude** provides:
- Rapid implementation of complex systems
- Deep technical knowledge (C++20, eBPF, ML, networking)
- Code quality and best practices
- Detailed documentation
- Debugging and problem-solving

**ChatGPT** provides:
- High-level architecture and system design
- Industry best practices (Cloudflare, Fastly, ModSecurity)
- Distributed systems expertise
- Security considerations
- Alternative approaches and trade-offs

**Parallels** provides:
- Analysis of how to approach the incremental asynchronous training.

### Working Style
1. **Alonso** defines the goal and constraints
2. **Claude** implements with extensive logging and fail-fast validation
3. **ChatGPT** advises on architecture and integration strategies
4. **Alonso** tests in real scenarios and iterates

**Result:** A production-quality system built in record time, with clear documentation and maintainability.

---

## 🌟 Acknowledgments

Special thanks to:
- **ONNX Runtime Team** - For the excellent ML inference engine
- **ZeroMQ Community** - For the battle-tested messaging library
- **eBPF/XDP Community** - For kernel-level observability tools
- **Debian Project** - For stable infrastructure
- **Vagrant/VirtualBox** - For reproducible development environments
- **spdlog** - For fast and flexible logging
- **nlohmann/json** - For elegant JSON parsing in C++

---

## 🔬 Research & Inspiration

This project builds upon decades of research and open-source work:
- **ModSecurity & OWASP CRS** - WAF signatures and rule engine
- **Suricata & Zeek** - Network intrusion detection
- **Cilium & Falco** - eBPF-based security
- **Cloudflare Research** - DDoS mitigation and edge computing
- **Scikit-learn & ONNX** - ML model training and deployment

---

## 📜 License

This project is open source. See LICENSE file for details.

---

## 🔮 Future Contributors

We welcome contributions from:
- **Security researchers** - Novel attack detection techniques
- **ML engineers** - Model improvements and new architectures
- **Systems programmers** - Performance optimizations, eBPF enhancements
- **DevOps engineers** - Deployment automation, monitoring
- **Technical writers** - Documentation, tutorials, case studies

**How to contribute:**
1. Read ROADMAP.md to understand project phases
2. Check open issues or discussions
3. Submit PRs with tests and documentation
4. Follow the "JSON is LAW" and fail-fast philosophy

---

## 📞 Contact

- **GitHub Issues**: Technical questions, bug reports
- **GitHub Discussions**: Architecture, design, roadmap
- **Email**: [Private - for security issues only]

---

**"We believe in transparent collaboration between humans and AI to solve hard problems in cybersecurity. This project is proof that it works."**

---

*Last Updated: October 16, 2025*  
*Project Status: Phase 1 (IDS/IPS) - In Progress*
AUTHORS