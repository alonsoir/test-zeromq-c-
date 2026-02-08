# Day 53 Continuity Prompt - HMAC-Based Log Integrity for RAG

## ⚠️ CRITICAL: AUDIT FIRST - NO ASSUMPTIONS

**This prompt contains EXAMPLE code and structures that may NOT match reality.**

**Before any implementation**:
1. Audit actual rag-ingester state (may be empty, placeholder, or different structure)
2. Audit actual ml-detector RAG logging (may not exist, may be different format)
3. Document findings in `audit_day53.md`
4. Adjust implementation plan based on ACTUAL state

**DO NOT assume**:
- rag-ingester has parsers/ directory
- ml-detector writes RAG logs
- Any specific file structure exists

**START with discovery, then plan, then implement.**

---

## 🎯 Session Goals

Day 53 focuses on **preventing log poisoning attacks** against the RAG system by implementing HMAC-based integrity protection for all logs ingested by rag-ingester.

**Context**: Day 52 validated the crypto pipeline (36K events, 0 errors) and achieved config-driven architecture. However, RAG logs are currently vulnerable to tampering and injection attacks.

---

## 🔐 SECURITY THREAT: Log Poisoning

### Attack Vectors

**Current Vulnerability**:
```
ml-detector → /vagrant/logs/rag/ml_detector_events.jsonl (plaintext, NO integrity)
firewall    → /vagrant/logs/lab/firewall-agent.log (plaintext, NO integrity)
```

**Attacker with filesystem access can**:
1. **Log Injection**: Add malicious lines → contaminate RAG → manipulate LLM responses
2. **Log Modification**: Change existing lines → hide activity / create false narratives
3. **Prompt Injection via Logs**: Inject LLM manipulation → arbitrary behavior
4. **ML Poisoning**: Corrupt training data → degrade detection accuracy
5. **Cover Tracks**: Delete/modify evidence of compromise

### Why NOT ChaCha20 Encryption?

| Issue | ChaCha20 (Confidentiality) | HMAC (Integrity) |
|-------|---------------------------|------------------|
| FAISS indexing | ❌ Cannot index ciphertext | ✅ Can index plaintext |
| Detect tampering | ❌ Decrypts anything with key | ✅ HMAC mismatch detected |
| Detect injection | ❌ New encrypted lines look valid | ✅ No valid HMAC = rejected |
| Performance | Slower (~10μs) | Faster (~2μs) |
| Auditability | ❌ Logs unreadable | ✅ Logs human-readable |
| Key compromise | Reads everything | Only validates, cannot forge |

**Conclusion**: HMAC provides integrity + authenticity while preserving FAISS compatibility.

---

## 🏗️ Day 53 Architecture

### Secure Logging Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Component (ml-detector / firewall-acl-agent)                    │
│ ├─ Generate log line (plaintext)                                │
│ ├─ Retrieve HMAC key from etcd                                  │
│ ├─ Compute HMAC-SHA256(log_line, hmac_key)                      │
│ └─ Write: "log_line|HMAC:hex_value\n"                           │
│    Example:                                                      │
│    {"ip":"1.2.3.4","conf":0.95}|HMAC:a3f5c2d8e9b1...            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ rag-ingester (runs as non-root user)                           │
│ ├─ Read log file (permissions: 0400)                            │
│ ├─ For each line:                                               │
│ │   ├─ Split → (message, hmac_hex)                              │
│ │   ├─ Compute expected_hmac = HMAC-SHA256(message, hmac_key)   │
│ │   ├─ Constant-time compare: hmac_hex == expected_hmac?        │
│ │   ├─ If VALID → parse and ingest to RAG                       │
│ │   └─ If INVALID → REJECT + ALERT (tampering detected)         │
│ ├─ Metrics: tampering_attempts_total                            │
│ └─ Alerts: Slack/email on HMAC mismatch                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ RAG / FAISS                                                     │
│ ├─ Contains ONLY validated logs (HMAC verified)                 │
│ ├─ Plaintext indexable by FAISS                                 │
│ └─ Protected against log poisoning ✅                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Day 53 Implementation Plan

### Phase 1: Audit & Documentation (First Thing Morning)

**CRITICAL**: We DO NOT KNOW what exists in rag-ingester. First task is to discover the actual implementation.

```bash
# 1. Discover rag-ingester structure
cd /vagrant/rag-ingester
ls -la                          # What directories exist?
find . -type f -name "*.py"     # What Python files exist?
cat README.md 2>/dev/null       # Is there documentation?
ls -la config/ 2>/dev/null      # What config files exist?

# 2. Discover ml-detector RAG logging
cd /vagrant/ml-detector
grep -r "rag" src/              # Where is RAG logging code?
ls -la /vagrant/logs/rag/       # What log files exist?
head -20 /vagrant/logs/rag/*.jsonl 2>/dev/null  # What format?

# 3. Document ACTUAL implementation
# Create: audit_day53.md with findings:
# - What files/directories exist in rag-ingester?
# - How does ml-detector write RAG logs? (if at all)
# - What log format is used?
# - How does rag-ingester work? (if implemented)
# - What needs to be built from scratch?
```

**Expected Findings** (UNKNOWN until audit):
- rag-ingester might be: fully implemented, partially implemented, or placeholder
- ml-detector RAG logging might be: working, broken, or non-existent
- Log format might be: JSONL, plaintext, or something else
- Parsing logic might be: Python scripts, config-driven, or manual

**Deliverables**:
- `audit_day53.md`: Complete documentation of actual state
- Decision: Build new vs modify existing
- Implementation plan based on reality, not assumptions

### Phase 2: firewall-acl-agent → rag-ingester → rag

**Feature Branch**:
```bash
git checkout -b feature/rag-firewall-hmac-security
```

**Components to Modify**:

#### 2.1: etcd-server - HMAC Key Management

```cpp
// etcd-server/src/main.cpp

void initialize_hmac_keys() {
    // Generate 32-byte HMAC keys
    std::vector<uint8_t> firewall_hmac_key(32);
    randombytes_buf(firewall_hmac_key.data(), 32);
    
    // Store in etcd
    std::string key_hex = bytes_to_hex(firewall_hmac_key);
    etcd_->Put("/secrets/firewall/log_hmac_key", key_hex);
    
    LOG_INFO("Generated HMAC key for firewall logs");
}
```

**Config**:
```json
{
  "secrets": {
    "rotation_interval_hours": 168,  // Weekly rotation
    "keys": {
      "firewall_log_hmac": {
        "path": "/secrets/firewall/log_hmac_key",
        "length_bytes": 32,
        "algorithm": "hmac-sha256"
      }
    }
  }
}
```

#### 2.2: firewall-acl-agent - SecureLogger

**New File**: `firewall-acl-agent/src/core/secure_logger.hpp`

```cpp
#include <openssl/hmac.h>

class SecureLogger {
private:
    std::ofstream log_file_;
    std::vector<uint8_t> hmac_key_;
    
    std::string compute_hmac(const std::string& message) {
        unsigned char hmac[32];
        HMAC(EVP_sha256(),
             hmac_key_.data(), hmac_key_.size(),
             (unsigned char*)message.c_str(), message.size(),
             hmac, nullptr);
        
        return bytes_to_hex(hmac, 32);
    }
    
public:
    void initialize(const std::string& log_path, 
                   const std::vector<uint8_t>& hmac_key) {
        log_file_.open(log_path, std::ios::app);
        hmac_key_ = hmac_key;
    }
    
    void log_secure(const std::string& level,
                   const std::string& message,
                   const std::map<std::string, std::string>& context) {
        // Build log line
        std::string line = format_log_line(level, message, context);
        
        // Compute HMAC
        std::string hmac = compute_hmac(line);
        
        // Write: line|HMAC:hex
        log_file_ << line << "|HMAC:" << hmac << std::endl;
    }
};
```

**Integration in main.cpp**:
```cpp
// firewall-acl-agent/src/main.cpp

// Retrieve HMAC key from etcd
std::string hmac_key_hex = etcd_client.get("/secrets/firewall/log_hmac_key");
std::vector<uint8_t> hmac_key = hex_to_bytes(hmac_key_hex);

// Initialize secure logger
SecureLogger logger;
logger.initialize(config.logging.file, hmac_key);

// Log events with HMAC
logger.log_secure("INFO", "IP blocked", {
    {"ip", "1.2.3.4"},
    {"confidence", "0.95"}
});
```

**Config Update**:
```json
{
  "logging": {
    "file": "/vagrant/logs/lab/firewall-agent.log",
    "level": "debug",
    "integrity": {
      "enabled": true,
      "algorithm": "hmac-sha256",
      "key_source": "etcd",
      "key_path": "/secrets/firewall/log_hmac_key"
    }
  }
}
```

#### 2.3: rag-ingester - HMAC Validation

**New File**: `rag-ingester/parsers/secure_firewall_parser.py`

```python
import hmac
import hashlib
from typing import Optional, Dict

class SecureFirewallParser:
    """Parse firewall logs with HMAC validation"""
    
    def __init__(self, hmac_key: bytes):
        self.hmac_key = hmac_key
        self.tampering_count = 0
        self.valid_count = 0
    
    def verify_line(self, line: str) -> tuple[bool, str]:
        """
        Verify HMAC and extract message.
        
        Returns: (is_valid, message)
        """
        if "|HMAC:" not in line:
            self.tampering_count += 1
            return False, "Missing HMAC"
        
        # Split message and HMAC
        message, hmac_hex = line.rsplit("|HMAC:", 1)
        
        # Compute expected HMAC
        h = hmac.new(self.hmac_key, message.encode(), hashlib.sha256)
        expected = h.hexdigest()
        
        # Constant-time comparison (prevent timing attacks)
        if not hmac.compare_digest(hmac_hex.strip(), expected):
            self.tampering_count += 1
            return False, "HMAC mismatch"
        
        self.valid_count += 1
        return True, message
    
    def parse_file(self, filepath: str) -> list:
        """Parse entire file with HMAC validation"""
        events = []
        tampering_detected = []
        
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                is_valid, message = self.verify_line(line)
                
                if not is_valid:
                    tampering_detected.append({
                        'line_num': line_num,
                        'reason': message,
                        'content': line[:100]  # First 100 chars
                    })
                    continue
                
                # Parse validated message
                event = self.parse_message(message)
                if event:
                    events.append(event)
        
        # Alert if tampering detected
        if tampering_detected:
            self.alert_security_team(tampering_detected)
        
        return events
    
    def parse_message(self, message: str) -> Optional[Dict]:
        """Parse validated log line"""
        # Standard firewall log parsing
        # (existing logic from Day 52 backlog)
        pass
    
    def alert_security_team(self, tampering_events: list):
        """Send alert on log tampering"""
        alert_message = f"""
        🚨 SECURITY ALERT: Log Tampering Detected
        
        Component: firewall-acl-agent
        Tampering events: {len(tampering_events)}
        Details: {tampering_events[:5]}  # First 5
        
        Action required: Investigate potential security breach
        """
        
        # Send to Slack/email
        send_slack_alert(alert_message)
```

**Config Update**:
```json
{
  "parsers": {
    "firewall-acl-agent": {
      "class": "SecureFirewallParser",
      "watch_path": "/vagrant/logs/lab/firewall-agent.log",
      "hmac_validation": {
        "enabled": true,
        "key_source": "etcd",
        "key_path": "/secrets/firewall/log_hmac_key",
        "alert_on_tampering": true
      }
    }
  }
}
```

#### 2.4: Testing & Validation

```bash
# 1. Generate logs with HMAC
cd /vagrant/tools/build
./synthetic_ml_output_injector 100 10

# 2. Verify HMAC format
tail -5 /vagrant/logs/lab/firewall-agent.log
# Should see: message|HMAC:a3f5c2d8...

# 3. Test tampering detection
echo "FAKE LOG LINE|HMAC:fakehash123" >> /vagrant/logs/lab/firewall-agent.log

# 4. Run rag-ingester
cd /vagrant/rag-ingester
python3 ingest.py --source firewall

# Expected output:
# ✅ Parsed 100 valid lines
# 🚨 ALERT: 1 tampering attempt detected
# ✅ Rejected 1 invalid line

# 5. Verify metrics
grep "tampering_count" /vagrant/logs/rag-ingester.log
```

**Acceptance Criteria**:
- [ ] firewall-acl-agent writes logs with valid HMAC
- [ ] rag-ingester validates HMAC before parsing
- [ ] Tampering detection triggers alerts
- [ ] Invalid lines rejected (not ingested to RAG)
- [ ] Valid lines parsed correctly
- [ ] Metrics: valid_count, tampering_count
- [ ] Zero false positives (all legitimate logs pass)

### Phase 3: ml-detector → rag-ingester

**Apply same pattern to ml-detector**:
1. Add HMAC to existing RAG logger (`ml-detector/src/core/rag_logger.cpp`)
2. Update rag-ingester ML detector parser
3. Validate end-to-end

**Defer to separate session** after firewall is validated.

---

## 🧪 Testing Strategy

### Unit Tests

```cpp
// firewall-acl-agent/tests/test_secure_logger.cpp

TEST(SecureLogger, ComputesValidHMAC) {
    std::vector<uint8_t> key(32, 0xAB);
    SecureLogger logger;
    logger.initialize("/tmp/test.log", key);
    
    std::string hmac = logger.compute_hmac("test message");
    EXPECT_EQ(hmac.length(), 64);  // 32 bytes = 64 hex chars
}

TEST(SecureLogger, DifferentMessagesProduceDifferentHMACs) {
    std::vector<uint8_t> key(32, 0xAB);
    SecureLogger logger;
    
    std::string hmac1 = logger.compute_hmac("message1");
    std::string hmac2 = logger.compute_hmac("message2");
    
    EXPECT_NE(hmac1, hmac2);
}
```

```python
# rag-ingester/tests/test_secure_parser.py

def test_valid_hmac_passes():
    key = b'A' * 32
    parser = SecureFirewallParser(key)
    
    message = "test log line"
    h = hmac.new(key, message.encode(), hashlib.sha256)
    hmac_hex = h.hexdigest()
    
    line = f"{message}|HMAC:{hmac_hex}"
    is_valid, extracted = parser.verify_line(line)
    
    assert is_valid == True
    assert extracted == message

def test_invalid_hmac_rejected():
    key = b'A' * 32
    parser = SecureFirewallParser(key)
    
    line = "test log line|HMAC:fakehash123"
    is_valid, reason = parser.verify_line(line)
    
    assert is_valid == False
    assert "mismatch" in reason.lower()

def test_missing_hmac_rejected():
    key = b'A' * 32
    parser = SecureFirewallParser(key)
    
    line = "test log line without HMAC"
    is_valid, reason = parser.verify_line(line)
    
    assert is_valid == False
    assert "missing" in reason.lower()
```

### Integration Tests

```bash
# End-to-end tampering detection

# 1. Start etcd-server
cd /vagrant/etcd-server/build
sudo ./etcd_server

# 2. Start firewall-acl-agent
cd /vagrant/firewall-acl-agent/build
sudo ./firewall-acl-agent -c ../config/firewall.json

# 3. Generate 1000 events
cd /vagrant/tools/build
./synthetic_ml_output_injector 1000 100

# 4. Inject tampering
echo "MALICIOUS_IP|192.168.666.666|HMAC:fakehash" >> /vagrant/logs/lab/firewall-agent.log

# 5. Run rag-ingester
cd /vagrant/rag-ingester
python3 ingest.py --source firewall

# 6. Verify
# ✅ 1000 valid events ingested
# 🚨 1 tampering alert triggered
# ✅ Malicious line NOT in RAG
grep "tampering" /vagrant/logs/rag-ingester.log
```

---

## 📊 Success Metrics

### Security
- [ ] 0 false positives (legitimate logs pass)
- [ ] 100% tampering detection (injected lines rejected)
- [ ] Alerts triggered within 1 second of tampering
- [ ] RAG contains ONLY validated logs

### Performance
- [ ] HMAC computation: <5 μs per line
- [ ] HMAC validation: <5 μs per line
- [ ] No noticeable impact on logging throughput
- [ ] No noticeable impact on ingestion latency

### Functionality
- [ ] firewall logs parseable by rag-ingester
- [ ] Cross-component queries work (detection ↔ block)
- [ ] Timeline reconstruction functional
- [ ] FAISS search quality unchanged (plaintext)

---

## 🚨 Critical Reminders

### Security
- **ALWAYS validate HMAC before parsing** — reject first, parse second
- **Constant-time comparison** — prevent timing attacks
- **Key rotation** — weekly rotation planned (future work)
- **Separate user** — rag-ingester runs as non-root
- **File permissions** — logs 0400 (read-only)

### Implementation
- **Audit first** — understand current state before changes
- **Backwards compatibility** — support logs without HMAC during transition
- **Metrics** — track tampering attempts, valid/invalid counts
- **Alerts** — notify security team on suspicious activity

### Testing
- **Unit tests** — HMAC computation/validation logic
- **Integration tests** — end-to-end tampering detection
- **Performance tests** — no significant overhead
- **Chaos tests** — wrong key, corrupted HMAC, network issues

---

## 📁 Files to Create/Modify

### firewall-acl-agent
```
src/core/secure_logger.hpp          (NEW)
src/core/secure_logger.cpp          (NEW)
src/main.cpp                         (UPDATE: use SecureLogger)
config/firewall.json                 (UPDATE: add integrity config)
tests/test_secure_logger.cpp        (NEW)
```

### etcd-server
```
src/main.cpp                         (UPDATE: generate HMAC keys)
config/etcd-server.json              (UPDATE: add secrets section)
```

### rag-ingester
```
parsers/secure_firewall_parser.py   (NEW)
parsers/__init__.py                  (UPDATE: import SecureFirewallParser)
config/ingester_config.json          (UPDATE: add hmac_validation)
tests/test_secure_parser.py         (NEW)
```

---

## 🎯 Session Workflow (Day 53)

### Morning (2-3 hours)
```bash
# 1. DISCOVERY PHASE - NO ASSUMPTIONS
cd /vagrant

# What exists in rag-ingester?
ls -la rag-ingester/
find rag-ingester/ -type f
cat rag-ingester/README.md 2>/dev/null

# What exists for ml-detector RAG?
grep -r "rag" ml-detector/src/
ls -la logs/rag/ 2>/dev/null

# Document ACTUAL state in audit_day53.md
# DO NOT assume structure exists
# DO NOT assume parsers exist
# DO NOT assume config format

# Based on audit, decide:
# - Build from scratch?
# - Modify existing?
# - Fix broken implementation?
```

### Mid-day (3-4 hours)
```bash
# 2. Create feature branch
git checkout -b feature/rag-firewall-hmac-security

# 3. Implement HMAC in firewall-acl-agent
# - Add SecureLogger
# - Update main.cpp
# - Update config

# 4. Implement validation in rag-ingester
# - Add SecureFirewallParser
# - Update config
# - Add tests
```

### Afternoon (2-3 hours)
```bash
# 5. Testing & validation
# - Unit tests
# - Integration tests
# - Tampering detection tests
# - Performance benchmarks

# 6. Documentation
# - Update BACKLOG.md
# - Update README.md
# - Write commit message

# 7. Commit & push
git add .
git commit -F commit_message.txt
git push origin feature/rag-firewall-hmac-security
```

---

## 📝 Commit Message Template

```
feat(security): HMAC-based log integrity for RAG system

SUMMARY
=======
Implement HMAC-SHA256 integrity protection for firewall-acl-agent logs
to prevent log poisoning attacks against the RAG system. Enables detection
and rejection of tampered or injected log lines before ingestion.

SECURITY THREAT
===============
Log poisoning attacks can:
- Inject malicious content → contaminate RAG
- Modify existing logs → hide malicious activity
- Poison ML training data → degrade detection accuracy
- Manipulate LLM responses → arbitrary behavior

SOLUTION: HMAC-based Integrity
===============================
- firewall-acl-agent writes logs with HMAC-SHA256 signature
- rag-ingester validates HMAC before parsing
- Tampering detection triggers immediate alerts
- Invalid lines rejected (never reach RAG/FAISS)
- Plaintext logs remain indexable by FAISS

CHANGES
=======
1. etcd-server: HMAC key generation and management
   - Generate 32-byte HMAC key on startup
   - Store at /secrets/firewall/log_hmac_key
   - Weekly rotation planned (future work)

2. firewall-acl-agent: SecureLogger implementation
   - New: secure_logger.hpp/cpp
   - Compute HMAC-SHA256 for each log line
   - Write format: "message|HMAC:hex_value"
   - Retrieve HMAC key from etcd

3. rag-ingester: HMAC validation
   - New: secure_firewall_parser.py
   - Validate HMAC before parsing
   - Constant-time comparison (prevent timing attacks)
   - Alert on tampering: Slack/email
   - Metrics: valid_count, tampering_count

VALIDATION
==========
- Generated 1,000 firewall events with valid HMAC
- Injected 10 malicious lines (invalid HMAC)
- Result: 1,000 ingested, 10 rejected, 10 alerts triggered
- Performance: <5μs HMAC overhead per line
- Zero false positives

SECURITY GUARANTEES
===================
✅ No log injection without valid HMAC
✅ No log modification without detection
✅ Tampering triggers immediate alerts
✅ RAG contains only validated logs
✅ ML retraining data integrity verified

BACKWARDS COMPATIBILITY
=======================
- Transition mode: accept logs without HMAC (with warning)
- Gradual rollout: enable HMAC validation after all components updated
- Migration period: 1 week

NEXT STEPS
==========
- Apply same HMAC pattern to ml-detector logs
- Implement key rotation (weekly)
- Add forensic logging (who/when/what)
- Performance optimization (batch HMAC validation)

Co-authored-by: Claude (Anthropic)
```

---

## ✅ Day 53 Checklist

### Pre-Session
- [ ] Read this continuity prompt
- [ ] Review Day 52 achievements (stress testing, config-driven)
- [ ] **Remember: This prompt has EXAMPLES, not facts about current code**

### Audit Phase (FIRST - MANDATORY)
- [ ] Discover rag-ingester actual structure (`find`, `ls -la`)
- [ ] Discover ml-detector RAG logging actual implementation
- [ ] Document findings in `audit_day53.md`
- [ ] List what exists vs what needs to be built
- [ ] Revise implementation plan based on reality

### Planning Phase (AFTER audit)
- [ ] Decide: build from scratch vs modify existing
- [ ] Identify actual files to create/modify (not guessed)
- [ ] Plan migration path if existing code found
- [ ] Create feature branch: `feature/rag-firewall-hmac-security`

### Implementation Phase (AFTER planning)
- [ ] etcd-server: Generate HMAC keys
- [ ] firewall-acl-agent: Implement SecureLogger
- [ ] rag-ingester: Implement/modify HMAC validation (based on audit)
- [ ] Add tests (structure depends on what was found)

### Validation Phase
- [ ] Generate 1K logs with valid HMAC
- [ ] Inject 10 tampering attempts
- [ ] Verify: valid ingested, invalid rejected, alerts triggered
- [ ] Benchmark: <5μs HMAC overhead

### Documentation Phase
- [ ] Update backlog files (based on actual implementation)
- [ ] Update CLAUDE.md (mark Day 53 complete)
- [ ] Write commit message (reflecting what was actually done)

### Completion Phase
- [ ] Commit all changes
- [ ] Push to feature branch
- [ ] Create Pull Request to main

---

**Status**: Day 53 Ready  
**Focus**: HMAC-based log integrity (firewall-acl-agent → rag-ingester)  
**Defer**: ml-detector HMAC implementation (separate session)  
**Via Appia Quality**: Secure by design, not by accident 🛡️