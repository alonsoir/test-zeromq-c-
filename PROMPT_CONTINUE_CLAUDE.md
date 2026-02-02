# 🔬 Day 50 Continuity Prompt - Firewall Hardening Observability

## 📍 CONTEXT
Branch: feature/contract-validation-142-features
Status: Day 49 stress testing revealed critical observability gaps
Document: docs/STRESS_TEST_FIREWALL.md (MUST READ)

## 🎯 OBJECTIVE
Add comprehensive observability to firewall-acl-agent before continuing stress tests.

## ⚠️ CRITICAL ISSUES DISCOVERED
1. **Firewall crash @ 50% load** - Unknown error, no diagnostics
2. **Zero visibility** - Can't see IPs entering ipsets
3. **etcd registration bug** - Crashes when config path wrong

## 📋 TODAY'S TASKS

### PRIORITY 1: Add Verbose Logging (2 hours)
**File:** `firewall-acl-agent/src/api/zmq_subscriber.cpp`
- [ ] Log every IP addition attempt (source IP, confidence, action)
- [ ] Log batch processor operations (batch size, IPs processed)
- [ ] Log ipset operations (add/remove with result)
- [ ] Add structured logging with severity levels

**File:** `firewall-acl-agent/src/batch_processor.cpp`
- [ ] Log batch assembly (count, threshold reached)
- [ ] Log ipset API calls with return codes
- [ ] Add performance metrics (batch latency)

### PRIORITY 2: Add Crash Diagnostics (1 hour)
**Files:** `firewall-acl-agent/src/main.cpp`, `zmq_subscriber.cpp`
- [ ] Wrap critical sections in try-catch with context
- [ ] Log exception type, message, and processing state
- [ ] Add signal handler for SIGSEGV with backtrace
- [ ] Log component state before crash (events processed, memory)

### PRIORITY 3: Fix etcd Registration Bug (30 min)
**File:** `firewall-acl-agent/src/main.cpp`
- [ ] Pass config path to etcd registration
- [ ] Remove hardcoded `../config/firewall.json`
- [ ] Test with absolute path from /vagrant/

### PRIORITY 4: Re-run Stress Test (1 hour)
- [ ] Start firewall with fixed build + verbose logging
- [ ] Run injector @ 1K/sec sustained (10K events)
- [ ] Monitor logs in real-time (tail -f)
- [ ] Identify exact failure point with diagnostic context
- [ ] Update STRESS_TEST_FIREWALL.md with Iteration 3 results

## 🔍 EXPECTED VISIBILITY
After changes, logs should show:
```
[DEBUG] ZMQ message received: 512 bytes
[DEBUG] Decrypted: 387 bytes
[DEBUG] Decompressed: 1024 bytes
[DEBUG] Protobuf parsed: NetworkSecurityEvent
[DEBUG]   Source IP: 192.168.1.100
[DEBUG]   Threat: DDOS, Confidence: 0.95
[DEBUG]   ML Analysis: attack_detected_level1=true
[DEBUG] Batch processor: Adding 192.168.1.100 to queue (9/10)
[DEBUG] Batch threshold reached: Flushing 10 IPs
[DEBUG] IPSet operation: ADD 192.168.1.100 to ml_defender_blacklist_test
[DEBUG] IPSet result: SUCCESS (timeout=600s)
```

## 📊 SUCCESS CRITERIA
- ✅ Can see every IP processed in logs
- ✅ Can identify exact crash location with context
- ✅ Can measure batch processor performance
- ✅ etcd registration succeeds reliably
- ✅ Stress test completes OR failure point clearly documented

## 📚 FILES TO READ FIRST
1. `docs/STRESS_TEST_FIREWALL.md` - Full context from Day 49
2. `firewall-acl-agent/src/api/zmq_subscriber.cpp` - Main processing loop
3. `firewall-acl-agent/src/batch_processor.cpp` - Batching logic

## 🏛️ VIA APPIA PRINCIPLE
"Que se haga la luz" - Build comprehensive observability BEFORE
continuing hardening. Evidence-based debugging requires visibility.