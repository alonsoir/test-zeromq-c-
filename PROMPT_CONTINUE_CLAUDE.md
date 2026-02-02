# 🔬 Day 50 - Firewall Observability & Hardening

## 📍 CONTEXT
- **Branch:** `feature/contract-validation-142-features`
- **Status:** Day 49 stress test revealed critical observability gaps
- **Document:** `docs/STRESS_TEST_FIREWALL.md` (READ FIRST - complete context)
- **Commit:** Stress test tools + Day 49 iterations documented

## 🎯 OBJECTIVE
Add comprehensive observability to firewall-acl-agent to diagnose crash @ 50% load.

## ⚠️ CRITICAL ISSUES (from Day 49)
1. **Firewall crash @ 50% load** - No diagnostic context, unknown error location
2. **Zero visibility** - Cannot see IPs entering ipsets, batch operations invisible
3. **etcd registration bug** - Hardcoded `../config/firewall.json` causes crash

## 📋 TODAY'S PRIORITIES

### P1: Verbose Logging (2h)
**Files:** `firewall-acl-agent/src/api/zmq_subscriber.cpp`, `batch_processor.cpp`
- [ ] Log every ZMQ message (size, decrypt/decompress steps)
- [ ] Log every protobuf parse (source IP, confidence, threat type)
- [ ] Log every batch operation (assembly, flush, ipset calls)
- [ ] Log ipset results (success/failure with context)

### P2: Crash Diagnostics (1h)
**Files:** `firewall-acl-agent/src/main.cpp`, `zmq_subscriber.cpp`
- [ ] Add signal handlers (SIGSEGV/SIGABRT) with backtrace
- [ ] Wrap critical sections in try-catch with state dump
- [ ] Log component state before crashes (memory, events processed)

### P3: Fix etcd Bug (30min)
**File:** `firewall-acl-agent/src/main.cpp`
- [ ] Pass config_path to etcd registration
- [ ] Remove hardcoded `../config/firewall.json`
- [ ] Test from /vagrant/ with absolute path

### P4: Re-run Stress Test (1h)
- [ ] Start firewall with verbose logging enabled
- [ ] Run injector @ 1K/sec (10K events total)
- [ ] Monitor logs real-time: `tail -f logs/firewall-acl-agent/*.log`
- [ ] Identify exact failure point with diagnostic context
- [ ] Document as Iteration 3 in STRESS_TEST_FIREWALL.md

## 🔍 EXPECTED LOG VISIBILITY
After changes, should see:
```
[DEBUG] ZMQ message: 512 bytes
[DEBUG] Decrypted: 387 bytes → Decompressed: 1024 bytes
[DEBUG] Protobuf: source_ip=192.168.1.100, threat=DDOS, confidence=0.95
[DEBUG] Batch: Adding 192.168.1.100 (9/10)
[DEBUG] Batch flush: 10 IPs
[DEBUG] IPSet ADD 192.168.1.100 → SUCCESS (timeout=600s)
```

## 📊 SUCCESS CRITERIA
- ✅ Every IP processed is visible in logs
- ✅ Crash location identified with backtrace
- ✅ Batch processor performance measured
- ✅ etcd registration succeeds reliably
- ✅ Stress test completes OR failure clearly documented

## 🏛️ VIA APPIA PRINCIPLE
"Fiat Lux" - Build observability BEFORE optimization.
Evidence-based debugging requires visibility.

## 📚 READ FIRST
1. `docs/STRESS_TEST_FIREWALL.md` - Full Day 49 context
2. `firewall-acl-agent/src/api/zmq_subscriber.cpp` - Main loop
3. `firewall-acl-agent/src/batch_processor.cpp` - Batching logic