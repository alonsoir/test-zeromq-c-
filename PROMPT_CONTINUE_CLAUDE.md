# ML Defender - Day 59 Continuation Prompt

## Day 59 Summary - COMPLETADO ✅

**Date:** 2026-02-16 (Monday)
**Duration:** ~3 hours productive work
**Status:** Service Discovery implemented, one bug discovered

### Achievements

**Priority 1: Service Discovery Architecture - COMPLETED ✅**

**Problem identified:** Components hardcoded paths to etcd resources (HMAC keys, crypto tokens, config)
- firewall.json: `"hmac_key_path": "/secrets/firewall/log_hmac_key"` ❌
- Violates Single Source of Truth principle
- Changes to path topology require recompiling all components

**Solution implemented:** etcd-server owns path topology, tells components via service discovery
- etcd-server extended POST /register to return paths:
```json
  {
    "status": "registered",
    "paths": {
      "hmac_key": "/secrets/firewall",
      "crypto_token": "/crypto/firewall/tokens",
      "config": "/config/firewall"
    }
  }
```
- etcd-client extended with `ServicePaths` struct and `get_service_paths()` method
- firewall-acl-agent modified to use discovered paths instead of hardcoded values

**Files Modified:**
```
etcd-server/src/etcd_server.cpp                      - Added paths to registration response
etcd-client/include/etcd_client/etcd_client.hpp      - Added ServicePaths struct
etcd-client/src/etcd_client.cpp                      - Parse and store paths from registration
etcd-client/tests/test_service_discovery.cpp         - NEW: End-to-end test
etcd-client/tests/CMakeLists.txt                     - Added new test
firewall-acl-agent/include/firewall/etcd_client.hpp  - Added get_service_paths() wrapper
firewall-acl-agent/src/core/etcd_client.cpp          - Implemented wrapper, temporary workaround
firewall-acl-agent/src/api/zmq_subscriber.cpp        - Use service discovery for HMAC key
firewall-acl-agent/config/firewall.json              - Removed hmac_key_path (etcd owns it)
```

**Test Results:**
```bash
$ ./test_service_discovery
========================================
✅ All service discovery tests PASSED
========================================

$ make run-firewall
🗺️  Service discovery paths received:
   - HMAC key: /secrets/firewall
   - Crypto token: /crypto/firewall/tokens
   - Config: /config/firewall
🔑 Retrieved HMAC key from: /secrets/firewall (32 bytes)
```

**Architectural Benefits:**
- ✅ Single source of truth: etcd-server controls all paths
- ✅ Flexibility: change `/secrets/*` convention without recompiling components
- ✅ No hardcoding in components
- ✅ Via Appia Quality: proper separation of concerns

---

### Bug Discovered (Day 60 Priority)

**Symptom:** ChaCha20 decryption fails when firewall uploads config to etcd-server
```
[CRYPTO] ❌ Error descifrando: ChaCha20 decryption failed (wrong key or corrupted data)
```

**Context:**
- firewall-acl-agent → `PUT /v1/config/firewall-acl-agent` (8258 bytes)
- etcd-server receives encrypted data
- Decryption fails with wrong key error

**Hypothesis:**
- May be related to Day 58 LZ4 change (removed 4-byte header)
- Possible key mismatch between client encryption and server decryption
- Need to trace crypto pipeline: compress → encrypt (client) vs decrypt → decompress (server)

**Temporary Workaround Applied (MUST REVERT Day 60):**
- Modified `firewall-acl-agent/src/core/etcd_client.cpp::registerService()`
- Changed `return false;` to continue on PUT failure
- **CRITICAL:** This is unsafe - component MUST abort if config upload fails
- Only applied to validate Day 59 service discovery work

**Files with temporary changes:**
```
firewall-acl-agent/src/core/etcd_client.cpp  - Line ~130 (REVERT THIS)
```

---

## Day 60 Priorities (In Order)

### Priority 1: Fix PUT Config Decryption Bug 🔴

**Goal:** Resolve ChaCha20 decryption failure in config upload

**Investigation Steps:**
1. **Trace encryption pipeline in etcd-client**
  - Check `put_config()` in etcd-client/src/etcd_client.cpp
  - Verify compress → encrypt order
  - Check if LZ4 header change (Day 58) affects encryption

2. **Trace decryption pipeline in etcd-server**
  - Check `PUT /v1/config/*` handler in etcd_server.cpp
  - Verify decrypt → decompress order
  - Check if server expects 4-byte LZ4 header

3. **Compare keys**
  - Firewall encrypts with: `encryption_key_` from registration
  - Server decrypts with: `crypto_manager_->decrypt()`
  - Verify both use same key from registration response

4. **Hypothesis testing**
  - If related to LZ4: check if server expects header client no longer sends
  - If key mismatch: verify registration response key == decryption key
  - Add debug logging to see exact bytes encrypted vs decrypted

5. **Fix and validate**
  - Apply fix to correct component
  - Test with `make run-firewall`
  - Verify config uploads successfully

**Success Criteria:**
```
✅ [firewall-acl-agent] Config uploaded encrypted + compressed
[ETCD-SERVER] ✅ Config saved for: firewall-acl-agent
```

### Priority 2: Revert Temporary Workaround 🟡

**Goal:** Restore proper error handling

**Tasks:**
1. Revert change in `firewall-acl-agent/src/core/etcd_client.cpp`
2. Restore original behavior: abort if PUT config fails
3. Verify firewall stops correctly on upload failure
4. Test that successful upload allows firewall to continue

**File to revert:**
```cpp
// firewall-acl-agent/src/core/etcd_client.cpp line ~130
// REVERT TO:
if (!pImpl->client_->put_config(full_config.dump(2))) {
    std::cerr << "❌ [firewall-acl-agent] Failed to upload config" << std::endl;
    return false;  // ✅ CORRECT: abort on failure
}
```

### Priority 3: ml-detector JSONL → CSV Migration 🟢

**Goal:** Eliminate nlohmann::json memory bug, unify format with firewall

**Tasks:**
1. Update RAGLogger to write CSV instead of JSONL
2. Add HMAC signing to CSV batches
3. Update ml_detector_config.json with csv_batch_logger section
4. Test migration and verify no memory issues

### Priority 4: rag-ingester Multi-Source CSV 🟢

**Goal:** Ingest CSV logs from both firewall and ml-detector

**Tasks:**
1. Refactor config for multiple input sources
2. Implement HMAC validation before decrypt
3. Create firewall-specific embedder
4. Test end-to-end pipeline

---

## Lessons Learned

**Lesson 1: Service Discovery > Hardcoding**
- Hardcoded paths create coupling and fragility
- Service discovery enables centralized control
- **Takeaway:** etcd-server is authority for topology, components are consumers

**Lesson 2: Incremental Testing Reveals Issues**
- Bug appeared when testing end-to-end integration
- Service discovery worked independently of encryption bug
- **Takeaway:** Layer validation catches bugs at right abstraction level

**Lesson 3: Temporary Workarounds Need Documentation**
- Applied unsafe workaround to validate other work
- Clearly marked as MUST REVERT with context
- **Takeaway:** Technical debt requires explicit tracking

**Lesson 4: Piano Piano Works**
- Day 58: Fixed LZ4 bug
- Day 59: Implemented service discovery
- Day 60: Will fix encryption bug discovered today
- **Takeaway:** "Rome wasn't debugged in a day"

---

## System Architecture (Current State)
```
┌──────────────────────────────────────────────────────────┐
│ etcd-server (Day 59 - Service Discovery)                │
├──────────────────────────────────────────────────────────┤
│ ✅ POST /register returns service discovery paths       │
│ ✅ SecretsManager generates HMAC keys on-demand         │
│ ✅ GET /secrets/{component} returns active key          │
│ ⚠️  PUT /v1/config/* has decryption bug (Day 60)        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ etcd-client (Day 59 - Service Discovery)                │
├──────────────────────────────────────────────────────────┤
│ ✅ ServicePaths struct stores discovered paths          │
│ ✅ register_component() parses paths from response      │
│ ✅ get_service_paths() returns paths to components      │
│ ✅ get_hmac_key() fetches keys using discovered paths   │
│ ⚠️  put_config() has encryption bug (Day 60)            │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ firewall-acl-agent (Day 59 - Partial)                   │
├──────────────────────────────────────────────────────────┤
│ ✅ Uses service discovery for HMAC key path             │
│ ✅ No hardcoded paths in code                           │
│ ✅ HMAC key retrieved: 32 bytes from /secrets/firewall  │
│ ⏳ CSV logging ready but not tested (needs ml input)    │
│ ⚠️  Temporary workaround in registerService() (REVERT)  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ ml-detector (Day 59 - Not started)                      │
├──────────────────────────────────────────────────────────┤
│ ⚠️  Still uses JSONL (memory bug exists)                │
│ ⏳ Needs: CSV migration (Priority 3)                    │
│ ⏳ Needs: Service discovery integration                 │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ rag-ingester (Day 59 - Not started)                     │
├──────────────────────────────────────────────────────────┤
│ ⚠️  Single source hardcoded                             │
│ ⏳ Needs: Multi-source CSV support (Priority 4)         │
│ ⏳ Needs: HMAC validation                               │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start Commands (Day 60)
```bash
# 1. Start etcd-server
cd /vagrant/etcd-server/build
./etcd-server

# 2. Test service discovery (should pass)
cd /vagrant/etcd-client/build/tests
./test_service_discovery

# 3. Debug PUT config bug
cd /vagrant
make run-firewall
# Observe decryption error, trace crypto pipeline

# 4. After fix, verify end-to-end
make run-firewall
# Should see: ✅ Config uploaded encrypted + compressed

# Remember: Piano piano 🏛️
```

---

## Contact & Philosophy

**Developer:** Alonso (Extremadura, Spain)
**Motivation:** Friend's business destroyed by ransomware
**Vision:** Democratize enterprise-grade security for SMBs
**License:** Open source (TBD)

**Core Values:**
- Via Appia Quality (built to last decades)
- Scientific honesty over marketing hype
- Evidence-based validation
- Piano piano: stone by stone, validated at each step

---

Last Updated: 2026-02-16 (Day 59 Complete)
Co-authored-by: Claude (Anthropic)
Co-authored-by: Alonso

Piano piano 🏛️