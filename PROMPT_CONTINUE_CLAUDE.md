# Day 23 - Pipeline Integration + Stress Testing

## Context: Day 22 Achievements
✅ Heartbeat system 100% operational (etcd-server + etcd-client)
✅ ml-detector tested with automatic heartbeat (30s interval)
✅ Auto-unregister on SIGINT/SIGTERM + timeout (90s)
✅ Signal handlers working perfectly
✅ Config upload encrypted + compressed (11756 → 5115 bytes)

## Current State
- **etcd-server:** Monitoring 3 component types, auto-restart disabled by default
- **etcd-client:** Background heartbeat thread, signal handlers, transparent integration
- **ml-detector:** Fully integrated, tested, heartbeat operational
- **sniffer:** Needs recompilation with etcd-client integration
- **firewall-acl-agent:** Needs recompilation with etcd-client integration

## Day 23 Goals

### 1. Component Integration (2 hours)
**Objective:** All 3 components registered and sending heartbeats

**Tasks:**
- Recompile sniffer with etcd-client
- Recompile firewall-acl-agent with etcd-client
- Integrate etcd-server into root Makefile
- Verify all 3 components register on startup
- Verify heartbeats from all 3 components

**Success Criteria:**
```bash
curl http://localhost:2379/components | jq
# Expected:
{
  "component_count": 3,
  "components": ["sniffer", "ml-detector", "firewall-acl-agent"]
}
```

### 2. Encryption + Compression Activation (1 hour)
**Objective:** Verify all components have correct transport settings

**Files to verify:**
- `/vagrant/sniffer/config/sniffer.json`
- `/vagrant/ml-detector/config/ml_detector_config.json`
- `/vagrant/firewall-acl-agent/config/firewall.json`

**Required settings:**
```json
{
  "transport": {
    "compression": {
      "enabled": true,
      "algorithm": "lz4",
      "level": 1
    },
    "encryption": {
      "enabled": true,
      "algorithm": "chacha20-poly1305"
    }
  }
}
```

**Special case - firewall-acl-agent:**
- Input: decrypt + decompress (YES)
- Output: encrypt + compress (NO - end of pipeline)

### 3. Full Pipeline Stress Test (2 hours)
**Objective:** Verify complete data flow with encryption + compression

**Pipeline:**
```
sniffer → [capture] → encrypt+compress → ZMQ:5571
    ↓
ml-detector → decrypt+decompress → [ML inference] → encrypt+compress → ZMQ:5572
    ↓                                                  └→ RAG log (unencrypted)
firewall-acl-agent → decrypt+decompress → [IPTables/IPSet] → RAG log (unencrypted)
```

**Test procedure:**
```bash
# Terminal 1: etcd-server
cd /vagrant/etcd-server/build && ./etcd-server --port 2379

# Terminal 2: ml-detector
cd /vagrant/ml-detector/build && ./ml-detector -c ../config/ml_detector_config.json

# Terminal 3: firewall-acl-agent
cd /vagrant/firewall-acl-agent/build && sudo ./firewall-acl-agent -c ../config/firewall.json

# Terminal 4: sniffer
cd /vagrant/sniffer/build && sudo ./sniffer -c ../config/sniffer.json

# Terminal 5: Monitoring
watch -n 2 'curl -s http://localhost:2379/components | jq'

# Terminal 6: Generate traffic
# (script to generate test traffic for stress testing)
```

**Verification points:**
- ✅ All 3 components registered in etcd-server
- ✅ Heartbeats arriving every 30s from each component
- ✅ sniffer capturing packets and sending encrypted payloads
- ✅ ml-detector receiving, processing, logging to RAG (unencrypted)
- ✅ firewall-acl-agent receiving, processing, updating IPTables
- ✅ No errors in compression/decompression
- ✅ No errors in encryption/decryption
- ✅ RAG logs readable (not encrypted) for FAISS

### 4. Pipeline Coherence Validation
**Critical rule:** Encryption/compression settings must be coherent across pipeline

**Invalid configurations:**
- ❌ sniffer encrypts but ml-detector doesn't decrypt
- ❌ ml-detector compresses but firewall doesn't decompress
- ❌ Inconsistent algorithms (sniffer:lz4 but ml-detector:zstd)

**Valid configurations:**
- ✅ All components: encryption ON, compression ON (production)
- ✅ All components: encryption OFF, compression OFF (development)
- ✅ Mixed: encryption ON, compression OFF (debugging)

**Future RAG integration:**
- RAG will enforce coherence rules
- Warn users about invalid combinations
- Suggest fixes for broken pipelines

## Important Notes

### RAG Logger Behavior
- **ml-detector:** Saves unencrypted logs to `/vagrant/logs/rag/events/` for FAISS
- **firewall-acl-agent:** Verify if it also logs to RAG (check implementation)
- **sniffer:** No RAG logging (just capture and forward)

### Firewall Special Case
- Firewall is the pipeline endpoint
- Input: **Must** decrypt + decompress
- Output: **No need** to encrypt + compress (no downstream consumer)
- Can still log to RAG (unencrypted for FAISS)

### Performance Expectations
- No degradation from encryption/compression
- Latency targets maintained:
   - sniffer capture: <100μs
   - ml-detector inference: <1ms per event
   - firewall IPTables update: <500μs

## Deliverables
1. ✅ All 3 components compiled and integrated
2. ✅ etcd-server in root Makefile
3. ✅ Stress test passed (sustained load for 10+ minutes)
4. ✅ RAG logs verified (readable, unencrypted)
5. ✅ Pipeline coherence documented
6. 📝 Script for automated integration testing

## Success Metrics
- **Uptime:** All components running for 10+ minutes without crashes
- **Heartbeats:** 100% delivery rate (no missed heartbeats)
- **Throughput:** Sustained processing of test traffic
- **Logs:** RAG logs complete and readable for FAISS indexing
- **Zero errors:** No encryption/compression failures

---

**Via Appia Quality reminder:** Funciona > Perfecto. Get the pipeline working, then optimize.
```
