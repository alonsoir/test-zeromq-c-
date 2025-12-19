# Day 20-22 Continuity Prompt - Component Integration + Heartbeat + Quorum

**Date:** December 20-22, 2025
**Status:** RAG integrated, ready for full pipeline encryption
**Progress:** 82% → 92% (target)

---

## ✅ What We Completed (Days 18-19)

### Day 18: Bidirectional Config Management
- ✅ PUT endpoint with encryption + compression
- ✅ Server migrated from AES-CBC to ChaCha20-Poly1305
- ✅ LZ4 decompression on server
- ✅ Automatic encryption key exchange
- ✅ X-Original-Size header protocol
- ✅ Intelligent compression detection

### Day 19: RAG Integration
- ✅ RAG uses etcd-client library via adapter pattern
- ✅ Zero changes to main.cpp (backward compatible)
- ✅ Automatic encryption key exchange working
- ✅ Config upload/retrieval encrypted
- ✅ Connection time: <100ms
- ✅ Smart compression (only when beneficial)

---

## 🎯 Goals for Days 20-22

### Day 20: Component Integration (~6-8 hours)
**Integrate etcd-client in remaining components:**

1. **ml-detector integration** (~2-3 hours)
   - Link etcd-client library
   - Register on startup
   - Upload detector config
   - Fetch global config
   - Test encrypted communication

2. **sniffer integration** (~2 hours)
   - Link etcd-client library
   - Register on startup
   - Upload sniffer config
   - Test eBPF + encryption

3. **firewall integration** (~2 hours)
   - Link etcd-client library
   - Register on startup
   - Fetch blocking rules from etcd
   - Test end-to-end encrypted pipeline

**Success Criteria:**
- [ ] All components register with etcd-server
- [ ] All communication encrypted with ChaCha20
- [ ] Config distribution working
- [ ] End-to-end test passing

---

### Day 21: Heartbeat Implementation (~3-4 hours)

1. **Server-side heartbeat endpoint** (~1 hour)
   - POST /heartbeat endpoint in etcd-server
   - Store component health status
   - Timestamp tracking

2. **Client-side heartbeat** (~1 hour)
   - Already implemented in etcd-client library
   - Just enable in configs

3. **Health monitoring** (~1-2 hours)
   - GET /health endpoint improvements
   - Component status tracking
   - Dead component detection

**Success Criteria:**
- [ ] POST /heartbeat endpoint working
- [ ] All components sending heartbeats
- [ ] Health status visible via GET /health
- [ ] Stale component detection (>60s)

---

### Day 22: Basic Quorum (~4-6 hours)

**Simple quorum implementation:**

1. **Multi-instance support** (~2 hours)
   - etcd-server can start on different ports
   - Instance discovery via config
   - Peer-to-peer communication

2. **Leader election** (~2 hours)
   - Simple Raft-lite algorithm
   - Lowest port = leader (simple)
   - Or most recent timestamp

3. **Data replication** (~2 hours)
   - Leader broadcasts config changes
   - Followers sync on startup
   - Basic conflict resolution (last-write-wins)

**Success Criteria:**
- [ ] 3 etcd-server instances running
- [ ] Leader elected automatically
- [ ] Config replicated across instances
- [ ] Client connects to any instance
- [ ] Failover working (leader dies → new leader)

---

## 📁 Key Files to Modify

### Day 20 - Component Integration

**ml-detector:**
```
/vagrant/ml-detector/CMakeLists.txt       ← Add etcd-client
/vagrant/ml-detector/src/main.cpp         ← Register with etcd
/vagrant/ml-detector/config/*.json        ← Update config format
```

**sniffer:**
```
/vagrant/sniffer/CMakeLists.txt           ← Add etcd-client
/vagrant/sniffer/src/main.cpp             ← Register with etcd
/vagrant/sniffer/config/*.json            ← Update config format
```

**firewall:**
```
/vagrant/firewall/CMakeLists.txt          ← Add etcd-client
/vagrant/firewall/src/main.cpp            ← Register with etcd
```

### Day 21 - Heartbeat

**etcd-server:**
```
/vagrant/etcd-server/src/etcd_server.cpp  ← Add POST /heartbeat
/vagrant/etcd-server/src/component_registry.cpp ← Track heartbeats
```

### Day 22 - Quorum

**etcd-server:**
```
/vagrant/etcd-server/src/etcd_server.cpp     ← Multi-instance support
/vagrant/etcd-server/src/quorum_manager.cpp  ← New file (leader election)
/vagrant/etcd-server/src/replication.cpp     ← New file (data sync)
/vagrant/etcd-server/config/cluster.json     ← New file (peer config)
```

---

## 🧪 Testing Strategy

### Day 20 Tests
```bash
# Terminal 1: etcd-server
cd /vagrant/etcd-server/build && ./etcd-server --port 2379

# Terminal 2: ml-detector
cd /vagrant/ml-detector/build && ./ml-detector

# Terminal 3: sniffer
cd /vagrant/sniffer/build && sudo ./sniffer

# Terminal 4: firewall
cd /vagrant/firewall/build && ./firewall

# Verify: All components registered
curl http://localhost:2379/components | jq '.components[]'
```

### Day 21 Tests
```bash
# Verify heartbeats
watch -n 5 'curl -s http://localhost:2379/health | jq ".components"'

# Should show: last_heartbeat timestamps updating
```

### Day 22 Tests
```bash
# Start 3 instances
./etcd-server --port 2379 --peers "localhost:2380,localhost:2381" &
./etcd-server --port 2380 --peers "localhost:2379,localhost:2381" &
./etcd-server --port 2381 --peers "localhost:2379,localhost:2380" &

# Check leader
curl http://localhost:2379/status | jq '.leader'

# Kill leader, verify failover
kill $(pgrep -f "port 2379")
sleep 2
curl http://localhost:2380/status | jq '.leader'
```

---

## 💡 Implementation Tips

### Adapter Pattern (Like RAG Day 19)
For ml-detector/sniffer/firewall, use same pattern as RAG:
1. Keep existing code structure
2. Create thin adapter layer
3. Use etcd-client library internally
4. Zero breaking changes

### Heartbeat Simple Implementation
```cpp
// etcd-server: POST /heartbeat
server.Post("/heartbeat", [this](const Request& req, Response& res) {
    auto json_body = json::parse(req.body);
    std::string component = json_body["component"];
    component_registry_->update_heartbeat(component, time(nullptr));
    res.set_content(R"({"status":"ok"})", "application/json");
});
```

### Quorum Minimum Viable
- Use file-based sync initially (simple)
- Leader writes to `/tmp/etcd-data.json`
- Followers read every 5 seconds
- Upgrade to HTTP replication later

---

## 🎯 Success Metrics

**Day 20:**
- 4/4 components registered ✅
- 100% encrypted communication ✅
- Config distribution working ✅

**Day 21:**
- POST /heartbeat implemented ✅
- Heartbeats visible in logs ✅
- Health monitoring working ✅

**Day 22:**
- 3 instances running ✅
- Leader election working ✅
- Config replicated ✅
- Failover tested ✅

**Progress:** 82% → 92% 🚀

---

## 📝 Notes

**From Day 19:**
- Compression only happens when beneficial (smart!)
- Small configs (<100 bytes) → no compression
- Large configs (>100 bytes) → LZ4 compression
- Server detects based on size comparison

**Heartbeat Warning:**
- RAG shows "⚠️ HTTP POST failed: 404" for heartbeat
- This is expected - endpoint doesn't exist yet
- Non-critical, system works fine without it
- Will fix on Day 21

**Vagrantfile:**
- Probably doesn't need updates
- etcd-server can run multiple instances via CLI args
- No network changes needed

**Makefile:**
- May need targets for multi-instance etcd
- Update `make run-lab-dev` to start all with encryption

---

**Via Appia Quality** - Functional > Perfect 🛡️

*Ready for Days 20-22!*