cat > docs/DAY_18_CONTINUITY_PROMPT.md << 'EOF'
# Day 18 Continuity Prompt - RAG Integration with etcd-client Library

## 📊 Current Status (End of Day 17)

### ✅ Day 17 Achievements - etcd-client Library Complete

**Library Created (1,238 lines C++20):**
- `etcd-client/` directory structure
- `include/etcd_client/etcd_client.hpp` - Complete API (145 lines)
- `src/config_loader.cpp` - JSON config loading (110 lines)
- `src/compression_lz4.cpp` - LZ4 compression (82 lines)
- `src/crypto_chacha20.cpp` - ChaCha20-Poly1305 encryption (142 lines)
- `src/http_client.cpp` - HTTP wrapper with retry (178 lines)
- `src/component_registration.cpp` - Component discovery (119 lines)
- `src/etcd_client.cpp` - Main PIMPL implementation (607 lines)

**Tests Validated (515 lines, 100% passing):**
- `tests/test_compression.cpp` - LZ4 validation (136 lines)
- `tests/test_encryption.cpp` - ChaCha20 validation (202 lines)
- `tests/test_pipeline.cpp` - Complete pipeline (177 lines)
- CTest results: 3/3 passed (0.05 seconds)

**Performance Metrics:**
- Compression: 10KB → 59 bytes (0.59%)
- Encryption overhead: +40 bytes fixed (nonce + MAC)
- Pipeline: 100KB → 452 bytes (0.452% total size!)
- JSON config: 535 → 460 bytes (86% efficiency)

**Compilation:**
- Library: `libetcd_client.so.1.0.0` (1.1 MB)
- Compiler: g++ 12.2.0 with -std=c++20
- Dependencies: libsodium 1.0.18, liblz4 1.9.4
- Status: Zero warnings, zero errors

**Security Design:**
- ChaCha20-Poly1305 (TLS 1.3 standard)
- Key management: etcd-server generates (hardware RNG)
- Mutual TLS (mTLS) roadmap documented (Phase 2B)
- HSM integration planned (Phase 3)

---

## 🎯 Day 18 Objectives - RAG Integration

### **Goal:** Replace RAG's custom etcd_client with shared library

### **Priority 1: Update RAG Build System (1-2 hours)**

**Files to modify:**
1. `rag/CMakeLists.txt` - Add etcd-client library dependency
2. `rag/src/etcd_client.cpp` - REMOVE (replaced by library)
3. `rag/include/rag/etcd_client.hpp` - REMOVE (use library header)

**Changes in rag/CMakeLists.txt:**
```cmake
# Add etcd-client library
add_subdirectory(../etcd-client etcd-client)

target_link_libraries(rag-system
    PRIVATE
        etcd_client  # NEW: Shared library
        # ... other libs
)

target_include_directories(rag-system
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../etcd-client/include  # NEW
        # ... other includes
)
```

**Remove old files:**
- Delete `rag/src/etcd_client.cpp` (old stub implementation)
- Delete `rag/include/rag/etcd_client.hpp` (old header)

---

### **Priority 2: Update RAG Config Format (30 min)**

**Current format (rag/config/rag-config.json):**
```json
{
  "rag": {
    "host": "0.0.0.0",
    "port": 8080,
    "model_name": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    "embedding_dimension": 768
  },
  "etcd": {
    "host": "127.0.0.1",
    "port": 2379
  }
}
```

**New format (use etcd-client standard):**
```json
{
  "rag": {
    "host": "0.0.0.0",
    "port": 8080,
    "model_name": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    "embedding_dimension": 768
  },
  "etcd_client": {
    "server": {
      "host": "127.0.0.1",
      "port": 2379,
      "timeout_seconds": 5
    },
    "component": {
      "name": "rag-engine",
      "config_path": "/vagrant/rag/config/rag-config.json"
    },
    "encryption": {
      "enabled": true,
      "algorithm": "chacha20-poly1305",
      "key_source": "etcd-server"
    },
    "compression": {
      "enabled": true,
      "algorithm": "lz4",
      "min_size_bytes": 256
    },
    "heartbeat": {
      "enabled": true,
      "interval_seconds": 30
    },
    "retry": {
      "max_attempts": 3,
      "backoff_seconds": 2
    }
  }
}
```

---

### **Priority 3: Update RAG Code (2-3 hours)**

**Files to modify:**
1. `rag/src/rag_command_manager.cpp` - Update etcd client usage
2. `rag/src/main.cpp` - Update initialization

**Changes in rag_command_manager.cpp:**
```cpp
// OLD (remove):
#include "rag/etcd_client.hpp"
Rag::EtcdClient etcd_client(endpoint, "rag-engine");

// NEW (use library):
#include <etcd_client/etcd_client.hpp>
etcd_client::EtcdClient etcd_client(config_json_path);

// Then use same API:
etcd_client.connect();
etcd_client.register_component();
// ... etc
```

**Key API changes:**
- Constructor: `EtcdClient(config_path)` instead of `EtcdClient(endpoint, name)`
- Everything else: SAME API (designed to be drop-in replacement)

---

### **Priority 4: Test Integration (1 hour)**

**Test sequence:**
```bash
# 1. Compile RAG with new library
cd /vagrant
make rag

# 2. Start etcd-server
vagrant ssh defender -c "cd /vagrant/etcd-server && ./etcd-server &"

# 3. Start RAG
vagrant ssh defender -c "cd /vagrant/rag && ./rag-system &"

# 4. Verify registration
# Should see: "✅ Component registered: rag-engine"

# 5. Check heartbeat
# Wait 30s, verify heartbeat logs

# 6. Test commands
curl -X POST http://localhost:8080/command \
  -H "Content-Type: application/json" \
  -d '{"command": "get_pipeline_status"}'

# 7. Verify encryption
# etcd-server logs should show encrypted payloads
# RAG should decrypt successfully
```

---

## 📋 Key Files Reference

### **Library Files (read-only for Day 18):**
- `etcd-client/include/etcd_client/etcd_client.hpp` - API reference
- `etcd-client/config/etcd_client_config.json` - Example config
- `etcd-client/README.md` - Library documentation

### **RAG Files (to modify):**
- `rag/CMakeLists.txt` - Build system
- `rag/config/rag-config.json` - Config format
- `rag/src/rag_command_manager.cpp` - Main usage
- `rag/src/main.cpp` - Initialization
- `rag/src/etcd_client.cpp` - DELETE THIS
- `rag/include/rag/etcd_client.hpp` - DELETE THIS

### **Test Files:**
- `rag/test/` - Integration tests (if needed)

---

## 🔧 Technical Notes

### **API Compatibility:**

**OLD RAG API:**
```cpp
EtcdClient(const std::string& endpoint, const std::string& component_name);
bool registerService();
bool unregisterService();
```

**NEW Library API:**
```cpp
EtcdClient(const std::string& config_json_path);
bool register_component();
bool unregister_component();
```

**Migration strategy:**
- Replace constructor call
- Replace `registerService()` → `register_component()`
- Replace `unregisterService()` → `unregister_component()`
- Everything else stays same

### **Config Loading:**

**OLD:**
```cpp
auto& config_manager = ConfigManager::getInstance();
auto etcd_config = config_manager.getEtcdConfig();
std::string endpoint = etcd_config.host + ":" + std::to_string(etcd_config.port);
EtcdClient client(endpoint, "rag-engine");
```

**NEW:**
```cpp
EtcdClient client("/vagrant/rag/config/rag-config.json");
```

Much simpler!

### **Heartbeat:**

**OLD:** Manual heartbeat implementation (if any)

**NEW:** Automatic! Library starts heartbeat thread on `register_component()`
- Interval: 30 seconds (configurable in JSON)
- Automatic in background
- Stops on `unregister_component()` or destructor

### **Error Handling:**

**Library throws exceptions on critical errors:**
```cpp
try {
    etcd_client.connect();
    etcd_client.register_component();
} catch (const std::exception& e) {
    std::cerr << "Failed to connect to etcd: " << e.what() << std::endl;
    return 1;
}
```

---

## ⚠️ Potential Issues & Solutions

### **Issue 1: ConfigManager dependency**

**Problem:** RAG might have ConfigManager that reads old format

**Solution:**
- Keep ConfigManager for RAG-specific config
- Use etcd-client Config for etcd settings
- Two separate configs, no conflict

### **Issue 2: Linking errors**

**Problem:** Library not found during linking

**Solution:**
```cmake
# In rag/CMakeLists.txt
link_directories(${CMAKE_SOURCE_DIR}/../etcd-client/build)
```

### **Issue 3: Include paths**

**Problem:** Headers not found

**Solution:**
```cmake
target_include_directories(rag-system
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../etcd-client/include
)
```

### **Issue 4: Namespace conflicts**

**Problem:** Old `Rag::EtcdClient` vs new `etcd_client::EtcdClient`

**Solution:**
```cpp
// Use fully qualified names
etcd_client::EtcdClient client(...);

// Or namespace alias
namespace ec = etcd_client;
ec::EtcdClient client(...);
```

---

## 📊 Success Criteria for Day 18

### **Minimum (Must Achieve):**
- ✅ RAG compiles with etcd-client library
- ✅ Old etcd_client.cpp removed
- ✅ Config format updated
- ✅ Registration works (see logs)

### **Target (Should Achieve):**
- ✅ All of above +
- ✅ Heartbeat working (30s interval)
- ✅ Commands work through RAG
- ✅ Encryption validated (etcd-server logs)

### **Stretch (If Time Permits):**
- ✅ All of above +
- ✅ Other components (ml-detector, sniffer, firewall)
- ✅ End-to-end encrypted pipeline
- ✅ Performance benchmarks

---

## 🎯 Day 18 Timeline (Estimated)
```
Morning (2-3 hours):
  [30 min] Review library API and example config
  [60 min] Update RAG CMakeLists.txt
  [30 min] Update rag-config.json format
  [60 min] Update rag_command_manager.cpp

Afternoon (2-3 hours):
  [30 min] Remove old etcd_client files
  [60 min] Compile and fix errors
  [60 min] Test with etcd-server
  [30 min] Validate encryption/compression

Evening (optional):
  [60 min] Start ml-detector integration
  [30 min] Documentation updates
  [30 min] Commit and push
```

---

## 📝 Commands for Day 18
```bash
# Start fresh
cd /vagrant
git checkout feature/etcd-client-lib
git pull

# Review library
cat etcd-client/include/etcd_client/etcd_client.hpp
cat etcd-client/config/etcd_client_config.json

# Backup old RAG etcd_client
cp rag/src/etcd_client.cpp rag/src/etcd_client.cpp.OLD
cp rag/include/rag/etcd_client.hpp rag/include/rag/etcd_client.hpp.OLD

# Edit files
vim rag/CMakeLists.txt
vim rag/config/rag-config.json
vim rag/src/rag_command_manager.cpp

# Compile
make rag

# Test
vagrant ssh defender -c "cd /vagrant/etcd-server && ./etcd-server &"
vagrant ssh defender -c "cd /vagrant/rag && ./rag-system"

# Verify
curl http://localhost:8080/command -X POST -d '{"command":"get_pipeline_status"}'

# Commit
git add rag/
git commit -m "feat(rag): Integrate etcd-client library"
git push
```

---

## 🔐 Security Reminders

**Key Management (Day 17 decision):**
- ❌ NO custom keys in config (security anti-pattern)
- ✅ etcd-server generates keys (hardware RNG: /dev/random)
- ✅ Keys distributed via registration handshake
- ✅ Keys stored ONLY in memory (never disk)
- ✅ mTLS planned for Phase 2B (client certificates)

**For Day 18:**
- Focus on HTTP (unencrypted) first
- TLS/mTLS postponed to Phase 2B
- Current: Key distributed in plaintext HTTP (dev only)
- Production: Will use TLS (Phase 2B)

---

## 💡 Tips for Success

1. **Read library docs first** - `etcd-client/README.md`
2. **Use example config** - Copy from `etcd-client/config/`
3. **Test incrementally** - Compile after each change
4. **Keep old files as .OLD** - Easy rollback if needed
5. **Check etcd-server logs** - Verify registration
6. **Verify encryption** - Check payload sizes in logs
7. **Ask questions** - If unclear, ask before coding

---

## 📚 References

- [etcd-client README](../etcd-client/README.md)
- [etcd-client API](../etcd-client/include/etcd_client/etcd_client.hpp)
- [Example Config](../etcd-client/config/etcd_client_config.json)
- [Test Examples](../etcd-client/tests/)
- [Day 17 Summary](./DAY_17_ETCD_CLIENT_LIBRARY.md)
- [Security Roadmap](./SECURITY_ROADMAP.md)

---

## 🎉 Expected End State (Day 18)
```
✅ RAG using shared etcd-client library
✅ Old custom etcd_client code removed
✅ Config format standardized
✅ Registration working
✅ Heartbeat automatic
✅ Encryption validated
✅ Compression validated
✅ Ready for other components (Day 19)

Next Day 19:
  → ml-detector integration
  → sniffer integration
  → firewall integration
  → End-to-end encrypted pipeline
```

---

**Good luck with Day 18!** 🚀

*Via Appia Quality - One solid stone at a time*
EOF