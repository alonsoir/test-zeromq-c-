# 🚀 Day 19 - RAG Integration with etcd-client (Continuity Prompt)

**Date:** December 19, 2025  
**Status:** Ready to begin RAG integration  
**Context:** Day 18 completed - Bidirectional config management with ChaCha20+LZ4 working perfectly

---

## 📋 What We Accomplished on Day 18

### ✅ **Part 1: PUT Endpoint Implementation**

**Files Modified:**
- `/vagrant/etcd-client/src/http_client.cpp` - Added `put()` function with retry logic
- `/vagrant/etcd-client/src/etcd_client.cpp` - Added `put_config()` method
- `/vagrant/etcd-server/src/etcd_server.cpp` - Added `PUT /v1/config/:id` endpoint

### ✅ **Part 2: Encryption & Compression Integration**

**Major Changes:**
1. **Migrated server from AES-CBC to ChaCha20-Poly1305**
   - Modified `/vagrant/etcd-server/src/crypto_manager.cpp`
   - Updated header `/vagrant/etcd-server/include/etcd_server/crypto_manager.hpp`
   - Same algorithm as client for compatibility

2. **Added LZ4 compression to server**
   - Created `/vagrant/etcd-server/src/compression_lz4.cpp`
   - Created `/vagrant/etcd-server/include/etcd_server/compression_lz4.hpp`
   - Updated `/vagrant/etcd-server/CMakeLists.txt`

3. **Automatic encryption key exchange**
   - Server returns derived key on `/register`
   - Client automatically receives and uses key
   - Modified `register_component()` in both client and server

4. **Added X-Original-Size header**
   - Client sends original size before compression
   - Server uses it for proper LZ4 decompression
   - Modified `http::put()` signature to accept `original_size`

---

## 🎯 Current System State

### **Architecture Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT (ml-detector, rag, sniffer)                             │
│                                                                 │
│  1. connect() → POST /register                                  │
│     ← Receives encryption_key (32 bytes, hex-encoded)          │
│     ← Key converted from hex string to binary                  │
│                                                                 │
│  2. put_config(json_string)                                     │
│     → Validate JSON                                             │
│     → Compress with LZ4 (362B → 217B, 40% reduction)           │
│     → Encrypt with ChaCha20 (217B → 257B)                      │
│     → HTTP PUT with X-Original-Size: 362                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP PUT /v1/config/{component_id}
                              │ Content-Type: application/octet-stream
                              │ X-Original-Size: 362
                              │ Body: 257 bytes (encrypted)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ETCD-SERVER                                                     │
│                                                                 │
│  PUT /v1/config/{component_id}                                  │
│    1. Check Content-Type                                        │
│    2. If octet-stream: decrypt_data() → ChaCha20                │
│       (257B → 217B)                                             │
│    3. Check X-Original-Size header                              │
│    4. If present: decompress_lz4(217B, 362) → 362B              │
│    5. Parse JSON                                                │
│    6. register_component(component_id, json)                    │
│    7. Return 200 OK with metadata                               │
└─────────────────────────────────────────────────────────────────┘
```

### **Performance Metrics**

```
Original JSON:     362 bytes
After LZ4:         217 bytes (40% reduction)
After ChaCha20:    257 bytes (nonce + ciphertext + MAC)
Network transfer:  257 bytes

Decompression:     217B → 362B ✅
Decryption:        257B → 217B ✅
Total time:        < 50ms
```

---

## 📂 Key Files Modified (Day 18)

### **etcd-client/**
```
src/
├── etcd_client.cpp              ✅ put_config(), connect() with key exchange
├── http_client.cpp              ✅ put() with X-Original-Size header
└── crypto_chacha20.cpp          ✅ hex_to_bytes() helper

include/etcd_client/
└── etcd_client.hpp              ✅ Method signatures updated
```

### **etcd-server/**
```
src/
├── etcd_server.cpp              ✅ PUT endpoint with decrypt+decompress
├── crypto_manager.cpp           ✅ ChaCha20 (migrated from AES-CBC)
├── compression_lz4.cpp          ✅ LZ4 decompression
└── component_registry.cpp       ✅ get_encryption_key() method

include/etcd_server/
├── crypto_manager.hpp           ✅ Updated for ChaCha20
├── compression_lz4.hpp          ✅ New header
└── component_registry.hpp       ✅ get_encryption_key() declaration

CMakeLists.txt                   ✅ Added LZ4 dependency
```

### **Tests/**
```
etcd-client/tests/
└── test_put_config_integration.cpp  ✅ Full integration test passing
```

---

## ✅ Working Test Validation

**Test Status:** ALL PASSING ✅

```bash
# Start server
cd /vagrant/etcd-server/build
./etcd-server --port 2379

# Expected output:
# [CRYPTO] 🔑 Clave derivada con HKDF desde seed
# [CRYPTO]   Key: XXXXXXXX...
# 🚀 Servidor HTTP escuchando en: http://0.0.0.0:2379

# Run test
cd /vagrant/etcd-client/tests
export LD_LIBRARY_PATH=/vagrant/etcd-client/build:$LD_LIBRARY_PATH
./test_put_config_integration

# Expected output:
# ✅ Connected to etcd-server
# 🔑 Encryption key received from server (32 bytes)
# 📦 Compressed: 362 → 217 bytes
# 🔒 Encrypted: 257 bytes
# ✅ Config uploaded successfully!
# ✅ SUCCESS: Config uploaded with ChaCha20 encryption!
```

---

## 🎯 Day 19 Objectives

### **Goal: Integrate RAG with etcd-client**

RAG (Retrieval-Augmented Generation) needs to:
1. Register with etcd-server using the new `EtcdClient` library
2. Upload its configuration via `put_config()`
3. Fetch ML Defender configurations from etcd-server
4. Subscribe to configuration updates (watcher pattern)

### **Phase 1: Basic Integration**

**Step 1:** Link RAG with `etcd-client` library
- Modify RAG's CMakeLists.txt
- Add etcd-client as dependency
- Include headers

**Step 2:** Replace ZeroMQ registration with etcd-client
- Remove old ZeroMQ registration code
- Use `EtcdClient::connect()` and `register_component()`
- Automatic encryption key exchange

**Step 3:** Use `put_config()` for RAG configuration
- Create RAG config JSON structure
- Call `put_config()` on startup
- Validate upload success

**Step 4:** Fetch ML Defender config from etcd-server
- Use `get_config()` to fetch ml-detector configuration
- Parse JSON
- Apply to RAG behavior

---

## 📋 Implementation Plan

### **Task 1: Modify RAG CMakeLists.txt** (15 min)

**File:** `/vagrant/rag/CMakeLists.txt`

Add:
```cmake
# Find etcd-client library
find_library(ETCD_CLIENT_LIB
    NAMES etcd_client
    PATHS /vagrant/etcd-client/build
    REQUIRED
)

# Add include directories
include_directories(/vagrant/etcd-client/include)

# Link etcd-client
target_link_libraries(rag
    PRIVATE
    ${ETCD_CLIENT_LIB}
    # ... existing libs
)
```

---

### **Task 2: Create RAGConfig class** (20 min)

**New File:** `/vagrant/rag/include/rag_config.hpp`

```cpp
#pragma once
#include <string>
#include <nlohmann/json.hpp>

class RAGConfig {
public:
    // RAG configuration
    std::string component_name = "rag-logger";
    std::string llm_model_path = "/models/tinyllama.bin";
    std::string faiss_index_path = "/data/faiss/index.bin";
    size_t max_context_length = 2048;
    bool enabled = true;
    
    // etcd-server connection
    std::string etcd_host = "localhost";
    int etcd_port = 2379;
    
    // Serialize to JSON
    nlohmann::json to_json() const;
    
    // Deserialize from JSON
    static RAGConfig from_json(const std::string& json_str);
};
```

---

### **Task 3: Integrate EtcdClient in RAG** (30 min)

**Modify:** `/vagrant/rag/src/main.cpp`

```cpp
#include "etcd_client/etcd_client.hpp"
#include "rag_config.hpp"

int main() {
    std::cout << "🚀 Iniciando RAG Logger..." << std::endl;
    
    // 1. Load RAG config
    RAGConfig rag_config;
    
    // 2. Configure etcd-client
    etcd_client::Config etcd_config;
    etcd_config.component_name = rag_config.component_name;
    etcd_config.host = rag_config.etcd_host;
    etcd_config.port = rag_config.etcd_port;
    etcd_config.encryption_enabled = true;
    etcd_config.compression_enabled = true;
    
    // 3. Create client and connect
    etcd_client::EtcdClient client(etcd_config);
    
    if (!client.connect()) {
        std::cerr << "❌ Failed to connect to etcd-server" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Connected to etcd-server" << std::endl;
    std::cout << "🔑 Encryption key received automatically" << std::endl;
    
    // 4. Upload RAG configuration
    std::string config_json = rag_config.to_json().dump(2);
    if (!client.put_config(config_json)) {
        std::cerr << "❌ Failed to upload RAG config" << std::endl;
        return 1;
    }
    
    std::cout << "✅ RAG configuration uploaded" << std::endl;
    
    // 5. Fetch ML Defender configuration
    std::string ml_config = client.get_config("ml-detector");
    if (!ml_config.empty()) {
        std::cout << "✅ ML Defender config received" << std::endl;
        auto ml_json = nlohmann::json::parse(ml_config);
        // Apply configuration...
    }
    
    // 6. Start RAG main loop
    // ...
    
    return 0;
}
```

---

### **Task 4: Testing** (20 min)

**Test Checklist:**
- [ ] RAG compiles with etcd-client
- [ ] RAG connects to etcd-server
- [ ] RAG receives encryption key automatically
- [ ] RAG uploads config successfully
- [ ] RAG fetches ML Defender config
- [ ] Encrypted communication works end-to-end

---

## 🧪 Testing Commands

```bash
# Terminal 1: Start etcd-server
cd /vagrant/etcd-server/build
./etcd-server --port 2379

# Terminal 2: Compile RAG
cd /vagrant/rag/build
cmake ..
make -j$(nproc)

# Terminal 3: Run RAG
cd /vagrant/rag/build
./rag

# Expected output:
# 🚀 Iniciando RAG Logger...
# 🔗 Connecting to etcd-server: localhost:2379
# ✅ Connected to etcd-server
# 🔑 Encryption key received from server (32 bytes)
# ✅ Component registered: rag-logger
# 📤 Uploading config...
# ✅ RAG configuration uploaded
# ✅ ML Defender config received
```

---

## 🔍 Troubleshooting Guide

### **Issue: RAG can't find etcd_client library**
```bash
# Check if library exists
ls -lh /vagrant/etcd-client/build/libetcd_client.so*

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/vagrant/etcd-client/build:$LD_LIBRARY_PATH
```

### **Issue: Compilation errors with etcd-client headers**
```bash
# Verify include path
ls /vagrant/etcd-client/include/etcd_client/

# Check CMakeLists.txt has correct path
grep "include_directories" /vagrant/rag/CMakeLists.txt
```

### **Issue: Connection refused**
```bash
# Check server is running
ps aux | grep etcd-server

# Check port
netstat -tlnp | grep 2379

# Restart server
cd /vagrant/etcd-server/build
./etcd-server --port 2379
```

---

## 📊 Success Criteria for Day 19

- [ ] RAG successfully links with etcd-client library
- [ ] RAG connects to etcd-server without errors
- [ ] Automatic encryption key exchange works
- [ ] RAG uploads configuration with ChaCha20+LZ4
- [ ] RAG fetches ML Defender configuration
- [ ] All communication encrypted and compressed
- [ ] Performance: <100ms for config operations
- [ ] Code compiles without warnings
- [ ] Tests pass consistently

---

## 🚀 Future Tasks (Post Day 19)

### **Day 20: FAISS Semantic Search**
- Integrate FAISS with RAG
- Vector embeddings for logs
- Similarity search queries

### **Day 21: Unified Watcher Library**
- Configuration change notifications
- Real-time updates
- Pub/sub pattern implementation

### **Day 22: Production Hardening**
- Error recovery
- Graceful degradation
- Monitoring and metrics

---

## 💡 Important Notes

1. **Encryption is automatic:** No need to manually set keys, server provides them on registration
2. **Compression is transparent:** Library handles it automatically if enabled
3. **Thread-safe:** All etcd-client operations are mutex-protected
4. **Heartbeat optional:** Can be enabled/disabled via config
5. **Memory efficient:** ChaCha20 + LZ4 use minimal memory overhead

---

## 🎉 Day 18 Summary

**What worked perfectly:**
- ✅ Bidirectional config management
- ✅ ChaCha20-Poly1305 encryption
- ✅ LZ4 compression (40% size reduction)
- ✅ Automatic key exchange
- ✅ Production-ready performance

**Key learnings:**
- Always match encryption algorithms (client & server)
- Send metadata (like original_size) via HTTP headers
- Test end-to-end before committing
- Via Appia Quality: functional > perfect

---

**Via Appia Quality** - Built to last decades! 🛡️

*Generated: December 18, 2025*  
*Status: Ready for Day 19 - RAG Integration*  
*Estimated Time: 1.5 - 2 hours*

---

¿Listo para Day 19, Alonso? 🚀