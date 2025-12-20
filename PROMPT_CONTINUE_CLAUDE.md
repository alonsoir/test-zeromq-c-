# Day 21 Context - ML Defender Encrypted Component Integration

## What We Completed Yesterday (Day 20)

Successfully integrated the **sniffer** component with the etcd-client library using a PIMPL adapter pattern:

### Achievements
1. **Zero Breaking Changes** - Maintained main.cpp compatibility through adapter
2. **Complete Config Upload** - Full 17,391-byte sniffer.json uploaded (no filtering)
3. **End-to-End Encryption** - ChaCha20-Poly1305 working: 17391 → 8569 → 8609 bytes
4. **Bug Fixes**:
   - Fixed `config_types.cpp` missing `etcd.enabled` mapping
   - Fixed `config_types.h` endpoint vs endpoints[] discrepancy
   - Fixed etcd-server validation to accept JSON objects (not just strings)

### Files Modified (Day 20)
- `/vagrant/sniffer/include/etcd_client.hpp` - PIMPL adapter header
- `/vagrant/sniffer/src/userspace/etcd_client.cpp` - Adapter implementation
- `/vagrant/sniffer/src/userspace/main.cpp` - Integration code (lines ~305-330)
- `/vagrant/sniffer/CMakeLists.txt` - etcd-client library linkage
- `/vagrant/sniffer/include/config_types.h` - Changed `endpoint` to `endpoints[]`
- `/vagrant/sniffer/src/userspace/config_types.cpp` - Added etcd mapping
- `/vagrant/etcd-server/src/component_registry.cpp` - Fixed validation

### Current Pipeline Status
```
✅ RAG → etcd-client → etcd-server (Day 19)
✅ Sniffer → etcd-client → etcd-server (Day 20)
⏳ ml-detector → etcd-client → etcd-server (Day 21)
⏳ firewall → etcd-client → etcd-server (Day 21)
⏳ Heartbeat endpoint (Day 21)
```

## Today's Goals (Day 21)

### Priority 1: ml-detector Integration (3-4 hours)
**Goal:** Same PIMPL adapter pattern as sniffer

**Files to Create:**
1. `/vagrant/ml-detector/include/etcd_client.hpp`
2. `/vagrant/ml-detector/src/etcd_client.cpp`

**Files to Modify:**
1. `/vagrant/ml-detector/src/main.cpp` - Add integration after config load
2. `/vagrant/ml-detector/CMakeLists.txt` - Link etcd-client library

**Integration Pattern (copy from sniffer):**
```cpp
// In main.cpp, after config loading:
std::unique_ptr<mldetector::EtcdClient> etcd_client;

if (config.etcd.enabled) {
    std::string etcd_endpoint = config.etcd.endpoints[0];
    etcd_client = std::make_unique<mldetector::EtcdClient>(etcd_endpoint, "ml-detector");
    
    if (!etcd_client->initialize()) {
        std::cerr << "⚠️  [etcd] Failed to initialize" << std::endl;
        etcd_client.reset();
    } else if (!etcd_client->registerService()) {
        std::cerr << "⚠️  [etcd] Failed to register" << std::endl;
        etcd_client.reset();
    } else {
        std::cout << "✅ [etcd] ml-detector registered" << std::endl;
    }
}
```

**Config File:** `/vagrant/ml-detector/config/ml-detector.json`
- Add `"etcd": {"enabled": true, "endpoints": ["localhost:2379"]}`

### Priority 2: firewall Integration (2-3 hours)
Same pattern as ml-detector, different component name.

**Files to Create:**
1. `/vagrant/firewall/include/etcd_client.hpp`
2. `/vagrant/firewall/src/etcd_client.cpp`

**Files to Modify:**
1. `/vagrant/firewall/src/main.cpp`
2. `/vagrant/firewall/CMakeLists.txt`

### Priority 3: Heartbeat Endpoint (2-3 hours)
**Current Issue:** Sniffer shows `⚠️ HTTP POST failed: 404` on heartbeat

**Implementation:**
1. Add `POST /v1/heartbeat/:component_name` endpoint to etcd-server
2. Update component's `last_heartbeat` timestamp
3. Mark component as `status: active` or `inactive` based on timeout
4. Return component status in `/components` endpoint

**Files to Modify:**
- `/vagrant/etcd-server/src/etcd_server.cpp` - Add heartbeat endpoint
- `/vagrant/etcd-server/src/component_registry.cpp` - Add heartbeat() method

**Endpoint Behavior:**
```cpp
// POST /v1/heartbeat/sniffer
{
  "timestamp": 1766225024,
  "status": "active"
}

// Response:
{
  "status": "ok",
  "last_heartbeat": 1766225024,
  "next_heartbeat_expected": 1766225054  // +30s
}
```

## Key Technical Details

### PIMPL Adapter Pattern
```cpp
// Header (include/etcd_client.hpp)
namespace component {
    class EtcdClient {
    public:
        EtcdClient(const std::string& endpoint, const std::string& component_name);
        ~EtcdClient();
        bool initialize();
        bool registerService();
    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };
}

// Implementation (src/etcd_client.cpp)
struct EtcdClient::Impl {
    std::unique_ptr<etcd_client::EtcdClient> client_;
    std::string component_name_;
    // Impl details...
};
```

### CMakeLists.txt Pattern
```cmake
# Find etcd-client library
set(ETCD_CLIENT_LIB /vagrant/etcd-client/build/libetcd_client.so)
set(ETCD_CLIENT_INCLUDE_DIR /vagrant/etcd-client/include)

if(EXISTS ${ETCD_CLIENT_LIB})
    message(STATUS "✅ Found etcd-client library")
    include_directories(${ETCD_CLIENT_INCLUDE_DIR})
    list(APPEND COMPONENT_LIBRARIES ${ETCD_CLIENT_LIB})
endif()
```

## Testing Strategy

### Test 1: ml-detector Registration
```bash
# Terminal 1: etcd-server
cd /vagrant/etcd-server/build && ./etcd-server --port 2379

# Terminal 2: ml-detector
cd /vagrant/ml-detector/build && ./ml-detector -c ../config/ml-detector.json

# Expected:
# ✅ [etcd] ml-detector registered and config uploaded
# 🔐 [etcd] Config encrypted with ChaCha20-Poly1305

# Verify:
curl http://localhost:2379/components | jq
```

### Test 2: firewall Registration
Same as ml-detector, different component.

### Test 3: Heartbeat
```bash
# Should see periodic heartbeats in etcd-server logs
# No more 404 errors in component logs
```

### Test 4: Full Pipeline
```bash
# All components registered:
curl http://localhost:2379/components | jq
# Expected: sniffer, ml-detector, firewall, rag all listed
```

## Known Issues to Watch

1. **Config Structure Differences** - Each component has different JSON schemas
2. **Namespace Conflicts** - Use `component::EtcdClient` not `sniffer::EtcdClient`
3. **Library Path** - May need `LD_LIBRARY_PATH=/vagrant/etcd-client/build`
4. **Heartbeat Thread** - Already running in sniffer, just needs endpoint

## Success Criteria

✅ ml-detector compiles and links with etcd-client  
✅ ml-detector registers with etcd-server  
✅ ml-detector uploads config encrypted  
✅ firewall compiles and links with etcd-client  
✅ firewall registers with etcd-server  
✅ firewall uploads config encrypted  
✅ Heartbeat endpoint implemented  
✅ No more 404 errors in logs  
✅ All 4 components show in `/components` endpoint

## Progress Target
- **Start:** 92%
- **End:** 98% (if all 3 priorities complete)

## Via Appia Quality Reminder
- **Funciona > Perfecto** - Get it working, then refine
- **Scientific Honesty** - Document what works and what doesn't
- **KISS** - Copy the sniffer pattern, don't reinvent

---

Good luck with Day 21! The pattern is proven from Day 20, so this should be smoother. Focus on ml-detector first (highest priority for the pipeline), then firewall, then heartbeat.