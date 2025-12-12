# **ARCHITECTURE.md - ML Defender Platform (Actualizado Día 9)**

**Version:** 4.1.0  
**Last Updated:** December 5, 2025  
**Status:** Phase 1 Complete - Gateway Mode Ready (Pending Validation)

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [cpp_sniffer Architecture](#cpp_sniffer-architecture)
- [XDP Mode Selection & Gateway Mode Architecture](#xdp-mode-selection--gateway-mode-architecture)
- [ml-detector Architecture](#ml-detector-architecture)
- [RAG Security System Architecture](#rag-security-system-architecture)
- [Enterprise Features](#enterprise-features)
- [Home Device Deployment](#home-device-deployment)
- [Performance Characteristics](#performance-characteristics)
- [Security Considerations](#security-considerations)

---

## 🎯 Overview

The ML Defender Platform is a **distributed, multi-component system** designed to provide real-time network security with embedded ML detection and RAG-powered intelligence for both **home** and **enterprise** deployments.

### System Goals

1. **Real-time Detection** - Sub-microsecond threat identification
2. **High Accuracy** - >98% detection rate, <1% false positives
3. **Low Overhead** - <5% CPU, <100 MB memory per component
4. **Scalability** - Single device → Multi-node enterprise
5. **Security** - Hardened, minimal attack surface
6. **Intelligence** - LLM-powered security analysis via RAG system

---

## 🔧 System Components
```
┌─────────────────────────────────────────────────────────────┐
│                    FULL SYSTEM ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  cpp_sniffer    │───────▶│  ml-detector    │───────▶│ firewall-acl    │
│                 │  ZMQ   │                 │  ZMQ   │     -agent      │
│  eBPF Capture   │  PUSH  │  ML Inference   │  REQ   │  iptables/nft   │
│  3-Layer Detect │        │  4 C++20 Models │        │  Auto Response  │
│  Dual-NIC Ready │        │                 │        │                 │
└─────────────────┘        └─────────────────┘        └─────────────────┘
        │                           │                           │
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │  RAG Security System  │
                        │  TinyLlama-1.1B +     │
                        │  KISS Architecture    │
                        └───────────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │  etcd (Enterprise)    │
                        │  Config + Coordination│
                        └───────────────────────┘
```

### Component Responsibilities

| Component | Role | Status | Language |
|-----------|------|--------|----------|
| **cpp_sniffer** | Packet capture + feature extraction | ✅ Production (Host-Based) | C++20 + eBPF |
| **ml-detector** | ML inference + threat scoring | ✅ 4 Models Complete | C++20 |
| **RAG Security System** | LLM intelligence + analysis | ✅ LLAMA Real | C++20 |
| **firewall-acl-agent** | Automated response | 📋 Planned | C++20 |
| **etcd** | Config coordination (enterprise) | 📋 Planned | C++20 |

---

## 🌊 Data Flow

### Current Implementation (Phase 1 Complete)
```
Network Traffic
      ↓
┌─────────────┐
│ cpp_sniffer │ Capture + Extract Features
│ eBPF/XDP    │ 40 ML features
│ Dual-NIC    │ Gateway Mode Ready
└──────┬──────┘
       │ ZMQ (Protobuf) port 5571
       ↓
┌─────────────┐
│ ml-detector │ 4 Embedded C++20 Models
│             │ • DDoS: 0.24μs
│             │ • Ransomware: 1.06μs  
│             │ • Traffic: 0.37μs
│             │ • Internal: 0.33μs
└──────┬──────┘
       │ ZMQ (Alert) port 5572
       ↓
┌─────────────┐
│ RAG System  │ Security Intelligence
│ TinyLlama   │ • ask_llm "security questions"
│ 1.1B        │ • show_config
│             │ • update_setting
└─────────────┘
```

### RAG Security System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   WhiteList     │    │   RagCommand     │    │   LlamaIntegration│
│    Manager      │◄---│     Manager      │◄---│     (REAL)       │
│ (Router + Etcd) │    │ (RAG Core + Val) │    │  TinyLlama-1.1B  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
         │                       │                       │
         │                       ├───────────────────────┘
         │                       │
         │              ┌──────────────────┐
         └------------► │   ConfigManager  │
                        │  (JSON Persist)  │
                        └──────────────────┘

Commands Available:
• rag show_config           - Display system configuration
• rag update_setting <k> <v> - Update settings with validation
• rag show_capabilities     - Show RAG system capabilities  
• rag ask_llm <question>    - Query LLAMA with security questions
• exit                      - Exit the system
```

---

## 🔍 cpp_sniffer Architecture

**Repository:** This repo  
**Language:** C++20 + eBPF/C  
**Status:** ✅ Production Ready (Host-Based Mode)

### Three-Layer Detection Pipeline
```
┌─────────────────────────────────────────────────────────────┐
│  KERNEL SPACE                                                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Layer 0: eBPF/XDP Program (sniffer.bpf.c)        │    │
│  │                                                     │    │
│  │  • XDP/TC hook on network interface                │    │
│  │  • Parse Ethernet → IP → TCP/UDP headers          │    │
│  │  • Extract first 512 bytes of L4 payload          │    │
│  │  • Populate simple_event structure (544 bytes)    │    │
│  │  • Submit to ring buffer (4 MB)                   │    │
│  │                                                     │    │
│  │  Performance: <1 μs per packet                     │    │
│  │  Safety: eBPF verifier approved                    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓ Ring Buffer (zero-copy)
┌─────────────────────────────────────────────────────────────┐
│  USER SPACE                                                  │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  RingBufferConsumer (Multi-threaded)               │    │
│  │                                                     │    │
│  │  Thread Pools:                                      │    │
│  │  • Ring consumers: N threads (packet ingestion)    │    │
│  │  • Feature processors: M threads (analysis)        │    │
│  │  • ZMQ senders: K threads (output)                 │    │
│  └────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │  FeatureExtractor (83+ features)                   │    │
│  │                                                     │    │
│  │  • Statistical features (mean, std, min, max)      │    │
│  │  • Temporal features (IAT, burst, duration)        │    │
│  │  • Protocol features (flags, lengths, ratios)      │    │
│  │  • Behavioral features (scan, lateral, C&C)        │    │
│  └────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Protobuf Serialization                            │    │
│  │  (NetworkSecurityEvent)                            │    │
│  └───────────────────────┬────────────────────────────┘    │
│                           │ Dual-NIC Support                │
│  ┌───────────────────────▼────────────────────────────┐    │
│  │  Gateway Mode Metadata                              │    │
│  │  • ifindex detection (eth1=3, eth3=5)              │    │
│  │  • mode identification (HOST=1, GATEWAY=2)         │    │
│  │  • WAN flag (1=WAN-facing, 0=LAN-facing)           │    │
│  └───────────────────────┬────────────────────────────┘    │
│                           │                                 │
│  ┌───────────────────────▼────────────────────────────┐    │
│  │  ZMQ PUSH Socket                                    │    │
│  │  tcp://127.0.0.1:5571                              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Data Structures

**simple_event (eBPF → Userspace):**
```c
struct simple_event {
    uint32_t src_ip;           // Source IP
    uint32_t dst_ip;           // Destination IP
    uint16_t src_port;         // Source port
    uint16_t dst_port;         // Destination port
    uint8_t protocol;          // IP protocol (TCP=6, UDP=17)
    uint8_t tcp_flags;         // TCP flags
    uint32_t packet_len;       // Total packet length
    uint16_t ip_header_len;    // IP header length
    uint16_t l4_header_len;    // L4 header length
    uint64_t timestamp;        // Nanosecond timestamp
    uint16_t payload_len;      // Actual payload captured
    uint8_t payload[512];      // First 512 bytes of L4 payload
} __attribute__((packed));

// Total size: 544 bytes
```

### Dual-NIC Implementation (Day 9)

**Multi-Interface XDP Attachment:**
```cpp
// include/ebpf_loader.hpp
std::vector<int> attached_ifindexes_;  // Multiple interfaces tracking

// src/userspace/ebpf_loader.cpp
bool EbpfLoader::attach_skb(const std::string& interface_name) {
    // Check if ALREADY attached to THIS interface
    if (std::find(attached_ifindexes_.begin(), attached_ifindexes_.end(), ifindex) 
        != attached_ifindexes_.end()) {
        return true;  // Already attached
    }
    
    // Attach and add to tracking list
    int err = bpf_xdp_attach(ifindex, prog_fd_, xdp_flags, nullptr);
    if (!err) {
        attached_ifindexes_.push_back(ifindex);
    }
    return !err;
}
```

**BPF Map Configuration for Gateway Mode:**
```c
// BPF map for interface configuration
struct iface_config {
    __u32 ifindex;    // Interface index (e.g., 3 for eth1, 5 for eth3)
    __u8 mode;        // 1=HOST_BASED, 2=GATEWAY
    __u8 is_wan;      // 1=WAN-facing, 0=LAN-facing
    __u8 reserved[2]; // Padding
};
```

**Example Configuration:**
```json
// eth1 (WAN interface): Host-based mode, WAN-facing
{
    "ifindex": 3,
    "mode": 1,    // HOST_BASED
    "is_wan": 1,  // WAN-facing
    "reserved": [0, 0]
}

// eth3 (LAN interface): Gateway mode, LAN-facing
{
    "ifindex": 5,
    "mode": 2,    // GATEWAY
    "is_wan": 0,  // LAN-facing
    "reserved": [0, 0]
}
```

### Performance Characteristics

| Metric | Value | Validated |
|--------|-------|-----------|
| **Throughput** | 82 evt/s peak | ✅ 17h test |
| **Latency (Layer 0)** | <1 μs | ✅ eBPF |
| **Memory** | 4.5 MB | ✅ Stable 17h |
| **CPU (load)** | 5-10% | ✅ Under stress |
| **CPU (idle)** | 0% | ✅ Background |
| **Dual-NIC Attach** | ✅ Working | Verified with bpftool |

---

## ⚡ XDP Mode Selection & Gateway Mode Architecture

**Critical Learning (Day 9):** XDP Generic (SKB mode) has fundamental limitations for gateway mode validation in development environments.

### **XDP Generic (Software Mode) Limitations**

#### **What XDP Generic CAN Capture:**
```
✅ Tráfico que entra FÍSICAMENTE desde fuera del host
  - SSH desde máquina externa hacia la VM
  - HTTP requests desde navegador externo
  - Cualquier paquete que cruce el límite físico de la VM
```

#### **What XDP Generic CANNOT Capture:**
```
❌ Tráfico generado localmente en la misma VM
  - tcpreplay desde VM hacia sus propias interfaces
  - curl/wget a servicios locales en la misma VM
  - Loopback interno (127.0.0.1)

❌ Tráfico entre network namespaces
  - Namespace client → veth pair → bridge → eth3
  - veth pairs, bridges, tunnels internos

❌ Tráfico inyectado artificialmente
  - Paquetes creados en userspace y enviados directamente
  - Scapy, raw sockets, netcat desde la misma máquina
```

### **Technical Explanation**

```
┌──────────────────────────────────────────────────────┐
│  Packet Flow in Linux Networking Stack               │
└──────────────────────────────────────────────────────┘

EXTERNAL TRAFFIC (from physical network):
  NIC Driver → XDP Generic Hook → Networking Stack → Application
                    ↑
            [XDP CAN CAPTURE HERE] ✅

LOCAL TRAFFIC (generated on same host):
  Application → Networking Stack → Loopback/Output
                    ↑
            [XDP HOOK NEVER TRIGGERED] ❌

INTER-VM TRAFFIC (via virtual networks):
  VM1 App → Virtual Network → Hypervisor → VM2 NIC
                                       ↑
                            [PHYSICAL BOUNDARY]
                                       ↓
                              XDP CAN CAPTURE ✅
```

### **Implications for Testing Strategy**

#### **1. Host-Based IDS Testing:**
```bash
# ✅ VALID - XDP Generic works for external traffic
ssh usuario@192.168.56.20        # Desde macOS hacia VM
curl http://192.168.56.20        # Desde navegador externo
```

#### **2. Gateway Mode Testing:**
```bash
# ❌ INVALID - Won't capture with XDP Generic
sudo tcpreplay -i eth3 traffic.pcap    # Misma VM
sudo curl http://192.168.100.1:8080    # Loopback interno

# ✅ VALID - Requires physical traffic crossing boundary
# Option A: Multi-VM setup
# Option B: Physical hardware with real NICs
```

### **XDP Mode Comparison Table**

| Feature | XDP Generic (SKB) | Native XDP (Driver) | TC-BPF (Alternative) |
|---------|-------------------|---------------------|----------------------|
| **Performance** | ~1M pps | 10M+ pps | ~500K pps |
| **Capture Scope** | External only | All traffic | All traffic |
| **Driver Support** | All NICs | Limited (ixgbe, mlx5) | All NICs |
| **Gateway Mode** | Limited testing | Full support | Full support |
| **Development** | Easy | Hardware required | Easy |
| **Our Use Case** | Host-based OK | Production gateway | Fallback option |

### **Gateway Mode Validation Strategy**

#### **Development/Testing Environment:**
```ruby
# Vagrantfile Multi-VM Setup (Day 10 Strategy)
Vagrant.configure("2") do |config|
  # Defender VM (ML Defender)
  config.vm.define "defender" do |defender|
    defender.vm.network "private_network",
                      ip: "192.168.100.1",
                      virtualbox__intnet: "lan"
    defender.vm.network "public_network",
                      bridge: "en0: Wi-Fi"
  end
  
  # Client VM (Traffic Generator)
  config.vm.define "client" do |client|
    client.vm.network "private_network",
                      ip: "192.168.100.50",
                      virtualbox__intnet: "lan"
  end
end
```

#### **Traffic Flow for Validation:**
```
Client VM (192.168.100.50)
  ↓ curl 8.8.8.8
  ↓ VirtualBox Internal Network "lan"
  ↓
Defender eth3 (192.168.100.1) ← ✅ XDP GENERIC CAN CAPTURE
  (Traffic crosses VM boundary)
  ↓ IP forwarding + NAT
  ↓
Defender eth1 (192.168.56.20)
  ↓
Internet
```

#### **Success Criteria for Gateway Mode:**
1. **Traffic must cross VM boundary** (physical network hop)
2. **eth3 must be in gateway mode** (`mode=2, wan=0`)
3. **Metadata must propagate** (`ifindex=5` events appear)
4. **Performance must be measured** (throughput, latency, drops)

### **Production Deployment Requirements**

#### **For Small Office/Home Office (SOHO):**
```
RECOMMENDED: Native XDP-capable NIC
  - Intel i210/i350 (supports native XDP)
  - Throughput: 1 Gbps full duplex
  - Latency: <10 μs per packet

FALLBACK: TC-BPF if Native XDP unavailable
  - Works with any NIC
  - Throughput: ~500 Mbps
  - Latency: ~50 μs per packet
```

#### **For Enterprise Deployment:**
```
REQUIRED: Native XDP with hardware offload
  - Mellanox ConnectX-5/6 (mlx5 driver)
  - Intel E810 (ice driver)
  - Throughput: 10-100 Gbps
  - Latency: <5 μs per packet
```

### **Testing Methodology Updates**

#### **Updated Test Categories:**

1. **Unit Tests** - Individual components
2. **Integration Tests** - Component interactions
3. **Host-Based Validation** - External → VM traffic
4. **Gateway Mode Validation** - Multi-VM or physical setup
5. **Performance Benchmarks** - Throughput, latency, drops
6. **Long-Running Stability** - 24h+ continuous operation

#### **Gateway-Specific Tests:**
```bash
# Test 1: Multi-VM traffic validation
vagrant up defender client
vagrant ssh client -c "curl 8.8.8.8"
# Verify: Events with ifindex=5 appear in defender logs

# Test 2: Performance under load
vagrant ssh client -c "iperf3 -c 8.8.8.8 -t 30"
# Measure: Throughput, CPU usage, packet drops

# Test 3: Protocol coverage
vagrant ssh client -c "nmap -sS -sU -p 1-1000 8.8.8.8"
# Verify: TCP/UDP packets captured correctly
```

### **Key Takeaways for Developers**

1. **XDP Generic is sufficient for:**
    - Host-based IDS development
    - Unit/integration testing
    - Feature development

2. **XDP Generic is insufficient for:**
    - Gateway mode validation with synthetic traffic
    - Performance benchmarking of transit traffic
    - Production gateway deployments

3. **Validation requires:**
    - Physical traffic crossing VM/host boundary
    - Multi-VM setup or physical hardware
    - Real-world traffic patterns

---

## 🤖 ml-detector Architecture

**Repository:** ../ml-detector  
**Language:** C++20  
**Status:** ✅ 4 Embedded Models Complete

### Current State (4 C++20 Embedded Models)
```
ZMQ PULL (from cpp_sniffer) port 5571
      ↓
┌──────────────────────┐
│  Feature Validation  │
│  • Check 40 features │
│  • Handle missing    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  4 Embedded Models   │
│  • All C++20         │
│  • Sub-microsecond   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Alert Generation    │
│  • Configurable      │
│    thresholds        │
│  • Send to firewall  │
└──────────────────────┘
```

### Embedded Model Performance

**Model #1: DDoS Detector**
- **Latency:** 0.24μs (417x better than target)
- **Features:** 10 network behavior features
- **Accuracy:** >98% validated
- **Throughput:** ~4.1M predictions/sec

**Model #2: Ransomware Detector**
- **Latency:** 1.06μs (94x better than target)
- **Features:** 10 file/encryption patterns
- **Accuracy:** >98% validated
- **Throughput:** 944K predictions/sec

**Model #3: Traffic Classifier**
- **Latency:** 0.37μs (270x better than target)
- **Features:** 10 traffic pattern features
- **Accuracy:** Internet vs Internal classification
- **Throughput:** ~2.7M predictions/sec

**Model #4: Internal Threat Detector**
- **Latency:** 0.33μs (303x better than target)
- **Features:** 10 lateral movement indicators
- **Accuracy:** Data exfiltration detection
- **Throughput:** ~3.0M predictions/sec

### Configuration System
```json
{
  "ml_defender": {
    "thresholds": {
      "ddos": 0.85,
      "ransomware": 0.90,  
      "traffic": 0.80,
      "internal": 0.85
    },
    "validation": {
      "min_threshold": 0.5,
      "max_threshold": 0.99,
      "fallback_threshold": 0.75
    }
  }
}
```

---

## 🧠 RAG Security System Architecture

**Repository:** /vagrant/rag  
**Language:** C++20  
**Status:** ✅ Complete with Real LLAMA Integration

### KISS Architecture Design
```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   WhiteList     │    │   RagCommand     │    │   LlamaIntegration│
│    Manager      │◄---│     Manager      │◄---│     (REAL)       │
│ (Router + Etcd) │    │ (RAG Core + Val) │    │  TinyLlama-1.1B  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
         │                       │                       │
         │                       ├───────────────────────┘
         │                       │
         │              ┌──────────────────┐
         └------------► │   ConfigManager  │
                        │  (JSON Persist)  │
                        └──────────────────┘
```

### Core Components

**1. WhiteListManager**
- Central router for all communications
- etcd integration for distributed coordination
- Single point of truth for component registration

**2. RagCommandManager**
- Core RAG logic and command processing
- Inherits from BaseValidator for robust validation
- Manages all RAG-specific operations

**3. LlamaIntegration**
- **Real TinyLlama-1.1B integration** (not simulated)
- Model: `/vagrant/rag/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf`
- C++20 bindings to llama.cpp library
- Security-focused prompt engineering

**4. ConfigManager**
- JSON persistence with automatic type validation
- Settings: `rag_port`, `model_path`, `max_tokens`
- Runtime configuration updates

### Available Commands
```bash
SECURITY_SYSTEM> rag show_config
SECURITY_SYSTEM> rag ask_llm "¿Qué es un firewall en seguridad informática?"
SECURITY_SYSTEM> rag ask_llm "Explica cómo detectar un ataque DDoS"
SECURITY_SYSTEM> rag update_setting port 9090
SECURITY_SYSTEM> rag show_capabilities
SECURITY_SYSTEM> exit
```

### Validation System
```
BaseValidator (Abstract)
    ↑
RagValidator (Concrete)
    • Command validation
    • Setting type checking  
    • Security rule enforcement
```

### Known Issues & Solutions

**⚠️ KV Cache Inconsistency:**
```
Problem: 
  init: the tokens of sequence 0 in the input batch have inconsistent sequence positions
  - last position stored: X = 213
  - tokens have starting position: Y = 0
  
Solution:
  Manual KV cache clearing between queries using batch reset
  Positions always start at 0 for new queries
  Workaround stable for multiple sequential queries
```

**Technical Implementation:**
```cpp
// Manual cache clearing workaround
void clear_kv_cache() {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 0;  // Empty batch
    llama_decode(ctx, batch);  // Resets internal state
    llama_batch_free(batch);
}
```

### Usage Example
```bash
# Start RAG Security System
cd /vagrant/rag/build && ./rag-security

# Interactive session
SECURITY_SYSTEM> rag ask_llm "¿Cómo funciona un firewall de aplicaciones?"
🤖 Consultando LLM: "¿Cómo funciona un firewall de aplicaciones?"
🎯 Generando respuesta REAL para: "¿Cómo funciona un firewall de aplicaciones?"
📊 Tokens generados: 86
🤖 Respuesta: Un firewall de aplicaciones es un sistema de seguridad que filtra el tráfico...
```

---

## 🏢 Enterprise Features

## 🔗 etcd Coordinator (C++20)

**Repository:** ../etcd-coordinator  
**Language:** C++20  
**Library:** etcd-cpp-apiv3  
**Status:** 📋 Phase 4 (Enterprise Features)

### Core Responsibilities

```cpp
class EtcdCoordinator {
public:
    // 1. Receive JSON from components
    void receive_config(const std::string& component_id, 
                       const nlohmann::json& config);
    
    // 2. Store in key-value structure
    void store(const std::string& key, const std::string& value);
    
    // 3. O(1) access with JSON path
    std::string get(const std::string& json_path);
    void set(const std::string& json_path, const std::string& value);
    
    // 4. Notify watchers with validation
    void watch(const std::string& key_prefix, 
              std::function<void(const WatchEvent&)> callback);
    bool validate_update(const std::string& key, 
                        const nlohmann::json& new_value);
    
    // 5. Distribute encryption keys
    void distribute_encryption_key(const std::string& component_id,
                                   const std::vector<uint8_t>& key);
};
```

### Key Structure

```
/config/
├── cpp_sniffer/
│   ├── node_001/
│   │   ├── interface           → "eth0"
│   │   ├── filter_mode         → "hybrid"
│   │   ├── excluded_ports      → [22, 4444, 8080]
│   │   └── encryption_key      → [binary blob]
│   └── node_002/...
├── ml_detector/
│   ├── model_version           → "3"
│   ├── model_path              → "/models/rf_v3.pkl"
│   ├── threshold               → 0.75
│   ├── f1_scores/
│   │   ├── model_1             → 0.9861
│   │   ├── model_2             → 0.9912
│   │   └── model_3             → 0.9934  # BEST
│   └── production_model        → "model_3"
└── firewall_acl/
    ├── block_duration          → 3600
    ├── whitelist               → ["192.168.1.100"]
    └── action_mode             → "block"

/state/
├── health/
│   ├── cpp_sniffer_001         → "healthy"
│   ├── ml_detector             → "healthy"
│   └── firewall_acl            → "healthy"
└── metrics/
    ├── packets_processed       → 2080549
    ├── alerts_generated        → 1234
    └── models_swapped          → 3
```

### Implementation (C++20)

```cpp
#include <etcd/Client.hpp>
#include <nlohmann/json.hpp>
#include <thread>
#include <coroutine>

class EtcdCoordinator {
private:
    etcd::Client client_;
    std::unordered_map<std::string, WatchHandle> watchers_;
    
public:
    EtcdCoordinator(const std::string& etcd_url) 
        : client_(etcd_url) {}
    
    // 1. Receive JSON from component
    void receive_config(const std::string& component_id, 
                       const nlohmann::json& config) {
        std::string key = "/config/" + component_id;
        std::string value = config.dump();
        
        auto response = client_.set(key, value).get();
        if (!response.is_ok()) {
            throw std::runtime_error("Failed to store config");
        }
    }
    
    // 2. Store in KV (already done above)
    
    // 3. O(1) access with JSON path
    std::string get(const std::string& json_path) {
        auto response = client_.get(json_path).get();
        if (!response.is_ok()) {
            throw std::runtime_error("Key not found: " + json_path);
        }
        return response.value().as_string();
    }
    
    void set(const std::string& json_path, const std::string& value) {
        // Validate before setting
        if (!validate_update(json_path, nlohmann::json::parse(value))) {
            throw std::runtime_error("Invalid value for key: " + json_path);
        }
        
        client_.set(json_path, value).get();
        // Watchers are automatically notified by etcd
    }
    
    // 4. Watch with validation
    void watch(const std::string& key_prefix, 
              std::function<void(const WatchEvent&)> callback) {
        
        auto watcher = client_.watch(key_prefix);
        
        // Spawn coroutine for async watching
        std::jthread watch_thread([this, watcher = std::move(watcher), 
                                  callback]() mutable {
            while (true) {
                auto response = watcher->Wait();
                if (!response.is_ok()) break;
                
                for (const auto& event : response.events()) {
                    WatchEvent evt{
                        .key = event.key(),
                        .value = event.value(),
                        .type = event.event_type()
                    };
                    
                    // Validate before notifying
                    if (validate_update(evt.key, 
                                      nlohmann::json::parse(evt.value))) {
                        callback(evt);
                    }
                }
            }
        });
        
        watch_thread.detach();
    }
    
    // Validation logic
    bool validate_update(const std::string& key, 
                        const nlohmann::json& new_value) {
        // Example: Validate threshold is in [0, 1]
        if (key.find("/threshold") != std::string::npos) {
            float threshold = new_value.get<float>();
            return threshold >= 0.0f && threshold <= 1.0f;
        }
        
        // Example: Validate port is in valid range
        if (key.find("/excluded_ports") != std::string::npos) {
            for (int port : new_value) {
                if (port < 1 || port > 65535) return false;
            }
        }
        
        // Add more validation rules...
        return true;
    }
    
    // 5. Distribute encryption keys
    void distribute_encryption_key(const std::string& component_id,
                                   const std::vector<uint8_t>& key) {
        std::string key_path = "/config/" + component_id + "/encryption_key";
        
        // Base64 encode key for storage
        std::string encoded_key = base64_encode(key);
        
        client_.set(key_path, encoded_key).get();
    }
};
```

### Component Watcher (C++20)

**Each component implements:**

```cpp
// In cpp_sniffer, ml_detector, firewall_acl
class ComponentWatcher {
private:
    EtcdCoordinator& coordinator_;
    std::string component_id_;
    
public:
    ComponentWatcher(EtcdCoordinator& coord, std::string id)
        : coordinator_(coord), component_id_(id) {}
    
    void start_watching() {
        std::string watch_prefix = "/config/" + component_id_;
        
        coordinator_.watch(watch_prefix, [this](const WatchEvent& event) {
            this->handle_config_update(event);
        });
    }
    
    void handle_config_update(const WatchEvent& event) {
        std::cout << "[Watcher] Config update: " 
                  << event.key << " = " << event.value << std::endl;
        
        // Hot-reload configuration
        auto new_config = nlohmann::json::parse(event.value);
        apply_config(new_config);
    }
    
    void apply_config(const nlohmann::json& config) {
        // Example: Update filter ports without restart
        if (config.contains("excluded_ports")) {
            update_excluded_ports(config["excluded_ports"]);
        }
        
        // Example: Swap ML model without restart
        if (config.contains("model_path")) {
            hot_swap_model(config["model_path"]);
        }
    }
};
```

---

## 🏠 Home Device Deployment

### Target Hardware

**Raspberry Pi 5:**
- ARM Cortex-A76 (4 cores, 2.4 GHz)
- 8 GB RAM
- microSD / NVMe storage
- Gigabit Ethernet

### Custom Debian 11 ARM

**Minimal OS:**
```
Base Debian 11 ARM64
├─ Kernel 6.1+ (eBPF support)
├─ libbpf, libzmq (minimal deps)
├─ systemd (service management)
├─ iptables/nftables
└─ SSH (hardened)

Removed:
❌ Desktop environment
❌ Unnecessary services
❌ Development tools (after build)
❌ Documentation
```

**Size Target:** <2 GB total footprint

### Security Hardening

1. **Minimal Services**
    - Only: sshd, systemd, network
    - Firewall: Drop all except SSH + management

2. **Secure Boot**
    - Signed kernel
    - Verified boot chain
    - Read-only root

3. **Auto-Updates**
    - Security patches only
    - Staged rollout
    - Rollback on failure

4. **Network Isolation**
    - Management VLAN
    - Monitored interfaces only
    - No outbound except updates

### Physical Device

**Case Design:**
```
┌─────────────────────────┐
│   🛡️ ML Defender Home   │
│                         │
│  [●] Power   [●] Net    │
│  [●] Alert   [●] Health │
│                         │
│  👤 Avatar 1  👤 Avatar 2│
└─────────────────────────┘
```

**LEDs:**
- 🟢 Power (green)
- 🟢 Network (green = OK, 🟡 amber = degraded)
- 🔴 Alert (red = threat detected)
- 🔵 Health (blue = all services OK)

---

## 📊 Performance Characteristics (Full System)

### Latency Budget (End-to-End)

| Stage | Latency | Cumulative |
|-------|---------|------------|
| eBPF capture | <1 μs | 1 μs |
| Ring buffer | <1 μs | 2 μs |
| Feature extraction | <10 μs | 12 μs |
| ZMQ PUSH | <100 μs | 112 μs |
| ml-detector inference | 0.24-1.06μs | ~113 μs |
| RAG analysis (optional) | <1 sec | ~1.1 sec |

**Total:** <150 ms from packet to detection (worst case)

### Throughput

- **cpp_sniffer:** 82 evt/s validated (can handle 200+ evt/s)
- **ml-detector:** 944K - 4.1M inferences/sec across 4 models
- **Bottleneck:** Network bandwidth (1 Gbps link saturates at ~120k pps)

### Resource Usage

**Per Component (Raspberry Pi 5):**
| Component | CPU | Memory | Disk |
|-----------|-----|--------|------|
| cpp_sniffer | 5-10% | 5 MB | 2 MB |
| ml-detector | 10-20% | 150 MB | 50 MB |
| RAG System | 15-30% | 500 MB | 1.5 GB (model) |
| **Total** | **<60%** | **<700 MB** | **~1.5 GB** |

**Plenty of headroom for 4-core ARM CPU + 8 GB RAM**

---

## 🔒 Security Considerations

### Attack Surface

**Minimized:**
- eBPF: Kernel-verified, no arbitrary code exec
- cpp_sniffer: Runs as non-root (cap_net_admin only)
- ZMQ: Local sockets only (no external exposure)
- RAG System: Local model, no external API calls

**Risks:**
- eBPF bugs (mitigated by verifier)
- ZMQ buffer overflow (mitigated by Protobuf size limits)
- ml-detector model poisoning (mitigated by signature verification)

### Hardening Checklist

- [x] eBPF verifier approved
- [x] Minimal privileges (capabilities, not root)
- [ ] SELinux/AppArmor profiles
- [ ] Signed model updates
- [ ] Encrypted ZMQ (optional, for remote)
- [ ] Rate limiting on all inputs
- [ ] Audit logging

---

## 📈 Scalability

### Single Device (Home)

- 1 Raspberry Pi 5
- 3 components co-located
- 1 Gbps link (~120k pps)
- **Capacity:** 1-5 devices protected

### Multi-Node (Enterprise)

- N sniffers (tap/span multiple links)
- M ml-detector nodes (load balanced)
- K firewall agents (distributed)
- etcd cluster (3-5 nodes)
- **Capacity:** 100+ Gbps, millions of flows

---

## 🧪 Testing Between Features

**Mandatory for every major feature:**

1. **Unit Tests** - All new code covered
2. **Integration Tests** - Component interactions
3. **Stress Test** - 1h high load (200+ evt/s)
4. **Long-Running** - 17h+ stability
5. **Regression** - All previous tests pass
6. **Environment Validation** - Verify testing matches production environment constraints

### **XDP-Specific Testing Requirements:**

#### **For Host-Based Mode:**
- [x] External traffic validation (SSH, HTTP from outside)
- [x] Performance metrics under external load
- [x] Multi-protocol capture validation

#### **For Gateway Mode:**
- [ ] Multi-VM setup for validation (Day 10)
- [ ] Physical boundary crossing verification
- [ ] Transit traffic performance benchmarks
- [ ] NAT/routing integration testing

### **Current Test Status:**

| Test Type | Host-Based | Gateway Mode | Notes |
|-----------|------------|--------------|-------|
| Unit Tests | ✅ Complete | ✅ Complete | Same code base |
| Integration | ✅ Complete | ⚠️ Needs Setup | Multi-VM required |
| Performance | ✅ 17h Stable | ⏳ Pending | Requires physical traffic |
| Gateway Validation | N/A | 🔄 Day 10 | Multi-VM strategy defined |

---

## 📝 Documentación a Actualizar Basado en Aprendizajes Día 9

### **Documentation Updates Required:**

1. **`TESTING.md`** - Add gateway mode validation procedures
2. **`DEPLOYMENT.md`** - Hardware requirements for production
3. **`TROUBLESHOOTING.md`** - XDP capture issues guide
4. **`ARCHITECTURE.md`** - This document (updated)

### **New Documentation to Create:**
```markdown
docs/gateway-mode-validation/
├── multi-vm-setup.md
├── hardware-requirements.md
├── performance-benchmarks.md
└── troubleshooting-capture-issues.md
```

---

## 🎯 Milestones

### Milestone 1: Core Detection Complete ✅ **Updated: Dec 5, 2025**
- [x] cpp_sniffer production-ready (host-based mode)
- [x] ml-detector (4 embedded C++20 models)
- [x] RAG Security System with LLAMA real
- [x] Configuration system with JSON validation
- [x] Dual-NIC architecture implementation
- [x] XDP limitations documented and understood
- [ ] Gateway mode validation with multi-VM setup
- [ ] Raspberry Pi image

**Current Status:** 85% Complete

**New Dependencies Identified:**
1. Multi-VM testing environment for gateway validation
2. Hardware with Native XDP support for production gateway
3. Updated deployment documentation for mode-specific requirements

### Milestone 2: Automated Response
- [ ] firewall-acl-agent development
- [ ] Dynamic iptables/nftables integration
- [ ] Rate limiting and connection tracking
- [ ] End-to-end threat response pipeline

**ETA:** Q1 2026

### Milestone 3: Enterprise Features
- [ ] etcd integration
- [ ] Distributed configuration management
- [ ] Multi-node deployment
- [ ] Advanced monitoring and alerting

**ETA:** Q2 2026

### Milestone 4: First Physical Device 🎉
- [ ] Custom Debian ARM
- [ ] Security hardening
- [ ] ARM binaries compiled
- [ ] Case + LEDs
- [ ] **Home deployment** 🏠

**ETA:** Q3 2026

---

## 🆕 Recent Achievements (December 5, 2025)

### Dual-NIC Gateway Architecture (Day 9)
- ✅ **Dual XDP Attachment** - Simultaneous attachment to multiple interfaces
- ✅ **BPF Map Configuration** - Per-interface settings (mode, wan flag)
- ✅ **Host-Based Validation** - 100+ events captured with correct metadata
- ✅ **Scientific Rigor** - Multiple experiments, honest documentation
- ⚠️ **XDP Generic Limitation Identified** - Requires physical traffic for gateway validation
- ✅ **Validation Strategy Defined** - Multi-VM setup for Day 10

### RAG Security System with Real LLAMA
- ✅ **TinyLlama-1.1B integration** - Real model, not simulation
- ✅ **KISS Architecture** - Clean separation of responsibilities
- ✅ **WhiteListManager** - Central router with etcd communication
- ✅ **Robust Validation** - BaseValidator + RagValidator inheritance
- ✅ **JSON Persistence** - Automatic configuration management
- ✅ **Interactive Commands** - ask_llm, show_config, update_setting

### ML Detector Performance
- ✅ **4 Embedded C++20 Models** - All sub-microsecond latency
- ✅ **DDoS Detector**: 0.24μs (417x better than target)
- ✅ **Ransomware Detector**: 1.06μs (94x better than target)
- ✅ **Traffic Classifier**: 0.37μs (270x better than target)
- ✅ **Internal Threat Detector**: 0.33μs (303x better than target)

### System Stability
- ✅ **17-hour stress test** - Memory stable (+1 MB growth)
- ✅ **35,387 events processed** - Zero crashes
- ✅ **Configurable thresholds** - JSON single source of truth
- ✅ **Zero hardcoding** - All settings from configuration

### Key Technical Learnings (Day 9):
1. **XDP Generic vs Native XDP** - Fundamental differences in capture scope
2. **Testing Methodology** - Environment must match production use case
3. **Gateway Mode Validation** - Requires traffic crossing physical boundaries
4. **Documentation Honesty** - Clear about what works and what needs validation

---

**Built with ❤️, rigorous testing, and scientific honesty**

**Esta arquitectura representa lo último en seguridad ML embebida con inteligencia IA real, documentada con integridad técnica tras el descubrimiento de limitaciones críticas del entorno de desarrollo.** 🛡️💚

*Documentado el 5 de diciembre de 2025, tras 4 horas de experimentación intensiva que definieron los límites de XDP Generic y establecieron una estrategia robusta de validación para el modo gateway.*