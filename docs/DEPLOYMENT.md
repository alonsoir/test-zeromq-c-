# 📦 **DEPLOYMENT.md - ML Defender Platform Deployment Guide**

# 📦 Deployment Guide - ML Defender Platform

**Version:** 4.0.0  
**Last Updated:** November 20, 2025  
**Target:** Production deployment with RAG + ML detectors on bare metal, VM, or Raspberry Pi

---

## 📋 Table of Contents

- [System Overview](#system-overview)
- [System Requirements](#system-requirements)
- [Pre-Installation Checklist](#pre-installation-checklist)
- [Complete Installation](#complete-installation)
- [Component Configuration](#component-configuration)
- [Systemd Services Setup](#systemd-services-setup)
- [Verification & Testing](#verification--testing)
- [Monitoring & Operations](#monitoring--operations)
- [Troubleshooting](#troubleshooting)
- [Updates & Maintenance](#updates--maintenance)
- [Production Best Practices](#production-best-practices)

---

## 🎯 System Overview

### Architecture Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  cpp_sniffer    │────▶│  ml-detector    │────▶│  RAG System     │
│                 │ ZMQ │                 │ ZMQ │                 │
│  eBPF Capture   │ 5571│  4 ML Models    │ 5572│  TinyLlama-1.1B │
│  40+ Features   │     │  Sub-microsecond│     │  Security AI    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Deployment Modes

**1. Single Node (All-in-One)**
- All components on one machine
- Ideal for: Home use, small networks, testing

**2. Distributed Deployment**
- Components across multiple servers
- Ideal for: Enterprise, high-traffic networks

---

## 💻 System Requirements

### Minimum Requirements (Single Node)

**Hardware:**
```
CPU:        4 cores (x86_64 or ARM64)
RAM:        4 GB available (2 GB for LLAMA model)
Storage:    5 GB (1.5 GB for LLAMA + binaries + logs)
Network:    1 Gbps Ethernet
```

**Software:**
```
OS:         Linux (Debian 12+, Ubuntu 22.04+)
Kernel:     6.1+ (eBPF CO-RE support required)
- BTF enabled (/sys/kernel/btf/vmlinux must exist)
- eBPF support (CONFIG_BPF=y, CONFIG_BPF_SYSCALL=y)
```

### Recommended (Production)

**Hardware:**
```
CPU:        8+ cores (for multi-component processing)
RAM:        8 GB available (4 GB for LLAMA + ML models)
Storage:    10 GB (for logs rotation and model storage)
Network:    10 Gbps (for high-traffic environments)
```

### Target Platforms
- ✅ **Debian 12 (Bookworm)** - Fully tested and validated
- ✅ **Ubuntu 22.04 LTS** - Compatible
- ✅ **Raspberry Pi 5 (ARM64)** - Primary target for home deployment
- ⚠️ **CentOS/RHEL 9+** - Should work (untested)
- ⚠️ **Arch Linux** - Should work (untested)

### Kernel Compatibility

**Validated Kernels:**
- ✅ **Kernel 6.1.x** (Debian 12) - Fully tested (17h stability)
- ✅ **Kernel 6.5.x** (Ubuntu 23.10+) - Compatible

**Required Kernel Features:**
```bash
# Verify BTF support (required for eBPF CO-RE)
ls -l /sys/kernel/btf/vmlinux

# Verify eBPF support
zgrep CONFIG_BPF /proc/config.gz
# Should show: CONFIG_BPF=y, CONFIG_BPF_SYSCALL=y
```

---

## 📋 Pre-Installation Checklist

### 1. System Verification

```bash
# Check kernel version
uname -r
# Required: 6.1 or higher

# Check available resources
free -h
# Minimum: 4 GB RAM available

df -h /
# Minimum: 5 GB free space

# Check CPU architecture
lscpu | grep Architecture
# Should be: x86_64 or aarch64 (ARM64)
```

### 2. Network Interface Verification

```bash
# List available interfaces
ip link show

# Verify target interface exists and is UP
ip link show eth0
# Replace 'eth0' with your monitoring interface

# Check interface capabilities
ethtool -i eth0
```

### 3. Permission Requirements

```bash
# For eBPF, you need either:
# Option A: Run as root (simpler)
sudo -v

# Option B: Grant capabilities (production preferred)
# We'll set this up during installation
```

---

## 🔧 Complete Installation

### Step 1: Install System Dependencies

#### Debian/Ubuntu
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    clang-14 \
    llvm-14 \
    libbpf-dev \
    libelf-dev \
    cmake \
    git \
    pkg-config \
    libzmq3-dev \
    protobuf-compiler \
    libprotobuf-dev \
    libjsoncpp-dev \
    liblz4-dev \
    libzstd-dev \
    zlib1g-dev \
    nlohmann-json3-dev

# Verify clang version
clang --version
# Need: clang 12+ (clang 14 recommended)
```

#### Raspberry Pi OS (ARM64)
```bash
# Same as Debian, all packages available
sudo apt-get update
sudo apt-get install -y \
    build-essential clang-14 llvm-14 libbpf-dev \
    libelf-dev cmake git pkg-config \
    libzmq3-dev protobuf-compiler libprotobuf-dev \
    libjsoncpp-dev liblz4-dev libzstd-dev \
    nlohmann-json3-dev
```

### Step 2: Clone and Setup Repository

```bash
# Clone main repository
cd /opt
sudo git clone https://github.com/yourusername/ml-defender.git
sudo chown -R $USER:$USER ml-defender
cd ml-defender

# Clone RAG subsystem
cd /opt
sudo git clone https://github.com/yourusername/rag-security.git
sudo chown -R $USER:$USER rag-security
```

### Step 3: Download LLAMA Model

```bash
# Create model directory
sudo mkdir -p /opt/rag-security/models
sudo chown $USER:$USER /opt/rag-security/models

# Download TinyLlama-1.1B model (1.5 GB)
cd /opt/rag-security/models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Verify download
ls -lh tinyllama-1.1b-chat-v1.0.Q4_0.gguf
# Should be ~1.5 GB
```

### Step 4: Build All Components

#### Build cpp_sniffer
```bash
cd /opt/ml-defender/sniffer
mkdir -p build
cd build

# Configure (Release mode for production)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-14 \
      -DCMAKE_CXX_COMPILER=clang++-14 \
      ..

# Build
make -j$(nproc)

# Verify build
ls -lh sniffer sniffer.bpf.o
```

#### Build ml-detector
```bash
cd /opt/ml-defender/ml-detector
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Verify build
ls -lh ml-detector
```

#### Build RAG Security System
```bash
cd /opt/rag-security
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Verify build
ls -lh rag-security
```

### Step 5: Install Binaries

```bash
# Install sniffer
cd /opt/ml-defender/sniffer/build
sudo make install
# or manually:
sudo cp sniffer /usr/local/bin/
sudo cp sniffer.bpf.o /usr/local/lib/sniffer/

# Install ml-detector
cd /opt/ml-defender/ml-detector/build
sudo cp ml-detector /usr/local/bin/

# Install RAG system
cd /opt/rag-security/build
sudo cp rag-security /usr/local/bin/
```

---

## ⚙️ Component Configuration

### 1. cpp_sniffer Configuration

```bash
sudo mkdir -p /etc/sniffer
sudo tee /etc/sniffer/sniffer.json > /dev/null << 'EOFCONFIG'
{
  "interface": "eth0",
  "profile": "production",
  "node_id": "ml_defender_node_001",
  
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
  },
  
  "threading": {
    "ring_consumer_threads": 2,
    "feature_processor_threads": 2,
    "zmq_sender_threads": 1
  },
  
  "zmq": {
    "output_endpoint": "tcp://127.0.0.1:5571",
    "socket_type": "PUSH"
  },
  
  "buffers": {
    "flow_state_buffer_entries": 500000
  },
  
  "logging": {
    "level": "info",
    "file": "/var/log/ml-defender/sniffer.log"
  }
}
EOFCONFIG
```

### 2. ml-detector Configuration

```bash
sudo mkdir -p /etc/ml-detector
sudo tee /etc/ml-detector/ml_detector_config.json > /dev/null << 'EOFMLCONFIG'
{
  "zmq": {
    "input_endpoint": "tcp://127.0.0.1:5571",
    "output_endpoint": "tcp://127.0.0.1:5572",
    "socket_type": "PULL"
  },
  
  "models": {
    "ddos": {
      "path": "/opt/ml-defender/models/ddos_model.bin",
      "threshold": 0.85
    },
    "ransomware": {
      "path": "/opt/ml-defender/models/ransomware_model.bin", 
      "threshold": 0.90
    },
    "traffic": {
      "path": "/opt/ml-defender/models/traffic_model.bin",
      "threshold": 0.80
    },
    "internal": {
      "path": "/opt/ml-defender/models/internal_model.bin",
      "threshold": 0.85
    }
  },
  
  "performance": {
    "batch_size": 32,
    "max_queue_size": 1000
  },
  
  "logging": {
    "level": "info",
    "file": "/var/log/ml-defender/ml-detector.log"
  }
}
EOFMLCONFIG
```

### 3. RAG System Configuration

```bash
sudo mkdir -p /etc/rag-security
sudo tee /etc/rag-security/system_config.json > /dev/null << 'EOFRAGCONFIG'
{
  "system": {
    "name": "ML Defender RAG Security",
    "version": "4.0.0",
    "rag_port": 9090
  },
  
  "llama": {
    "model_path": "/opt/rag-security/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    "max_tokens": 256,
    "temperature": 0.7,
    "context_size": 1024
  },
  
  "security": {
    "whitelist_enabled": true,
    "max_connections": 10,
    "request_timeout_sec": 30
  },
  
  "logging": {
    "level": "info",
    "file": "/var/log/ml-defender/rag-security.log"
  }
}
EOFRAGCONFIG
```

### 4. Create Log Directories

```bash
sudo mkdir -p /var/log/ml-defender
sudo chown -R $USER:$USER /var/log/ml-defender
```

---

## 🚀 Systemd Services Setup

### 1. cpp_sniffer Service

```bash
sudo tee /etc/systemd/system/ml-defender-sniffer.service > /dev/null << 'EOFSNIFFER'
[Unit]
Description=ML Defender Network Sniffer
After=network.target
Wants=network-online.target
Before=ml-defector.service

[Service]
Type=simple
User=root
Group=root

# Security: Capabilities instead of full root
CapabilityBoundingSet=CAP_NET_ADMIN CAP_NET_RAW CAP_BPF CAP_PERFMON
AmbientCapabilities=CAP_NET_ADMIN CAP_NET_RAW CAP_BPF CAP_PERFMON

# Paths
WorkingDirectory=/usr/local/lib/sniffer
ExecStart=/usr/local/bin/sniffer -c /etc/sniffer/sniffer.json

# Restart policy
Restart=on-failure
RestartSec=5s

# Logging
StandardOutput=append:/var/log/ml-defender/sniffer-stdout.log
StandardError=append:/var/log/ml-defender/sniffer-stderr.log

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOFSNIFFER
```

### 2. ml-detector Service

```bash
sudo tee /etc/systemd/system/ml-defender-detector.service > /dev/null << 'EOFDETECTOR'
[Unit]
Description=ML Defender Threat Detector
After=ml-defender-sniffer.service
Wants=ml-defender-sniffer.service

[Service]
Type=simple
User=root
Group=root

# Paths
WorkingDirectory=/opt/ml-defender/ml-detector
ExecStart=/usr/local/bin/ml-detector --config /etc/ml-detector/ml_detector_config.json

# Restart policy
Restart=on-failure
RestartSec=5s

# Logging
StandardOutput=append:/var/log/ml-defender/detector-stdout.log
StandardError=append:/var/log/ml-defender/detector-stderr.log

# Resource limits
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOFDETECTOR
```

### 3. RAG Security Service

```bash
sudo tee /etc/systemd/system/ml-defender-rag.service > /dev/null << 'EOFRAG'
[Unit]
Description=ML Defender RAG Security System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root

# Paths
WorkingDirectory=/opt/rag-security
ExecStart=/usr/local/bin/rag-security --config /etc/rag-security/system_config.json

# Restart policy
Restart=on-failure
RestartSec=10s

# Logging
StandardOutput=append:/var/log/ml-defender/rag-stdout.log
StandardError=append:/var/log/ml-defender/rag-stderr.log

# Resource limits (LLAMA needs more memory)
LimitNOFILE=65536
LimitAS=infinity

[Install]
WantedBy=multi-user.target
EOFRAG
```

### 4. Set Capabilities and Enable Services

```bash
# Grant capabilities to sniffer
sudo setcap cap_net_admin,cap_net_raw,cap_bpf,cap_perfmon=eip /usr/local/bin/sniffer

# Verify capabilities
getcap /usr/local/bin/sniffer

# Reload systemd and enable services
sudo systemctl daemon-reload

sudo systemctl enable ml-defender-sniffer
sudo systemctl enable ml-defender-detector  
sudo systemctl enable ml-defender-rag

# Start services in order
sudo systemctl start ml-defender-sniffer
sudo systemctl start ml-defender-detector
sudo systemctl start ml-defender-rag
```

---

## ✅ Verification & Testing

### 1. Check Service Status

```bash
# Check all services
sudo systemctl status ml-defender-sniffer
sudo systemctl status ml-defender-detector
sudo systemctl status ml-defender-rag

# All should show: Active: active (running)

# View logs
sudo journalctl -u ml-defender-sniffer -f
sudo tail -f /var/log/ml-defender/sniffer-stdout.log
```

### 2. Verify eBPF Program Loaded

```bash
# Check loaded eBPF programs
sudo bpftool prog list | grep sniffer

# Check eBPF maps
sudo bpftool map list | grep sniffer
```

### 3. Test ML Detectors

```bash
# Generate test traffic
curl http://example.com
ping -c 5 8.8.8.8

# Check ml-detector logs for processing
tail -f /var/log/ml-defender/detector-stdout.log
# Should show inference results and threat scores
```

### 4. Test RAG System

```bash
# Connect to RAG system
telnet localhost 9090

# Test commands (in interactive session)
SECURITY_SYSTEM> rag show_config
SECURITY_SYSTEM> rag show_capabilities
SECURITY_SYSTEM> rag ask_llm "What is a DDoS attack?"

# Expected: Should return coherent security-focused response
```

### 5. Performance Validation

```bash
# Monitor resource usage
watch -n 1 'ps aux | grep -E "(sniffer|ml-detector|rag-security)" | grep -v grep'

# Check for memory leaks (run for several hours)
watch -n 60 'free -h && echo "---" && df -h /'

# Verify no crashes in logs
grep -i "error\|exception\|segmentation" /var/log/ml-defender/*.log
```

---

## 📊 Monitoring & Operations

### 1. Health Check Script

```bash
sudo tee /usr/local/bin/ml-defender-health-check << 'EOFHEALTH'
#!/bin/bash
# ML Defender health check script

COMPONENTS=("ml-defender-sniffer" "ml-defender-detector" "ml-defender-rag")
HEALTHY=true

for component in "${COMPONENTS[@]}"; do
    STATUS=$(systemctl is-active "$component")
    if [ "$STATUS" != "active" ]; then
        echo "❌ $component not running (status: $STATUS)"
        HEALTHY=false
    else
        echo "✅ $component running"
    fi
done

# Check memory usage
MEM_USAGE=$(ps aux | grep -E "(sniffer|ml-detector|rag-security)" | grep -v grep | awk '{sum += $6} END {print sum}')
if [ "$MEM_USAGE" -gt 4194304 ]; then  # 4 GB in KB
    echo "⚠️  High memory usage: $((MEM_USAGE / 1024)) MB"
    HEALTHY=false
fi

# Check recent logs for errors
RECENT_ERRORS=$(find /var/log/ml-defender -name "*.log" -mmin -5 -exec grep -l -i "error\|exception" {} \; | wc -l)
if [ "$RECENT_ERRORS" -gt 0 ]; then
    echo "⚠️  Recent errors found in logs"
    HEALTHY=false
fi

if [ "$HEALTHY" = true ]; then
    echo "✅ ML Defender system healthy"
    exit 0
else
    echo "❌ ML Defender system has issues"
    exit 1
fi
EOFHEALTH

sudo chmod +x /usr/local/bin/ml-defender-health-check
```

### 2. Log Rotation

```bash
sudo tee /etc/logrotate.d/ml-defender << 'EOFLOGROTATE'
/var/log/ml-defender/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    postrotate
        systemctl reload ml-defender-sniffer > /dev/null 2>&1 || true
        systemctl reload ml-defender-detector > /dev/null 2>&1 || true  
        systemctl reload ml-defender-rag > /dev/null 2>&1 || true
    endscript
}
EOFLOGROTATE
```

### 3. Performance Monitoring

```bash
# Create monitoring script
sudo tee /usr/local/bin/ml-defender-monitor << 'EOFMONITOR'
#!/bin/bash
echo "=== ML Defender Performance Monitor ==="
echo "CPU Usage:"
ps aux | grep -E "(sniffer|ml-detector|rag-security)" | grep -v grep | awk '{print $3 "% " $11}'

echo -e "\nMemory Usage:"
ps aux | grep -E "(sniffer|ml-detector|rag-security)" | grep -v grep | awk '{print $6/1024 " MB " $11}'

echo -e "\nNetwork Stats:"
ifconfig eth0 | grep -E "(RX packets|TX packets)"

echo -e "\nRecent Events:"
tail -5 /var/log/ml-defender/sniffer-stdout.log | grep -E "(processed|alerts)"
EOFMONITOR

sudo chmod +x /usr/local/bin/ml-defender-monitor
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. RAG System - KV Cache Errors
**Problem:** `inconsistent sequence positions (X=213, Y=0)`

**Solution:** This is a known issue with workaround implemented
```bash
# Check RAG logs
tail -f /var/log/ml-defender/rag-stderr.log

# If persistent, restart RAG service
sudo systemctl restart ml-defender-rag

# The system includes automatic cache clearing between queries
```

#### 2. ML Detector Not Receiving Data
**Problem:** No inference results in logs

**Solution:**
```bash
# Check ZMQ connectivity
netstat -tlnp | grep 5571
netstat -tlnp | grep 5572

# Verify sniffer is sending data
tail -f /var/log/ml-defender/sniffer-stdout.log | grep "ZMQ"

# Restart services in order
sudo systemctl restart ml-defender-sniffer
sudo systemctl restart ml-defender-detector
```

#### 3. High Memory Usage
**Problem:** Memory usage >4 GB

**Solution:**
```bash
# Identify memory-hungry component
ps aux | grep -E "(sniffer|ml-detector|rag-security)" | grep -v grep | sort -nk6

# RAG system (LLAMA) typically uses most memory
# Consider reducing context_size in RAG config if needed
```

#### 4. eBPF Program Failed to Load
**Problem:** Sniffer fails to start

**Solution:**
```bash
# Check kernel support
ls -l /sys/kernel/btf/vmlinux

# Verify interface
ip link show eth0

# Check existing XDP programs
sudo bpftool net list

# Remove conflicting programs
sudo ip link set dev eth0 xdp off
```

### Debug Mode

```bash
# Run components in foreground for debugging
sudo /usr/local/bin/sniffer -c /etc/sniffer/sniffer.json -vvv
sudo /usr/local/bin/ml-detector --config /etc/ml-detector/ml_detector_config.json --verbose
sudo /usr/local/bin/rag-security --config /etc/rag-security/system_config.json --debug
```

---

## 🔄 Updates & Maintenance

### Update Procedure

```bash
# 1. Stop services
sudo systemctl stop ml-defender-rag
sudo systemctl stop ml-defender-detector
sudo systemctl stop ml-defender-sniffer

# 2. Backup configurations
sudo cp -r /etc/sniffer /etc/sniffer.backup
sudo cp -r /etc/ml-detector /etc/ml-detector.backup
sudo cp -r /etc/rag-security /etc/rag-security.backup

# 3. Update code
cd /opt/ml-defender && git pull
cd /opt/rag-security && git pull

# 4. Rebuild components
cd /opt/ml-defender/sniffer/build && make clean && make -j$(nproc)
cd /opt/ml-defender/ml-detector/build && make clean && make -j$(nproc)  
cd /opt/rag-security/build && make clean && make -j$(nproc)

# 5. Reinstall
sudo cp /opt/ml-defender/sniffer/build/sniffer /usr/local/bin/
sudo cp /opt/ml-defender/ml-detector/build/ml-detector /usr/local/bin/
sudo cp /opt/rag-security/build/rag-security /usr/local/bin/

# 6. Restart services
sudo systemctl start ml-defender-sniffer
sudo systemctl start ml-defender-detector
sudo systemctl start ml-defender-rag
```

### Backup Strategy

```bash
# Daily backup script
sudo tee /usr/local/bin/ml-defender-backup << 'EOFBACKUP'
#!/bin/bash
BACKUP_DIR="/backup/ml-defender-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configurations
cp -r /etc/sniffer $BACKUP_DIR/
cp -r /etc/ml-detector $BACKUP_DIR/ 
cp -r /etc/rag-security $BACKUP_DIR/

# Backup logs (last 7 days)
find /var/log/ml-defender -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/ \;

# Create archive
tar czf $BACKUP_DIR.tar.gz $BACKUP_DIR
echo "Backup created: $BACKUP_DIR.tar.gz"
EOFBACKUP

sudo chmod +x /usr/local/bin/ml-defender-backup
```

---

## 🏆 Production Best Practices

### Security Hardening

1. **Network Segmentation**
   ```bash
   # Isolate management interface
   sudo ufw allow from 192.168.1.0/24 to any port 22
   sudo ufw allow from 192.168.1.0/24 to any port 9090
   sudo ufw enable
   ```

2. **Service Isolation**
   ```bash
   # Create dedicated user
   sudo useradd -r -s /bin/false ml-defender
   sudo chown -R ml-defender:ml-defender /var/log/ml-defender
   ```

3. **Configuration Security**
   ```bash
   sudo chmod 600 /etc/sniffer/sniffer.json
   sudo chmod 600 /etc/ml-detector/ml_detector_config.json
   sudo chmod 600 /etc/rag-security/system_config.json
   ```

### Performance Optimization

1. **CPU Affinity** (High-traffic environments)
   ```bash
   # Pin components to specific cores
   sudo taskset -c 0,1 /usr/local/bin/sniffer
   sudo taskset -c 2,3 /usr/local/bin/ml-detector
   sudo taskset -c 4,5 /usr/local/bin/rag-security
   ```

2. **Network Tuning**
   ```bash
   # Increase ring buffers for high packet rates
   sudo ethtool -G eth0 rx 4096 tx 4096
   
   # Enable RSS for multi-queue
   sudo ethtool -L eth0 combined 4
   ```

3. **Memory Optimization**
   ```bash
   # Adjust RAG context size if memory constrained
   # In /etc/rag-security/system_config.json:
   # "context_size": 512  # Reduce from 1024 if needed
   ```

### High Availability

1. **Multiple Node Deployment**
   ```bash
   # Deploy on 3+ nodes with load balancing
   # Each node: sniffer + detector + RAG
   # Use etcd for coordination (future enhancement)
   ```

2. **Health Monitoring**
   ```bash
   # Add to crontab
   */5 * * * * /usr/local/bin/ml-defender-health-check || /usr/local/bin/alert-admin.sh
   ```

3. **Auto-Recovery**
   ```bash
   # Systemd already configured with:
   # Restart=on-failure
   # RestartSec=5s
   ```

### Scaling Strategies

**Vertical Scaling:**
- Increase RAM for larger LLAMA models
- Add CPU cores for more detection threads
- Use faster storage for model loading

**Horizontal Scaling:**
- Deploy multiple sniffers on different network segments
- Centralized ml-detector cluster
- Distributed RAG systems with load balancing

---

## 📞 Support & Resources

### Log Locations
```
/var/log/ml-defender/sniffer-stdout.log
/var/log/ml-defender/sniffer-stderr.log
/var/log/ml-defender/detector-stdout.log  
/var/log/ml-defender/detector-stderr.log
/var/log/ml-defender/rag-stdout.log
/var/log/ml-defender/rag-stderr.log
```

### Configuration Files
```
/etc/sniffer/sniffer.json
/etc/ml-detector/ml_detector_config.json
/etc/rag-security/system_config.json
```

### Documentation
- **Architecture**: `/opt/ml-defender/docs/ARCHITECTURE.md`
- **Troubleshooting**: `/opt/ml-defender/docs/TROUBLESHOOTING.md`
- **API Documentation**: `/opt/rag-security/docs/API.md`

### Getting Help

1. **Check Logs**: Always start with component logs
2. **Health Check**: Run `ml-defender-health-check`
3. **Debug Mode**: Start components with `--verbose` flag
4. **Community**: GitHub Issues and Discussions

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Start all services
sudo systemctl start ml-defender-sniffer ml-defender-detector ml-defender-rag

# Stop all services  
sudo systemctl stop ml-defender-rag ml-defender-detector ml-defender-sniffer

# Check status
sudo systemctl status ml-defender-*

# View logs
sudo journalctl -u ml-defender-sniffer -f
sudo tail -f /var/log/ml-defender/rag-stdout.log

# Health check
/usr/local/bin/ml-defender-health-check

# Performance monitor
/usr/local/bin/ml-defender-monitor
```

### Key Files and Directories

```
/usr/local/bin/sniffer              # Network sniffer
/usr/local/bin/ml-detector          # ML threat detection
/usr/local/bin/rag-security         # AI security analysis

/etc/sniffer/sniffer.json           # Sniffer configuration
/etc/ml-detector/ml_detector_config.json # Detector config
/etc/rag-security/system_config.json     # RAG system config

/var/log/ml-defender/               # All component logs
/opt/rag-security/models/           # LLAMA model storage

/opt/ml-defender/                   # Source code
/opt/rag-security/                  # RAG system source
```

### Default Ports
- **ZMQ Sniffer Output**: 5571
- **ZMQ Detector Output**: 5572
- **RAG System Console**: 9090

---

**Deployed with ❤️ and validated on 35,387+ events**

**Status:** ✅ Production-Ready with RAG + ML  
**Version:** 4.0.0  
**Architecture:** ML Defender Platform Complete

---

<div align="center">

**🛡️ ML Defender - Protecting Critical Infrastructure with Embedded ML and AI**  
*Zero crashes in 17h stability testing • Sub-microsecond detection latency • Real LLAMA integration*

</div>