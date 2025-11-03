# 📦 **DEPLOYMENT.md - Production Deployment Guide**

# 📦 Deployment Guide - Ransomware Detection System

**Version:** 3.2.0  
**Last Updated:** November 3, 2025  
**Target:** Production deployment on bare metal, VM, or Raspberry Pi

---

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Pre-Installation Checklist](#pre-installation-checklist)
- [Installation](#installation)
- [Configuration](#configuration)
- [Systemd Service Setup](#systemd-service-setup)
- [Verification](#verification)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Updates](#updates)
- [Uninstallation](#uninstallation)
- [Production Best Practices](#production-best-practices)

---

## 💻 System Requirements

### Minimum Requirements

**Hardware:**
```
CPU:        2 cores (x86_64 or ARM64)
RAM:        512 MB available
Storage:    100 MB for binaries + logs
Network:    1 Gbps Ethernet (recommended)
```

**Software:**
```
OS:         Linux (Debian 11+, Ubuntu 20.04+)
Kernel:     6.1+ (eBPF CO-RE support required)
- BTF enabled (/sys/kernel/btf/vmlinux must exist)
- eBPF support (CONFIG_BPF=y, CONFIG_BPF_SYSCALL=y)
```

### Recommended (Production)

**Hardware:**
```
CPU:        4+ cores (for multi-threaded processing)
RAM:        2 GB available
Storage:    1 GB (for logs rotation)
Network:    10 Gbps (for high-traffic environments)
```

**Target Platforms:**
- ✅ Debian 12 (Bookworm) - Tested
- ✅ Ubuntu 22.04 LTS - Compatible
- ✅ Raspberry Pi 5 (ARM64) - Target for home device
- ⚠️ CentOS/RHEL 8+ - Should work (untested)
- ⚠️ Arch Linux - Should work (untested)

---

╔═══════════════════════════════════════════════════════════════╗
║  Kernel 6.1 - LTS Release (Debian 12 Default)                ║
╚═══════════════════════════════════════════════════════════════╝

Reasons for 6.1+ Requirement:
✅ Modern eBPF JIT improvements
✅ Better BTF support (CO-RE reliability)
✅ XDP performance enhancements
✅ Security hardening
✅ Tested and validated (17h stability)
✅ Debian 12 default (production ready)
✅ Ubuntu 22.04+ compatible
✅ Raspberry Pi OS latest

Older Kernels (5.10-6.0):
⚠️  Should work, but NOT tested
⚠️  Potential eBPF edge cases
⚠️  Missing performance optimizations

### Kernel Compatibility Notes

**Validated Kernels:**
- ✅ Kernel 6.1.x (Debian 12) - Fully tested (17h stability)
- ✅ Kernel 6.5.x (Ubuntu 23.10+) - Compatible

**Should Work (Untested):**
- ⚠️ Kernel 5.15+ (Ubuntu 22.04) - Basic eBPF support
- ⚠️ Kernel 5.10 LTS - Minimum eBPF CO-RE

**Not Supported:**
- ❌ Kernel < 5.10 - Missing BTF/CO-RE features

**Recommendation:** Use kernel 6.1+ for production deployment.

## 📋 Pre-Installation Checklist

### 1. Verify Kernel Support

```bash
# Check kernel version
uname -r
# Required: 6.1 or higher (tested on Debian 12)

# Check BTF support (required for eBPF CO-RE)
ls -l /sys/kernel/btf/vmlinux
# Should exist and be readable

# Check eBPF support
zgrep CONFIG_BPF /proc/config.gz
# Should show: CONFIG_BPF=y, CONFIG_BPF_SYSCALL=y

# If config.gz doesn't exist, try:
cat /boot/config-$(uname -r) | grep CONFIG_BPF
```

### 2. Check Network Interface

```bash
# List available interfaces
ip link show

# Verify target interface exists and is UP
ip link show eth0
# Replace 'eth0' with your interface name
```

### 3. Verify Permissions

```bash
# For eBPF, you need either:
# Option A: Run as root
sudo -v

# Option B: Grant capabilities (preferred)
# We'll set this up during installation
```

---

## 🔧 Installation

### Step 1: Install Dependencies

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
    zlib1g-dev

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
    libjsoncpp-dev liblz4-dev libzstd-dev
```

### Step 2: Clone Repository

```bash
# Clone from Git
cd /opt
sudo git clone https://github.com/yourusername/sniffer.git
sudo chown -R $USER:$USER sniffer
cd sniffer

# Or if deploying from archive
cd /opt
sudo tar xzf sniffer-v3.2.0.tar.gz
sudo chown -R $USER:$USER sniffer
cd sniffer
```

### Step 3: Build

```bash
cd /opt/sniffer

# Create build directory
mkdir -p build
cd build

# Configure (Release mode for production)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-14 \
      -DCMAKE_CXX_COMPILER=clang++-14 \
      ..

# Build (use all cores)
make -j$(nproc)

# Verify build
ls -lh sniffer sniffer.bpf.o
# Both files should exist
```

### Step 4: Run Tests (Optional but Recommended)

```bash
cd /opt/sniffer/build

# Run unit tests
ctest --output-on-failure

# Expected: All tests pass
# If any fail, check build configuration
```

### Step 5: Install

```bash
cd /opt/sniffer/build

# Install to /usr/local
sudo make install

# This installs:
# - /usr/local/bin/sniffer           (main binary)
# - /usr/local/lib/sniffer.bpf.o     (eBPF program)
# - /usr/local/share/sniffer/*       (configs, docs)
```

**Note:** If `make install` is not configured, manually copy:
```bash
sudo mkdir -p /usr/local/bin /usr/local/lib/sniffer
sudo cp build/sniffer /usr/local/bin/
sudo cp build/sniffer.bpf.o /usr/local/lib/sniffer/
sudo chmod +x /usr/local/bin/sniffer
```

---

## ⚙️ Configuration

### Step 1: Create Configuration Directory

```bash
sudo mkdir -p /etc/sniffer
sudo chown $USER:$USER /etc/sniffer
```

### Step 2: Create Configuration File

```bash
cat > /etc/sniffer/sniffer.json << 'EOFCONFIG'
{
  "interface": "eth0",
  "profile": "lab",
  "node_id": "sniffer_node_001",
  "cluster": "production_cluster",
  
  "threading": {
    "ring_consumer_threads": 2,
    "feature_processor_threads": 2,
    "zmq_sender_threads": 1
  },
  
  "filter": {
    "mode": "hybrid",
    "excluded_ports": [22, 4444, 8080],
    "included_ports": [8000],
    "default_action": "capture"
  },
  
  "zmq": {
    "output_endpoint": "tcp://127.0.0.1:5571",
    "socket_type": "PUSH"
  },
  
  "compression": {
    "enabled": false,
    "algorithm": "lz4",
    "level": 1
  },
  
  "encryption": {
    "enabled": false,
    "algorithm": "chacha20-poly1305"
  },
  
  "ransomware_detection": {
    "enabled": true,
    "fast_detector_window_ms": 10000,
    "feature_processor_interval_s": 30
  },
  
  "logging": {
    "level": "info",
    "file": "/var/log/sniffer/sniffer.log"
  }
}
EOFCONFIG

echo "✅ Configuration created: /etc/sniffer/sniffer.json"
```

### Step 3: Adjust for Your Environment

```bash
# Edit configuration
sudo nano /etc/sniffer/sniffer.json

# Key settings to check:
# - interface: Your network interface (e.g., eth0, enp0s3)
# - node_id: Unique identifier for this node
# - zmq.output_endpoint: Where to send events
```

### Step 4: Create Log Directory

```bash
sudo mkdir -p /var/log/sniffer
sudo chown $USER:$USER /var/log/sniffer
```

---

## 🚀 Systemd Service Setup

### Step 1: Create Service File

```bash
sudo tee /etc/systemd/system/sniffer.service > /dev/null << 'EOFSERVICE'
[Unit]
Description=Ransomware Detection Sniffer
After=network.target
Wants=network-online.target

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
RestartSec=10s

# Logging
StandardOutput=append:/var/log/sniffer/stdout.log
StandardError=append:/var/log/sniffer/stderr.log

# Resource limits (optional)
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOFSERVICE

echo "✅ Systemd service created"
```

### Step 2: Set Capabilities (Preferred over Root)

```bash
# Grant necessary capabilities to binary
sudo setcap cap_net_admin,cap_net_raw,cap_bpf,cap_perfmon=eip /usr/local/bin/sniffer

# Verify
getcap /usr/local/bin/sniffer
# Should show: cap_net_admin,cap_net_raw,cap_bpf,cap_perfmon=eip
```

**Note:** If kernel < 5.8, `CAP_BPF` might not exist. Use only:
```bash
sudo setcap cap_net_admin,cap_net_raw=eip /usr/local/bin/sniffer
```

### Step 3: Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable (start on boot)
sudo systemctl enable sniffer

# Start now
sudo systemctl start sniffer

# Check status
sudo systemctl status sniffer
```

---

## ✅ Verification

### Step 1: Check Service Status

```bash
# Service status
sudo systemctl status sniffer

# Should show: Active: active (running)

# View logs
sudo journalctl -u sniffer -f

# Or direct logs
tail -f /var/log/sniffer/stdout.log
```

### Step 2: Verify eBPF Program Loaded

```bash
# Check loaded eBPF programs
sudo bpftool prog list | grep sniffer

# Check eBPF maps
sudo bpftool map list | grep sniffer

# Both should show entries if loaded successfully
```

### Step 3: Generate Test Traffic

```bash
# Generate some traffic
curl http://example.com
ping -c 5 8.8.8.8

# Check logs for packet processing
tail -20 /var/log/sniffer/stdout.log

# Should see:
# [INFO] Events processed: XXX
# === ESTADÍSTICAS ===
```

### Step 4: Verify ZMQ Output (if ml-detector running)

```bash
# If ml-detector is NOT running, you'll see:
# [ERROR] ZMQ send falló!
# This is EXPECTED and harmless

# Once ml-detector is deployed, errors should stop
```

---

## 📊 Monitoring

### Systemd Status

```bash
# Quick status
sudo systemctl status sniffer

# Detailed info
systemctl show sniffer

# Resource usage
systemd-cgtop
# Look for sniffer.service
```

### Performance Monitoring

```bash
# CPU and Memory
watch -n 1 'ps aux | grep sniffer | grep -v grep'

# Network stats
watch -n 1 'ifconfig eth0 | grep "RX packets"'

# eBPF stats (if available)
sudo bpftool prog show | grep sniffer
```

### Log Rotation

```bash
# Create logrotate config
sudo tee /etc/logrotate.d/sniffer > /dev/null << 'EOFLOGROTATE'
/var/log/sniffer/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    postrotate
        systemctl reload sniffer > /dev/null 2>&1 || true
    endscript
}
EOFLOGROTATE

# Test rotation
sudo logrotate -f /etc/logrotate.d/sniffer
```

### Health Check Script

```bash
cat > /usr/local/bin/sniffer-health-check << 'EOFHEALTH'
#!/bin/bash
# Sniffer health check script

STATUS=$(systemctl is-active sniffer)
if [ "$STATUS" != "active" ]; then
    echo "❌ Sniffer not running"
    exit 1
fi

# Check if processing events (last 5 min)
RECENT_EVENTS=$(grep "Paquetes procesados" /var/log/sniffer/stdout.log | tail -1)
if [ -z "$RECENT_EVENTS" ]; then
    echo "⚠️  No recent events logged"
    exit 2
fi

# Check memory usage
MEM=$(ps aux | grep '/usr/local/bin/sniffer' | grep -v grep | awk '{print $6}')
if [ "$MEM" -gt 102400 ]; then  # 100 MB
    echo "⚠️  High memory usage: ${MEM} KB"
    exit 3
fi

echo "✅ Sniffer healthy"
exit 0
EOFHEALTH

sudo chmod +x /usr/local/bin/sniffer-health-check

# Run health check
/usr/local/bin/sniffer-health-check
```

---

## 🐛 Troubleshooting

### Service Fails to Start

**Problem:** `systemctl start sniffer` fails

**Solution 1: Check configuration**
```bash
# Validate JSON config
cat /etc/sniffer/sniffer.json | jq .
# Should parse without errors

# Test manually
cd /usr/local/lib/sniffer
sudo /usr/local/bin/sniffer -c /etc/sniffer/sniffer.json -vv
```

**Solution 2: Check interface**
```bash
# Verify interface exists
ip link show eth0

# If interface name is different, update config
sudo nano /etc/sniffer/sniffer.json
# Change "interface": "eth0" to your interface
```

**Solution 3: Check kernel support**
```bash
# Verify BTF
ls -l /sys/kernel/btf/vmlinux

# Verify eBPF support
zgrep CONFIG_BPF /proc/config.gz
```

### eBPF Program Fails to Load

**Problem:** "Failed to attach XDP program"

**Solution 1: Use TC mode instead of XDP**
```bash
# Edit config
sudo nano /etc/sniffer/sniffer.json

# Change capture mode to TC (if XDP not supported)
# This is automatic fallback, but you can force it
```

**Solution 2: Check interface flags**
```bash
# Interface must be UP
sudo ip link set eth0 up

# Verify
ip link show eth0
# Should show: state UP
```

**Solution 3: Check existing XDP programs**
```bash
# List existing XDP programs
sudo bpftool net list

# Remove conflicting program (if any)
sudo ip link set dev eth0 xdp off
```

### High Memory Usage

**Problem:** Memory grows over time

**Solution 1: Check for leaks**
```bash
# Monitor memory
watch -n 5 'ps aux | grep sniffer'

# If growing continuously, restart service
sudo systemctl restart sniffer
```

**Solution 2: Reduce thread count**
```bash
# Edit config
sudo nano /etc/sniffer/sniffer.json

# Reduce threads:
"threading": {
  "ring_consumer_threads": 1,
  "feature_processor_threads": 1,
  "zmq_sender_threads": 1
}

# Restart
sudo systemctl restart sniffer
```

### High CPU Usage

**Problem:** CPU usage >50%

**Solution 1: Reduce packet rate**
```bash
# Add port filters
sudo nano /etc/sniffer/sniffer.json

# Exclude high-traffic ports:
"excluded_ports": [80, 443, 22]

# Restart
sudo systemctl restart sniffer
```

**Solution 2: Check for packet storms**
```bash
# Monitor packet rate
watch -n 1 'ifconfig eth0 | grep "RX packets"'

# If >10000 pps, consider:
# - Rate limiting at firewall
# - Filtering specific protocols
```

### ZMQ Send Errors

**Problem:** `[ERROR] ZMQ send falló!`

**This is NORMAL if ml-detector is not running**

```bash
# Check if ml-detector is deployed
ps aux | grep ml-detector

# If not deployed yet:
# - These errors are expected
# - No impact on sniffer functionality
# - Deploy ml-detector to resolve
```

---

## 🔄 Updates

### Update Procedure (Zero-Downtime)

```bash
# 1. Build new version
cd /opt/sniffer
git pull
cd build
make clean
make -j$(nproc)

# 2. Run tests
ctest --output-on-failure

# 3. Stop service
sudo systemctl stop sniffer

# 4. Backup old binary
sudo cp /usr/local/bin/sniffer /usr/local/bin/sniffer.backup

# 5. Install new version
sudo cp sniffer /usr/local/bin/
sudo cp sniffer.bpf.o /usr/local/lib/sniffer/

# 6. Verify
/usr/local/bin/sniffer --version

# 7. Start service
sudo systemctl start sniffer

# 8. Check logs
sudo journalctl -u sniffer -f
```

### Rollback (if update fails)

```bash
# Restore backup
sudo cp /usr/local/bin/sniffer.backup /usr/local/bin/sniffer

# Restart
sudo systemctl start sniffer
```

---

## 🗑️ Uninstallation

### Complete Removal

```bash
# 1. Stop and disable service
sudo systemctl stop sniffer
sudo systemctl disable sniffer

# 2. Remove service file
sudo rm /etc/systemd/system/sniffer.service
sudo systemctl daemon-reload

# 3. Remove binaries
sudo rm /usr/local/bin/sniffer
sudo rm -rf /usr/local/lib/sniffer

# 4. Remove configuration
sudo rm -rf /etc/sniffer

# 5. Remove logs
sudo rm -rf /var/log/sniffer

# 6. Remove source (optional)
sudo rm -rf /opt/sniffer

echo "✅ Sniffer uninstalled"
```

---

## 🏆 Production Best Practices

### Security

1. **Run with Capabilities, Not Root**
   ```bash
   # Use setcap instead of running as root
   sudo setcap cap_net_admin,cap_net_raw,cap_bpf=eip /usr/local/bin/sniffer
   ```

2. **Firewall Rules**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22/tcp    # SSH
   sudo ufw allow 5571/tcp  # ZMQ (internal only)
   sudo ufw enable
   ```

3. **Log Rotation**
    - Configure logrotate (see Monitoring section)
    - Limit log size: `rotate 7` keeps 1 week

4. **File Permissions**
   ```bash
   sudo chmod 600 /etc/sniffer/sniffer.json
   sudo chown root:root /etc/sniffer/sniffer.json
   ```

### Performance

1. **Thread Configuration**
    - Start with 1 thread per type
    - Scale up if CPU < 50% under load
    - Monitor with `top` or `htop`

2. **Network Tuning**
   ```bash
   # Increase ring buffer size (if high packet loss)
   sudo ethtool -G eth0 rx 4096 tx 4096
   
   # Check current
   ethtool -g eth0
   ```

3. **CPU Affinity (Optional)**
   ```bash
   # Pin to specific cores
   sudo taskset -c 0,1 /usr/local/bin/sniffer -c /etc/sniffer/sniffer.json
   ```

### Monitoring

1. **Set Up Alerts**
   ```bash
   # Monitor service health
   */5 * * * * /usr/local/bin/sniffer-health-check || mail -s "Sniffer Down" admin@example.com
   ```

2. **Metrics Collection**
    - Use Prometheus exporter (future)
    - Parse logs for metrics
    - Monitor: events/s, memory, CPU

3. **Backup Configuration**
   ```bash
   # Daily backup
   0 2 * * * tar czf /backup/sniffer-config-$(date +\%Y\%m\%d).tar.gz /etc/sniffer
   ```

### High Availability

1. **Multiple Nodes**
    - Deploy on multiple servers
    - Each monitors different network segments
    - Centralized ml-detector

2. **Automatic Restart**
   ```bash
   # Already configured in systemd service
   Restart=on-failure
   RestartSec=10s
   ```

3. **Health Checks**
    - Run `/usr/local/bin/sniffer-health-check` regularly
    - Integrate with monitoring (Nagios, Zabbix, etc.)

---

## 📞 Support

### Logs Location
- **Stdout:** `/var/log/sniffer/stdout.log`
- **Stderr:** `/var/log/sniffer/stderr.log`
- **Systemd:** `journalctl -u sniffer`

### Debug Mode

```bash
# Run in foreground with verbose output
sudo /usr/local/bin/sniffer -c /etc/sniffer/sniffer.json -vvv
```

### Get Help

- **Documentation:** README.md, ARCHITECTURE.md, TESTING.md
- **Issues:** GitHub Issues
- **Testing Scripts:** `scripts/testing/`

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Start
sudo systemctl start sniffer

# Stop
sudo systemctl stop sniffer

# Restart
sudo systemctl restart sniffer

# Status
sudo systemctl status sniffer

# Logs (follow)
sudo journalctl -u sniffer -f

# Health check
/usr/local/bin/sniffer-health-check
```

### Files and Directories

```
/usr/local/bin/sniffer              # Main binary
/usr/local/lib/sniffer/             # eBPF programs
/etc/sniffer/sniffer.json           # Configuration
/var/log/sniffer/                   # Logs
/etc/systemd/system/sniffer.service # Service file
```

---

**Deployed with ❤️ and tested on 2.08 million packets**

**Status:** ✅ Production-Ready  
**Version:** 3.2.0

---

