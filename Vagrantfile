# -*- mode: ruby -*-
# vi: set ft=ruby :

# ══════════════════════════════════════════════════════════════════════════════
# ML DEFENDER LABORATORY - MULTI-VM CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE:
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ML Defender Complete Pipeline Laboratory                               │
# │                                                                          │
# │  ┌─────────────────────────┐         ┌──────────────────────────────┐   │
# │  │  DEFENDER VM            │         │  CLIENT VM                   │   │
# │  │  (Full ML Pipeline)     │         │  (Traffic Generator)         │   │
# │  │                         │         │                              │   │
# │  │  • eBPF/XDP Sniffer     │◄────────│  • Attack simulation         │   │
# │  │  • ML Detector          │   LAN   │  • Gateway testing           │   │
# │  │  • Firewall ACL Agent   │  eth2   │  • PCAP dataset replay       │   │
# │  │  • RAG Security System  │         │  • Performance benchmarks    │   │
# │  │  • FAISS Ingestion      │         │                              │   │
# │  │                         │         │                              │   │
# │  │  eth1: 192.168.56.20    │         │  eth1: 192.168.100.50        │   │
# │  │  eth2: 192.168.100.1    │         │  Gateway: 192.168.100.1      │   │
# │  └─────────────────────────┘         └──────────────────────────────┘   │
# └──────────────────────────────────────────────────────────────────────────┘
#
# PHASE 2A: FAISS Ingestion Support
#   • FAISS v1.8.0 (CPU-only, shared library)
#   • ONNX Runtime v1.17.1
#   • Cron restart every 72h (memory leak mitigation)
#
# USAGE:
#   Development (defender only):   vagrant up defender
#   Gateway testing (both VMs):    vagrant up defender client
#   Full demo:                     vagrant up
#
# CONTROL:
#   autostart: false → Client VM disabled by default
#   autostart: true  → Client VM starts automatically
#
# ══════════════════════════════════════════════════════════════════════════════

Vagrant.configure("2") do |config|

  # ════════════════════════════════════════════════════════════════════════════
  # DEFENDER VM - Full ML Pipeline (Primary)
  # ════════════════════════════════════════════════════════════════════════════
  config.vm.define "defender", primary: true do |defender|
    defender.vm.box = "debian/bookworm64"
    defender.vm.box_version = "12.20240905.1"

    defender.vm.provider "virtualbox" do |vb|
      vb.name = "ml-defender-gateway-lab"
      vb.memory = "8192"
      vb.cpus = 6

      # Network optimizations - Simple virtio (stable, proven)
      vb.customize ["modifyvm", :id, "--nictype1", "virtio"]  # NAT
      vb.customize ["modifyvm", :id, "--nictype2", "virtio"]  # WAN (eth1)
      vb.customize ["modifyvm", :id, "--nictype3", "virtio"]  # Gateway (eth2)

      # Promiscuous mode para captura de paquetes
      vb.customize ["modifyvm", :id, "--nicpromisc2", "allow-all"]  # eth1 (WAN)
      vb.customize ["modifyvm", :id, "--nicpromisc3", "allow-all"]  # eth2 (Gateway)

      # General optimizations
      vb.customize ["modifyvm", :id, "--ioapic", "on"]
      vb.customize ["modifyvm", :id, "--audio", "none"]
      vb.customize ["modifyvm", :id, "--usb", "off"]
      vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    end

    # ════════════════════════════════════════════════════════════════════════
    # RED - Configuración Dual-NIC para Testing (STABLE)
    # ════════════════════════════════════════════════════════════════════════
    # eth0: NAT (Vagrant management)
    # eth1: 192.168.56.20 (WAN-facing, host-only) - Host-based IDS
    # eth2: 192.168.100.1 (LAN-facing, internal) - Gateway mode

    defender.vm.network "private_network", ip: "192.168.56.20"  # eth1: WAN-facing
    defender.vm.network "private_network", ip: "192.168.100.1",
      virtualbox__intnet: "ml_defender_gateway_lan"  # eth2: Gateway LAN

    defender.vm.network "forwarded_port", guest: 5571, host: 5571
    defender.vm.network "forwarded_port", guest: 5572, host: 5572
    defender.vm.network "forwarded_port", guest: 2379, host: 2379

    defender.vm.synced_folder ".", "/vagrant", type: "virtualbox",
        mount_options: ["dmode=775,fmode=775,exec"]

    # ════════════════════════════════════════════════════════════════════════
    # Provisioning: Configuración de Red DUAL-NIC + Modo Promiscuo
    # ════════════════════════════════════════════════════════════════════════
    defender.vm.provision "shell", run: "always", inline: <<-SHELL
      echo "🔧 Configurando interfaces de red para Dual-NIC testing..."

      # 1. Instalar herramientas de red
      apt-get update -qq
      apt-get install -y ethtool tcpdump iptables iproute2

      # 2. Configurar IP forwarding para gateway mode
      echo "🌐 Activando IP forwarding para gateway mode..."
      sysctl -w net.ipv4.ip_forward=1
      sysctl -w net.ipv6.conf.all.forwarding=1
      if ! grep -q "net.ipv4.ip_forward=1" /etc/sysctl.conf; then
        echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
        echo "net.ipv6.conf.all.forwarding=1" >> /etc/sysctl.conf
      fi

      # 3. CRITICAL: Disable rp_filter (prevents routing issues)
      echo "🔧 Disabling rp_filter..."
      sysctl -w net.ipv4.conf.all.rp_filter=0
      sysctl -w net.ipv4.conf.eth1.rp_filter=0
      sysctl -w net.ipv4.conf.eth2.rp_filter=0
      if ! grep -q "net.ipv4.conf.all.rp_filter" /etc/sysctl.conf; then
        echo "net.ipv4.conf.all.rp_filter=0" >> /etc/sysctl.conf
        echo "net.ipv4.conf.eth1.rp_filter=0" >> /etc/sysctl.conf
        echo "net.ipv4.conf.eth2.rp_filter=0" >> /etc/sysctl.conf
      fi

      # 4. Configure NAT for gateway mode
      echo "🔥 Configuring NAT/MASQUERADE..."
      iptables -t nat -F POSTROUTING
      iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE
      iptables -A FORWARD -i eth2 -o eth1 -j ACCEPT
      iptables -A FORWARD -i eth1 -o eth2 -m state --state RELATED,ESTABLISHED -j ACCEPT

      # 5. Detectar interfaz gateway automáticamente
      GATEWAY_IFACE=$(ip -o addr show | grep "192.168.100.1" | awk '{print $2}')
      if [ -z "$GATEWAY_IFACE" ]; then
        echo "⚠️  Gateway interface not found, defaulting to eth2"
        GATEWAY_IFACE="eth2"
      fi

      echo "═══════════════════════════════════════════════════════════"
      echo "🎯 CONFIGURACIÓN DUAL-NIC ML DEFENDER"
      echo "═══════════════════════════════════════════════════════════"
      echo "eth0: NAT (Vagrant management)"
      echo "eth1: 192.168.56.20 (WAN-facing, host-only) - Host-Based IDS"
      echo "eth2: 192.168.100.1 (LAN-facing, internal) - Gateway Mode"
      echo "IP Forwarding: $(sysctl net.ipv4.ip_forward | cut -d= -f2)"
      echo "rp_filter: $(sysctl net.ipv4.conf.all.rp_filter | cut -d= -f2)"
      echo "Gateway Interface: $GATEWAY_IFACE"
      echo "═══════════════════════════════════════════════════════════"

      # 6. Configurar modo promiscuo en interfaces de captura
      echo "🔍 Configurando eth1 (WAN-facing, host-based)..."
      if ip link show eth1 >/dev/null 2>&1; then
        ip link set eth1 up
        ip link set eth1 promisc on
        ethtool -K eth1 gro off tso off gso off 2>/dev/null || true

        if ip link show eth1 | grep -q PROMISC; then
          echo "✅ eth1: Modo promiscuo ACTIVO (Host-Based IDS)"
        else
          echo "❌ eth1: Modo promiscuo INACTIVO"
        fi
      fi

      echo "🔍 Configurando $GATEWAY_IFACE (LAN-facing, gateway mode)..."
      if ip link show $GATEWAY_IFACE >/dev/null 2>&1; then
        ip link set $GATEWAY_IFACE up
        ip link set $GATEWAY_IFACE promisc on
        ethtool -K $GATEWAY_IFACE gro off tso off gso off 2>/dev/null || true

        if ip link show $GATEWAY_IFACE | grep -q PROMISC; then
          echo "✅ $GATEWAY_IFACE: Modo promiscuo ACTIVO (Gateway Mode)"
        else
          echo "❌ $GATEWAY_IFACE: Modo promiscuo INACTIVO"
        fi
      else
        echo "⚠️  $GATEWAY_IFACE no encontrada"
      fi

      # 7. Verificación final
      echo ""
      echo "═══════════════════════════════════════════════════════════"
      echo "✅ CONFIGURACIÓN DE RED COMPLETADA"
      echo "═══════════════════════════════════════════════════════════"
      echo "Interfaces disponibles:"
      ip addr show | grep -E '^[0-9]+:|inet ' | grep -v '127.0.0.1'
      echo ""
      echo "═══════════════════════════════════════════════════════════"
      echo ""
    SHELL

    # ════════════════════════════════════════════════════════════════════════
    # Provisioning: ALL Dependencies
    # ════════════════════════════════════════════════════════════════════════
    defender.vm.provision "shell", name: "all-dependencies", inline: <<-DEPENDENCIES_EOF
      export DEBIAN_FRONTEND=noninteractive
      set -x

      echo "╔════════════════════════════════════════════════════════════╗"
      echo "║  Installing ALL dependencies - Phase 2A (FAISS)           ║"
      echo "╚════════════════════════════════════════════════════════════╝"

      # Core system packages
      apt-get update
      apt-get install -y build-essential git wget curl vim jq make rsync locales libc-bin file

      # eBPF toolchain
      apt-get install -y clang llvm bpftool linux-headers-amd64

      # CRITICAL: libbpf 1.4.6 (FIX PERMANENTE)
      CURRENT_LIBBPF_VERSION=$(PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}" pkg-config --modversion libbpf 2>/dev/null || echo "0.0.0")
      if [ "$(printf '%s\n' "1.2.0" "$CURRENT_LIBBPF_VERSION" | sort -V | head -n1)" != "1.2.0" ]; then
        echo "🔧 Upgrading libbpf to 1.4.6..."
        apt-get install -y libelf-dev zlib1g-dev pkg-config
        cd /tmp && rm -rf libbpf
        git clone --depth 1 --branch v1.4.6 https://github.com/libbpf/libbpf.git
        cd libbpf/src
        make -j$(nproc) BUILD_STATIC_ONLY=y
        make install install_headers
        ldconfig

        if ! grep -q "PKG_CONFIG_PATH.*usr/lib64/pkgconfig" /etc/environment 2>/dev/null; then
          echo 'PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"' >> /etc/environment
        fi

        cat > /etc/profile.d/libbpf.sh << 'LIBBPF_PROFILE'
export PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="/usr/lib64:/usr/local/lib:${LD_LIBRARY_PATH}"
LIBBPF_PROFILE
        chmod +x /etc/profile.d/libbpf.sh
        export PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}"
        export LD_LIBRARY_PATH="/usr/lib64:/usr/local/lib:${LD_LIBRARY_PATH}"
        echo "/usr/lib64" > /etc/ld.so.conf.d/libbpf.conf
        ldconfig
        cd /tmp && rm -rf libbpf
      fi

      # Networking libraries
      apt-get install -y libjsoncpp-dev libcurl4-openssl-dev libzmq3-dev

      # Protobuf
      apt-get install -y protobuf-compiler libprotobuf-dev libprotobuf32

      # Compression
      apt-get install -y liblz4-dev libzstd-dev

      # ML Detector
      apt-get install -y pkg-config libspdlog-dev nlohmann-json3-dev

      # Firewall
      apt-get install -y iptables ipset libxtables-dev

      # RAG dependencies
      apt-get install -y libboost-all-dev libtool autoconf automake libgrpc-dev libgrpc++-dev \
        protobuf-compiler-grpc libc-ares-dev libre2-dev libabsl-dev libbenchmark-dev \
        libgtest-dev libssl-dev libcpprest-dev cmake

      # Python
      apt-get install -y python3 python3-pip python3-venv python3-dev

      # Testing tools (para gateway testing)
      apt-get install -y hping3 nmap tcpreplay netcat-openbsd iperf3 net-tools dnsutils

      # CMake 3.25+
      CMAKE_VERSION=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
      if [ -z "$CMAKE_VERSION" ] || [ "$(printf '%s\n' "3.20" "$CMAKE_VERSION" | sort -V | head -n1)" != "3.20" ]; then
        cd /tmp
        wget -q https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.sh
        sh cmake-3.25.0-linux-x86_64.sh --prefix=/usr/local --skip-license
        rm cmake-3.25.0-linux-x86_64.sh
      fi

      # ONNX Runtime v1.17.1
      if [ ! -f /usr/local/lib/libonnxruntime.so ]; then
        echo "🧠 Installing ONNX Runtime v1.17.1..."
        cd /tmp
        wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
        tar -xzf onnxruntime-linux-x64-1.17.1.tgz
        cp -r onnxruntime-linux-x64-1.17.1/include/* /usr/local/include/
        cp -r onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/
        ldconfig

        # Fix: Create /usr/local/lib64 symlinks for CMake compatibility
        echo "🔗 Creating /usr/local/lib64 symlinks for ONNX Runtime..."
        mkdir -p /usr/local/lib64
        ln -sf /usr/local/lib/libonnxruntime.so* /usr/local/lib64/
        ln -sf /usr/local/lib/libonnxruntime_providers_shared.so /usr/local/lib64/

        rm -rf onnxruntime-linux-*
        echo "✅ ONNX Runtime installed with lib64 symlinks"
      else
        echo "✅ ONNX Runtime already installed"

        # Ensure lib64 symlinks exist even if ONNX was previously installed
        if [ ! -d /usr/local/lib64 ]; then
          echo "🔗 Creating missing /usr/local/lib64 symlinks..."
          mkdir -p /usr/local/lib64
          ln -sf /usr/local/lib/libonnxruntime.so* /usr/local/lib64/
          ln -sf /usr/local/lib/libonnxruntime_providers_shared.so /usr/local/lib64/ 2>/dev/null || true
          echo "✅ lib64 symlinks created"
        fi
      fi

      # FAISS v1.8.0 (CPU-only, shared library) - Phase 2A
      if [ ! -f /usr/local/lib/libfaiss.so ]; then
        echo "🔍 Installing FAISS v1.8.0 (CPU-only, shared library)..."

        # Dependencies
        apt-get install -y libblas-dev liblapack-dev

        # Build from source
        cd /tmp && rm -rf faiss
        git clone --depth 1 --branch v1.8.0 https://github.com/facebookresearch/faiss.git
        cd faiss

        mkdir -p build && cd build
        cmake .. \
          -DFAISS_ENABLE_GPU=OFF \
          -DFAISS_ENABLE_PYTHON=OFF \
          -DBUILD_TESTING=OFF \
          -DBUILD_SHARED_LIBS=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local

        make -j$(nproc)
        make install
        ldconfig

        cd /tmp && rm -rf faiss
        echo "✅ FAISS installed successfully"
      else
        echo "✅ FAISS already installed"
      fi

      # Verify FAISS installation
      if [ -f /usr/local/lib/libfaiss.so ]; then
        echo "✅ FAISS library verified: $(ls -lh /usr/local/lib/libfaiss.so | awk '{print $5}')"
      fi

      # etcd-cpp-api
      if [ ! -f /usr/local/lib/libetcd-cpp-api.so ] && [ ! -f /usr/local/lib/libetcd-cpp-api.a ]; then
        cd /tmp && rm -rf etcd-cpp-apiv3
        git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git
        cd etcd-cpp-apiv3 && git checkout v0.15.3
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local
        make -j4 && make install
        ldconfig
      fi

      # cpp-httplib
      if [ ! -f /usr/local/include/httplib.h ]; then
        cd /tmp && rm -rf cpp-httplib
        git clone https://github.com/yhirose/cpp-httplib.git
        mkdir -p /usr/local/include
        cp cpp-httplib/httplib.h /usr/local/include/
      fi

      # Crypto++
      if [ ! -f /usr/include/cryptopp/cryptlib.h ] && [ ! -f /usr/local/include/cryptopp/cryptlib.h ]; then
        apt-get install -y libcrypto++-dev libcrypto++-doc libcrypto++-utils || {
          cd /tmp
          wget https://www.cryptopp.com/cryptopp870.zip
          unzip cryptopp870.zip -d cryptopp
          cd cryptopp && make -j4 && make install
        }
      fi

      # llama.cpp
      if [ ! -f /vagrant/third_party/llama.cpp/build/src/libllama.a ]; then
        cd /vagrant/third_party/llama.cpp
        mkdir -p build && cd build
        cmake .. -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
          -DLLAMA_NATIVE=OFF -DLLAMA_NO_ACCELERATE=ON -DLLAMA_METAL=OFF -DCMAKE_BUILD_TYPE=Release
        cmake --build . --target all -- -j4
      fi

      # Download LLM model
      mkdir -p /vagrant/rag/models
      cd /vagrant/rag/models
      if [ ! -f "tinyllama-1.1b-chat-v1.0.Q4_0.gguf" ]; then
        wget -q --show-progress --continue --timeout=120 --tries=3 \
          "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" || \
        curl -L -C - --progress-bar \
          "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" \
          -o tinyllama-1.1b-chat-v1.0.Q4_0.gguf
      fi

      # Sudoers
      mkdir -p /etc/sudoers.d
      cat > /etc/sudoers.d/ml-defender << 'EOF'
vagrant ALL=(ALL) NOPASSWD: /vagrant/sniffer/build/sniffer
vagrant ALL=(ALL) NOPASSWD: /vagrant/firewall-acl-agent/build/firewall-acl-agent
vagrant ALL=(ALL) NOPASSWD: /usr/sbin/iptables
vagrant ALL=(ALL) NOPASSWD: /usr/sbin/ipset
vagrant ALL=(ALL) NOPASSWD: /usr/bin/pkill
vagrant ALL=(ALL) NOPASSWD: /bin/kill
vagrant ALL=(ALL) NOPASSWD: /usr/bin/killall
EOF
      chmod 0440 /etc/sudoers.d/ml-defender

      # System config
      sed -i '/es_ES.UTF-8/s/^# //g' /etc/locale.gen
      locale-gen es_ES.UTF-8
      update-locale LANG=es_ES.UTF-8 LC_ALL=es_ES.UTF-8

      if [ -f /proc/sys/net/core/bpf_jit_enable ]; then
        echo 1 > /proc/sys/net/core/bpf_jit_enable
        mountpoint -q /sys/fs/bpf || mount -t bpf none /sys/fs/bpf
        grep -q "/sys/fs/bpf" /etc/fstab || echo "none /sys/fs/bpf bpf defaults 0 0" >> /etc/fstab
      fi

      # Directory structure
      mkdir -p /vagrant/ml-detector/models/production/{level1,level2,level3}
      mkdir -p /vagrant/ml-training/outputs/onnx
      mkdir -p /vagrant/firewall-acl-agent/build/logs
      mkdir -p /vagrant/rag/build/logs
      mkdir -p /vagrant/logs/lab
      mkdir -p /var/log/ml-defender
      chown -R vagrant:vagrant /var/log/ml-defender
      chmod 755 /var/log/ml-defender

      # Protobuf generation
      if [ -f /vagrant/protobuf/generate.sh ] && [ ! -f /vagrant/protobuf/network_security.pb.cc ]; then
        cd /vagrant/protobuf && ./generate.sh
      fi

      if [ -f /vagrant/protobuf/network_security.pb.cc ]; then
        mkdir -p /vagrant/firewall-acl-agent/proto
        cp /vagrant/protobuf/network_security.pb.cc /vagrant/firewall-acl-agent/proto/
        cp /vagrant/protobuf/network_security.pb.h /vagrant/firewall-acl-agent/proto/
      fi

      # Build components
      if [ ! -f /vagrant/firewall-acl-agent/build/firewall-acl-agent ]; then
        mkdir -p /vagrant/firewall-acl-agent/build
        cd /vagrant/firewall-acl-agent/build
        cmake .. && make -j4
      fi

      # Bash aliases
      if ! grep -q "FAISS Ingestion aliases" /home/vagrant/.bashrc; then
        cat >> /home/vagrant/.bashrc << 'BASHRC_EOF'
# ML Defender aliases
alias build-sniffer='cd /vagrant/sniffer && make'
alias build-detector='cd /vagrant/ml-detector/build && rm -rf * && cmake .. && make -j4'
alias build-firewall='cd /vagrant/firewall-acl-agent/build && rm -rf * && cmake .. && make -j4'
alias build-rag='cd /vagrant/rag/build && rm -rf * && cmake .. && make -j4'
alias proto-regen='cd /vagrant/protobuf && ./generate.sh && cp network_security.pb.* /vagrant/firewall-acl-agent/proto/'
alias run-firewall='cd /vagrant/firewall-acl-agent/build && sudo ./firewall-acl-agent -c ../config/firewall.json'
alias run-detector='cd /vagrant/ml-detector/build && ./ml-detector -c ../config/ml_detector_config.json'
alias run-sniffer='cd /vagrant/sniffer/build && sudo ./sniffer -c ../config/sniffer.json'
alias run-rag='cd /vagrant/rag/build && ./rag-security -c ../config/rag_config.json'
alias run-lab='cd /vagrant && bash scripts/run_lab_dev.sh'
alias kill-lab='sudo pkill -9 firewall-acl-agent; pkill -9 ml-detector; sudo pkill -9 sniffer; pkill -9 rag-security'
alias status-lab='pgrep -a firewall-acl-agent; pgrep -a ml-detector; pgrep -a sniffer; pgrep -a rag-security'
alias logs-firewall='tail -f /vagrant/firewall-acl-agent/build/logs/*.log 2>/dev/null'
alias logs-detector='tail -f /vagrant/ml-detector/build/logs/*.log 2>/dev/null'
alias logs-sniffer='tail -f /vagrant/logs/lab/sniffer.log 2>/dev/null'
alias logs-rag='tail -f /vagrant/rag/build/logs/*.log 2>/dev/null'
alias logs-lab='cd /vagrant && bash scripts/monitor_lab.sh'

# Gateway testing aliases
alias test-gateway='/vagrant/scripts/gateway/defender/validate_gateway.sh'
alias start-gateway='/vagrant/scripts/gateway/defender/start_gateway_test.sh'
alias gateway-dash='/vagrant/scripts/gateway/defender/gateway_dashboard.sh'

# FAISS Ingestion aliases (Phase 2A)
alias explore-logs='/vagrant/scripts/explore_rag_logs.sh'
alias verify-faiss='ls -lh /usr/local/lib/libfaiss.so && ls -d /usr/local/include/faiss'
alias verify-onnx='ls -lh /usr/local/lib/libonnxruntime.so && find /usr/local/include -name "onnxruntime*.h"'

export PROJECT_ROOT="/vagrant"
export MODELS_DIR="/vagrant/ml-detector/models/production"

cat << 'WELCOME'
╔════════════════════════════════════════════════════════════╗
║  ML Defender - Network Security Pipeline                   ║
║  Development Environment - PHASE 2A (FAISS)                ║
╚════════════════════════════════════════════════════════════╝
🎯 Dual-NIC Configuration:
   eth1: 192.168.56.20 (WAN-facing, host-based IDS)
   eth2: 192.168.100.1 (LAN-facing, gateway mode)
🔍 FAISS Ingestion Ready:
   explore-logs     # Explore available RAG logs
   verify-faiss     # Verify FAISS installation
   verify-onnx      # Verify ONNX Runtime
🚀 Gateway Testing:
   start-gateway    # Start sniffer in gateway mode
   test-gateway     # Validate gateway capture
   gateway-dash     # Live monitoring dashboard
WELCOME
BASHRC_EOF
      fi

      echo "✅ PROVISIONING COMPLETED SUCCESSFULLY!"
    DEPENDENCIES_EOF

    # ════════════════════════════════════════════════════════════════════════
    # Provisioning: Auto-configure sniffer.json
    # ════════════════════════════════════════════════════════════════════════
    defender.vm.provision "shell", name: "configure-sniffer", run: "always", inline: <<-SNIFFER_CONFIG
      echo "🔧 Auto-configuring sniffer.json for current network topology..."

      # Detect gateway interface
      GATEWAY_IFACE=$(ip -o addr show | grep "192.168.100.1" | awk '{print $2}')

      if [ -z "$GATEWAY_IFACE" ]; then
        echo "⚠️  Gateway interface not found, defaulting to eth2"
        GATEWAY_IFACE="eth2"
      fi

      echo "✅ Gateway interface detected: $GATEWAY_IFACE"

      # Update sniffer.json with correct interface
      if [ -f /vagrant/sniffer/config/sniffer.json ]; then
        # Backup
        cp /vagrant/sniffer/config/sniffer.json /vagrant/sniffer/config/sniffer.json.auto.backup

        # Update using sed (simple and reliable)
        sed -i "s/\\"interface\\": \\"eth[0-9]\\"/\\"interface\\": \\"$GATEWAY_IFACE\\"/g" /vagrant/sniffer/config/sniffer.json

        echo "✅ sniffer.json updated with gateway interface: $GATEWAY_IFACE"
      else
        echo "⚠️  sniffer.json not found at /vagrant/sniffer/config/sniffer.json"
      fi

      echo "═══════════════════════════════════════════════════════════"
      echo "🎯 SNIFFER AUTO-CONFIGURATION COMPLETE"
      echo "═══════════════════════════════════════════════════════════"
      echo "WAN interface:     eth1 (192.168.56.20)"
      echo "Gateway interface: $GATEWAY_IFACE (192.168.100.1)"
      echo "═══════════════════════════════════════════════════════════"
    SNIFFER_CONFIG

    # ════════════════════════════════════════════════════════════════════════
    # Provisioning: Cron restart every 72h (memory leak mitigation)
    # ════════════════════════════════════════════════════════════════════════
    defender.vm.provision "shell", name: "configure-cron-restart", run: "once", inline: <<-CRON
      echo "⏰ Configurando cron para restart automático cada 72h..."

      # Crear entrada cron para vagrant user
      CRON_ENTRY="0 3 */3 * * /vagrant/scripts/restart_ml_defender.sh"

      # Verificar si ya existe
      if ! crontab -u vagrant -l 2>/dev/null | grep -q "restart_ml_defender"; then
        (crontab -u vagrant -l 2>/dev/null; echo "# ML Defender restart every 72h (memory leak mitigation)") | crontab -u vagrant -
        (crontab -u vagrant -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -u vagrant -
        echo "✅ Cron configurado: Restart cada 3 días a las 3:00 AM"
      else
        echo "✅ Cron ya configurado"
      fi

      # Mostrar crontab
      crontab -u vagrant -l
    CRON

  end  # End defender VM

  # ════════════════════════════════════════════════════════════════════════════
  # CLIENT VM - Traffic Generator & Gateway Testing
  # ════════════════════════════════════════════════════════════════════════════

  config.vm.define "client", autostart: false do |client|
    client.vm.box = "debian/bookworm64"
    client.vm.box_version = "12.20240905.1"
    client.vm.hostname = "ml-client"

    client.vm.provider "virtualbox" do |vb|
      vb.name = "ml-defender-client"
      vb.memory = "1024"
      vb.cpus = 2
      vb.customize ["modifyvm", :id, "--nictype1", "virtio"]
      vb.customize ["modifyvm", :id, "--nictype2", "virtio"]
    end

    # Network: LAN only (connects to defender eth2)
    client.vm.network "private_network",
      ip: "192.168.100.50",
      virtualbox__intnet: "ml_defender_gateway_lan"

    client.vm.provision "shell", name: "client-setup", run: "always", inline: <<-CLIENT
      export DEBIAN_FRONTEND=noninteractive
      set -x
      echo "═══════════════════════════════════════════════════════════════════"
      echo "║  ML CLIENT - Traffic Generator Setup                            ║"
      echo "═══════════════════════════════════════════════════════════════════"

      # Install tools
      apt-get update -qq
      apt-get install -y --no-install-recommends \
        --allow-downgrades --allow-remove-essential --allow-change-held-packages \
        -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
        curl wget hping3 nmap iproute2 \
        tcpdump tcpreplay netcat-openbsd dnsutils \
        iputils-ping net-tools

      # Install iperf3 separately
      apt-get install -y --no-install-recommends \
        -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
        iperf3 || echo "⚠️  iperf3 install had issues (non-critical)"

      # Configure routing
      ip route del default 2>/dev/null || true
      ip route add default via 192.168.100.1 dev eth1

      # DNS
      echo "nameserver 8.8.8.8" > /etc/resolv.conf
      echo "nameserver 1.1.1.1" >> /etc/resolv.conf

      echo "✅ CLIENT READY"
      echo "   IP: 192.168.100.50"
      echo "   Gateway: 192.168.100.1 (defender eth2)"
    CLIENT

  end  # End client VM

end  # End Vagrant configuration