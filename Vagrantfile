Vagrant.configure("2") do |config|
  config.vm.box = "debian/bookworm64"
  config.vm.box_version = "12.20240905.1"

  config.vm.provider "virtualbox" do |vb|
    vb.name = "ml-detector-lab"
    vb.memory = "8192"
    vb.cpus = 6

    # Optimizaciones para red
    vb.customize ["modifyvm", :id, "--nictype1", "virtio"]
    vb.customize ["modifyvm", :id, "--nictype2", "virtio"]
    vb.customize ["modifyvm", :id, "--nictype3", "virtio"]
    vb.customize ["modifyvm", :id, "--nicpromisc3", "allow-all"]

    # Optimizaciones adicionales
    vb.customize ["modifyvm", :id, "--ioapic", "on"]
    vb.customize ["modifyvm", :id, "--audio", "none"]
    vb.customize ["modifyvm", :id, "--usb", "off"]
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
  end

  # ════════════════════════════════════════════════════════════════════════
  # Provisioning: Configuración de Red para Sniffer XDP + Modo Promiscuo
  # ════════════════════════════════════════════════════════════════════════
  config.vm.provision "shell", run: "always", inline: <<-SHELL
    echo "🔧 Configurando eth2 para captura de tráfico..."

    # 1. Instalar ethtool (idempotente)
    if ! command -v ethtool &> /dev/null; then
      echo "📦 Instalando ethtool..."
      apt-get update -qq
      apt-get install -y ethtool
    else
      echo "✅ ethtool ya instalado"
    fi

    # 2. Activar modo promiscuo en eth2
    echo "🔍 Activando modo promiscuo en eth2..."
    ip link set eth2 promisc on

    # 3. Desactivar offloading features incompatibles con XDP
    echo "⚙️  Desactivando offloading features..."
    ethtool -K eth2 gro off 2>/dev/null || true
    ethtool -K eth2 tx-checksum-ip-generic off 2>/dev/null || true
    ethtool -K eth2 tso off 2>/dev/null || true
    ethtool -K eth2 gso off 2>/dev/null || true

    # 4. Verificar configuración
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "✅ CONFIGURACIÓN DE eth2 APLICADA"
    echo "═══════════════════════════════════════════════════════════"

    # Modo promiscuo
    if ip link show eth2 | grep -q PROMISC; then
      echo "✅ Modo promiscuo: ACTIVO"
    else
      echo "❌ Modo promiscuo: INACTIVO"
    fi

    # Offloading features
    echo ""
    echo "Offloading features:"
    ethtool -k eth2 | grep -E 'generic-receive-offload|tx-checksumming|tcp-segmentation-offload|generic-segmentation-offload' | head -4

    echo "═══════════════════════════════════════════════════════════"
    echo ""
  SHELL

  config.vm.network "private_network", ip: "192.168.56.20"
  config.vm.network "public_network", bridge: "en0: Wi-Fi"

  config.vm.network "forwarded_port", guest: 5571, host: 5571
  config.vm.network "forwarded_port", guest: 5572, host: 5572
  config.vm.network "forwarded_port", guest: 2379, host: 2379

  config.vm.synced_folder ".", "/vagrant", type: "virtualbox",
      mount_options: ["dmode=775,fmode=775,exec"]

  # ========================================
  # SINGLE PHASE: ALL DEPENDENCIES
  # ========================================
  config.vm.provision "shell", name: "all-dependencies", inline: <<-SHELL
    set -e  # Exit on error

    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Installing ALL dependencies - Single Phase                ║"
    echo "╚════════════════════════════════════════════════════════════╝"

    # Update package lists
    apt-get update

    # ========================================
    # CORE SYSTEM PACKAGES
    # ========================================
    echo "📦 Installing core system packages..."
    apt-get install -y \
      build-essential \
      git \
      wget \
      curl \
      vim \
      jq \
      make \
      rsync \
      locales

    # ========================================
    # eBPF TOOLCHAIN
    # ========================================
    echo "📦 Installing eBPF toolchain..."
    apt-get install -y \
      clang \
      llvm \
      bpftool \
      libbpf-dev \
      linux-headers-amd64

    # ========================================
    # NETWORKING & COMMUNICATION LIBRARIES
    # ========================================
    echo "📦 Installing networking libraries..."
    apt-get install -y \
      libjsoncpp-dev \
      libcurl4-openssl-dev \
      libzmq3-dev

    # ========================================
    # PROTOBUF (BOTH COMPILER AND RUNTIME)
    # ========================================
    echo "📦 Installing Protobuf..."
    apt-get install -y \
      protobuf-compiler \
      libprotobuf-dev \
      libprotobuf32

    # ========================================
    # COMPRESSION LIBRARIES
    # ========================================
    echo "📦 Installing compression libraries..."
    apt-get install -y \
      liblz4-dev \
      libzstd-dev

    # ========================================
    # ML DETECTOR SPECIFIC
    # ========================================
    echo "📦 Installing ML Detector dependencies..."
    apt-get install -y \
      pkg-config \
      libspdlog-dev \
      nlohmann-json3-dev

    # ========================================
    # FIREWALL ACL AGENT SPECIFIC
    # ========================================
    echo "📦 Installing Firewall dependencies..."
    apt-get install -y \
      iptables \
      ipset \
      libxtables-dev

    # ========================================
    # PYTHON ENVIRONMENT (Minimal - only for scripts, not ML)
    # ========================================
    echo "📦 Installing Python (minimal)..."
    apt-get install -y \
      python3 \
      python3-pip \
      python3-venv \
      python3-dev

    # NOTE: ML training (pandas, sklearn, etc.) should be done on host macOS
    # This Python is only for utility scripts in the VM

    # ========================================
    # DOCKER & DOCKER COMPOSE
    # ========================================
    if ! command -v docker >/dev/null 2>&1; then
      echo "📦 Installing Docker..."
      curl -fsSL https://get.docker.com | sh
      usermod -aG docker vagrant
      systemctl enable docker
      systemctl start docker
    else
      echo "✅ Docker already installed"
    fi

    if ! command -v docker-compose >/dev/null 2>&1; then
      echo "📦 Installing Docker Compose..."
      curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
      chmod +x /usr/local/bin/docker-compose
    else
      echo "✅ Docker Compose already installed"
    fi

    # ========================================
    # CMAKE 3.25+
    # ========================================
    CMAKE_VERSION=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
    if [ -z "$CMAKE_VERSION" ] || [ "$(printf '%s\n' "3.20" "$CMAKE_VERSION" | sort -V | head -n1)" != "3.20" ]; then
      echo "📦 Installing CMake 3.25..."
      cd /tmp
      wget -q https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.sh
      sh cmake-3.25.0-linux-x86_64.sh --prefix=/usr/local --skip-license
      rm cmake-3.25.0-linux-x86_64.sh
    else
      echo "✅ CMake $CMAKE_VERSION already installed"
    fi

    # ========================================
    # ONNX RUNTIME 1.17.1 (C++ only)
    # ========================================
    if [ ! -f /usr/local/lib/libonnxruntime.so ]; then
      echo "📦 Installing ONNX Runtime 1.17.1..."
      cd /tmp
      wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
      tar -xzf onnxruntime-linux-x64-1.17.1.tgz
      cp -r onnxruntime-linux-x64-1.17.1/include/* /usr/local/include/
      cp -r onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/
      ldconfig
      rm -rf onnxruntime-linux-*
    else
      echo "✅ ONNX Runtime already installed"
    fi

    # ========================================
    # SUDOERS CONFIGURATION (NO PASSWORD FOR SNIFFER + FIREWALL)
    # ========================================
    echo "🔐 Configuring sudoers for sniffer and firewall..."
    if ! grep -q "vagrant ALL=(ALL) NOPASSWD: /vagrant/sniffer/build/sniffer" /etc/sudoers.d/ml-defender 2>/dev/null; then
      cat > /etc/sudoers.d/ml-defender << 'EOF'
# ML Defender - Allow sniffer and firewall to run without password
vagrant ALL=(ALL) NOPASSWD: /vagrant/sniffer/build/sniffer
vagrant ALL=(ALL) NOPASSWD: /vagrant/firewall-acl-agent/build/firewall-acl-agent
vagrant ALL=(ALL) NOPASSWD: /usr/sbin/iptables
vagrant ALL=(ALL) NOPASSWD: /usr/sbin/ipset
vagrant ALL=(ALL) NOPASSWD: /usr/bin/pkill
EOF
      chmod 0440 /etc/sudoers.d/ml-defender
      echo "✅ Sudoers configured"
    else
      echo "✅ Sudoers already configured"
    fi

    # ========================================
    # CONFIGURATION
    # ========================================
    echo "⚙️  Configuring system..."

    # Locales
    sed -i '/es_ES.UTF-8/s/^# //g' /etc/locale.gen
    locale-gen es_ES.UTF-8
    update-locale LANG=es_ES.UTF-8 LC_ALL=es_ES.UTF-8

    # BPF JIT
    if [ -f /proc/sys/net/core/bpf_jit_enable ]; then
      echo 1 > /proc/sys/net/core/bpf_jit_enable
      mountpoint -q /sys/fs/bpf || mount -t bpf none /sys/fs/bpf
      grep -q "/sys/fs/bpf" /etc/fstab || echo "none /sys/fs/bpf bpf defaults 0 0" >> /etc/fstab
    fi

    # ========================================
    # DIRECTORY STRUCTURE
    # ========================================
    echo "📁 Creating directory structure..."
    mkdir -p /vagrant/ml-detector/models/production/level1
    mkdir -p /vagrant/ml-detector/models/production/level2
    mkdir -p /vagrant/ml-detector/models/production/level3
    mkdir -p /vagrant/ml-training/outputs/onnx
    mkdir -p /vagrant/firewall-acl-agent/build/logs
    mkdir -p /var/log/ml-defender
    chown -R vagrant:vagrant /var/log/ml-defender
    chmod 755 /var/log/ml-defender

    # ========================================
    # PROTOBUF GENERATION
    # ========================================
    if [ -f /vagrant/protobuf/generate.sh ] && [ ! -f /vagrant/protobuf/network_security.pb.cc ]; then
      echo "🔨 Generating protobuf files..."
      cd /vagrant/protobuf && ./generate.sh
    fi

    # Copy protobuf to firewall
    if [ -f /vagrant/protobuf/network_security.pb.cc ]; then
      echo "📋 Copying protobuf to firewall..."
      mkdir -p /vagrant/firewall-acl-agent/proto
      cp /vagrant/protobuf/network_security.pb.cc /vagrant/firewall-acl-agent/proto/
      cp /vagrant/protobuf/network_security.pb.h /vagrant/firewall-acl-agent/proto/
    fi

    # ========================================
    # BUILD FIREWALL ON FIRST PROVISION
    # ========================================
    if [ ! -f /vagrant/firewall-acl-agent/build/firewall-acl-agent ]; then
      echo "🔨 Building Firewall ACL Agent..."
      mkdir -p /vagrant/firewall-acl-agent/build
      cd /vagrant/firewall-acl-agent/build
      cmake .. && make -j4 || echo "⚠️  Firewall build failed (will retry with 'make firewall')"
    fi

    # ========================================
    # BASH ALIASES (UPDATED WITH FIREWALL)
    # ========================================
    if ! grep -q "build-firewall" /home/vagrant/.bashrc; then
      cat >> /home/vagrant/.bashrc << 'EOF'

# ========================================
# ML Defender Development Aliases
# ========================================

# Building
alias build-sniffer='cd /vagrant/sniffer && make'
alias build-detector='cd /vagrant/ml-detector/build && rm -rf * && cmake .. && make -j4'
alias build-firewall='cd /vagrant/firewall-acl-agent/build && rm -rf * && cmake .. && make -j4'
alias proto-regen='cd /vagrant/protobuf && ./generate.sh && cp network_security.pb.* /vagrant/firewall-acl-agent/proto/'

# Running (individual components)
alias run-firewall='cd /vagrant/firewall-acl-agent/build && sudo ./firewall-acl-agent -c ../config/firewall.json'
alias run-detector='cd /vagrant/ml-detector/build && ./ml-detector -c config/ml_detector_config.json'
alias run-sniffer='cd /vagrant/sniffer/build && sudo ./sniffer -c config/sniffer.json'

# Running (full lab)
alias run-lab='cd /vagrant && bash scripts/run_lab_dev.sh'
alias kill-lab='sudo pkill -9 firewall-acl-agent; pkill -9 ml-detector; sudo pkill -9 sniffer'
alias status-lab='pgrep -a firewall-acl-agent; pgrep -a ml-detector; pgrep -a sniffer'

# ML Model Deployment (from host macOS training)
alias sync-models='rsync -av /vagrant/ml-training/outputs/onnx/*.onnx /vagrant/ml-detector/models/production/ 2>/dev/null && echo "✅ Models synced from host" || echo "⚠️  No models found in ml-training/outputs/onnx/"'
alias list-models='echo "Available ONNX models:" && find /vagrant/ml-detector/models/production -name "*.onnx" -exec ls -lh {} \;'

# Logs
alias logs-firewall='tail -f /vagrant/firewall-acl-agent/build/logs/*.log /var/log/ml-defender/firewall-acl-agent.log 2>/dev/null || echo "No logs yet"'
alias logs-detector='tail -f /vagrant/ml-detector/build/logs/*.log 2>/dev/null || echo "No logs yet"'
alias logs-sniffer='tail -f /vagrant/sniffer/build/logs/*.log 2>/dev/null || echo "No logs yet"'
alias logs-lab='cd /vagrant && bash scripts/monitor_lab.sh'

# Shortcuts
export PROJECT_ROOT="/vagrant"
export MODELS_DIR="/vagrant/ml-detector/models/production"

# Welcome message
cat << 'WELCOME'

╔════════════════════════════════════════════════════════════╗
║  ML Defender - Network Security Pipeline                   ║
║  Development Environment                                   ║
╚════════════════════════════════════════════════════════════╝

🎯 Pipeline Architecture:
   Sniffer (eBPF/XDP) → ML Detector → Firewall ACL Agent
      PUSH 5571           PUB 5572       SUB 5572

🚀 Quick Start:
   run-lab              # Start full pipeline (background + monitor)
   kill-lab             # Stop all components
   status-lab           # Check component status
   logs-lab             # View combined logs

📦 Individual Components:
   run-firewall         # Start firewall (Terminal 1)
   run-detector         # Start detector (Terminal 2)
   run-sniffer          # Start sniffer (Terminal 3)

🔨 Building:
   build-sniffer        # Compile sniffer
   build-detector       # Compile ml-detector
   build-firewall       # Compile firewall-acl-agent
   proto-regen          # Regenerate protobuf + sync

📚 ML Model Workflow:
   1. Train on HOST macOS: cd ml-training && python scripts/train_*.py
   2. Models auto-sync: ml-training/outputs/onnx/ → detector/models/
   3. Deploy: sync-models && build-detector

📊 Monitoring:
   logs-firewall        # Firewall logs
   logs-detector        # Detector logs
   logs-sniffer         # Sniffer logs
   logs-lab             # Combined monitoring

WELCOME
EOF
    fi

    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ✅ ALL DEPENDENCIES INSTALLED                             ║"
    echo "╚════════════════════════════════════════════════════════════╝"
  SHELL

  # ========================================
  # VERIFICATION & SUMMARY
  # ========================================
  config.vm.provision "shell", name: "verification", inline: <<-SHELL
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Dependency Verification                                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    echo "🔧 Core Tools:"
    echo "  clang:    $(clang --version 2>/dev/null | head -1 || echo '❌ MISSING')"
    echo "  cmake:    $(cmake --version 2>/dev/null | head -1 || echo '❌ MISSING')"
    echo "  protoc:   $(protoc --version 2>/dev/null || echo '❌ MISSING')"
    echo "  docker:   $(docker --version 2>/dev/null || echo '❌ MISSING')"
    echo ""

    echo "📚 Libraries (pkg-config):"
    echo "  libbpf:   $(pkg-config --modversion libbpf 2>/dev/null || echo '❌ MISSING')"
    echo "  jsoncpp:  $(pkg-config --modversion jsoncpp 2>/dev/null || echo '❌ MISSING')"
    echo "  libzmq:   $(pkg-config --modversion libzmq 2>/dev/null || echo '❌ MISSING')"
    echo "  libcurl:  $(pkg-config --modversion libcurl 2>/dev/null || echo '❌ MISSING')"
    echo "  liblz4:   $(pkg-config --modversion liblz4 2>/dev/null || echo '❌ MISSING')"
    echo "  libzstd:  $(pkg-config --modversion libzstd 2>/dev/null || echo '❌ MISSING')"
    echo "  spdlog:   $(pkg-config --modversion spdlog 2>/dev/null || echo 'header-only')"
    echo ""

    echo "🔥 Firewall:"
    echo "  iptables: $(iptables --version 2>/dev/null | head -1 || echo '❌ MISSING')"
    echo "  ipset:    $(ipset --version 2>/dev/null | head -1 || echo '❌ MISSING')"
    echo ""

    echo "📦 Protobuf:"
    echo "  Runtime:  $(dpkg -l | grep libprotobuf32 | awk '{print $3}' || echo '❌ MISSING')"
    echo "  Dev:      $(dpkg -l | grep libprotobuf-dev | awk '{print $3}' || echo '❌ MISSING')"
    echo ""

    echo "🧠 ML:"
    echo "  ONNX C++: $([ -f /usr/local/lib/libonnxruntime.so ] && echo 'v1.17.1 ✅' || echo '❌ MISSING')"
    echo "  JSON:     $(dpkg -l | grep nlohmann-json3 | awk '{print $3}' || echo '❌ MISSING')"
    echo ""

    echo "🐍 Python (minimal - ML training on host macOS):"
    echo "  Version:  $(python3 --version 2>/dev/null || echo '❌ MISSING')"
    echo ""

    echo "📁 Build Status:"
    [ -f /vagrant/sniffer/build/sniffer ] && echo "  Sniffer:  ✅" || echo "  Sniffer:  ❌"
    [ -f /vagrant/ml-detector/build/ml-detector ] && echo "  Detector: ✅" || echo "  Detector: ❌"
    [ -f /vagrant/firewall-acl-agent/build/firewall-acl-agent ] && echo "  Firewall: ✅" || echo "  Firewall: ❌"
    echo ""

    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ✅ VM Ready - Pipeline Components                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Quick Start:"
    echo "  vagrant ssh"
    echo "  run-lab              # Start full pipeline"
    echo "  logs-lab             # Monitor components"
    echo ""
  SHELL
end