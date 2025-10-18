# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  # Debian 12 Bookworm
  config.vm.box = "debian/bookworm64"
  config.vm.box_version = "12.20240905.1"

  # VM Configuration
  config.vm.provider "virtualbox" do |vb|
    vb.name = "zeromq-etcd-lab-debian"
    vb.memory = "6144"
    vb.cpus = 4

    # VirtualBox optimizations
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
    vb.customize ["modifyvm", :id, "--nictype1", "virtio"]
    vb.customize ["modifyvm", :id, "--audio", "none"]
    vb.customize ["modifyvm", :id, "--usb", "off"]
    vb.customize ["modifyvm", :id, "--usbehci", "off"]
    vb.customize ["modifyvm", :id, "--nested-hw-virt", "on"]
  end

  # Network configuration
  config.vm.network "private_network", ip: "192.168.56.20"
  config.vm.network "public_network", bridge: "en0: Wi-Fi"

  # Port forwarding
  config.vm.network "forwarded_port", guest: 5555, host: 5555, protocol: "tcp"
  config.vm.network "forwarded_port", guest: 2379, host: 2379, protocol: "tcp"
  config.vm.network "forwarded_port", guest: 2380, host: 2380, protocol: "tcp"
  config.vm.network "forwarded_port", guest: 3000, host: 3000, protocol: "tcp"
  config.vm.network "forwarded_port", guest: 5571, host: 5571, protocol: "tcp"
  config.vm.network "forwarded_port", guest: 5572, host: 5572, protocol: "tcp"

  # Synced folder
  config.vm.synced_folder ".", "/vagrant", type: "virtualbox",
      mount_options: ["dmode=775,fmode=775,exec"]

  # Network environment setup
  config.vm.provision "shell", inline: <<-SHELL
    echo "Configurando entorno de red..."
    HOST_IP=$(ip route | grep '^default' | awk '{print $3}' | head -1)
    [ -z "$HOST_IP" ] && HOST_IP="10.0.2.2"
    echo "Host IP detectada: $HOST_IP"
    echo "Interfaces de red disponibles:"
    ip -4 addr show | grep inet | awk '{print "  " $NF ": " $2}'

    BRIDGED_IP=$(ip -4 addr show | grep "inet.*192.168\|inet.*10\." | grep -v "192.168.56" | awk '{print $2}' | cut -d'/' -f1 | head -1)
    [ -n "$BRIDGED_IP" ] && echo "IP bridged detectada: $BRIDGED_IP"

    {
      echo "VAGRANT_HOST_IP=$HOST_IP"
      echo "SNIFFER_HOST_IP=$HOST_IP"
      echo "SNIFFER_ENDPOINT=tcp://$HOST_IP:5571"
      echo "PRIVATE_NETWORK_IP=192.168.56.20"
      [ -n "$BRIDGED_IP" ] && echo "BRIDGED_NETWORK_IP=$BRIDGED_IP"
    } >> /etc/environment

    {
      echo "$HOST_IP host.docker.internal"
      echo "$HOST_IP docker.host.internal"
    } >> /etc/hosts
  SHELL

  # Locales y herramientas básicas
  config.vm.provision "shell", inline: <<-SHELL
    echo "Instalando herramientas básicas..."
    apt-get update
    apt-get install -y rsync locales curl wget git vim jq make build-essential

    # Configurar locales españolas
    sed -i '/es_ES.UTF-8/s/^# //g' /etc/locale.gen
    locale-gen es_ES.UTF-8
    update-locale LANG=es_ES.UTF-8 LC_ALL=es_ES.UTF-8
    echo 'export LANG=es_ES.UTF-8' >> /etc/profile.d/locale.sh
    echo 'export LC_ALL=es_ES.UTF-8' >> /etc/profile.d/locale.sh

    # Docker
    if ! command -v docker >/dev/null 2>&1; then
      echo "Instalando Docker..."
      curl -fsSL https://get.docker.com -o get-docker.sh
      sh get-docker.sh
      usermod -aG docker vagrant
      systemctl enable docker
      systemctl start docker
      rm get-docker.sh
    fi

    # Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
      echo "Instalando Docker Compose..."
      curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
      chmod +x /usr/local/bin/docker-compose
    fi

    # BPF JIT y filesystem
    if [ -f /proc/sys/net/core/bpf_jit_enable ]; then
        echo 1 > /proc/sys/net/core/bpf_jit_enable
        if ! mountpoint -q /sys/fs/bpf; then
            mount -t bpf none /sys/fs/bpf
        fi
        if ! grep -q "/sys/fs/bpf" /etc/fstab; then
            echo "none /sys/fs/bpf bpf defaults 0 0" >> /etc/fstab
        fi
    fi
  SHELL

  # ========================================
  # eBPF Sniffer Dependencies
  # ========================================
  config.vm.provision "shell", name: "sniffer-deps", inline: <<-SHELL
    echo ""
    echo "=========================================="
    echo "Instalando dependencias Sniffer eBPF..."
    echo "=========================================="

    apt-get update
    apt-get install -y \
      clang \
      llvm \
      bpftool \
      libbpf-dev \
      linux-headers-$(uname -r) \
      libjsoncpp-dev \
      libcurl4-openssl-dev \
      libzmq3-dev \
      libprotobuf-dev \
      protobuf-compiler \
      liblz4-dev \
      libzstd-dev

    echo ""
    echo "Verificando dependencias Sniffer:"
    echo "  - clang: $(clang --version | head -1)"
    echo "  - llvm: $(llc --version | head -1)"
    echo "  - bpftool: $(bpftool version 2>/dev/null || echo 'installed')"
    echo "  - libbpf: $(pkg-config --modversion libbpf)"
    echo "  - jsoncpp: $(pkg-config --modversion jsoncpp)"
    echo "  - curl: $(pkg-config --modversion libcurl)"
    echo "  - ZeroMQ: $(pkg-config --modversion libzmq)"
    echo "  - Protobuf: $(protoc --version)"
    echo "  - LZ4: $(pkg-config --modversion liblz4)"
    echo "  - Zstd: $(pkg-config --modversion libzstd)"
    echo ""
    echo "=========================================="
    echo "✅ Sniffer dependencies installed"
    echo "=========================================="
  SHELL

  config.vm.provision "shell", name: "protobuf-compiler", inline: <<-SHELL
    echo "📦 Installing protobuf compiler..."
    apt-get install -y protobuf-compiler

    # Generate protobuf files on first boot
    if [ ! -f /vagrant/protobuf/network_security.pb.cc ]; then
      echo "🔨 Generating initial protobuf files..."
      cd /vagrant/protobuf && ./generate.sh
    fi
  SHELL

  # ========================================
  # ML Detector Dependencies
  # ========================================
  config.vm.provision "shell", name: "ml-detector-deps", inline: <<-SHELL
    echo ""
    echo "=========================================="
    echo "Instalando dependencias ML Detector..."
    echo "=========================================="

    # CMake moderno (>= 3.20)
    CMAKE_VERSION=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
    REQUIRED_VERSION="3.20"

    if [ -z "$CMAKE_VERSION" ]; then
      INSTALL_CMAKE=1
    else
      if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$CMAKE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        INSTALL_CMAKE=1
      else
        INSTALL_CMAKE=0
      fi
    fi

    if [ "$INSTALL_CMAKE" -eq 1 ]; then
      echo "Instalando CMake 3.25..."
      apt-get remove -y cmake 2>/dev/null || true
      cd /tmp
      wget -q https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.sh
      sh cmake-3.25.0-linux-x86_64.sh --prefix=/usr/local --skip-license
      rm cmake-3.25.0-linux-x86_64.sh
    fi

    # Dependencias C++ (ZMQ y Protobuf ya instalados por sniffer)
    apt-get install -y \
      pkg-config \
      libspdlog-dev \
      nlohmann-json3-dev \
      libgtest-dev \
      libbenchmark-dev

    # ONNX Runtime
    if [ ! -f /usr/local/lib/libonnxruntime.so ]; then
      echo "Instalando ONNX Runtime 1.17.1..."
      ONNX_VERSION="1.17.1"
      cd /tmp
      wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz
      tar -xzf onnxruntime-linux-x64-${ONNX_VERSION}.tgz
      cp -r onnxruntime-linux-x64-${ONNX_VERSION}/include/* /usr/local/include/
      cp -r onnxruntime-linux-x64-${ONNX_VERSION}/lib/* /usr/local/lib/
      ldconfig
      rm -rf onnxruntime-linux-*
      echo "✅ ONNX Runtime instalado"
    else
      echo "✅ ONNX Runtime ya instalado"
    fi

    echo ""
    echo "Verificando dependencias ML Detector:"
    echo "  - CMake: $(cmake --version | head -1)"
    echo "  - spdlog: $(pkg-config --modversion spdlog 2>/dev/null || echo 'header-only')"
    echo "  - nlohmann/json: $(dpkg -l | grep nlohmann-json3 | awk '{print $3}')"
    echo "  - ONNX Runtime: $([ -f /usr/local/lib/libonnxruntime.so ] && echo 'installed' || echo 'missing')"
    echo ""
    echo "=========================================="
    echo "✅ ML Detector dependencies installed"
    echo "=========================================="
  SHELL

  # ========================================
  # Python Environment for ML Training
  # ========================================
  config.vm.provision "shell", name: "python-ml-env", inline: <<-SHELL
    echo ""
    echo "=========================================="
    echo "Instalando Python ML environment..."
    echo "=========================================="

    apt-get update
    apt-get install -y \
      python3 \
      python3-pip \
      python3-venv \
      python3-dev \
      python3-setuptools \
      python3-wheel

    python3 -m pip install --upgrade pip --break-system-packages

    # Crear venv para ml-training
    if [ ! -d /vagrant/ml-training/.venv ]; then
      echo "Creando Python venv..."
      su - vagrant -c "cd /vagrant/ml-training && python3 -m venv .venv"
      echo "✅ Python venv creado"
    fi

    # Instalar dependencias ML
    if [ -f /vagrant/ml-training/requirements.txt ]; then
      echo "Instalando dependencias Python..."
      su - vagrant -c "cd /vagrant/ml-training && source .venv/bin/activate && pip install -r requirements.txt"
      echo "✅ Dependencias Python instaladas"
    fi

    # Aliases útiles
    if ! grep -q "ml-activate" /home/vagrant/.bashrc; then
      cat >> /home/vagrant/.bashrc << 'EOF'

# ============================================================================
# ML Training & Development Shortcuts
# ============================================================================
alias ml-activate='source /vagrant/ml-training/.venv/bin/activate'
alias ml-explore='cd /vagrant/ml-training && ml-activate && python scripts/explore.py'
alias ml-train='cd /vagrant/ml-training && ml-activate && python scripts/train_level1.py'
alias ml-convert='cd /vagrant/ml-training && ml-activate && python scripts/convert_to_onnx.py'

alias build-detector='cd /vagrant/ml-detector/build && cmake .. && make -j4'
alias build-sniffer='cd /vagrant/sniffer/build && cmake .. && make -j4'
alias run-detector='cd /vagrant/ml-detector/build && ./ml-detector --verbose'
alias run-sniffer='cd /vagrant/sniffer/build && sudo ./sniffer --verbose'

export PROJECT_ROOT="/vagrant"
EOF
      echo "✅ Aliases añadidos"
    fi

    echo ""
    echo "Verificando Python environment:"
    echo "  - Python: $(python3 --version)"
    echo "  - pip: $(python3 -m pip --version)"
    echo "  - venv: $([ -d /vagrant/ml-training/.venv ] && echo 'created' || echo 'missing')"
    echo ""
    echo "=========================================="
    echo "✅ Python ML environment ready"
    echo "=========================================="
  SHELL

  # Final message
  config.vm.provision "shell", inline: <<-SHELL
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║  VM LISTA - Upgraded Happiness Lab         ║"
    echo "╚════════════════════════════════════════════╝"
    echo ""
    echo "📊 Configuración de red:"
    echo "  • Private: 192.168.56.20"
    echo "  • Bridged: $(ip -4 addr show | grep "inet.*192.168\|inet.*10\." | grep -v "192.168.56" | awk '{print $2}' | cut -d'/' -f1 | head -1 || echo 'no detectada')"
    echo ""
    echo "🔧 Componentes instalados:"
    echo "  ✅ Docker + Docker Compose"
    echo "  ✅ eBPF toolchain (clang, llvm, bpftool, libbpf)"
    echo "  ✅ Sniffer dependencies (jsoncpp, curl, zmq, protobuf, lz4, zstd)"
    echo "  ✅ ML Detector dependencies (CMake 3.25, spdlog, nlohmann-json, ONNX)"
    echo "  ✅ Python ML environment (venv, pip packages)"
    echo ""
    echo "🚀 Comandos útiles:"
    echo "  vagrant ssh              # Conectar a la VM"
    echo "  build-sniffer            # Compilar sniffer"
    echo "  build-detector           # Compilar ml-detector"
    echo "  run-sniffer              # Ejecutar sniffer (necesita sudo)"
    echo "  run-detector             # Ejecutar ml-detector"
    echo "  ml-activate              # Activar Python venv"
    echo ""
    echo "📂 Estructura:"
    echo "  /vagrant/sniffer/        # Sniffer eBPF"
    echo "  /vagrant/ml-detector/    # ML Detector C++"
    echo "  /vagrant/ml-training/    # Python ML training"
    echo "  /vagrant/protobuf/       # Protobuf schemas"
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "✨ Ready for development!"
    echo "═══════════════════════════════════════════════"
  SHELL
end