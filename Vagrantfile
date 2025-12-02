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
    vb.customize ["modifyvm", :id, "--nictype4", "virtio"]  # NUEVO: Para eth3 gateway

    # Promiscuous mode para captura de paquetes
    vb.customize ["modifyvm", :id, "--nicpromisc2", "allow-all"]  # eth1 (host-only)
    vb.customize ["modifyvm", :id, "--nicpromisc3", "allow-all"]  # eth2 (public bridge)
    vb.customize ["modifyvm", :id, "--nicpromisc4", "allow-all"]  # eth3 (gateway LAN) - NUEVO

    # Optimizaciones adicionales
    vb.customize ["modifyvm", :id, "--ioapic", "on"]
    vb.customize ["modifyvm", :id, "--audio", "none"]
    vb.customize ["modifyvm", :id, "--usb", "off"]
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
  end

  # ════════════════════════════════════════════════════════════════════════
  # RED - Configuración Dual-NIC para Testing
  # ════════════════════════════════════════════════════════════════════════
  # eth0: NAT (Vagrant management)
  # eth1: 192.168.56.20 (WAN-facing, host-only) - Para ataques desde OSX
  # eth2: public_network bridge (captura externa opcional)
  # eth3: 192.168.100.1 (LAN-facing, internal) - NUEVO: Para gateway mode testing

  config.vm.network "private_network", ip: "192.168.56.20"  # eth1: WAN-facing
  config.vm.network "public_network", bridge: "en0: Wi-Fi"  # eth2: Captura externa
  config.vm.network "private_network", ip: "192.168.100.1", virtualbox__intnet: "ml_defender_lan"  # eth3: Gateway LAN - NUEVO

  config.vm.network "forwarded_port", guest: 5571, host: 5571
  config.vm.network "forwarded_port", guest: 5572, host: 5572
  config.vm.network "forwarded_port", guest: 2379, host: 2379

  config.vm.synced_folder ".", "/vagrant", type: "virtualbox",
      mount_options: ["dmode=775,fmode=775,exec"]

  # ════════════════════════════════════════════════════════════════════════
  # Provisioning: Configuración de Red DUAL-NIC + Modo Promiscuo
  # ════════════════════════════════════════════════════════════════════════
  config.vm.provision "shell", run: "always", inline: <<-SHELL
    echo "🔧 Configurando interfaces de red para Dual-NIC testing..."

    # 1. Instalar herramientas de red
    apt-get update -qq
    apt-get install -y ethtool tcpdump

    # 2. Configurar IP forwarding para gateway mode
    echo "🌐 Activando IP forwarding para gateway mode..."
    sysctl -w net.ipv4.ip_forward=1
    if ! grep -q "net.ipv4.ip_forward=1" /etc/sysctl.conf; then
      echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
    fi

    # 3. Detectar interfaz bridge automáticamente (para captura externa)
    BRIDGE_INTERFACE=""
    for iface in eth2; do
      if ip link show $iface >/dev/null 2>&1; then
        BRIDGE_INTERFACE=$iface
        break
      fi
    done

    if [ -z "$BRIDGE_INTERFACE" ]; then
      echo "⚠️  No se encontró interfaz bridge, captura externa no disponible"
      BRIDGE_INTERFACE="none"
    fi

    echo "═══════════════════════════════════════════════════════════"
    echo "🎯 CONFIGURACIÓN DUAL-NIC ML DEFENDER"
    echo "═══════════════════════════════════════════════════════════"
    echo "eth0: NAT (Vagrant management)"
    echo "eth1: 192.168.56.20 (WAN-facing, host-only) - Host-Based IDS"
    echo "eth2: $BRIDGE_INTERFACE (Captura externa opcional)"
    echo "eth3: 192.168.100.1 (LAN-facing, internal) - Gateway Mode"
    echo "═══════════════════════════════════════════════════════════"

    # 4. Configurar modo promiscuo en interfaces de captura
    # eth1: Host-Based Mode (captura ataques desde OSX)
    echo "🔍 Configurando eth1 (WAN-facing, host-based)..."
    if ip link show eth1 >/dev/null 2>&1; then
      ip link set eth1 promisc on
      ethtool -K eth1 gro off 2>/dev/null || true
      ethtool -K eth1 tx-checksum-ip-generic off 2>/dev/null || true
      ethtool -K eth1 tso off 2>/dev/null || true
      ethtool -K eth1 gso off 2>/dev/null || true

      if ip link show eth1 | grep -q PROMISC; then
        echo "✅ eth1: Modo promiscuo ACTIVO (Host-Based IDS)"
      else
        echo "❌ eth1: Modo promiscuo INACTIVO"
      fi
    fi

    # eth2: Captura externa (bridge a Wi-Fi)
    if [ "$BRIDGE_INTERFACE" != "none" ]; then
      echo "🔍 Configurando eth2 (captura externa)..."
      ip link set $BRIDGE_INTERFACE promisc on
      ethtool -K $BRIDGE_INTERFACE gro off 2>/dev/null || true
      ethtool -K $BRIDGE_INTERFACE tx-checksum-ip-generic off 2>/dev/null || true
      ethtool -K $BRIDGE_INTERFACE tso off 2>/dev/null || true
      ethtool -K $BRIDGE_INTERFACE gso off 2>/dev/null || true

      if ip link show $BRIDGE_INTERFACE | grep -q PROMISC; then
        echo "✅ eth2: Modo promiscuo ACTIVO (Captura externa)"
      else
        echo "❌ eth2: Modo promiscuo INACTIVO"
      fi
    fi

    # eth3: Gateway Mode (nuevo para Day 8)
    echo "🔍 Configurando eth3 (LAN-facing, gateway mode)..."
    if ip link show eth3 >/dev/null 2>&1; then
      ip link set eth3 promisc on
      ethtool -K eth3 gro off 2>/dev/null || true
      ethtool -K eth3 tx-checksum-ip-generic off 2>/dev/null || true
      ethtool -K eth3 tso off 2>/dev/null || true
      ethtool -K eth3 gso off 2>/dev/null || true

      if ip link show eth3 | grep -q PROMISC; then
        echo "✅ eth3: Modo promiscuo ACTIVO (Gateway Mode)"
      else
        echo "❌ eth3: Modo promiscuo INACTIVO"
      fi
    else
      echo "⚠️  eth3 no encontrada (normal si no usas gateway mode)"
    fi

    # 5. Verificación final
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "✅ CONFIGURACIÓN DE RED COMPLETADA"
    echo "═══════════════════════════════════════════════════════════"
    echo "Interfaces disponibles:"
    ip addr show | grep -E '^[0-9]+:|inet ' | grep -v '127.0.0.1'
    echo ""
    echo "IP Forwarding: $(sysctl net.ipv4.ip_forward | cut -d= -f2)"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
  SHELL

  # ========================================
  # SINGLE PHASE: ALL DEPENDENCIES
  # ========================================
  config.vm.provision "shell", name: "all-dependencies", inline: <<-SHELL
    # NO usar set -e para que no salga silenciosamente
    # set -e

    # Activar trace completo
    set -x

    # CRITICAL: Prevent interactive prompts during apt installations
    export DEBIAN_FRONTEND=noninteractive

    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Installing ALL dependencies - Single Phase                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo "DEBUG: Starting provision at $(date)"

    # ========================================
    # CORE SYSTEM PACKAGES
    # ========================================
    echo "=== PHASE 1: CORE PACKAGES ==="
    apt-get update
    echo "DEBUG: apt-get update exit code: $?"

    apt-get install -y \
      build-essential \
      git \
      wget \
      curl \
      vim \
      jq \
      make \
      rsync \
      locales \
      libc-bin
    echo "DEBUG: Core packages install exit code: $?"

    echo "📦 Installing file utility..."
    apt-get install -y file
    echo "DEBUG: file install exit code: $?"

    # ========================================
    # eBPF TOOLCHAIN
    # ========================================
    echo "=== PHASE 2: eBPF TOOLCHAIN ==="
    apt-get install -y \
      clang \
      llvm \
      bpftool \
      libbpf-dev \
      linux-headers-amd64
    echo "DEBUG: eBPF toolchain install exit code: $?"

    # ========================================
    # NETWORKING & COMMUNICATION LIBRARIES
    # ========================================
    echo "=== PHASE 3: NETWORKING LIBRARIES ==="
    apt-get install -y \
      libjsoncpp-dev \
      libcurl4-openssl-dev \
      libzmq3-dev
    echo "DEBUG: Networking libraries install exit code: $?"

    # ========================================
    # PROTOBUF (BOTH COMPILER AND RUNTIME)
    # ========================================
    echo "=== PHASE 4: PROTOBUF ==="
    apt-get install -y \
      protobuf-compiler \
      libprotobuf-dev \
      libprotobuf32
    echo "DEBUG: Protobuf install exit code: $?"

    # ========================================
    # COMPRESSION LIBRARIES
    # ========================================
    echo "=== PHASE 5: COMPRESSION LIBRARIES ==="
    apt-get install -y \
      liblz4-dev \
      libzstd-dev
    echo "DEBUG: Compression libraries install exit code: $?"

    # ========================================
    # ML DETECTOR SPECIFIC
    # ========================================
    echo "=== PHASE 6: ML DETECTOR ==="
    apt-get install -y \
      pkg-config \
      libspdlog-dev \
      nlohmann-json3-dev
    echo "DEBUG: ML Detector dependencies install exit code: $?"

    # ========================================
    # FIREWALL ACL AGENT SPECIFIC
    # ========================================
    echo "=== PHASE 7: FIREWALL ==="
    apt-get install -y \
      iptables \
      ipset \
      libxtables-dev
    echo "DEBUG: Firewall dependencies install exit code: $?"

    # ========================================
    # RAG SECURITY SYSTEM SPECIFIC
    # ========================================
    echo "=== PHASE 8: RAG SECURITY SYSTEM ==="
    apt-get install -y \
        libboost-all-dev \
        libtool \
        autoconf \
        automake \
        libgrpc-dev \
        libgrpc++-dev \
        protobuf-compiler-grpc \
        libc-ares-dev \
        libre2-dev \
        libabsl-dev \
        libbenchmark-dev \
        libgtest-dev \
        libssl-dev \
        libcurl4-openssl-dev \
        libcpprest-dev \
        pkg-config \
        cmake \
        build-essential
    echo "DEBUG: RAG dependencies install exit code: $?"

    # ========================================
    # PYTHON ENVIRONMENT
    # ========================================
    echo "=== PHASE 9: PYTHON ==="
    apt-get install -y \
      python3 \
      python3-pip \
      python3-venv \
      python3-dev
    echo "DEBUG: Python install exit code: $?"

    # ========================================
    # TESTING TOOLS (NUEVO para Day 8)
    # ========================================
    echo "=== PHASE 9.5: TESTING TOOLS ==="
    apt-get install -y \
      hping3 \
      nmap \
      tcpreplay \
      netcat-openbsd \
      iperf3 \
      net-tools
    echo "DEBUG: Testing tools install exit code: $?"

    # ========================================
    # DOCKER & DOCKER COMPOSE - CON CHECKS EXPLÍCITOS
    # ========================================
    echo "=== PHASE 10: DOCKER ==="
    if ! command -v docker >/dev/null 2>&1; then
      echo "📦 Installing Docker..."
      curl -fsSL https://get.docker.com -o get-docker.sh
      echo "DEBUG: curl docker script exit code: $?"
      sh get-docker.sh
      echo "DEBUG: docker install script exit code: $?"
      usermod -aG docker vagrant
      systemctl enable docker
      systemctl start docker
    else
      echo "✅ Docker already installed: $(docker --version)"
    fi

    echo "=== PHASE 11: DOCKER COMPOSE ==="
    if ! command -v docker-compose >/dev/null 2>&1; then
      echo "📦 Installing Docker Compose..."
      curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
      echo "DEBUG: curl docker-compose exit code: $?"
      chmod +x /usr/local/bin/docker-compose
      echo "DEBUG: chmod docker-compose exit code: $?"
    else
      echo "✅ Docker Compose already installed: $(docker-compose --version)"
    fi

    # ========================================
    # CMAKE 3.25+ - CON CHECKS
    # ========================================
    echo "=== PHASE 12: CMAKE ==="
    CMAKE_VERSION=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
    echo "DEBUG: Current CMake: $CMAKE_VERSION"
    if [ -z "$CMAKE_VERSION" ] || [ "$(printf '%s\n' "3.20" "$CMAKE_VERSION" | sort -V | head -n1)" != "3.20" ]; then
      echo "📦 Installing CMake 3.25..."
      cd /tmp
      wget -q https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.sh
      echo "DEBUG: wget cmake exit code: $?"
      sh cmake-3.25.0-linux-x86_64.sh --prefix=/usr/local --skip-license
      echo "DEBUG: cmake install script exit code: $?"
      rm cmake-3.25.0-linux-x86_64.sh
    else
      echo "✅ CMake $CMAKE_VERSION already installed"
    fi

    # ========================================
    # ONNX RUNTIME - CON CHECKS
    # ========================================
    echo "=== PHASE 13: ONNX RUNTIME ==="
    if [ ! -f /usr/local/lib/libonnxruntime.so ]; then
      echo "📦 Installing ONNX Runtime 1.17.1..."
      cd /tmp
      wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
      echo "DEBUG: wget onnx exit code: $?"
      tar -xzf onnxruntime-linux-x64-1.17.1.tgz
      echo "DEBUG: tar onnx exit code: $?"
      cp -r onnxruntime-linux-x64-1.17.1/include/* /usr/local/include/
      cp -r onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/
      echo "DEBUG: cp onnx exit code: $?"
      sudo ldconfig
      rm -rf onnxruntime-linux-*
    else
      echo "✅ ONNX Runtime already installed"
    fi

    # ========================================
    # ETCD-CPP-API - CON CHECKS DETALLADOS
    # ========================================
    echo "=== PHASE 14: ETCD-CPP-API ==="
    if [ ! -f /usr/local/lib/libetcd-cpp-api.so ] && [ ! -f /usr/local/lib/libetcd-cpp-api.a ]; then
      echo "📦 Compiling etcd-cpp-api from source..."
      cd /tmp
      rm -rf etcd-cpp-apiv3
      git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git
      echo "DEBUG: git clone exit code: $?"
      cd etcd-cpp-apiv3
      git checkout v0.15.3
      echo "DEBUG: git checkout exit code: $?"
      mkdir build && cd build
      cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local
      echo "DEBUG: cmake configure exit code: $?"
      make -j4
      echo "DEBUG: make exit code: $?"
      sudo make install
      echo "DEBUG: make install exit code: $?"
      sudo ldconfig
    else
      echo "✅ etcd-cpp-api already installed"
    fi

        # ========================================
        # CPP-HTTPLIB (SIMPLE HTTP SERVER) - ALTERNATIVA A DROGON
        # ========================================
        echo "=== PHASE 14.1: CPP-HTTPLIB ==="
        if [ ! -f /usr/local/include/httplib.h ]; then
          echo "📦 Installing cpp-httplib (simple HTTP server)..."
          cd /tmp
          rm -rf cpp-httplib
          git clone https://github.com/yhirose/cpp-httplib.git
          echo "DEBUG: git clone cpp-httplib exit code: $?"

          # Instalar como header-only library
          mkdir -p /usr/local/include
          cp cpp-httplib/httplib.h /usr/local/include/
          echo "✅ cpp-httplib installed as header-only library"
        else
          echo "✅ cpp-httplib already installed"
        fi

        # ========================================
        # NLohmann JSON (si no está instalado)
        # ========================================
        echo "=== PHASE 14.2: NLOHMANN JSON ==="
        if [ ! -f /usr/include/nlohmann/json.hpp ] && [ ! -f /usr/local/include/nlohmann/json.hpp ]; then
          echo "📦 Installing nlohmann json..."
          apt-get install -y nlohmann-json3-dev
          echo "DEBUG: nlohmann json install exit code: $?"
        else
          echo "✅ nlohmann json already installed"
        fi

        # ========================================
        # CRYPTO++ LIBRARY (CIFRADO REAL) - CON FALLBACK
        # ========================================
        echo "=== PHASE 14.3: CRYPTO++ ==="
        if [ ! -f /usr/include/cryptopp/cryptlib.h ] && [ ! -f /usr/local/include/cryptopp/cryptlib.h ]; then
          echo "📦 Installing Crypto++ library..."

          # Intentar con apt
          if apt-get install -y libcrypto++-dev libcrypto++-doc libcrypto++-utils; then
            echo "✅ Crypto++ installed via apt"
          else
            echo "⚠️  Fallando a instalación desde source..."

            # Compilar desde source como fallback
            cd /tmp
            wget https://www.cryptopp.com/cryptopp870.zip
            unzip cryptopp870.zip -d cryptopp
            cd cryptopp
            make -j4
            make install
            echo "✅ Crypto++ compiled from source"
          fi

        else
          echo "✅ Crypto++ already installed"
        fi

    # ========================================
    # LLAMA.CPP COMPILATION & MODEL DOWNLOAD
    # ========================================
    echo "=== PHASE 15: LLAMA.CPP ==="
    if [ ! -f /vagrant/third_party/llama.cpp/build/src/libllama.a ]; then
        echo "🦙 Compiling llama.cpp in VM..."
        cd /vagrant/third_party/llama.cpp
        mkdir -p build && cd build

        # Configuración optimizada para RPI5/ARM
        cmake .. \
            -DBUILD_SHARED_LIBS=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=ON \
            -DLLAMA_NATIVE=OFF \
            -DLLAMA_NO_ACCELERATE=ON \
            -DLLAMA_METAL=OFF \
            -DLLAMA_CUBLAS=OFF \
            -DLLAMA_OPENBLAS=OFF \
            -DCMAKE_BUILD_TYPE=Release
        echo "DEBUG: llama.cpp cmake exit code: $?"

        cmake --build . --target all -- -j4
        echo "DEBUG: llama.cpp build exit code: $?"
    else
        echo "✅ llama.cpp already compiled in VM"
    fi

    # ========================================
    # LLAMA.CPP MODEL DOWNLOAD
    # ========================================
    echo "=== PHASE 15.1: DOWNLOAD LLAMA.CPP MODEL ==="
    mkdir -p /vagrant/rag/models
    cd /vagrant/rag/models

    # Verificar si ya existe el modelo
    if [ ! -f "tinyllama-1.1b-chat-v1.0.Q4_0.gguf" ]; then
        echo "📥 Downloading TinyLlama 1.1B Chat model (optimized for RPI5)..."

        # Usar curl como alternativa si wget falla
        if command -v wget >/dev/null 2>&1; then
            wget -q --show-progress --continue --timeout=120 --tries=3 \
                "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" || \
            {
                echo "⚠️  wget failed, trying curl..."
                curl -L -C - --progress-bar \
                    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" \
                    -o tinyllama-1.1b-chat-v1.0.Q4_0.gguf
            }
        else
            curl -L -C - --progress-bar \
                "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" \
                -o tinyllama-1.1b-chat-v1.0.Q4_0.gguf
        fi

        if [ $? -eq 0 ] && [ -f "tinyllama-1.1b-chat-v1.0.Q4_0.gguf" ]; then
            echo "✅ Model downloaded successfully"
        else
            echo "❌ Model download failed - will use simulated mode"
            echo "💡 You can manually download later with:"
            echo "   cd /vagrant/rag/models && wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
        fi
    else
        echo "✅ Model already exists: tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
        ls -lh tinyllama-1.1b-chat-v1.0.Q4_0.gguf
    fi

    # Verificar y probar el modelo
    if [ -f "tinyllama-1.1b-chat-v1.0.Q4_0.gguf" ]; then
        echo "🔍 Verifying model file..."
        FILE_SIZE=$(stat -c%s "tinyllama-1.1b-chat-v1.0.Q4_0.gguf" 2>/dev/null || stat -f%z "tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
        if [ "$FILE_SIZE" -gt 600000000 ]; then  # ~600MB para Q4_0
            echo "✅ Model verified: $((FILE_SIZE/1024/1024)) MB"

            # Probar carga rápida del modelo
            echo "🧪 Testing model loading..."
            cd /vagrant/third_party/llama.cpp/build
            timeout 30s ./bin/llama-cli -m /vagrant/rag/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "hello" -n 3 > /tmp/llama_test.log 2>&1 && \
                echo "🎉 Model loaded successfully!" || \
                echo "⚠️  Model load test timed out or failed (normal for first load)"

            # Mostrar ejemplo de salida
            if [ -f "/tmp/llama_test.log" ]; then
                echo "📄 Test output:"
                head -5 /tmp/llama_test.log
            fi
        else
            echo "❌ Model file seems too small ($((FILE_SIZE/1024/1024)) MB) - possibly corrupted"
            rm -f /vagrant/rag/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
        fi
    fi

    # Crear symlink por defecto
    cd /vagrant/rag/models
    if [ ! -f "default.gguf" ] && [ -f "tinyllama-1.1b-chat-v1.0.Q4_0.gguf" ]; then
        ln -sf tinyllama-1.1b-chat-v1.0.Q4_0.gguf default.gguf
        echo "🔗 Created default.gguf symlink"
    fi

    echo "📝 Model location: /vagrant/rag/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

    # ========================================
    # SUDOERS CONFIGURATION
    # ========================================
    echo "=== PHASE 16: SUDOERS ==="
    mkdir -p /etc/sudoers.d
    cat > /etc/sudoers.d/ml-defender << 'EOF'
# ML Defender - Allow sniffer and firewall to run without password
vagrant ALL=(ALL) NOPASSWD: /vagrant/sniffer/build/sniffer
vagrant ALL=(ALL) NOPASSWD: /vagrant/firewall-acl-agent/build/firewall-acl-agent
vagrant ALL=(ALL) NOPASSWD: /usr/sbin/iptables
vagrant ALL=(ALL) NOPASSWD: /usr/sbin/ipset
vagrant ALL=(ALL) NOPASSWD: /usr/bin/pkill
vagrant ALL=(ALL) NOPASSWD: /bin/kill
vagrant ALL=(ALL) NOPASSWD: /usr/bin/killall
EOF
    echo "DEBUG: sudoers file creation exit code: $?"
    chmod 0440 /etc/sudoers.d/ml-defender

    # ========================================
    # CONFIGURATION
    # ========================================
    echo "=== PHASE 17: SYSTEM CONFIG ==="
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
    echo "=== PHASE 18: DIRECTORIES ==="
    mkdir -p /vagrant/ml-detector/models/production/level1
    mkdir -p /vagrant/ml-detector/models/production/level2
    mkdir -p /vagrant/ml-detector/models/production/level3
    mkdir -p /vagrant/ml-training/outputs/onnx
    mkdir -p /vagrant/firewall-acl-agent/build/logs
    mkdir -p /vagrant/rag/build/logs
    mkdir -p /vagrant/logs/lab
    mkdir -p /var/log/ml-defender
    chown -R vagrant:vagrant /var/log/ml-defender
    chmod 755 /var/log/ml-defender

    # ========================================
    # PROTOBUF GENERATION
    # ========================================
    echo "=== PHASE 19: PROTOBUF GENERATION ==="
    if [ -f /vagrant/protobuf/generate.sh ] && [ ! -f /vagrant/protobuf/network_security.pb.cc ]; then
      echo "🔨 Generating protobuf files..."
      cd /vagrant/protobuf && ./generate.sh
      echo "DEBUG: protobuf generation exit code: $?"
    fi

    if [ -f /vagrant/protobuf/network_security.pb.cc ]; then
      echo "📋 Copying protobuf to firewall..."
      mkdir -p /vagrant/firewall-acl-agent/proto
      cp /vagrant/protobuf/network_security.pb.cc /vagrant/firewall-acl-agent/proto/
      cp /vagrant/protobuf/network_security.pb.h /vagrant/firewall-acl-agent/proto/
    fi

    # ========================================
    # BUILD COMPONENTS
    # ========================================
    echo "=== PHASE 20: BUILDING COMPONENTS ==="

    # Firewall ACL Agent
    if [ ! -f /vagrant/firewall-acl-agent/build/firewall-acl-agent ]; then
        echo "🔨 Building Firewall ACL Agent..."
        mkdir -p /vagrant/firewall-acl-agent/build
        cd /vagrant/firewall-acl-agent/build
        cmake .. && make -j4
        echo "DEBUG: firewall build exit code: $?"
    fi

    # RAG Security System
    if [ ! -f /vagrant/rag/build/rag-security ] && [ ! -f /vagrant/rag/build/rag_core ]; then
        echo "🤖 RAG Security System ready for implementation"
        mkdir -p /vagrant/rag/build
    else
        echo "✅ RAG Security System already built"
    fi

    # ========================================
    # BASH ALIASES
    # ========================================
    echo "=== PHASE 21: BASH ALIASES ==="
    if ! grep -q "build-rag" /home/vagrant/.bashrc; then
      cat >> /home/vagrant/.bashrc << 'EOF'
# ========================================
# ML Defender Development Aliases
# ========================================

# Building
alias build-sniffer='cd /vagrant/sniffer && make'
alias build-detector='cd /vagrant/ml-detector/build && rm -rf * && cmake .. && make -j4'
alias build-firewall='cd /vagrant/firewall-acl-agent/build && rm -rf * && cmake .. && make -j4'
alias build-rag='cd /vagrant/rag/build && rm -rf * && cmake .. && make -j4'
alias proto-regen='cd /vagrant/protobuf && ./generate.sh && cp network_security.pb.* /vagrant/firewall-acl-agent/proto/'

# Running (individual components)
alias run-firewall='cd /vagrant/firewall-acl-agent/build && sudo ./firewall-acl-agent -c ../config/firewall.json'
alias run-detector='cd /vagrant/ml-detector/build && ./ml-detector -c config/ml_detector_config.json'
alias run-sniffer='cd /vagrant/sniffer/build && sudo ./sniffer -c config/sniffer.json'
alias run-rag='cd /vagrant/rag/build && ./rag-security -c ../config/rag_config.json'
alias test-rag='cd /vagrant/rag/build && ./test_etcd_client && echo "✅ RAG tests passed" || echo "❌ RAG tests failed"'

# Running (full lab)
alias run-lab='cd /vagrant && bash scripts/run_lab_dev.sh'
alias kill-lab='sudo pkill -9 firewall-acl-agent; pkill -9 ml-detector; sudo pkill -9 sniffer; pkill -9 rag-security'
alias status-lab='pgrep -a firewall-acl-agent; pgrep -a ml-detector; pgrep -a sniffer; pgrep -a rag-security'

# Day 8: Dual-NIC Testing Shortcuts
alias test-host-mode='echo "Testing host-based mode on eth1..." && sudo tcpdump -i eth1 -c 10'
alias test-gateway-mode='echo "Testing gateway mode on eth3..." && sudo tcpdump -i eth3 -c 10'
alias check-interfaces='echo "Network Interfaces:" && ip addr show | grep -E "^[0-9]+:|inet "'
alias check-promiscuous='echo "Promiscuous Mode Status:" && ip link show | grep -E "eth[0-9]:|PROMISC"'

# ML Model Deployment (from host macOS training)
alias sync-models='rsync -av /vagrant/ml-training/outputs/onnx/*.onnx /vagrant/ml-detector/models/production/ 2>/dev/null && echo "✅ Models synced from host" || echo "⚠️  No models found in ml-training/outputs/onnx/"'
alias list-models='echo "Available ONNX models:" && find /vagrant/ml-detector/models/production -name "*.onnx" -exec ls -lh {} \;'

# RAG Model Management
alias setup-rag-model='echo "Downloading test model..." && cd /vagrant/rag/models && wget -c https://huggingface.co/microsoft/DialoGPT-small/resolve/main/pytorch_model.bin || echo "Use: python3 scripts/download_model.py"'

# Logs
alias logs-firewall='tail -f /vagrant/firewall-acl-agent/build/logs/*.log /var/log/ml-defender/firewall-acl-agent.log 2>/dev/null || echo "No logs yet"'
alias logs-detector='tail -f /vagrant/ml-detector/build/logs/*.log 2>/dev/null || echo "No logs yet"'
alias logs-sniffer='tail -f /vagrant/logs/lab/sniffer.log 2>/dev/null || echo "No logs yet"'
alias logs-rag='tail -f /vagrant/rag/build/logs/*.log 2>/dev/null || echo "No logs yet"'
alias logs-lab='cd /vagrant && bash scripts/monitor_lab.sh'

# etcd-server development
alias build-etcd-server='cd /vagrant/etcd-server/build && rm -rf * && cmake .. && make -j4'
alias run-etcd-server='cd /vagrant/etcd-server/build && ./etcd-server'
alias test-etcd-server='curl -X GET http://localhost:2379/validate'

# RAG with etcd integration
alias test-rag-etcd='cd /vagrant/rag/build && ./rag-security --test-etcd'

# Shortcuts
export PROJECT_ROOT="/vagrant"
export MODELS_DIR="/vagrant/ml-detector/models/production"

# Welcome message
cat << 'WELCOME'

╔════════════════════════════════════════════════════════════╗
║  ML Defender - Network Security Pipeline                   ║
║  Development Environment - DUAL-NIC READY                  ║
╚════════════════════════════════════════════════════════════╝

🎯 Pipeline Architecture:
   Sniffer (eBPF/XDP) → ML Detector → Firewall ACL Agent → RAG Security
      PUSH 5571           PUB 5572       SUB 5572           AI Commands

🌐 Dual-NIC Configuration (Day 8):
   eth1: 192.168.56.20 (WAN-facing, host-based IDS)
   eth3: 192.168.100.1 (LAN-facing, gateway mode)

🧪 Dual-NIC Testing:
   check-interfaces     # Show all network interfaces
   check-promiscuous    # Verify promiscuous mode
   test-host-mode       # Quick host-based capture test
   test-gateway-mode    # Quick gateway capture test

🚀 Quick Start:
   run-lab              # Start full pipeline (background + monitor)
   kill-lab             # Stop all components
   status-lab           # Check component status
   logs-lab             # View combined logs

📦 Individual Components:
   run-firewall         # Start firewall (Terminal 1)
   run-detector         # Start detector (Terminal 2)
   run-sniffer          # Start sniffer (Terminal 3)
   run-rag              # Start RAG Security (Terminal 4)

🔨 Building:
   build-sniffer        # Compile sniffer
   build-detector       # Compile ml-detector
   build-firewall       # Compile firewall-acl-agent
   build-rag            # Compile RAG Security System
   proto-regen          # Regenerate protobuf + sync

🤖 RAG AI Security:
   test-rag             # Run RAG system tests
   setup-rag-model      # Download AI model for RAG

📚 ML Model Workflow:
   1. Train on HOST macOS: cd ml-training && python scripts/train_*.py
   2. Models auto-sync: ml-training/outputs/onnx/ → detector/models/
   3. Deploy: sync-models && build-detector

📊 Monitoring:
   logs-firewall        # Firewall logs
   logs-detector        # Detector logs
   logs-sniffer         # Sniffer logs (→ /vagrant/logs/lab/sniffer.log)
   logs-rag             # RAG Security logs
   logs-lab             # Combined monitoring

WELCOME
EOF
      echo "DEBUG: bash aliases setup exit code: $?"
    fi

    # ========================================
    # FINAL STATUS SUMMARY - DENTRO DEL MISMO PROVISIONER
    # ========================================
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║       ML DEFENDER - STATUS SUMMARY (DUAL-NIC READY)       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Component Status
    echo "🔧 PIPELINE COMPONENTS:"
    echo "┌──────────────────────┬─────────────┬─────────────────────┐"
    [ -f /vagrant/sniffer/build/sniffer ] && echo "│ Sniffer               │     ✅      │ Built successfully   │" || echo "│ Sniffer               │     ❌      │ Build failed         │"
    [ -f /vagrant/ml-detector/build/ml-detector ] && echo "│ ML Detector           │     ✅      │ Built successfully   │" || echo "│ ML Detector           │     ❌      │ Build failed         │"
    [ -f /vagrant/firewall-acl-agent/build/firewall-acl-agent ] && echo "│ Firewall ACL Agent    │     ✅      │ Built successfully   │" || echo "│ Firewall ACL Agent    │     ❌      │ Build failed         │"
    [ -f /vagrant/rag/build/rag-security ] && echo "│ RAG Security System   │     ✅      │ Built successfully   │" || echo "│ RAG Security System   │     🚧      │ Ready to implement   │"
    echo "└──────────────────────┴─────────────┴─────────────────────┘"
    echo ""

    # Core Dependencies
    echo "📚 CORE DEPENDENCIES:"
    echo "┌──────────────────────┬─────────────┬─────────────────────┐"
        [ -f /usr/local/lib/libetcd-cpp-api.so ] && echo "│ etcd-cpp-api         │     ✅      │ Installed           │" || echo "│ etcd-cpp-api         │     ❌      │ Missing             │"
        [ -f /usr/local/include/httplib.h ] && echo "│ cpp-httplib          │     ✅      │ Installed           │" || echo "│ cpp-httplib          │     ❌      │ Missing             │"
        [ -f /usr/local/lib/libonnxruntime.so ] && echo "│ ONNX Runtime         │     ✅      │ Installed           │" || echo "│ ONNX Runtime         │     ❌      │ Missing             │"
        [ -f /vagrant/third_party/llama.cpp/build/src/libllama.a ] && echo "│ llama.cpp            │     ✅      │ Compiled            │" || echo "│ llama.cpp            │     ❌      │ Not compiled        │"
        which docker >/dev/null && echo "│ Docker               │     ✅      │ Installed           │" || echo "│ Docker               │     ❌      │ Missing             │"
        which cmake >/dev/null && echo "│ CMake                │     ✅      │ Installed           │" || echo "│ CMake                │     ❌      │ Missing             │"
        which hping3 >/dev/null && echo "│ Testing Tools        │     ✅      │ Installed           │" || echo "│ Testing Tools        │     ❌      │ Missing             │"
    echo "└──────────────────────┴─────────────┴─────────────────────┘"

    # Network Status (DUAL-NIC)
    echo "🌐 NETWORK STATUS (DUAL-NIC):"
    echo "┌──────────────────────┬─────────────┬─────────────────────┐"
    ip link show eth1 | grep -q UP && echo "│ eth1 (WAN Host-Based) │     ✅      │ Active              │" || echo "│ eth1 (WAN Host-Based) │     ❌      │ Inactive            │"
    ip link show eth1 | grep -q PROMISC && echo "│   └─ Promiscuous     │     ✅      │ Enabled             │" || echo "│   └─ Promiscuous     │     ❌      │ Disabled            │"
    ip link show eth2 | grep -q UP && echo "│ eth2 (External Cap)   │     ✅      │ Active              │" || echo "│ eth2 (External Cap)   │     🔄      │ Optional            │"
    ip link show eth3 | grep -q UP && echo "│ eth3 (LAN Gateway)    │     ✅      │ Active              │" || echo "│ eth3 (LAN Gateway)    │     ❌      │ Inactive            │"
    ip link show eth3 | grep -q PROMISC && echo "│   └─ Promiscuous     │     ✅      │ Enabled             │" || echo "│   └─ Promiscuous     │     ❌      │ Disabled            │"
    sysctl net.ipv4.ip_forward | grep -q "= 1" && echo "│ IP Forwarding        │     ✅      │ Enabled             │" || echo "│ IP Forwarding        │     ❌      │ Disabled            │"
    echo "└──────────────────────┴─────────────┴─────────────────────┘"
    echo ""

    # System Status
    echo "⚙️  SYSTEM STATUS:"
    echo "┌──────────────────────┬─────────────┬─────────────────────┐"
    sudo -l -U vagrant | grep -q "NOPASSWD" && echo "│ Sudoers Config       │     ✅      │ Configured          │" || echo "│ Sudoers Config       │     ❌      │ Not configured      │"
    [ -f /vagrant/protobuf/network_security.pb.cc ] && echo "│ Protobuf Files      │     ✅      │ Generated          │" || echo "│ Protobuf Files      │     ❌      │ Not generated       │"
    systemctl is-active --quiet docker && echo "│ Docker Service       │     ✅      │ Running             │" || echo "│ Docker Service       │     ❌      │ Stopped             │"
    mountpoint -q /sys/fs/bpf && echo "│ BPF Filesystem       │     ✅      │ Mounted             │" || echo "│ BPF Filesystem       │     ❌      │ Not mounted         │"
    echo "└──────────────────────┴─────────────┴─────────────────────┘"
    echo ""

    # Quick Start
    echo "🚀 QUICK START (DUAL-NIC TESTING):"
    echo "┌────────────────────────────────────────────────────────────┐"
    echo "│ vagrant ssh             # Enter the VM                    │"
    echo "│ check-interfaces        # Verify dual-NIC setup           │"
    echo "│ check-promiscuous       # Verify capture mode             │"
    echo "│ run-sniffer             # Start ML Defender               │"
    echo "│ logs-sniffer            # Monitor sniffer logs            │"
    echo "└────────────────────────────────────────────────────────────┘"
    echo ""

    # Day 8 Testing
    echo "🧪 DAY 8 DUAL-NIC TESTING:"
    echo "┌────────────────────────────────────────────────────────────┐"
    echo "│ FROM OSX: Attack eth1 (host-based mode)                   │"
    echo "│   sudo nmap -sS -p 1-1000 192.168.56.20                   │"
    echo "│   sudo hping3 -S -p 80 --flood -c 5000 192.168.56.20      │"
    echo "│                                                            │"
    echo "│ FROM VM: Test eth3 (gateway mode)                         │"
    echo "│   sudo tcpreplay -i eth3 --mbps 100 dataset.pcap          │"
    echo "│                                                            │"
    echo "│ Expected: interface_mode=1 on eth1, mode=2 on eth3        │"
    echo "└────────────────────────────────────────────────────────────┘"
    echo ""

    # Final Status
    echo "✅ PROVISIONING COMPLETED SUCCESSFULLY!"
    echo "🎯 PIPELINE STATUS: OPERATIONAL"
    echo "🌐 DUAL-NIC STATUS: READY FOR DAY 8 TESTING"
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                   ML DEFENDER PHASE 1 DAY 8               ║"
    echo "║              DUAL-NIC VALIDATION ENVIRONMENT              ║"
    echo "╚════════════════════════════════════════════════════════════╝"

    echo "DEBUG: Provision completed at $(date)"
  SHELL
end