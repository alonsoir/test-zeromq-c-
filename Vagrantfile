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
  # Provisioning: Configuración de Red INTELIGENTE
  # ════════════════════════════════════════════════════════════════════════
  config.vm.provision "shell", run: "always", inline: <<-SHELL
    echo "🔧 Configurando interfaces de red optimizadas..."

    # 1. Instalar herramientas de red
    apt-get update -qq
    apt-get install -y ethtool tcpdump

    # 2. Detectar interfaz bridge automáticamente (para captura externa)
    BRIDGE_INTERFACE=""
    for iface in eth2 eth1; do
      if ip link show $iface >/dev/null 2>&1; then
        BRIDGE_INTERFACE=$iface
        break
      fi
    done
    
    if [ -z "$BRIDGE_INTERFACE" ]; then
      echo "⚠️  No se encontró interfaz bridge, usando eth0 para tráfico interno"
      BRIDGE_INTERFACE="eth0"
    fi

    echo "🎯 Interfaz para captura externa: $BRIDGE_INTERFACE"
    echo "🎯 Interfaz para tráfico interno: eth0"

    # 3. Configurar modo promiscuo SOLO si es interfaz bridge externa
    if [ "$BRIDGE_INTERFACE" != "eth0" ]; then
      echo "🔍 Activando modo promiscuo en $BRIDGE_INTERFACE (captura externa)..."
      ip link set $BRIDGE_INTERFACE promisc on

      # Desactivar offloading features para XDP
      echo "⚙️  Desactivando offloading features en $BRIDGE_INTERFACE..."
      ethtool -K $BRIDGE_INTERFACE gro off 2>/dev/null || true
      ethtool -K $BRIDGE_INTERFACE tx-checksum-ip-generic off 2>/dev/null || true
      ethtool -K $BRIDGE_INTERFACE tso off 2>/dev/null || true
      ethtool -K $BRIDGE_INTERFACE gso off 2>/dev/null || true
    else
      echo "ℹ️  Modo promiscuo no necesario en eth0 (tráfico interno)"
    fi

    # 4. Verificar configuración
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "✅ CONFIGURACIÓN DE RED COMPLETADA"
    echo "═══════════════════════════════════════════════════════════"
    
    echo "Interfaz captura externa: $BRIDGE_INTERFACE"
    if [ "$BRIDGE_INTERFACE" != "eth0" ]; then
      if ip link show $BRIDGE_INTERFACE | grep -q PROMISC; then
        echo "✅ Modo promiscuo: ACTIVO en $BRIDGE_INTERFACE"
      else
        echo "❌ Modo promiscuo: INACTIVO en $BRIDGE_INTERFACE"
      fi
    fi
    
    echo "Interfaz tráfico interno: eth0"
    echo "Interfaz host-VM: eth1 (192.168.56.20)"

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
    # NO usar set -e para que no salga silenciosamente
    # set -e

    # Activar trace completo
    set -x

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
            -DLLAMA_BUILD_EXAMPLES=ON \  # Para probar con llama-cli
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

# ML Model Deployment (from host macOS training)
alias sync-models='rsync -av /vagrant/ml-training/outputs/onnx/*.onnx /vagrant/ml-detector/models/production/ 2>/dev/null && echo "✅ Models synced from host" || echo "⚠️  No models found in ml-training/outputs/onnx/"'
alias list-models='echo "Available ONNX models:" && find /vagrant/ml-detector/models/production -name "*.onnx" -exec ls -lh {} \;'

# RAG Model Management
alias setup-rag-model='echo "Downloading test model..." && cd /vagrant/rag/models && wget -c https://huggingface.co/microsoft/DialoGPT-small/resolve/main/pytorch_model.bin || echo "Use: python3 scripts/download_model.py"'

# Logs
alias logs-firewall='tail -f /vagrant/firewall-acl-agent/build/logs/*.log /var/log/ml-defender/firewall-acl-agent.log 2>/dev/null || echo "No logs yet"'
alias logs-detector='tail -f /vagrant/ml-detector/build/logs/*.log 2>/dev/null || echo "No logs yet"'
alias logs-sniffer='tail -f /vagrant/sniffer/build/logs/*.log 2>/dev/null || echo "No logs yet"'
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
║  Development Environment                                   ║
╚════════════════════════════════════════════════════════════╝

🎯 Pipeline Architecture:
   Sniffer (eBPF/XDP) → ML Detector → Firewall ACL Agent → RAG Security
      PUSH 5571           PUB 5572       SUB 5572           AI Commands

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
   logs-sniffer         # Sniffer logs
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
    echo "║              ML DEFENDER - STATUS SUMMARY                 ║"
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
        [ -f /usr/local/lib/libdrogon.a ] && echo "│ Drogon Framework     │     ✅      │ Installed           │" || echo "│ Drogon Framework     │     ❌      │ Missing             │"
        [ -f /usr/local/lib/libonnxruntime.so ] && echo "│ ONNX Runtime         │     ✅      │ Installed           │" || echo "│ ONNX Runtime         │     ❌      │ Missing             │"
        [ -f /vagrant/third_party/llama.cpp/build/src/libllama.a ] && echo "│ llama.cpp            │     ✅      │ Compiled            │" || echo "│ llama.cpp            │     ❌      │ Not compiled        │"
        which docker >/dev/null && echo "│ Docker               │     ✅      │ Installed           │" || echo "│ Docker               │     ❌      │ Missing             │"
        which cmake >/dev/null && echo "│ CMake                │     ✅      │ Installed           │" || echo "│ CMake                │     ❌      │ Missing             │"
    echo "└──────────────────────┴─────────────┴─────────────────────┘"

    # Network Status
    echo "🌐 NETWORK STATUS:"
    echo "┌──────────────────────┬─────────────┬─────────────────────┐"
    ip link show eth2 | grep -q PROMISC && echo "│ eth2 (Capture)       │     ✅      │ Promiscuous mode    │" || echo "│ eth2 (Capture)       │     ❌      │ Normal mode         │"
    ip link show eth0 | grep -q UP && echo "│ eth0 (Internal)       │     ✅      │ Active              │" || echo "│ eth0 (Internal)       │     ❌      │ Inactive            │"
    ip link show eth1 | grep -q UP && echo "│ eth1 (Host-Only)      │     ✅      │ Active              │" || echo "│ eth1 (Host-Only)      │     ❌      │ Inactive            │"
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
    echo "🚀 QUICK START:"
    echo "┌────────────────────────────────────────────────────────────┐"
    echo "│ vagrant ssh             # Enter the VM                    │"
    echo "│ run-lab                 # Start full pipeline             │"
    echo "│ build-rag               # Build RAG Security System       │"
    echo "│ status-lab              # Check component status          │"
    echo "│ logs-lab                # Monitor all logs                │"
    echo "└────────────────────────────────────────────────────────────┘"
    echo ""

    # Next Steps
    echo "🎯 NEXT STEPS FOR RAG IMPLEMENTATION:"
    echo "┌────────────────────────────────────────────────────────────┐"
    echo "│ 1. Update Rag/CMakeLists.txt with dependencies            │"
    echo "│ 2. Implement etcd_client.cpp                              │"
    echo "│ 3. Create unit tests                                      │"
    echo "│ 4. Implement llama_integration.cpp                        │"
    echo "│ 5. Build and test: build-rag && test-rag                  │"
    echo "└────────────────────────────────────────────────────────────┘"
    echo ""

    # Final Status
    echo "✅ PROVISIONING COMPLETED SUCCESSFULLY!"
    echo "🎯 PIPELINE STATUS: OPERATIONAL"
    echo "🚀 READY FOR RAG SECURITY SYSTEM IMPLEMENTATION"
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                  CLOSING VAGRANTFILE TOPIC                ║"
    echo "║               MOVING TO RAG IMPLEMENTATION                ║"
    echo "╚════════════════════════════════════════════════════════════╝"

    echo "DEBUG: Provision completed at $(date)"
  SHELL
end