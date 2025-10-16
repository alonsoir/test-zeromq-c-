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
  # 1. NAT (implícita, siempre presente en interfaz 1)
  # 2. Private network para acceso host-VM
  config.vm.network "private_network", ip: "192.168.56.20"

  # 3. Bridged network para acceso desde LAN
  # Vagrant preguntará qué interfaz usar durante 'vagrant up'
  # Útil para: acceso desde otros dispositivos, pruebas de red real
  config.vm.network "public_network", bridge: "en0: Wi-Fi"

  # Port forwarding (útil cuando no se usa bridged o para acceso local)
  config.vm.network "forwarded_port", guest: 5555, host: 5555, protocol: "tcp"  # ZeroMQ
  config.vm.network "forwarded_port", guest: 2379, host: 2379, protocol: "tcp"  # etcd client
  config.vm.network "forwarded_port", guest: 2380, host: 2380, protocol: "tcp"  # etcd peer
  config.vm.network "forwarded_port", guest: 3000, host: 3000, protocol: "tcp"  # monitoring
  config.vm.network "forwarded_port", guest: 5571, host: 5571, protocol: "tcp"  # sniffer

  # Synced folder
  config.vm.synced_folder ".", "/vagrant", type: "virtualbox",
      mount_options: ["dmode=775,fmode=775,exec"]

  # Network environment setup for sniffer
  config.vm.provision "shell", inline: <<-SHELL
    echo "Configurando entorno de red para sniffer..."

    # Detectar IP del host
    HOST_IP=$(ip route | grep '^default' | awk '{print $3}' | head -1)
    if [ -z "$HOST_IP" ]; then
      HOST_IP="10.0.2.2"
    fi

    echo "Host IP detectada: $HOST_IP"

    # Detectar IPs de todas las interfaces
    echo "Interfaces de red disponibles:"
    ip -4 addr show | grep inet | awk '{print "  " $NF ": " $2}'

    # Detectar IP de la interfaz bridged (si existe)
    BRIDGED_IP=$(ip -4 addr show | grep "inet.*192.168\|inet.*10\." | grep -v "192.168.56" | awk '{print $2}' | cut -d'/' -f1 | head -1)
    if [ -n "$BRIDGED_IP" ]; then
      echo "IP bridged detectada: $BRIDGED_IP"
    fi

    # Variables de entorno
    {
      echo "VAGRANT_HOST_IP=$HOST_IP"
      echo "SNIFFER_HOST_IP=$HOST_IP"
      echo "SNIFFER_ENDPOINT=tcp://$HOST_IP:5571"
      echo "PRIVATE_NETWORK_IP=192.168.56.20"
      [ -n "$BRIDGED_IP" ] && echo "BRIDGED_NETWORK_IP=$BRIDGED_IP"
    } >> /etc/environment

    # Docker host resolution
    {
      echo "$HOST_IP host.docker.internal"
      echo "$HOST_IP docker.host.internal"
    } >> /etc/hosts
  SHELL

  config.vm.provision "shell", inline: <<-SHELL
    # Instalar herramientas básicas
    echo "Instalar rsync..."
    apt-get update
    apt-get install -y rsync

    # Configurar locales españolas
    echo "Instalar y configurar locales españolas..."
    apt-get install -y locales
    sed -i '/es_ES.UTF-8/s/^# //g' /etc/locale.gen
    locale-gen es_ES.UTF-8
    update-locale LANG=es_ES.UTF-8 LC_ALL=es_ES.UTF-8
    echo 'export LANG=es_ES.UTF-8' >> /etc/profile.d/locale.sh
    echo 'export LC_ALL=es_ES.UTF-8' >> /etc/profile.d/locale.sh
  SHELL

  # Basic provisioning
  config.vm.provision "shell", inline: <<-SHELL
    echo "Instalando dependencias básicas..."
    apt-get update
    apt-get install -y curl wget git vim jq make build-essential

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

    # Habilitar BPF JIT si está disponible
    if [ -f /proc/sys/net/core/bpf_jit_enable ]; then
        echo "Habilitando BPF JIT..."
        echo 1 > /proc/sys/net/core/bpf_jit_enable

        # Montar BPF filesystem
        if ! mountpoint -q /sys/fs/bpf; then
            echo "Montando BPF filesystem..."
            mount -t bpf none /sys/fs/bpf
        fi

        # Hacerlo permanente
        if ! grep -q "/sys/fs/bpf" /etc/fstab; then
            echo "Configurando BPF en fstab..."
            echo "none /sys/fs/bpf bpf defaults 0 0" >> /etc/fstab
        fi
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
      # Comparar versiones
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

    # Dependencias C++ para ML Detector
    echo "Instalando librerías C++..."
    apt-get update
    apt-get install -y \
      pkg-config \
      libzmq3-dev \
      libprotobuf-dev \
      protobuf-compiler \
      liblz4-dev \
      libspdlog-dev \
      nlohmann-json3-dev \
      libgtest-dev \
      libbenchmark-dev

    # ONNX Runtime (descarga e instalación)
    if [ ! -f /usr/local/lib/libonnxruntime.so ]; then
      echo "Instalando ONNX Runtime 1.16.0..."
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

    # Verificar instalación
    echo ""
    echo "Verificando dependencias ML Detector:"
    echo "  - CMake: $(cmake --version | head -1)"
    echo "  - ZeroMQ: $(pkg-config --modversion libzmq)"
    echo "  - Protobuf: $(pkg-config --modversion protobuf)"
    echo "  - LZ4: $(pkg-config --modversion liblz4)"
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

    # Python 3.11 desde Debian repos
    apt-get update
    apt-get install -y \
      python3 \
      python3-pip \
      python3-venv \
      python3-dev \
      python3-setuptools \
      python3-wheel

    # Actualizar pip
    python3 -m pip install --upgrade pip --break-system-packages

    # Crear venv para ml-training (como vagrant user)
    if [ ! -d /vagrant/ml-training/.venv ]; then
      echo "Creando Python venv en /vagrant/ml-training/.venv..."
      su - vagrant -c "cd /vagrant/ml-training && python3 -m venv .venv"
      echo "✅ Python venv creado"
    else
      echo "✅ Python venv ya existe"
    fi

    # Instalar dependencias ML (si existe requirements.txt)
    if [ -f /vagrant/ml-training/requirements.txt ]; then
      echo "Instalando dependencias Python..."
      su - vagrant -c "cd /vagrant/ml-training && source .venv/bin/activate && pip install -r requirements.txt"
      echo "✅ Dependencias Python instaladas"
    else
      echo "⚠️  requirements.txt no encontrado (se instalará manualmente)"
    fi

    # Añadir alias útiles al .bashrc
    if ! grep -q "ml-activate" /home/vagrant/.bashrc; then
      cat >> /home/vagrant/.bashrc << 'EOF'

# ============================================================================
# ML Training Environment Shortcuts
# ============================================================================
alias ml-activate='source /vagrant/ml-training/.venv/bin/activate'
alias ml-explore='cd /vagrant/ml-training && ml-activate && python scripts/explore.py'
alias ml-train='cd /vagrant/ml-training && ml-activate && python scripts/train_level1.py'
alias ml-convert='cd /vagrant/ml-training && ml-activate && python scripts/convert_to_onnx.py'

# Build shortcuts
alias build-detector='cd /vagrant/ml-detector/build && cmake .. && make -j4'
alias build-sniffer='cd /vagrant/sniffer-ebpf && make clean && make'

export PROJECT_ROOT="/vagrant"
EOF
      echo "✅ Aliases añadidos a .bashrc"
    fi

    # Verificar
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
    echo "=== VM LISTA ==="
    echo "Configuración de red:"
    echo "  - Private network: 192.168.56.20"
    echo "  - Bridged network: $(ip -4 addr show | grep "inet.*192.168\|inet.*10\." | grep -v "192.168.56" | awk '{print $2}' | cut -d'/' -f1 | head -1 || echo 'no detectada')"
    echo ""
    echo "Para usar el laboratorio:"
    echo "  vagrant ssh"
    echo "  cd /vagrant"
    echo "  make help                # Ver opciones disponibles"
    echo "  make lab-start          # Iniciar pipeline básico"
    echo "  make sniffer-build      # Compilar sniffer eBPF"
    echo "  make ml-detector-build  # Compilar ML detector"
    echo "================"
  SHELL
end