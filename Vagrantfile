# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  # Debian 12 Bookworm - kernel actual es suficiente para eBPF avanzado
  config.vm.box = "debian/bookworm64"
  config.vm.box_version = "12.20240905.1"  # Latest stable Debian 12

  # VM Configuration
  config.vm.provider "virtualbox" do |vb|
    vb.name = "zeromq-etcd-lab-debian"
    vb.memory = "6144"  # 6GB RAM para mejor performance con kernel 6.12
    vb.cpus = 4         # 4 cores para compilación paralela

    # VirtualBox optimizations for Debian Bookworm
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
    vb.customize ["modifyvm", :id, "--nictype1", "virtio"]
    vb.customize ["modifyvm", :id, "--audio", "none"]  # Disable audio for performance
    vb.customize ["modifyvm", :id, "--usb", "off"]     # Disable USB for performance
    vb.customize ["modifyvm", :id, "--usbehci", "off"] # Disable USB 2.0

    # Enable nested virtualization for better Docker performance
    vb.customize ["modifyvm", :id, "--nested-hw-virt", "on"]
  end

  # Network configuration
  config.vm.network "private_network", ip: "192.168.56.20"  # Different IP to avoid conflicts

  # Port forwarding for services
  config.vm.network "forwarded_port", guest: 5555, host: 5555, protocol: "tcp"  # ZeroMQ
  config.vm.network "forwarded_port", guest: 2379, host: 2379, protocol: "tcp"  # etcd client
  config.vm.network "forwarded_port", guest: 2380, host: 2380, protocol: "tcp"  # etcd peer
  config.vm.network "forwarded_port", guest: 3000, host: 3000, protocol: "tcp"  # Future monitoring
  config.vm.network "forwarded_port", guest: 5571, host: 5571, protocol: "tcp"  # Sniffer output

  # Synced folder with better performance
  config.vm.synced_folder ".", "/vagrant", type: "virtualbox",
      mount_options: ["dmode=775,fmode=775,exec"]

  # Kernel upgrade para eBPF avanzado
  config.vm.provision "shell", name: "kernel-upgrade", inline: <<-SHELL
    set -euo pipefail

    echo ""
    echo "=== UPGRADE KERNEL PARA eBPF AVANZADO ==="

    CURRENT_KERNEL=$(uname -r)
    echo "Kernel actual: $CURRENT_KERNEL"

    # Verificar si ya tenemos kernel 6.11+
    KERNEL_VERSION=$(uname -r | cut -d. -f1-2)
    MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1)
    MINOR=$(echo $KERNEL_VERSION | cut -d. -f2)

    if [ "$MAJOR" -gt 6 ] || ([ "$MAJOR" -eq 6 ] && [ "$MINOR" -ge 11 ]); then
      echo "Kernel $KERNEL_VERSION ya es suficiente para eBPF avanzado"
    else
      echo "Kernel $KERNEL_VERSION es demasiado antiguo, actualizando..."

      if [ -f /etc/debian_version ]; then
        # Debian - usar backports
        echo "Detectado Debian, usando backports..."
        echo "deb http://deb.debian.org/debian bookworm-backports main" >> /etc/apt/sources.list
        apt update

        # Instalar kernel más nuevo de backports
        if apt install -t bookworm-backports -y linux-image-amd64 linux-headers-amd64; then
          echo "Kernel actualizado desde backports"
          echo "REINICIO REQUERIDO para cargar nuevo kernel"
        else
          echo "ADVERTENCIA: No se pudo actualizar kernel desde backports"
        fi

      elif [ -f /etc/lsb-release ] && grep -q Ubuntu /etc/lsb-release; then
        # Ubuntu - usar mainline kernel PPA
        echo "Detectado Ubuntu, usando mainline kernel..."
        apt update
        apt install -y wget

        # Descargar kernel mainline 6.11
        cd /tmp
        wget -q https://kernel.ubuntu.com/~kernel-ppa/mainline/v6.11/amd64/linux-image-6.11.0-061100-generic_6.11.0-061100.202409151536_amd64.deb
        wget -q https://kernel.ubuntu.com/~kernel-ppa/mainline/v6.11/amd64/linux-headers-6.11.0-061100-generic_6.11.0-061100.202409151536_amd64.deb
        wget -q https://kernel.ubuntu.com/~kernel-ppa/mainline/v6.11/amd64/linux-headers-6.11.0-061100_6.11.0-061100.202409151536_all.deb

        if dpkg -i *.deb; then
          echo "Kernel 6.11 instalado exitosamente"
          echo "REINICIO REQUERIDO para cargar nuevo kernel"
        else
          echo "ADVERTENCIA: Error instalando kernel mainline"
          apt --fix-broken install -y
        fi
      fi

      echo "Para aplicar el nuevo kernel:"
      echo "  vagrant reload"
      echo "Después verificar con: uname -r"
    fi

    echo "========================================="
  SHELL
  config.vm.provision "shell", inline: <<-SHELL
    set -euo pipefail

    echo "Configurando entorno para comunicación Sniffer <-> Service3..."

    # 1. Detectar IP del host dinámicamente (más robusto que hardcodear)
    echo "Detectando IP del host..."

    # Método 1: IP del gateway por defecto (más confiable)
    HOST_IP=$(ip route | grep '^default' | awk '{print $3}' | head -1)

    if [ -z "$HOST_IP" ]; then
      # Método 2: Fallback a la IP estándar de VirtualBox NAT
      echo "No se pudo detectar gateway, usando IP estándar de VirtualBox NAT"
      HOST_IP="10.0.2.2"
    fi

    echo "IP del host detectada: $HOST_IP"

    # 2. Configurar variables de entorno del sistema
    echo "Configurando variables de entorno..."
    {
      echo "# Host IP for Docker containers to access Vagrant host"
      echo "VAGRANT_HOST_IP=$HOST_IP"
      echo "SNIFFER_HOST_IP=$HOST_IP"
      echo "SNIFFER_ENDPOINT=tcp://$HOST_IP:5571"
    } >> /etc/environment

    # 3. Configurar Docker para resolver host.docker.internal
    echo "Configurando Docker host resolution..."
    {
      echo "# Map host.docker.internal to Vagrant host IP"
      echo "$HOST_IP host.docker.internal"
      echo "$HOST_IP docker.host.internal"  # Alternative name
    } >> /etc/hosts

    # 4. Configurar script para actualizar service3 config dinámicamente
    echo "Creando script de configuración automática..."
    cat > /usr/local/bin/update-service3-config << 'SCRIPT_EOF'
#!/bin/bash
# Script para actualizar automáticamente la configuración de service3
set -euo pipefail

CONFIG_FILE="/vagrant/service3/config/service3.json"
HOST_IP=${1:-$(ip route | grep '^default' | awk '{print $3}' | head -1)}

if [ -z "$HOST_IP" ]; then
  HOST_IP="10.0.2.2"  # Fallback
fi

echo "Actualizando configuración de service3 con IP: $HOST_IP"

# Backup del config original
cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"

# Usar jq si está disponible, sino sed
if command -v jq >/dev/null 2>&1; then
  # Método preferido con jq
  tmp_file=$(mktemp)
  jq --arg ip "$HOST_IP" '.connection.sniffer_endpoint = ("tcp://" + $ip + ":5571")' "$CONFIG_FILE" > "$tmp_file"
  mv "$tmp_file" "$CONFIG_FILE"
else
  # Fallback con sed
  sed -i.bak 's|"sniffer_endpoint": *"[^"]*"|"sniffer_endpoint": "tcp://'$HOST_IP':5571"|g' "$CONFIG_FILE"
fi

echo "Configuración actualizada: tcp://$HOST_IP:5571"
SCRIPT_EOF

    chmod +x /usr/local/bin/update-service3-config

    # 5. Ejecutar el script de actualización inmediatamente
    if [ -f "/vagrant/service3/config/service3.json" ]; then
      /usr/local/bin/update-service3-config "$HOST_IP"
    else
      echo "Archivo service3.json no encontrado, se configurará cuando se cree"
    fi

    # 6. Mostrar información de configuración
    echo ""
    echo "=== Configuración de Red ==="
    echo "Host IP (para Docker): $HOST_IP"
    echo "Sniffer endpoint: tcp://$HOST_IP:5571"
    echo "Variables configuradas en /etc/environment"
    echo "host.docker.internal → $HOST_IP"
    echo ""
    echo "Para reconfigurar manualmente:"
    echo "   sudo /usr/local/bin/update-service3-config [nueva_ip]"
    echo "==============================="
  SHELL

  # Sección de compilación del sniffer
  config.vm.provision "shell", name: "sniffer-build", inline: <<-SHELL
    set -euo pipefail

    echo ""
    echo "=== COMPILACION DEL SNIFFER ==="
    echo "Iniciando compilación automática del Enhanced Sniffer v3.1..."

    # Verificar que el directorio existe
    if [ ! -d "/vagrant/sniffer" ]; then
      echo "ERROR: Directorio /vagrant/sniffer no encontrado"
      exit 1
    fi

    cd /vagrant/sniffer

    # Verificar dependencias del sniffer
    echo "Verificando dependencias del sniffer..."

    # Instalar dependencias si no están instaladas
    apt-get update -qq

    # Lista de paquetes requeridos para el sniffer
    REQUIRED_PACKAGES=(
      "build-essential"
      "cmake"
      "pkg-config"
      "libbpf-dev"
      "libzmq3-dev"
      "libjsoncpp-dev"
      "liblz4-dev"
      "libzstd-dev"
      "libprotobuf-dev"
      "protobuf-compiler"
      "clang"
      "bpftool"
      "linux-headers-amd64"
    )

    for package in "${REQUIRED_PACKAGES[@]}"; do
      if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
        echo "Instalando $package..."
        if ! apt-get install -y "$package"; then
          echo "ADVERTENCIA: No se pudo instalar $package"
          # Continuar con otros paquetes - algunos pueden no ser críticos
        fi
      fi
    done

    # Verificación especial para headers del kernel
    if ! dpkg-query -W -f='${Status}' "linux-headers-amd64" 2>/dev/null | grep -q "install ok installed"; then
      echo "Intentando instalar headers específicos del kernel..."
      KERNEL_VERSION=$(uname -r)
      if apt-get install -y "linux-headers-$KERNEL_VERSION" 2>/dev/null; then
        echo "Headers específicos instalados: linux-headers-$KERNEL_VERSION"
      else
        echo "Headers específicos no disponibles, usando genéricos"
        apt-get install -y linux-headers-generic || echo "ADVERTENCIA: No se pudieron instalar headers del kernel"
      fi
    fi

    echo "Dependencias verificadas correctamente"

    # Limpiar compilación anterior si existe
    if [ -d "build" ]; then
      echo "Limpiando compilación anterior..."
      rm -rf build/*
    fi

    # Compilar usando el Makefile existente
    echo "Iniciando compilación del sniffer..."
    echo "Comando: make sniffer-build"

    if make sniffer-build; then
      echo ""
      echo "COMPILACION EXITOSA!"
      echo "Binario del sniffer compilado en: /vagrant/sniffer/build/sniffer"

      # Verificar el binario
      if [ -f "build/sniffer" ]; then
        echo "Verificando binario compilado..."
        ls -la build/sniffer

        # Hacer una verificación rápida de la configuración
        echo "Probando carga de configuración JSON..."
        cd build
        if ./sniffer --help >/dev/null 2>&1; then
          echo "Sniffer responde correctamente a --help"
        fi

        # Probar dry-run para validar JSON
        echo "Validando configuración JSON..."
        if ./sniffer --dry-run --verbose 2>/dev/null; then
          echo "Configuración JSON validada exitosamente"
        else
          echo "ADVERTENCIA: Validación JSON falló, revisar configuración"
        fi
      else
        echo "ERROR: Binario no encontrado después de compilación"
        exit 1
      fi

      echo ""
      echo "=== SNIFFER LISTO PARA USO ==="
      echo "Para ejecutar el sniffer:"
      echo "  cd /vagrant/sniffer/build"
      echo "  sudo ./sniffer --verbose"
      echo ""
      echo "Para validar solo la configuración:"
      echo "  sudo ./sniffer --dry-run --verbose"
      echo ""
      echo "Para ver configuración parseada:"
      echo "  sudo ./sniffer --show-config --verbose"
      echo "================================="

    else
      echo ""
      echo "ERROR EN COMPILACION!"
      echo "La compilación del sniffer falló"
      echo "Revisar logs arriba para detalles del error"
      echo "Directorios disponibles:"
      ls -la /vagrant/sniffer/
      exit 1
    fi
  SHELL

  # Provisioning script original (si existe)
  if File.exist?("scripts/vagrant-provision.sh")
    config.vm.provision "shell", path: "scripts/vagrant-provision.sh"
  else
    # Provisioning básico en caso de que no exista el script
    config.vm.provision "shell", inline: <<-SHELL
      echo "Instalando dependencias básicas..."
      apt-get update
      apt-get install -y curl wget git vim jq

      # Instalar Docker si no está instalado
      if ! command -v docker >/dev/null 2>&1; then
        echo "Instalando Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        usermod -aG docker vagrant
        systemctl enable docker
        systemctl start docker
        rm get-docker.sh
      fi

      # Instalar Docker Compose si no está instalado
      if ! command -v docker-compose >/dev/null 2>&1; then
        echo "Instalando Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
      fi

      echo "Provisioning básico completado"
    SHELL
  end
end