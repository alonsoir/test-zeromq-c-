# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  # Debian 12 Bookworm (kernel 6.1 base → upgrade to 6.12 mainline)
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

  # Enhanced provisioning with dynamic IP detection
  config.vm.provision "shell", inline: <<-SHELL
    set -euo pipefail

    echo "🔧 Configurando entorno para comunicación Sniffer <-> Service3..."

    # 1. Detectar IP del host dinámicamente (más robusto que hardcodear)
    echo "🔍 Detectando IP del host..."

    # Método 1: IP del gateway por defecto (más confiable)
    HOST_IP=$(ip route | grep '^default' | awk '{print $3}' | head -1)

    if [ -z "$HOST_IP" ]; then
      # Método 2: Fallback a la IP estándar de VirtualBox NAT
      echo "⚠️  No se pudo detectar gateway, usando IP estándar de VirtualBox NAT"
      HOST_IP="10.0.2.2"
    fi

    echo "✅ IP del host detectada: $HOST_IP"

    # 2. Configurar variables de entorno del sistema
    echo "📝 Configurando variables de entorno..."
    {
      echo "# Host IP for Docker containers to access Vagrant host"
      echo "VAGRANT_HOST_IP=$HOST_IP"
      echo "SNIFFER_HOST_IP=$HOST_IP"
      echo "SNIFFER_ENDPOINT=tcp://$HOST_IP:5571"
    } >> /etc/environment

    # 3. Configurar Docker para resolver host.docker.internal
    echo "🐳 Configurando Docker host resolution..."
    {
      echo "# Map host.docker.internal to Vagrant host IP"
      echo "$HOST_IP host.docker.internal"
      echo "$HOST_IP docker.host.internal"  # Alternative name
    } >> /etc/hosts

    # 4. Configurar script para actualizar service3 config dinámicamente
    echo "📋 Creando script de configuración automática..."
    cat > /usr/local/bin/update-service3-config << 'EOF'
#!/bin/bash
# Script para actualizar automáticamente la configuración de service3
set -euo pipefail

CONFIG_FILE="/vagrant/service3/config/service3.json"
HOST_IP=${1:-$(ip route | grep '^default' | awk '{print $3}' | head -1)}

if [ -z "$HOST_IP" ]; then
  HOST_IP="10.0.2.2"  # Fallback
fi

echo "🔧 Actualizando configuración de service3 con IP: $HOST_IP"

# Backup del config original
cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"

# Usar jq si está disponible, sino sed
if command -v jq >/dev/null 2>&1; then
  # Método preferido con jq
  tmp_file=$(mktemp)
  jq --arg ip "$HOST_IP" '.connection.sniffer_endpoint = "tcp://\($ip):5571"' "$CONFIG_FILE" > "$tmp_file"
  mv "$tmp_file" "$CONFIG_FILE"
else
  # Fallback con sed
  sed -i.bak "s|\"sniffer_endpoint\": *\"[^\"]*\"|\"sniffer_endpoint\": \"tcp://$HOST_IP:5571\"|g" "$CONFIG_FILE"
fi

echo "✅ Configuración actualizada: tcp://$HOST_IP:5571"
EOF

    chmod +x /usr/local/bin/update-service3-config

    # 5. Ejecutar el script de actualización inmediatamente
    if [ -f "/vagrant/service3/config/service3.json" ]; then
      /usr/local/bin/update-service3-config "$HOST_IP"
    else
      echo "⚠️  Archivo service3.json no encontrado, se configurará cuando se cree"
    fi

    # 6. Mostrar información de configuración
    echo ""
    echo "=== 🌐 Configuración de Red ==="
    echo "Host IP (para Docker): $HOST_IP"
    echo "Sniffer endpoint: tcp://$HOST_IP:5571"
    echo "Variables configuradas en /etc/environment"
    echo "host.docker.internal → $HOST_IP"
    echo ""
    echo "💡 Para reconfigurar manualmente:"
    echo "   sudo /usr/local/bin/update-service3-config [nueva_ip]"
    echo "==============================="
  SHELL

  # Provisioning script original (si existe)
  if File.exist?("scripts/vagrant-provision.sh")
    config.vm.provision "shell", path: "scripts/vagrant-provision.sh"
  else
    # Provisioning básico en caso de que no exista el script
    config.vm.provision "shell", inline: <<-SHELL
      echo "📦 Instalando dependencias básicas..."
      apt-get update
      apt-get install -y curl wget git vim jq

      # Instalar Docker si no está instalado
      if ! command -v docker >/dev/null 2>&1; then
        echo "🐳 Instalando Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        usermod -aG docker vagrant
        systemctl enable docker
        systemctl start docker
        rm get-docker.sh
      fi

      # Instalar Docker Compose si no está instalado
      if ! command -v docker-compose >/dev/null 2>&1; then
        echo "🐙 Instalando Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
      fi

      echo "✅ Provisioning básico completado"
    SHELL
  end
end