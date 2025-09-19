# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  # Ubuntu 22.04 LTS - Necesario para C++20 y herramientas modernas
  config.vm.box = "ubuntu/jammy64"
  config.vm.box_version = "20231215.0.0"

  # Configuración de la VM
  config.vm.provider "virtualbox" do |vb|
    vb.name = "zeromq-protobuf-dev"
    vb.memory = "4096"  # 4GB RAM para compilación cómoda
    vb.cpus = 4         # 4 cores para compilación paralela

    # Optimizaciones para desarrollo
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
    vb.customize ["modifyvm", :id, "--nictype1", "virtio"]
  end

  # Configurar red - Importante para ZeroMQ
  config.vm.network "private_network", ip: "192.168.56.10"

  # Port forwarding para acceder desde host si necesario
  config.vm.network "forwarded_port", guest: 5555, host: 5555, protocol: "tcp"

  # Sincronizar el directorio del proyecto
  config.vm.synced_folder ".", "/vagrant", type: "virtualbox"

  # Provisioning script
  config.vm.provision "shell", inline: <<-SHELL
    set -e  # Exit on any error

    echo "🚀 Setting up ZeroMQ + Protobuf Development Environment"
    echo "======================================================"

    # Update system
    echo "📦 Updating package lists..."
    apt-get update

    # Install essential development tools
    echo "🔧 Installing development tools..."
    apt-get install -y \
      curl \
      wget \
      vim \
      git \
      htop \
      net-tools \
      tree \
      unzip

    # Install Docker
    echo "🐳 Installing Docker..."
    apt-get install -y apt-transport-https ca-certificates gnupg lsb-release
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io

    # Install Docker Compose (latest version)
    echo "🔧 Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Add vagrant user to docker group
    usermod -aG docker vagrant

    # Install build tools for potential debugging
    echo "🛠️ Installing build tools..."
    apt-get install -y \
      build-essential \
      cmake \
      pkg-config \
      protobuf-compiler \
      libprotobuf-dev

    # Enable Docker service
    systemctl enable docker
    systemctl start docker

    # Optimize system for container development
    echo "⚡ Optimizing system..."
    echo 'vm.max_map_count=262144' >> /etc/sysctl.conf
    echo 'fs.file-max=65536' >> /etc/sysctl.conf
    sysctl -p

    # Verify installations
    echo ""
    echo "✅ Installation verification:"
    echo "   Docker version: $(docker --version)"
    echo "   Docker Compose version: $(docker-compose --version)"
    echo "   Protoc version: $(protoc --version)"
    echo "   G++ version: $(g++ --version | head -1)"

    # Set up convenient aliases
    echo 'alias ll="ls -la"' >> /home/vagrant/.bashrc
    echo 'alias dc="docker-compose"' >> /home/vagrant/.bashrc
    echo 'alias dps="docker ps"' >> /home/vagrant/.bashrc

    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. vagrant ssh"
    echo "   2. cd /vagrant"
    echo "   3. chmod +x build_and_run.sh"
    echo "   4. ./build_and_run.sh"
    echo ""
    echo "🔧 Useful commands:"
    echo "   - docker-compose build --no-cache"
    echo "   - docker-compose up"
    echo "   - docker-compose logs -f service1"
    echo "   - ./debug.sh"

  SHELL
end