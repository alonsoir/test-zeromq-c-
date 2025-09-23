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
  end

  # Network configuration
  config.vm.network "private_network", ip: "192.168.56.20"  # Different IP to avoid conflicts

  # Port forwarding for services
  config.vm.network "forwarded_port", guest: 5555, host: 5555, protocol: "tcp"  # ZeroMQ
  config.vm.network "forwarded_port", guest: 2379, host: 2379, protocol: "tcp"  # etcd client
  config.vm.network "forwarded_port", guest: 2380, host: 2380, protocol: "tcp"  # etcd peer
  config.vm.network "forwarded_port", guest: 3000, host: 3000, protocol: "tcp"  # Future monitoring

  # Synced folder with better performance
  config.vm.synced_folder ".", "/vagrant", type: "virtualbox",
    mount_options: ["dmode=775,fmode=664"]

  # Provisioning script
  config.vm.provision "shell", inline: <<-SHELL
    set -e  # Exit on any error

    echo "🚀 Setting up ZeroMQ + Protobuf + etcd Lab on Debian 12 Bookworm"
    echo "Target: Kernel 6.12.x for modern container/networking features"
    echo "================================================================="

    # Update system packages
    echo "📦 Updating system packages..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get upgrade -y

    # Install essential development tools
    echo "🔧 Installing core development stack..."
    apt-get install -y \
      curl wget vim git htop net-tools tree unzip jq \
      build-essential cmake pkg-config \
      software-properties-common lsb-release gnupg \
      ca-certificates apt-transport-https \
      linux-headers-amd64 dkms sudo

    # Install modern C++ toolchain
    echo "⚙️ Installing C++23 toolchain..."
    apt-get install -y \
      gcc-12 g++-12 \
      protobuf-compiler libprotobuf-dev \
      libzmq3-dev libzmq5 \
      libcurl4-openssl-dev \
      nlohmann-json3-dev

    # Set GCC 12 as default for C++23 support
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

    # Add Docker's official GPG key and repository
    echo "🐳 Installing Docker CE for Debian Bookworm..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Install Docker Compose standalone (latest version)
    echo "🔧 Installing Docker Compose standalone..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

    # Configure Docker for vagrant user
    usermod -aG docker vagrant
    systemctl enable docker
    systemctl start docker

    # Install kernel 6.12 from Debian backports/experimental (safer than PPA)
    echo "⬇️ Setting up mainline kernel installation..."
    echo "deb http://deb.debian.org/debian bookworm-backports main" > /etc/apt/sources.list.d/backports.list
    echo "deb http://deb.debian.org/debian experimental main" > /etc/apt/sources.list.d/experimental.list

    # Set lower priority for experimental repo
    cat > /etc/apt/preferences.d/experimental << 'EOF'
Package: *
Pin: release a=experimental
Pin-Priority: 50
EOF

    apt-get update

    # Try to install newer kernel from backports first, then experimental
    echo "🔄 Installing latest available kernel..."
    apt-get install -y -t bookworm-backports linux-image-amd64 linux-headers-amd64 || {
      echo "Trying experimental repo for kernel 6.12..."
      apt-get install -y -t experimental linux-image-amd64 linux-headers-amd64 || {
        echo "Using stable kernel for now"
      }
    }

    # System optimizations for containerized workloads
    echo "⚡ Applying system optimizations..."
    cat >> /etc/sysctl.conf << EOF
# Container and network optimizations
vm.max_map_count=262144
fs.file-max=1048576
net.core.somaxconn=32768
net.ipv4.ip_local_port_range=1024 65000
net.ipv4.tcp_tw_reuse=1
net.core.netdev_max_backlog=5000
EOF

    # Apply sysctl settings
    sysctl -p

    # Set up development environment
    echo "🛠️ Configuring development environment..."

    # Convenient aliases for vagrant user
    cat >> /home/vagrant/.bashrc << 'EOF'
# Development aliases
alias ll='ls -la'
alias la='ls -la'
alias dc='docker-compose'
alias dps='docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"'
alias dlogs='docker-compose logs -f'
alias dclean='docker system prune -f'

# Lab specific
alias lab-up='cd /vagrant && docker-compose up -d'
alias lab-down='cd /vagrant && docker-compose down'
alias lab-build='cd /vagrant && docker-compose build --no-cache'
alias lab-logs='cd /vagrant && docker-compose logs -f'
alias lab-status='cd /vagrant && docker-compose ps && echo && ./etcd-health.sh'
alias lab-restart='cd /vagrant && docker-compose restart'

# Git helpers
alias gs='git status'
alias gl='git log --oneline -10'
alias gd='git diff'

export EDITOR=vim
export VAGRANT_LAB=1
EOF

    # Create lab directory structure
    mkdir -p /home/vagrant/lab-scripts
    chown -R vagrant:vagrant /home/vagrant/lab-scripts

    # Verification and summary
    echo ""
    echo "✅ Installation verification:"
    echo "   Debian version: $(lsb_release -d | cut -f2)"
    echo "   Current kernel: $(uname -r)"
    echo "   Docker version: $(docker --version)"
    echo "   Docker Compose: $(docker-compose --version)"
    echo "   Protoc version: $(protoc --version)"
    echo "   GCC version: $(gcc --version | head -1)"
    echo "   G++ standard: C++$(g++ -dumpversion | cut -d. -f1)3 capable"

    echo ""
    echo "🎉 Debian 12 Bookworm + Modern Kernel setup completed!"
    echo ""
    echo "🔄 REBOOT RECOMMENDED to activate latest kernel:"
    echo "   vagrant reload"
    echo ""
    echo "📋 Then run:"
    echo "   vagrant ssh"
    echo "   uname -r  # Check kernel version"
    echo "   cd /vagrant"
    echo "   ./build_and_run.sh"
    echo ""
    echo "🔧 New lab commands available:"
    echo "   lab-up, lab-down, lab-build, lab-logs, lab-status"
    echo ""
    echo "🎯 Debian 12 advantages for this lab:"
    echo "   • Rock-solid stability (better than Ubuntu for labs)"
    echo "   • Latest kernels via backports/experimental"
    echo "   • Excellent Docker support"
    echo "   • Faster package installation"
    echo "   • Better memory usage (ideal for 6GB VM)"
    echo "   • Superior C++ toolchain stability"

  SHELL
end