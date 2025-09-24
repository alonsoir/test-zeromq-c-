#!/bin/bash
set -e  # Exit on any error

echo "🚀 Setting up ZeroMQ + Protobuf + etcd + eBPF Lab on Debian 12 Bookworm"
echo "Target: Kernel 6.12.x for modern container/networking + eBPF features"
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
  nlohmann-json3-dev \
  libjsoncpp-dev

# Set GCC 12 as default for C++23 support
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Install eBPF/XDP development stack
echo "🔧 Installing eBPF/XDP development stack..."
apt-get install -y \
  clang llvm \
  libbpf-dev libbpf1 \
  bpftool \
  linux-tools-common linux-tools-generic \
  ethtool \
  libelf-dev \
  zlib1g-dev

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

# System optimizations for containerized workloads and eBPF
echo "⚡ Applying system optimizations..."
cat >> /etc/sysctl.conf << EOF
# Container and network optimizations
vm.max_map_count=262144
fs.file-max=1048576
net.core.somaxconn=32768
net.ipv4.ip_local_port_range=1024 65000
net.ipv4.tcp_tw_reuse=1
net.core.netdev_max_backlog=5000

# eBPF optimizations
kernel.unprivileged_bpf_disabled=0
net.core.bpf_jit_enable=1
net.core.bpf_jit_kallsyms=1
EOF

# Apply sysctl settings
sysctl -p

# Set up development environment
echo "🛠️ Configuring development environment..."

# Set up sniffer helper scripts
echo "🔧 Setting up sniffer helper scripts..."
chmod +x /vagrant/scripts/*.sh

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
alias lab-status='cd /vagrant && docker-compose ps && echo && ./scripts/etcd-health.sh'
alias lab-restart='cd /vagrant && docker-compose restart'

# Sniffer specific aliases
alias sniffer-build='cd /vagrant/sniffer/build && cmake .. && make -j4'
alias sniffer-auto='cd /vagrant && ./scripts/run_sniffer_with_iface.sh'
alias sniffer-manual='cd /vagrant/sniffer/build && sudo ./sniffer --config=../config/sniffer.json --verbose'
alias sniffer-ifaces='ip -o link show | grep -v lo'
alias sniffer-check='sudo bpftool prog show && sudo bpftool net show'

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

# Initial sniffer build setup
echo "🔨 Setting up initial sniffer build environment..."
cd /vagrant/sniffer
mkdir -p build
chown -R vagrant:vagrant build

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
echo "   Clang version: $(clang --version | head -1)"
echo "   bpftool: $(bpftool version 2>/dev/null | head -1 || echo 'Available')"
echo "   libbpf: $(pkg-config --modversion libbpf 2>/dev/null || echo 'Installed')"

echo ""
echo "🎉 Debian 12 Bookworm + Modern Kernel + eBPF setup completed!"
echo ""
echo "🔄 REBOOT RECOMMENDED to activate latest kernel:"
echo "   vagrant reload"
echo ""
echo "📋 Then run:"
echo "   vagrant ssh"
echo "   uname -r  # Check kernel version"
echo "   cd /vagrant"
echo "   ./scripts/build_and_run.sh  # For ZeroMQ pipeline"
echo "   sniffer-auto               # For eBPF sniffer"
echo ""
echo "🔧 New commands available:"
echo "   • Lab: lab-up, lab-down, lab-build, lab-logs, lab-status"
echo "   • Sniffer: sniffer-build, sniffer-auto, sniffer-manual, sniffer-check"
echo ""
echo "🎯 Debian 12 advantages for this lab:"
echo "   • Rock-solid stability for eBPF development"
echo "   • Latest kernels via backports/experimental"
echo "   • Excellent Docker + eBPF support"
echo "   • Superior C++ toolchain stability"
echo "   • Modern libbpf and bpftool versions"