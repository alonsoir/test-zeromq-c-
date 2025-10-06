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

  # Port forwarding
  config.vm.network "forwarded_port", guest: 5555, host: 5555, protocol: "tcp"  # ZeroMQ
  config.vm.network "forwarded_port", guest: 2379, host: 2379, protocol: "tcp"  # etcd client
  config.vm.network "forwarded_port", guest: 2380, host: 2380, protocol: "tcp"  # etcd peer
  config.vm.network "forwarded_port", guest: 3000, host: 3000, protocol: "tcp"  # monitoring
  config.vm.network "forwarded_port", guest: 5571, host: 5571, protocol: "tcp"  # sniffer

  # Synced folder
  config.vm.synced_folder ".", "/vagrant", type: "virtualbox",
      mount_options: ["dmode=775,fmode=775,exec"]

  # Configuración de red para sniffer
  config.vm.provision "shell", inline: <<-SHELL
    echo "Configurando entorno para sniffer..."

    # Detectar IP del host
    HOST_IP=$(ip route | grep '^default' | awk '{print $3}' | head -1)
    if [ -z "$HOST_IP" ]; then
      HOST_IP="10.0.2.2"
    fi

    echo "Host IP: $HOST_IP"

    # Variables de entorno
    {
      echo "VAGRANT_HOST_IP=$HOST_IP"
      echo "SNIFFER_HOST_IP=$HOST_IP"
      echo "SNIFFER_ENDPOINT=tcp://$HOST_IP:5571"
    } >> /etc/environment

    # Docker host resolution
    {
      echo "$HOST_IP host.docker.internal"
      echo "$HOST_IP docker.host.internal"
    } >> /etc/hosts
  SHELL

  # Provisioning básico
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
        echo "Habilitando BPF JIT si está disponible..."
        echo 1 > /proc/sys/net/core/bpf_jit_enable
        # Montar BPF filesystem
        if ! mountpoint -q /sys/fs/bpf; then
            echo "Montar BPF filesystem..."
            mount -t bpf none /sys/fs/bpf
        fi

        # Hacerlo permanente en /etc/fstab
        if ! grep -q "/sys/fs/bpf" /etc/fstab; then
            echo "Hacerlo permanente en /etc/fstab..."
            echo "none /sys/fs/bpf bpf defaults 0 0" >> /etc/fstab
        fi
    fi

    echo ""
    echo "=== VM LISTA ==="
    echo "Para usar el laboratorio:"
    echo "  vagrant ssh"
    echo "  cd /vagrant"
    echo "  make help                # Ver opciones disponibles"
    echo "  make lab-start          # Iniciar pipeline básico"
    echo "  make sniffer-build      # Compilar sniffer eBPF"
    echo "================"
  SHELL
end