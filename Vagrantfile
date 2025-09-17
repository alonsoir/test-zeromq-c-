# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/focal64"
  config.vm.hostname = "ubuntu-server"
  config.vm.network "forwarded_port", guest: 5555, host: 5555

  config.vm.provider "virtualbox" do |vb|
    vb.memory = "2048"
    vb.cpus = 2
  end

  config.vm.provision "shell", inline: <<-SHELL
    apt-get update -y
    apt-get upgrade -y

    # Instalar herramientas de desarrollo primero
    apt-get install -y build-essential cmake git curl wget unzip pkg-config

    # Instalar Docker usando el método oficial
    apt-get install -y ca-certificates curl gnupg lsb-release
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Crear grupo docker y agregar usuario vagrant
    groupadd -f docker
    usermod -aG docker vagrant

    # Habilitar y arrancar Docker
    systemctl enable docker
    systemctl start docker

    # Instalar docker-compose standalone (para compatibilidad)
    curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Crear enlace simbólico para docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

    echo "✅ Setup completado. Reinicia la VM con 'vagrant reload' para aplicar cambios de grupo."
    echo "📋 Versiones instaladas:"
    docker --version || echo "❌ Docker no disponible"
    docker-compose --version || echo "❌ Docker Compose no disponible"
  SHELL
end