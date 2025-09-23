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
  config.vm.network "forwarded_port", guest: 5571, host: 5571, protocol: "tcp"  # Sniffer output

  # Synced folder with better performance
  config.vm.synced_folder ".", "/vagrant", type: "virtualbox",
      mount_options: ["dmode=775,fmode=775,exec"]

  # Provisioning script
  config.vm.provision "shell", path: "scripts/vagrant-provision.sh"
end