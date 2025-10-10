#!/bin/bash
set -e

echo "📦 Instalando dependencias para crear paquete Debian..."

# Configurar locales españolas
echo "🌍 Configurando locales españolas..."
sudo apt-get install -y locales
sudo sed -i '/es_ES.UTF-8/s/^# //g' /etc/locale.gen
sudo locale-gen es_ES.UTF-8
sudo update-locale LANG=es_ES.UTF-8 LC_ALL=es_ES.UTF-8

export LANG=es_ES.UTF-8
export LC_ALL=es_ES.UTF-8

sudo apt-get update -qq

echo "Instalando herramientas básicas..."
sudo apt-get install -y rsync  # ← AÑADIDO

echo "Instalando herramientas de build Debian..."
sudo apt-get install -y \
    debhelper \
    devscripts \
    build-essential \
    dpkg-dev \
    fakeroot

echo "Instalando dependencias específicas del sniffer..."
sudo apt-get install -y \
    libbpf-dev \
    libprotobuf-c-dev \
    liblz4-dev \
    libzmq3-dev \
    protobuf-c-compiler \
    clang \
    llvm \
    bpftool \
    linux-headers-$(uname -r)

echo "✅ Dependencias instaladas correctamente"