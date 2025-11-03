#!/bin/bash
# install-ml-detector-deps.sh - Instalar dependencias ML Detector

set -e

echo "=========================================="
echo "Instalando dependencias ML Detector..."
echo "=========================================="

# CMake moderno (>= 3.20)
CMAKE_VERSION=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')
REQUIRED_VERSION="3.20"

if [ -z "$CMAKE_VERSION" ]; then
  INSTALL_CMAKE=1
else
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

# Dependencias C++
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
  libgtest-dev

# ONNX Runtime
if [ ! -f /usr/local/lib/libonnxruntime.so ]; then
  echo "Instalando ONNX Runtime 1.16.0..."
  ONNX_VERSION="1.16.0"
  cd /tmp
  wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz
  tar -xzf onnxruntime-linux-x64-${ONNX_VERSION}.tgz
  cp -r onnxruntime-linux-x64-${ONNX_VERSION}/include/* /usr/local/include/
  cp -r onnxruntime-linux-x64-${ONNX_VERSION}/lib/* /usr/local/lib/
  ldconfig
  rm -rf onnxruntime-linux-*
fi

echo ""
echo "=========================================="
echo "✅ ML Detector dependencies installed"
echo "=========================================="