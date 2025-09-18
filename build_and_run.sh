#!/bin/bash

# Script para construir y ejecutar el proyecto ZeroMQ + Protobuf
# Ubicación: raíz del proyecto test-zeromq-c-

set -e  # Exit on any error

echo "🚀 Building and Running ZeroMQ + Protobuf Demo"
echo "=============================================="

# Verificar que estamos en el directorio correcto
if [[ ! -f "protobuf/network_security.proto" ]]; then
    echo "❌ Error: protobuf/network_security.proto not found. Are you in the project root?"
    exit 1
fi

if [[ ! -f "docker-compose.yaml" ]]; then
    echo "❌ Error: docker-compose.yaml not found. Are you in the project root?"
    exit 1
fi

# Limpiar contenedores anteriores
echo "🧹 Cleaning up previous containers..."
docker-compose down --remove-orphans 2>/dev/null || true
docker system prune -f >/dev/null 2>&1 || true

# Construir las imágenes
echo "🏗️  Building Docker images (this may take a few minutes)..."
echo "   - Compiling ZeroMQ from source"
echo "   - Compiling Protobuf from source"
echo "   - Compiling .proto files inside Ubuntu containers"
echo "   - Building service executables"

# Construir con output detallado para debug
docker-compose build --no-cache

if [[ $? -ne 0 ]]; then
    echo "❌ Build failed. Check the output above for errors."
    exit 1
fi

echo "✅ Build completed successfully"
echo ""

# Ejecutar los servicios
echo "🎯 Starting services..."
echo "   - Service1 (Producer): Generates protobuf messages with random data"
echo "   - Service2 (Consumer): Receives and displays protobuf messages"
echo ""

# Ejecutar con logs en tiempo real
docker-compose up

echo ""
echo "🔚 Demo completed"
echo ""
echo "📋 What happened:"
echo "   1. Service1 compiled network_security.proto inside Ubuntu container"
echo "   2. Service1 generated a NetworkSecurityEvent with realistic random data"
echo "   3. Service1 serialized the protobuf message and sent it via ZeroMQ"
echo "   4. Service2 received the message via ZeroMQ"
echo "   5. Service2 deserialized the protobuf message and displayed all fields"
echo ""
echo "🎉 ZeroMQ + Protobuf integration successful!"

# Cleanup
echo "🧹 Cleaning up containers..."
docker-compose down --remove-orphans
