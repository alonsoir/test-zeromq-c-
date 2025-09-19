#!/bin/bash

# Script de debug para troubleshooting del proyecto ZeroMQ + Protobuf

echo "🔍 ZeroMQ + Protobuf Debug Information"
echo "====================================="

# Verificar archivos necesarios
echo "📁 Checking required files..."
files_to_check=(
    "protobuf/network_security.proto"
    "docker-compose.yml"
    "Dockerfile.service1"
    "Dockerfile.service2"
    "service1/main.cpp"
    "service1/main.h"
    "service2/main.cpp"
    "service2/main.h"
)

missing_files=()
for file in "${files_to_check[@]}"; do
    if [[ -f "$file" ]]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (MISSING)"
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo ""
    echo "❌ Missing files detected. Please ensure all files are in place."
    exit 1
fi

echo ""
echo "🐳 Docker information..."
echo "Docker version:"
docker --version
echo "Docker Compose version:"
docker-compose --version

echo ""
echo "📦 Current Docker images related to project:"
docker images | grep -E "(zeromq|protobuf|service[12])" || echo "   No project images found"

echo ""
echo "🏃 Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🔧 Recent Docker logs for troubleshooting:"
echo "Service1 logs (last 20 lines):"
docker-compose logs --tail=20 service1 2>/dev/null || echo "   No service1 logs available"

echo ""
echo "Service2 logs (last 20 lines):"
docker-compose logs --tail=20 service2 2>/dev/null || echo "   No service2 logs available"

echo ""
echo "🌐 Network information:"
docker network ls | grep zeromq || echo "   No zeromq networks found"

echo ""
echo "💾 Disk space usage:"
df -h . | head -2

echo ""
echo "📊 System resources:"
echo "Memory usage:"
free -h | head -2
echo "CPU info:"
nproc --all
echo "Load average:"
uptime

echo ""
echo "🔍 Proto file validation:"
if [[ -f "protobuf/network_security.proto" ]]; then
    echo "Proto file size: $(wc -c < protobuf/network_security.proto) bytes"
    echo "Proto file lines: $(wc -l < protobuf/network_security.proto) lines"
    echo "Package declaration:"
    grep -n "^package " protobuf/network_security.proto || echo "   No package declaration found"
    echo "Main message types found:"
    grep -n "^message " protobuf/network_security.proto || echo "   No message types found"
else
    echo "   ❌ protobuf/network_security.proto not found"
fi

echo ""
echo "🛠️ Suggested debug commands:"
echo "   - Build with verbose output: docker-compose build --progress=plain"
echo "   - Run single service: docker-compose up service1"
echo "   - Check service logs: docker-compose logs -f service1"
echo "   - Enter container shell: docker-compose exec service1 bash"
echo "   - Clean everything: docker-compose down && docker system prune -af"

echo ""
echo "🎯 Quick test commands:"
echo "   - Test proto compilation: docker run --rm -v \$(pwd):/workspace ubuntu:22.04 bash -c 'apt-get update && apt-get install -y protobuf-compiler && cd /workspace && protoc --version && protoc --cpp_out=. protobuf/network_security.proto && ls -la protobuf/*.pb.*'"
echo "   - Test ZeroMQ: docker run --rm zeromq/zeromq:latest zmq_version"