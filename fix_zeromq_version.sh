#!/bin/bash

echo "Corrigiendo versión de ZeroMQ en Dockerfiles..."

# Corregir Dockerfile.service1
sed -i 's/--branch v4.4.7 --depth 1/--depth 1/' Dockerfile.service1

# Corregir Dockerfile.service2  
sed -i 's/--branch v4.4.7 --depth 1/--depth 1/' Dockerfile.service2

echo "✅ Corrección aplicada. Usando branch master de ZeroMQ."
