#!/bin/bash
set -e

echo "🚀 Construyendo contenedores..."
docker-compose build

echo "📤 Levantando contenedores en background..."
docker-compose up -d

echo "⏳ Esperando 3 segundos para que service1 esté listo..."
sleep 3

echo "📌 Mostrando logs de service1 y service2..."
docker-compose logs --tail=20 -f

echo "🛑 Para detener los contenedores, presiona Ctrl+C y luego ejecuta: docker-compose down"
