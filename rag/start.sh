#!/bin/bash
# Script de inicio para RAG Security System
# Uso: ./start.sh [debug|test|normal]

cd /vagrant/rag

CONFIG_PATH="config/rag-config.json"
LOG_FILE="/vagrant/logs/rag-$(date +%Y%m%d-%H%M%S).log"

case "$1" in
    debug)
        echo "🚀 Iniciando RAG en modo debug..."
        ./build/rag-security -c "$CONFIG_PATH" --debug 2>&1 | tee "$LOG_FILE"
        ;;
    test)
        echo "🧪 Iniciando RAG en modo test..."
        ./build/rag-security -c "$CONFIG_PATH" --test 2>&1 | tee "$LOG_FILE"
        ;;
    normal|"")
        echo "🚀 Iniciando RAG en modo normal..."
        ./build/rag-security -c "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"
        ;;
    *)
        echo "Modo desconocido: $1"
        echo "Uso: $0 [debug|test|normal]"
        exit 1
        ;;
esac
