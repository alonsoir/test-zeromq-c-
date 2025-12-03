#!/bin/bash
echo "=== VERIFICACIÓN PRECISA RAG ECOSYSTEM ==="

echo "1. Procesos activos..."
if pgrep -f rag-security > /dev/null; then
    echo "✅ RAG Security: ACTIVO (PID: $(pgrep -f rag-security))"
else
    echo "❌ RAG Security: INACTIVO"
fi

if pgrep -f etcd-server > /dev/null; then
    echo "✅ etcd-server: ACTIVO (PID: $(pgrep -f etcd-server))"
    echo "   Endpoint: http://localhost:2379"
else
    echo "❌ etcd-server: INACTIVO"
fi

echo -e "\n2. Verificación funcional..."
# RAG
echo "RAG Status:"
cd /vagrant/rag/build 2>/dev/null && timeout 1s ./rag-security 2>&1 | grep -E "✅|Iniciando|CRITICAL" | head -2

# etcd-server
echo -e "\netcd-server Status:"
if curl -s http://localhost:2379/health >/dev/null 2>&1; then
    echo "✅ API respondiendo"
    curl -s http://localhost:2379/info 2>/dev/null | head -3
else
    echo "❌ API no responde"
fi

echo -e "\n3. Registro de componentes..."
curl -s http://localhost:2379/components 2>/dev/null | grep -i "component" || echo "Sin componentes registrados o API no accesible"

echo -e "\n4. Configuración..."
[ -f "/vagrant/rag/config/rag-config.json" ] && echo "✅ Config RAG: $(jq -r '.id' /vagrant/rag/config/rag-config.json 2>/dev/null || echo 'EXISTE')" || echo "❌ Config RAG faltante"
[ -f "/vagrant/rag/models/default.gguf" ] && echo "✅ Modelo LLM: $(ls -lh /vagrant/rag/models/default.gguf | awk '{print $5}')" || echo "⚠️  Modelo LLM faltante"

echo -e "\n5. Makefile targets..."
echo "   Desde VM: cd /vagrant/rag && make run"
echo "   Desde VM: cd /vagrant/etcd-server && make run"
echo "   Desde host: make rag-etcd-start"
echo "   Desde host: make rag-etcd-status"

echo -e "\n=== ESTADO: $(pgrep -f 'rag-security|etcd-server' | wc -l)/2 componentes activos ==="
