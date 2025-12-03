#!/bin/bash
echo "=== RAG ECOSYSTEM VERIFICATION (FIXED) ==="

echo "1. Checking directories..."
[ -d "/vagrant/rag" ] && echo "✅ /vagrant/rag exists" || echo "❌ /vagrant/rag missing"
[ -d "/vagrant/etcd-server" ] && echo "✅ /vagrant/etcd-server exists" || echo "❌ /vagrant/etcd-server missing"
[ -d "/vagrant/third_party/llama.cpp" ] && echo "✅ llama.cpp exists" || echo "⚠️  llama.cpp missing"

echo -e "\n2. Checking binaries..."
if [ -f "/vagrant/rag/build/rag-security" ]; then
    echo "✅ RAG binary exists"
    # Test with absolute path
    cd /vagrant/rag && timeout 2s ./build/rag-security -c /vagrant/rag/config/rag-config.json 2>&1 | head -3 || echo "ℹ️  Test ejecutado"
else
    echo "❌ RAG binary missing"
fi

if [ -f "/vagrant/etcd-server/build/etcd-server" ]; then
    echo "✅ etcd-server binary exists"
    cd /vagrant/etcd-server && timeout 2s ./build/etcd-server 2>&1 | head -3 || echo "ℹ️  Test ejecutado"
else
    echo "❌ etcd-server binary missing"
fi

echo -e "\n3. Checking config files..."
if [ -f "/vagrant/rag/config/rag-config.json" ]; then
    echo "✅ RAG config exists"
    echo "   Size: $(stat -c%s /vagrant/rag/config/rag-config.json) bytes"
else
    echo "❌ RAG config missing - creating default..."
    cat > /vagrant/rag/config/rag-config.json << 'CONFIG'
{
  "name": "RAG Security System",
  "etcd": {"endpoints": ["http://localhost:2379"]},
  "llama": {"model_path": "/vagrant/rag/models/default.gguf"}
}
CONFIG
fi

echo -e "\n4. Checking dependencies..."
pkg-config --exists libcrypto++ && echo "✅ libcrypto++ available" || echo "❌ libcrypto++ missing"
[ -f "/usr/local/include/httplib.h" ] && echo "✅ cpp-httplib available" || echo "⚠️  cpp-httplib missing"
[ -f "/vagrant/third_party/llama.cpp/build/bin/libllama.a" ] && echo "✅ llama.cpp compiled" || echo "⚠️  llama.cpp not compiled"

echo -e "\n5. Checking LLM model..."
if [ -f "/vagrant/rag/models/default.gguf" ]; then
    SIZE=$(stat -c%s /vagrant/rag/models/default.gguf 2>/dev/null || echo "0")
    if [ "$SIZE" -gt 1000000 ]; then
        echo "✅ LLM model exists ($((SIZE/1024/1024)) MB)"
    else
        echo "⚠️  LLM model too small or corrupted"
    fi
else
    echo "❌ LLM model missing"
    echo "   Download with: make rag-download-model"
fi

echo -e "\n6. Checking processes..."
pgrep -f rag-security >/dev/null && echo "✅ RAG process running" || echo "❌ RAG not running"
pgrep -f etcd-server >/dev/null && echo "✅ etcd-server process running" || echo "❌ etcd-server not running"

echo -e "\n=== VERIFICATION COMPLETE ==="
echo ""
echo "Quick fixes if needed:"
echo "  sudo apt-get install libcrypto++-dev"
echo "  cd /vagrant/rag && make clean && make"
echo "  cd /vagrant/etcd-server && make clean && make"
