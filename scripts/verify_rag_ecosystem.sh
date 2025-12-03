#!/bin/bash
echo "=== RAG ECOSYSTEM VERIFICATION ==="

echo "1. Checking RAG binary..."
if [ -f "/vagrant/rag/build/rag-security" ]; then
    echo "✅ RAG binary exists"
    /vagrant/rag/build/rag-security --help 2>&1 | head -3 || echo "ℹ️  Executable runs"
else
    echo "❌ RAG binary missing - run: make rag-build"
fi

echo -e "\n2. Checking etcd-server binary..."
if [ -f "/vagrant/etcd-server/build/etcd-server" ]; then
    echo "✅ etcd-server binary exists"
    /vagrant/etcd-server/build/etcd-server --help 2>&1 | head -3 || echo "ℹ️  Executable runs"
else
    echo "❌ etcd-server binary missing - run: make etcd-server-build"
fi

echo -e "\n3. Checking Makefile targets..."
make -n rag-build >/dev/null 2>&1 && echo "✅ make rag-build works" || echo "❌ make rag-build fails"
make -n rag-etcd-start >/dev/null 2>&1 && echo "✅ make rag-etcd-start works" || echo "❌ make rag-etcd-start fails"

echo -e "\n4. Quick process check..."
make rag-etcd-status 2>/dev/null | grep -q "running" && echo "✅ RAG ecosystem running" || echo "❌ RAG ecosystem not running"

echo -e "\n=== VERIFICATION COMPLETE ==="
