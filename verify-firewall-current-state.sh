#!/bin/bash
# Script: verify-firewall-current-state.sh

echo "=== FIREWALL CURRENT STATE VERIFICATION ==="
echo ""

echo "📁 1. CHECKING FIREWALL STRUCTURE..."
ls -lah /vagrant/firewall-acl-agent/

echo ""
echo "📄 2. CHECKING CMakeLists.txt..."
if [ -f "/vagrant/firewall-acl-agent/CMakeLists.txt" ]; then
    echo "✅ CMakeLists.txt exists"
    grep -i "etcd" /vagrant/firewall-acl-agent/CMakeLists.txt && echo "  ⚠️  etcd already referenced" || echo "  ❌ etcd NOT referenced"
    grep -i "lz4" /vagrant/firewall-acl-agent/CMakeLists.txt && echo "  ⚠️  lz4 already referenced" || echo "  ❌ lz4 NOT referenced"
    grep -i "openssl" /vagrant/firewall-acl-agent/CMakeLists.txt && echo "  ⚠️  openssl already referenced" || echo "  ❌ openssl NOT referenced"
else
    echo "❌ CMakeLists.txt NOT FOUND"
fi

echo ""
echo "📄 3. CHECKING SOURCE FILES..."
find /vagrant/firewall-acl-agent/src -name "*.cpp" -o -name "*.h" 2>/dev/null | head -10

echo ""
echo "📄 4. CHECKING CONFIG FILE..."
if [ -f "/vagrant/firewall-acl-agent/config/firewall.json" ]; then
    echo "✅ firewall.json exists"
    cat /vagrant/firewall-acl-agent/config/firewall.json
else
    echo "❌ firewall.json NOT FOUND"
fi

echo ""
echo "🔧 5. CHECKING DEPENDENCIES..."
echo "  etcd-client library:"
ls -lh /vagrant/etcd-client/build/libetcd_client.so 2>/dev/null && echo "  ✅ libetcd_client.so found" || echo "  ❌ libetcd_client.so NOT found"

echo "  LZ4 library:"
dpkg -l | grep liblz4 || echo "  ❌ LZ4 NOT installed"

echo "  OpenSSL:"
dpkg -l | grep libssl || echo "  ❌ OpenSSL NOT installed"

echo ""
echo "=== VERIFICATION COMPLETE ==="