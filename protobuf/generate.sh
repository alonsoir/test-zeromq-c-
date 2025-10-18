#!/bin/bash
set -e

PROTO_FILE="network_security.proto"
PROTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Protobuf Schema Generator                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Schema: ${PROTO_FILE}"
echo "📂 Output: ${PROTO_DIR}"
echo ""

# Check protoc
if ! command -v protoc >/dev/null 2>&1; then
    echo "❌ protoc not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y protobuf-compiler
fi

PROTOC_VERSION=$(protoc --version)
echo "✅ ${PROTOC_VERSION}"
echo ""

# Generate C++ files
echo "🔨 Generating C++ protobuf files..."
protoc \
    --cpp_out="${PROTO_DIR}" \
    --proto_path="${PROTO_DIR}" \
    "${PROTO_DIR}/${PROTO_FILE}"

# Verify generation
if [ -f "${PROTO_DIR}/network_security.pb.cc" ] && [ -f "${PROTO_DIR}/network_security.pb.h" ]; then
    echo "✅ Generated successfully:"
    ls -lh "${PROTO_DIR}"/network_security.pb.*
    
    # Show file sizes
    CC_SIZE=$(wc -l < "${PROTO_DIR}/network_security.pb.cc")
    H_SIZE=$(wc -l < "${PROTO_DIR}/network_security.pb.h")
    echo ""
    echo "📊 Statistics:"
    echo "   network_security.pb.cc: ${CC_SIZE} lines"
    echo "   network_security.pb.h:  ${H_SIZE} lines"
    
    # Generate Python files too (for reference)
    echo ""
    echo "🐍 Generating Python protobuf files..."
    protoc \
        --python_out="${PROTO_DIR}" \
        --proto_path="${PROTO_DIR}" \
        "${PROTO_DIR}/${PROTO_FILE}"
    
    if [ -f "${PROTO_DIR}/network_security_pb2.py" ]; then
        PY_SIZE=$(wc -l < "${PROTO_DIR}/network_security_pb2.py")
        echo "✅ network_security_pb2.py: ${PY_SIZE} lines"
    fi
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ✅ Protobuf generation complete                           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Review generated files"
    echo "   2. Rebuild sniffer: cd /vagrant/sniffer && make"
    echo "   3. Rebuild ml-detector: cd /vagrant/ml-detector/build && cmake .. && make"
    echo ""
else
    echo "❌ Generation failed!"
    exit 1
fi
