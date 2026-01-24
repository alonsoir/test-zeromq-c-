#!/bin/bash
# Update ONNX Runtime C++ to v1.23.2
# Matches the Python version and supports IR version 10

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Update ONNX Runtime C++ to v1.23.2                  ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

VERSION="1.23.2"
ARCHIVE="onnxruntime-linux-x64-${VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${ARCHIVE}"
EXTRACT_DIR="onnxruntime-linux-x64-${VERSION}"

cd /tmp

echo "Step 1: Downloading ONNX Runtime v${VERSION}..."
wget -q --show-progress "${URL}"

if [ ! -f "${ARCHIVE}" ]; then
    echo "❌ Failed to download"
    exit 1
fi
echo "  ✅ Downloaded ${ARCHIVE}"

echo ""
echo "Step 2: Extracting..."
tar -xzf "${ARCHIVE}"
echo "  ✅ Extracted to ${EXTRACT_DIR}"

echo ""
echo "Step 3: Backing up old version..."
sudo mv /usr/local/include/onnxruntime_c_api.h /usr/local/include/onnxruntime_c_api.h.v1.17.1.bak 2>/dev/null || true
sudo mv /usr/local/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so.v1.17.1.bak 2>/dev/null || true
echo "  ✅ Backups created (if they existed)"

echo ""
echo "Step 4: Installing new version..."
sudo cp -r "${EXTRACT_DIR}/include/"* /usr/local/include/
sudo cp -r "${EXTRACT_DIR}/lib/"* /usr/local/lib/
echo "  ✅ Files copied to /usr/local"

echo ""
echo "Step 5: Updating library cache..."
sudo ldconfig
echo "  ✅ ldconfig complete"

echo ""
echo "Step 6: Verifying installation..."
if [ -f "/usr/local/lib/libonnxruntime.so.${VERSION}" ]; then
    echo "  ✅ libonnxruntime.so.${VERSION} installed"
else
    echo "  ⚠️  Version-specific .so not found (may be OK)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  ONNX Runtime C++ v${VERSION} INSTALLED               ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Recompile test: g++ -std=c++20 -o test_real_embedders test_real_embedders.cpp -I/usr/local/include -L/usr/local/lib -lonnxruntime"
echo "  2. Run test: ./test_real_embedders"
echo "  3. Should now support IR version 10 ✅"