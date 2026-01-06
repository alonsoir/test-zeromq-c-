#!/bin/bash
# Quick Fix Script - Day 34
# Corrige los 2 issues detectados por preflight_check

echo "╔════════════════════════════════════════════════════════╗"
echo "║  ML Defender - Day 34 Quick Fix                       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Issue 1: Install ONNX Runtime
echo "Fix 1: Installing ONNX Runtime Python..."
pip3 install onnxruntime --break-system-packages
if [ $? -eq 0 ]; then
    echo "   ✅ ONNX Runtime installed"
else
    echo "   ❌ Failed to install ONNX Runtime"
    exit 1
fi

echo ""
echo "Fix 2: Correcting paths in Python scripts..."

# Backup originals
cp preflight_check.py preflight_check.py.bak
cp test_real_inference.py test_real_inference.py.bak
cp test_batch_processing.py test_batch_processing.py.bak

# Fix paths using sed
sed -i 's|/vagrant/data/rag/events|/vagrant/logs/rag/events|g' preflight_check.py
sed -i 's|/vagrant/data/rag/events|/vagrant/logs/rag/events|g' test_real_inference.py
sed -i 's|/vagrant/data/rag/events|/vagrant/logs/rag/events|g' test_batch_processing.py

if [ $? -eq 0 ]; then
    echo "   ✅ Paths corrected in all scripts"
    echo "   ✅ Backups saved as *.bak"
else
    echo "   ❌ Failed to correct paths"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  FIXES COMPLETE - Re-run preflight_check.py           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Next step: python3 preflight_check.py"