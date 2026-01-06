#!/usr/bin/env python3
"""
ML Defender - Day 34 Pre-Flight Check

Verifica que todo está listo antes de empezar las pruebas.
"""

import os
import sys
from pathlib import Path

def check_models():
    """Verificar que los modelos ONNX existen"""
    print("1. Checking ONNX models...")

    models = [
        "chronos_embedder.onnx",
        "sbert_embedder.onnx",
        "attack_embedder.onnx"
    ]

    all_found = True
    for model in models:
        if os.path.exists(model):
            size = os.path.getsize(model)
            print(f"   ✅ {model:25s} ({size/1024:.1f} KB)")
        else:
            print(f"   ❌ {model:25s} NOT FOUND")
            all_found = False

    return all_found

def check_data():
    """Verificar que hay datos JSONL disponibles"""
    print("\n2. Checking JSONL data...")

    data_dir = Path("/vagrant/logs/rag/events")

    if not data_dir.exists():
        print(f"   ❌ Directory not found: {data_dir}")
        return False

    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"   ❌ No JSONL files found in {data_dir}")
        return False

    print(f"   ✅ Found {len(jsonl_files)} JSONL file(s):")
    for jf in jsonl_files[-3:]:  # Show last 3
        size = jf.stat().st_size
        print(f"      - {jf.name:40s} ({size/1024:.1f} KB)")

    return True

def check_onnxruntime():
    """Verificar que ONNX Runtime está instalado"""
    print("\n3. Checking ONNX Runtime...")

    try:
        import onnxruntime as ort
        version = ort.__version__
        print(f"   ✅ ONNX Runtime installed: v{version}")
        return True
    except ImportError:
        print("   ❌ ONNX Runtime not installed")
        print("      Run: pip3 install onnxruntime --break-system-packages")
        return False

def check_numpy():
    """Verificar que NumPy está instalado"""
    print("\n4. Checking NumPy...")

    try:
        import numpy as np
        version = np.__version__
        print(f"   ✅ NumPy installed: v{version}")
        return True
    except ImportError:
        print("   ❌ NumPy not installed")
        print("      Run: pip3 install numpy --break-system-packages")
        return False

def check_cpp_compiler():
    """Verificar que g++ está disponible"""
    print("\n5. Checking C++ compiler...")

    result = os.system("g++ --version > /dev/null 2>&1")

    if result == 0:
        print("   ✅ g++ compiler available")
        return True
    else:
        print("   ❌ g++ compiler not found")
        return False

def estimate_runtime():
    """Estimar tiempo de ejecución"""
    print("\n6. Estimated runtime:")
    print("   - Phase 1 (Python inference):    ~5-10 min")
    print("   - Phase 2 (C++ compile + test):  ~10-15 min")
    print("   - Phase 3 (Batch processing):    ~5-10 min")
    print("   " + "="*50)
    print("   Total estimated time:            ~20-35 min")

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  ML Defender - Day 34 Pre-Flight Check               ║")
    print("╚════════════════════════════════════════════════════════╝\n")

    checks = [
        ("ONNX Models", check_models),
        ("JSONL Data", check_data),
        ("ONNX Runtime", check_onnxruntime),
        ("NumPy", check_numpy),
        ("C++ Compiler", check_cpp_compiler),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Error during {name} check: {e}")
            results.append((name, False))

    # Runtime estimate
    estimate_runtime()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - READY TO START DAY 34")
        print("="*60)
        print("\nNext steps:")
        print("  1. python3 test_real_inference.py")
        print("  2. Compile and run test_real_embedders.cpp")
        print("  3. python3 test_batch_processing.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE STARTING")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())