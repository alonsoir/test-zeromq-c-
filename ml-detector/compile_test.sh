#!/bin/bash

echo "🔍 Verificando archivos..."
echo ""

# Verificar qué header existe y renombrar si es necesario
if [ -d "include/ml-defender" ]; then
    echo "⚠️  Renombrando include/ml-defender → include/ml_defender"
    mv include/ml-defender include/ml_defender
fi

if [ -f "include/ml_defender/ransomware_detector.hpp" ]; then
    echo "✅ Header encontrado en: include/ml_defender/"
else
    echo "❌ ERROR: No se encuentra ransomware_detector.hpp"
    exit 1
fi

# Verificar otros archivos
echo "✅ src/ransomware_detector.cpp: $(ls -lh src/ransomware_detector.cpp | awk '{print $5}')"
echo "✅ src/forest_trees_inline.hpp: $(ls -lh src/forest_trees_inline.hpp | awk '{print $5}')"
echo "✅ tests/unit/test_ransomware_detector.cpp: $(ls -lh tests/unit/test_ransomware_detector.cpp | awk '{print $5}')"
echo ""

# Crear directorio build
mkdir -p build_detector_test
echo "📁 Build directory: build_detector_test/"
echo ""

# Compilar detector
echo "🔧 Compilando ransomware_detector.cpp..."
g++ -std=c++20 -O3 -march=native \
    -I./include -I./src \
    -c src/ransomware_detector.cpp \
    -o build_detector_test/ransomware_detector.o

if [ $? -eq 0 ]; then
    echo "✅ ransomware_detector.o compilado correctamente"
else
    echo "❌ ERROR compilando ransomware_detector.cpp"
    exit 1
fi
echo ""

# Compilar test
echo "🔧 Compilando test_ransomware_detector.cpp..."
g++ -std=c++20 -O3 -march=native \
    -I./include -I./src \
    tests/unit/test_ransomware_detector.cpp \
    build_detector_test/ransomware_detector.o \
    -o build_detector_test/test_unit

if [ $? -eq 0 ]; then
    echo "✅ test_unit compilado correctamente"
else
    echo "❌ ERROR compilando test"
    exit 1
fi
echo ""

# Ejecutar test
echo "=========================================="
echo "🚀 EJECUTANDO TESTS"
echo "=========================================="
./build_detector_test/test_unit

TEST_RESULT=$?
echo ""
echo "=========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ TESTS PASSED"
else
    echo "❌ TESTS FAILED (exit code: $TEST_RESULT)"
fi
echo "=========================================="