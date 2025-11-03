#!/bin/bash
# verify_macos_setup.sh
# Verificar que el setup en macOS está correcto antes de entrenar

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Verificación de Setup macOS - ML Training                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

check_pass() {
    echo -e "${GREEN}✅${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}❌${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠️${NC}  $1"
    ((WARNINGS++))
}

# ========================================
# 1. PYTHON & VIRTUALENV
# ========================================
echo "🐍 PYTHON & VIRTUALENV"
echo "────────────────────────────────────────────────────────────"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        check_pass "Python $PYTHON_VERSION (>= 3.8 required)"
    else
        check_fail "Python $PYTHON_VERSION (3.8+ required)"
    fi
else
    check_fail "Python 3 not found"
fi

# Check virtualenv
if [ -d ".venv-macos" ]; then
    check_pass "Virtualenv .venv-macos exists"
    
    # Check if activated
    if [[ "$VIRTUAL_ENV" == *".venv-macos"* ]]; then
        check_pass "Virtualenv is activated"
    else
        check_warn "Virtualenv exists but not activated. Run: source .venv-macos/bin/activate"
    fi
else
    check_fail "Virtualenv .venv-macos not found. Run: python3 -m venv .venv-macos"
fi

echo ""

# ========================================
# 2. PYTHON DEPENDENCIES
# ========================================
echo "📦 PYTHON DEPENDENCIES"
echo "────────────────────────────────────────────────────────────"

DEPS=(
    "pandas"
    "numpy"
    "sklearn:scikit-learn"
    "matplotlib"
    "seaborn"
    "onnx"
    "onnxruntime"
    "skl2onnx"
    "imblearn:imbalanced-learn"
    "joblib"
    "psutil"
)

for dep in "${DEPS[@]}"; do
    IFS=':' read -r import_name package_name <<< "$dep"
    if [ -z "$package_name" ]; then
        package_name=$import_name
    fi
    
    if python3 -c "import $import_name" 2>/dev/null; then
        VERSION=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null)
        check_pass "$package_name ($VERSION)"
    else
        check_fail "$package_name not installed"
    fi
done

echo ""

# ========================================
# 3. DATASET
# ========================================
echo "📂 DATASET"
echo "────────────────────────────────────────────────────────────"

DATASET_PATH="datasets/CIC-DDoS-2019"

if [ -d "$DATASET_PATH" ]; then
    CSV_COUNT=$(find "$DATASET_PATH" -name "*.csv" 2>/dev/null | wc -l)
    
    if [ "$CSV_COUNT" -gt 0 ]; then
        DATASET_SIZE=$(du -sh "$DATASET_PATH" 2>/dev/null | awk '{print $1}')
        check_pass "Dataset found: $CSV_COUNT CSV files ($DATASET_SIZE)"
    else
        check_fail "Dataset directory exists but no CSV files found"
    fi
else
    check_fail "Dataset not found at: $DATASET_PATH"
    echo "       Download from: https://www.unb.ca/cic/datasets/ddos-2019.html"
fi

echo ""

# ========================================
# 4. SCRIPTS
# ========================================
echo "📜 SCRIPTS"
echo "────────────────────────────────────────────────────────────"

SCRIPTS=(
    "scripts/train_level2_ddos_binary_optimized.py"
    "scripts/convert_level2_ddos_to_onnx.py"
    "scripts/check_system_config.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        check_pass "$(basename $script)"
    else
        check_fail "$(basename $script) not found"
    fi
done

echo ""

# ========================================
# 5. DISK SPACE
# ========================================
echo "💾 DISK SPACE"
echo "────────────────────────────────────────────────────────────"

FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
FREE_SPACE_GB=$(df -g . | awk 'NR==2 {print $4}')

if [ "$FREE_SPACE_GB" -gt 10 ]; then
    check_pass "Free disk space: $FREE_SPACE (>10GB available)"
elif [ "$FREE_SPACE_GB" -gt 5 ]; then
    check_warn "Free disk space: $FREE_SPACE (10GB+ recommended)"
else
    check_fail "Free disk space: $FREE_SPACE (<5GB - need more)"
fi

echo ""

# ========================================
# 6. MEMORY
# ========================================
echo "🧠 MEMORY"
echo "────────────────────────────────────────────────────────────"

if command -v sysctl &> /dev/null; then
    TOTAL_MEM_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    
    if [ "$TOTAL_MEM_GB" -ge 16 ]; then
        check_pass "Total RAM: ${TOTAL_MEM_GB}GB (excellent for ML training)"
    elif [ "$TOTAL_MEM_GB" -ge 8 ]; then
        check_warn "Total RAM: ${TOTAL_MEM_GB}GB (okay, consider smaller sample_size)"
    else
        check_fail "Total RAM: ${TOTAL_MEM_GB}GB (<8GB - may have issues)"
    fi
fi

echo ""

# ========================================
# 7. VM STATUS (if Vagrant available)
# ========================================
if command -v vagrant &> /dev/null; then
    echo "🖥️  VM STATUS"
    echo "────────────────────────────────────────────────────────────"
    
    cd ..  # Go to project root
    VM_STATUS=$(vagrant status 2>/dev/null | grep "default" | awk '{print $2}')
    cd ml-training
    
    if [[ "$VM_STATUS" == "running" ]]; then
        check_pass "VM is running"
    elif [[ "$VM_STATUS" == "poweroff" ]]; then
        check_warn "VM is powered off. Start with: vagrant up"
    else
        check_warn "VM status: $VM_STATUS"
    fi
    
    echo ""
fi

# ========================================
# 8. OUTPUT DIRECTORIES
# ========================================
echo "📁 OUTPUT DIRECTORIES"
echo "────────────────────────────────────────────────────────────"

OUTPUT_DIRS=(
    "outputs"
    "outputs/models"
    "outputs/onnx"
    "outputs/metadata"
    "outputs/plots"
)

for dir in "${OUTPUT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "$dir/ exists"
    else
        check_warn "$dir/ not found (will be created automatically)"
    fi
done

echo ""

# ========================================
# SUMMARY
# ========================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  SUMMARY                                                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

TOTAL=$((PASSED + FAILED + WARNINGS))

echo "  Passed:   $PASSED / $TOTAL"
echo "  Failed:   $FAILED / $TOTAL"
echo "  Warnings: $WARNINGS / $TOTAL"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ Setup is ready for ML training!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Activate virtualenv: source .venv-macos/bin/activate"
    echo "  2. Check system config: python scripts/check_system_config.py"
    echo "  3. Train model: python scripts/train_level2_ddos_binary_optimized.py"
    exit 0
elif [ $FAILED -le 2 ]; then
    echo -e "${YELLOW}⚠️  Setup has minor issues but may work${NC}"
    echo ""
    echo "Fix the failed checks above, then:"
    echo "  1. Run this script again: ./scripts/verify_macos_setup.sh"
    echo "  2. Or proceed with caution: python scripts/train_level2_ddos_binary_optimized.py"
    exit 1
else
    echo -e "${RED}❌ Setup has critical issues - please fix them${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - Install Python deps: pip install pandas numpy scikit-learn ..."
    echo "  - Download dataset: https://www.unb.ca/cic/datasets/ddos-2019.html"
    echo "  - Create virtualenv: python3 -m venv .venv-macos"
    exit 1
fi
