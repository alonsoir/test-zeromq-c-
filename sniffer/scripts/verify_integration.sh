#!/bin/bash
# ============================================================================
# ML Defender Phase 1, Day 3 - Post-Integration Verification Script
# Purpose: Verify that integration was successful
# Run this AFTER applying changes and compiling
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ML Defender Phase 1, Day 3 - Verification Script          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="/vagrant/sniffer"
ERRORS=0
WARNINGS=0

# ============================================================================
# Check 1: Source files exist
# ============================================================================
echo "🔍 Check 1: Verifying source files..."

FILES=(
    "$PROJECT_ROOT/include/flow_manager.hpp"
    "$PROJECT_ROOT/include/ring_consumer.hpp"
    "$PROJECT_ROOT/include/ml_defender_features.hpp"
    "$PROJECT_ROOT/src/userspace/ring_consumer.cpp"
    "$PROJECT_ROOT/src/userspace/ml_defender_features.cpp"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ MISSING: $file"
        ((ERRORS++))
    fi
done

# ============================================================================
# Check 2: Verify flow_manager.hpp has new method
# ============================================================================
echo ""
echo "🔍 Check 2: Verifying flow_manager.hpp modifications..."

if grep -q "get_flow_stats_unsafe" "$PROJECT_ROOT/include/flow_manager.hpp" 2>/dev/null; then
    echo "   ✅ get_flow_stats_unsafe() method found"
else
    echo "   ❌ get_flow_stats_unsafe() method NOT found"
    echo "      → flow_manager.hpp needs modification"
    ((ERRORS++))
fi

# ============================================================================
# Check 3: Verify ring_consumer.hpp has includes
# ============================================================================
echo ""
echo "🔍 Check 3: Verifying ring_consumer.hpp includes..."

if grep -q "ml_defender_features.hpp" "$PROJECT_ROOT/include/ring_consumer.hpp" 2>/dev/null; then
    echo "   ✅ ml_defender_features.hpp include found"
else
    echo "   ❌ ml_defender_features.hpp include NOT found"
    ((ERRORS++))
fi

if grep -q "flow_manager.hpp" "$PROJECT_ROOT/include/ring_consumer.hpp" 2>/dev/null; then
    echo "   ✅ flow_manager.hpp include found"
else
    echo "   ❌ flow_manager.hpp include NOT found"
    ((ERRORS++))
fi

# ============================================================================
# Check 4: Verify ring_consumer.cpp has integration
# ============================================================================
echo ""
echo "🔍 Check 4: Verifying ring_consumer.cpp integration..."

if grep -q "thread_local FlowManager" "$PROJECT_ROOT/src/userspace/ring_consumer.cpp" 2>/dev/null; then
    echo "   ✅ thread_local FlowManager found"
else
    echo "   ❌ thread_local FlowManager NOT found"
    ((ERRORS++))
fi

if grep -q "thread_local MLDefenderExtractor" "$PROJECT_ROOT/src/userspace/ring_consumer.cpp" 2>/dev/null; then
    echo "   ✅ thread_local MLDefenderExtractor found"
else
    echo "   ❌ thread_local MLDefenderExtractor NOT found"
    ((ERRORS++))
fi

if grep -q "flow_manager_.add_packet" "$PROJECT_ROOT/src/userspace/ring_consumer.cpp" 2>/dev/null; then
    echo "   ✅ flow_manager_.add_packet() call found"
else
    echo "   ❌ flow_manager_.add_packet() call NOT found"
    ((WARNINGS++))
fi

if grep -q "ml_extractor_.populate_ml_defender_features" "$PROJECT_ROOT/src/userspace/ring_consumer.cpp" 2>/dev/null; then
    echo "   ✅ ml_extractor_.populate_ml_defender_features() call found"
else
    echo "   ❌ ml_extractor_.populate_ml_defender_features() call NOT found"
    ((ERRORS++))
fi

# ============================================================================
# Check 5: Verify binary exists
# ============================================================================
echo ""
echo "🔍 Check 5: Verifying compiled binary..."

if [ -f "$PROJECT_ROOT/build/sniffer" ]; then
    echo "   ✅ Binary exists: $PROJECT_ROOT/build/sniffer"

    # Check size (should be reasonable)
    SIZE=$(stat -f%z "$PROJECT_ROOT/build/sniffer" 2>/dev/null || stat -c%s "$PROJECT_ROOT/build/sniffer" 2>/dev/null)
    if [ "$SIZE" -gt 100000 ]; then
        echo "   ✅ Binary size: $(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo $SIZE bytes)"
    else
        echo "   ⚠️  Binary size seems small: $SIZE bytes"
        ((WARNINGS++))
    fi
else
    echo "   ❌ Binary NOT found - project not compiled?"
    echo "      → Run: cd $PROJECT_ROOT/build && cmake .. && make -j4"
    ((ERRORS++))
fi

# ============================================================================
# Check 6: Verify symbols in binary
# ============================================================================
if [ -f "$PROJECT_ROOT/build/sniffer" ] && command -v nm &> /dev/null; then
    echo ""
    echo "🔍 Check 6: Verifying symbols in binary..."

    if nm "$PROJECT_ROOT/build/sniffer" 2>/dev/null | grep -q "MLDefenderExtractor"; then
        echo "   ✅ MLDefenderExtractor symbols found"
    else
        echo "   ⚠️  MLDefenderExtractor symbols not found"
        echo "      → This might be OK if symbols are stripped"
        ((WARNINGS++))
    fi
fi

# ============================================================================
# Check 7: Test run (if requested)
# ============================================================================
echo ""
echo "🔍 Check 7: Test run available..."
echo "   ℹ️  To test: sudo $PROJECT_ROOT/build/sniffer -c config.json -vv"
echo "   Expected output: [ML Defender] Extracted features for flow..."

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                     VERIFICATION SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED!"
    echo ""
    echo "Integration appears successful. Next steps:"
    echo "  1. Test: sudo $PROJECT_ROOT/build/sniffer -c config.json -vv"
    echo "  2. Verify [ML Defender] logs appear"
    echo "  3. Check protobuf output contains ml_defender submessages"
    echo ""
    echo "Ready for Phase 1, Day 4: ML Detector Integration 🚀"
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  CHECKS PASSED WITH WARNINGS"
    echo ""
    echo "Warnings: $WARNINGS"
    echo "These are likely non-critical, but review them."
    echo ""
    echo "You can proceed with testing:"
    echo "  sudo $PROJECT_ROOT/build/sniffer -c config.json -vv"
else
    echo "❌ INTEGRATION INCOMPLETE"
    echo ""
    echo "Errors: $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Please review INTEGRATION_SUMMARY.md and apply missing changes."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"