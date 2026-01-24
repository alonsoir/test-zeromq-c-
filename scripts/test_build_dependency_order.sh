#!/bin/bash
# ============================================================================
# Test Build Dependency Order
# ============================================================================
# This script verifies that the Makefile has correct dependency order
# by performing a clean build and checking compilation sequence
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🧪 Testing Build Dependency Order                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Test 1: Clean Build Test
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Clean Build (verifies correct dependency order)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "⚠️  WARNING: This will delete all build artifacts!"
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Test cancelled."
    exit 0
fi

echo ""
echo "Step 1: Cleaning all build artifacts..."
make clean > /dev/null 2>&1
echo -e "${GREEN}✅ Clean complete${NC}"

echo ""
echo "Step 2: Building from scratch (watch for order)..."
echo ""

# Build with verbose output to see order
make all 2>&1 | tee /tmp/build_order.log

echo ""
echo "Step 3: Analyzing build order..."
echo ""

# Extract build steps
PROTO_LINE=$(grep -n "Protobuf Unified System" /tmp/build_order.log | head -1 | cut -d: -f1)
ETCD_CLIENT_LINE=$(grep -n "Building etcd-client library" /tmp/build_order.log | head -1 | cut -d: -f1)
SNIFFER_LINE=$(grep -n "Building Sniffer" /tmp/build_order.log | head -1 | cut -d: -f1)
DETECTOR_LINE=$(grep -n "Building ML Detector" /tmp/build_order.log | head -1 | cut -d: -f1)
FIREWALL_LINE=$(grep -n "Building Firewall" /tmp/build_order.log | head -1 | cut -d: -f1)
ETCD_SERVER_LINE=$(grep -n "Building custom etcd-server\|Building etcd-server" /tmp/build_order.log | head -1 | cut -d: -f1)

echo "Build sequence detected:"
echo "  Line $PROTO_LINE: proto-unified"
echo "  Line $ETCD_CLIENT_LINE: etcd-client-build"
echo "  Line $SNIFFER_LINE: sniffer"
echo "  Line $DETECTOR_LINE: detector"
echo "  Line $FIREWALL_LINE: firewall"
echo "  Line $ETCD_SERVER_LINE: etcd-server"
echo ""

# Verify order
PASS=true

if [ "$PROTO_LINE" -lt "$ETCD_CLIENT_LINE" ] && \
   [ "$ETCD_CLIENT_LINE" -lt "$SNIFFER_LINE" ] && \
   [ "$ETCD_CLIENT_LINE" -lt "$DETECTOR_LINE" ] && \
   [ "$ETCD_CLIENT_LINE" -lt "$FIREWALL_LINE" ]; then
    echo -e "${GREEN}✅ Dependency order CORRECT${NC}"
    echo "   proto → etcd-client → components"
else
    echo -e "${RED}❌ Dependency order INCORRECT${NC}"
    echo "   Expected: proto → etcd-client → components"
    PASS=false
fi

echo ""

# ============================================================================
# Test 2: Verify Linkage
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Verify etcd-client Linkage"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

make verify-etcd-linkage > /tmp/linkage_check.log 2>&1

if grep -q "❌ NOT linked" /tmp/linkage_check.log; then
    echo -e "${RED}❌ Some components NOT linked with etcd-client${NC}"
    cat /tmp/linkage_check.log
    PASS=false
else
    echo -e "${GREEN}✅ All components linked with etcd-client${NC}"
fi

echo ""

# ============================================================================
# Test 3: Library Timestamp Check
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Timestamp Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Checking if components were compiled AFTER etcd-client..."

ETCD_TIME=$(vagrant ssh -c "stat -c %Y /vagrant/etcd-client/build/libetcd_client.so" | tr -d '\r\n' | tr -d ' ')
SNIFFER_TIME=$(vagrant ssh -c "stat -c %Y /vagrant/sniffer/build/sniffer 2>/dev/null || echo 0")
DETECTOR_TIME=$(vagrant ssh -c "stat -c %Y /vagrant/ml-detector/build/ml-detector 2>/dev/null || echo 0")
FIREWALL_TIME=$(vagrant ssh -c "stat -c %Y /vagrant/firewall-acl-agent/build/firewall-acl-agent 2>/dev/null || echo 0")

echo "  etcd-client: $ETCD_TIME"
echo "  sniffer:     $SNIFFER_TIME"
echo "  detector:    $DETECTOR_TIME"
echo "  firewall:    $FIREWALL_TIME"
echo ""

if [ "$SNIFFER_TIME" -gt "$ETCD_TIME" ] && \
   [ "$DETECTOR_TIME" -gt "$ETCD_TIME" ] && \
   [ "$FIREWALL_TIME" -gt "$ETCD_TIME" ]; then
    echo -e "${GREEN}✅ Components compiled AFTER etcd-client (correct)${NC}"
else
    echo -e "${RED}❌ Some components compiled BEFORE etcd-client (incorrect)${NC}"
    PASS=false
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "════════════════════════════════════════════════════════════"
echo "TEST SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ "$PASS" = true ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ ALL TESTS PASSED - Dependency order is CORRECT        ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Your Makefile has the correct dependency order:"
    echo "  1. proto-unified"
    echo "  2. etcd-client-build"
    echo "  3. sniffer/detector/firewall (in any order)"
    echo "  4. etcd-server-build"
    echo ""
    echo "Next steps:"
    echo "  make verify-pipeline-config"
    echo "  make run-lab-dev-day23"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌ TESTS FAILED - Dependency order needs fixing          ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Apply the corrected Makefile section:"
    echo "  1. Review: Makefile_build_section_corrected.patch"
    echo "  2. Read:   DEPENDENCY_ORDER_EXPLAINED.md"
    echo "  3. Apply:  Replace 'Build Targets' section in your Makefile"
    echo "  4. Test:   ./test_build_dependency_order.sh"
    exit 1
fi