#!/bin/bash
# =============================================================================
# 🎯 RAG Logger Test Orchestrator (Day 15 - FIXED v2)
# =============================================================================
# Authors: Alonso Isidoro Roman + Claude (Anthropic)
# Date: 2025-12-14
# Purpose: Orchestrate full ML Defender test with RAG logging
# Strategy: DON'T restart if running + Use artifacts as source of truth
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PCAP_FILE=${1:-"datasets/ctu13/botnet-capture-20110810-neris.pcap"}
TEST_NAME=$(basename "$PCAP_FILE" .pcap)
PROJECT_ROOT="/vagrant"
LOG_DIR="${PROJECT_ROOT}/logs/lab"
RAG_DIR="${PROJECT_ROOT}/logs/rag"

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  🎯 ML DEFENDER - RAG Logger Test (Day 15)                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Test:    ${GREEN}${TEST_NAME}${NC}"
echo -e "PCAP:    ${PCAP_FILE}"
echo -e "Date:    $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================================================================
# Step 1: Pre-flight checks
# =============================================================================
echo -e "${CYAN}[1/6]${NC} Pre-flight checks..."

# Check VMs running
if ! vagrant status | grep -q "defender.*running"; then
    echo -e "${RED}❌ Defender VM not running${NC}"
    echo -e "   Run: ${YELLOW}vagrant up defender${NC}"
    exit 1
fi

if ! vagrant status | grep -q "client.*running"; then
    echo -e "${RED}❌ Client VM not running${NC}"
    echo -e "   Run: ${YELLOW}vagrant up client${NC}"
    exit 1
fi

# Check PCAP exists
if ! vagrant ssh client -c "test -f ${PROJECT_ROOT}/${PCAP_FILE}" 2>/dev/null; then
    echo -e "${RED}❌ PCAP file not found: ${PCAP_FILE}${NC}"
    echo -e "   Available files:"
    vagrant ssh client -c "ls -lh ${PROJECT_ROOT}/datasets/ctu13/*.pcap" 2>/dev/null || true
    exit 1
fi

# Check binaries built
if ! vagrant ssh defender -c "test -f ${PROJECT_ROOT}/ml-detector/build/ml-detector" 2>/dev/null; then
    echo -e "${RED}❌ ml-detector binary not found${NC}"
    echo -e "   Run: ${YELLOW}make detector${NC}"
    exit 1
fi

if ! vagrant ssh defender -c "test -f ${PROJECT_ROOT}/sniffer/build/sniffer" 2>/dev/null; then
    echo -e "${RED}❌ sniffer binary not found${NC}"
    echo -e "   Run: ${YELLOW}make sniffer${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}"
echo ""

# =============================================================================
# Step 2: Clean previous RAG logs
# =============================================================================
echo -e "${CYAN}[2/6]${NC} Cleaning previous RAG logs..."

TODAY=$(date +%Y-%m-%d)
vagrant ssh defender -c "
    # Clean today's RAG logs only
    rm -f ${RAG_DIR}/events/${TODAY}.jsonl
    rm -rf ${RAG_DIR}/artifacts/${TODAY}

    # Ensure directories exist
    mkdir -p ${RAG_DIR}/events ${RAG_DIR}/artifacts
" 2>/dev/null

echo -e "${GREEN}✅ RAG logs cleaned (kept lab logs for continuity)${NC}"
echo ""

# =============================================================================
# Step 3: Ensure ML Defender lab is running (DON'T RESTART IF RUNNING)
# =============================================================================
echo -e "${CYAN}[3/6]${NC} Checking ML Defender lab status..."

DETECTOR_RUNNING=false
SNIFFER_RUNNING=false
FIREWALL_RUNNING=false

if vagrant ssh defender -c "pgrep -f 'ml-detector' > /dev/null" 2>/dev/null; then
    DETECTOR_RUNNING=true
    DETECTOR_PID=$(vagrant ssh defender -c "pgrep -f 'ml-detector'" 2>/dev/null | head -1)
fi

if vagrant ssh defender -c "pgrep -f 'sniffer.*-c' > /dev/null" 2>/dev/null; then
    SNIFFER_RUNNING=true
    SNIFFER_PID=$(vagrant ssh defender -c "pgrep -f 'sniffer.*-c'" 2>/dev/null | head -1)
fi

if vagrant ssh defender -c "pgrep -f 'firewall-acl-agent' > /dev/null" 2>/dev/null; then
    FIREWALL_RUNNING=true
    FIREWALL_PID=$(vagrant ssh defender -c "pgrep -f 'firewall-acl-agent'" 2>/dev/null | head -1)
fi

# Only start if NOT all running
if $DETECTOR_RUNNING && $SNIFFER_RUNNING && $FIREWALL_RUNNING; then
    echo -e "${GREEN}✅ Lab already running, reusing existing processes${NC}"
    echo -e "   Sniffer:  PID ${SNIFFER_PID}"
    echo -e "   Detector: PID ${DETECTOR_PID}"
    echo -e "   Firewall: PID ${FIREWALL_PID}"
else
    echo -e "${YELLOW}⚠️  Some components not running, starting lab...${NC}"

    # Kill any partial processes
    vagrant ssh defender -c "sudo pkill -9 -f firewall-acl-agent 2>/dev/null || true"
    vagrant ssh defender -c "pkill -9 -f ml-detector 2>/dev/null || true"
    vagrant ssh defender -c "sudo pkill -9 -f sniffer 2>/dev/null || true"
    sleep 3

    # Start lab
    echo -e "   Using: ${YELLOW}run_lab_dev.sh${NC}"
    vagrant ssh defender -c "bash ${PROJECT_ROOT}/scripts/run_lab_dev.sh > /dev/null 2>&1 &" 2>/dev/null

    echo -e "   Waiting for initialization (15 seconds)..."
    sleep 15

    # Verify components started
    if vagrant ssh defender -c "pgrep -f 'sniffer.*-c' > /dev/null" 2>/dev/null; then
        SNIFFER_PID=$(vagrant ssh defender -c "pgrep -f 'sniffer.*-c'" 2>/dev/null | head -1)
        echo -e "   ${GREEN}✅ Sniffer:  PID ${SNIFFER_PID}${NC}"
    else
        echo -e "   ${RED}❌ Sniffer failed to start${NC}"
        exit 1
    fi

    if vagrant ssh defender -c "pgrep -f 'ml-detector' > /dev/null" 2>/dev/null; then
        DETECTOR_PID=$(vagrant ssh defender -c "pgrep -f 'ml-detector'" 2>/dev/null | head -1)
        echo -e "   ${GREEN}✅ Detector: PID ${DETECTOR_PID}${NC}"
    else
        echo -e "   ${RED}❌ Detector failed to start${NC}"
        exit 1
    fi

    if vagrant ssh defender -c "pgrep -f 'firewall-acl-agent' > /dev/null" 2>/dev/null; then
        FIREWALL_PID=$(vagrant ssh defender -c "pgrep -f 'firewall-acl-agent'" 2>/dev/null | head -1)
        echo -e "   ${GREEN}✅ Firewall: PID ${FIREWALL_PID}${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Firewall not running (optional)${NC}"
    fi
fi

echo ""

# =============================================================================
# Step 4: Execute PCAP Replay
# =============================================================================
echo -e "${CYAN}[4/6]${NC} Replaying PCAP on client VM..."

PCAP_SIZE=$(vagrant ssh client -c "du -h ${PROJECT_ROOT}/${PCAP_FILE} 2>/dev/null | cut -f1" 2>/dev/null)
echo -e "   File size: ${PCAP_SIZE}"
echo -e "   Speed:     10 Mbps"
echo -e "   Duration:  ~1-5 minutes (depending on size)"
echo ""

START_TIME=$(date +%s)

# Execute replay
vagrant ssh client -c "
    sudo tcpreplay -i eth1 --mbps=10 --stats=2 ${PROJECT_ROOT}/${PCAP_FILE} 2>&1 | \
    tee ${PROJECT_ROOT}/logs/lab/tcpreplay.log
" 2>/dev/null || {
    echo -e "${RED}❌ tcpreplay failed${NC}"
    exit 1
}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}✅ Replay complete (${DURATION} seconds)${NC}"
echo ""

# =============================================================================
# Step 5: Wait for processing
# =============================================================================
echo -e "${CYAN}[5/6]${NC} Waiting for event processing and RAG flush..."
echo -n "   "
for i in {1..45}; do
    echo -n "."
    sleep 1
done
echo ""
echo -e "${YELLOW}   Forcing final flush (waiting 10 more seconds)...${NC}"
sleep 10
echo ""

echo -e "${GREEN}✅ Processing buffer complete${NC}"
echo ""

# =============================================================================
# Step 6: Analyze Results (ARTIFACTS = SOURCE OF TRUTH)
# =============================================================================
echo -e "${CYAN}[6/6]${NC} Analyzing results..."
echo ""

ARTIFACT_DIR="${RAG_DIR}/artifacts/${TODAY}"
RAG_FILE="${RAG_DIR}/events/${TODAY}.jsonl"

# Count artifacts (always works)
ARTIFACT_COUNT=0
if vagrant ssh defender -c "test -d ${ARTIFACT_DIR}" 2>/dev/null; then
    ARTIFACT_COUNT=$(vagrant ssh defender -c "find ${ARTIFACT_DIR} -name '*.json' -printf '.' 2>/dev/null | wc -c" 2>/dev/null | tr -cd '0-9')
    ARTIFACT_SIZE=$(vagrant ssh defender -c "du -sh ${ARTIFACT_DIR} 2>/dev/null | cut -f1" 2>/dev/null | xargs)
fi

# Validate ARTIFACT_COUNT is a number
if ! [[ "$ARTIFACT_COUNT" =~ ^[0-9]+$ ]]; then
    ARTIFACT_COUNT=0
fi

echo "We have $ARTIFACT_COUNT artifacts..."
# Check if we have artifacts
if [ -n "$ARTIFACT_COUNT" ] && [ "$ARTIFACT_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ RAG Events Generated (via artifacts)${NC}"
    echo -e "   Events:      ${YELLOW}${ARTIFACT_COUNT}${NC}"
    echo -e "   Size:        ${ARTIFACT_SIZE}"
    echo -e "   Directory:   ${ARTIFACT_DIR}"
    echo ""

    # Check if .jsonl also exists (bonus)
    if vagrant ssh defender -c "test -f ${RAG_FILE}" 2>/dev/null; then
        JSONL_COUNT=$(vagrant ssh defender -c "wc -l < ${RAG_FILE} 2>/dev/null" 2>/dev/null | tr -d ' ')
        JSONL_SIZE=$(vagrant ssh defender -c "du -h ${RAG_FILE} 2>/dev/null | cut -f1" 2>/dev/null)
        echo -e "${GREEN}✅ Consolidated log also available${NC}"
        echo -e "   JSONL Events: ${YELLOW}${JSONL_COUNT}${NC}"
        echo -e "   JSONL Size:   ${JSONL_SIZE}"
        echo -e "   File:         ${RAG_FILE}"
        echo ""
    else
        echo -e "${YELLOW}⚠️  Consolidated .jsonl not yet flushed${NC}"
        echo -e "   (Known timing issue - artifacts are authoritative)"
        echo ""
    fi

    # Analysis using artifacts
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  📊 DETECTION SUMMARY (from artifacts)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Classification breakdown
    vagrant ssh defender -c "
        find ${ARTIFACT_DIR} -name 'event_*.json' -exec cat {} \; 2>/dev/null | \
        jq -r '.detection.classification.final_class' 2>/dev/null | \
        sort | uniq -c | awk '{printf \"   %-15s %s\n\", \$2\":\", \$1}'
    " 2>/dev/null || echo "   (Classification parsing failed)"

    echo ""
    echo -e "${CYAN}  🎯 SCORE STATISTICS${NC}"
    echo ""

    # Score statistics
    vagrant ssh defender -c "
        find ${ARTIFACT_DIR} -name 'event_*.json' -exec cat {} \; 2>/dev/null | \
        jq -s '
        {
            total: length,
            avg_final: ([.[] | .detection.scores.final_score] | add / length),
            avg_divergence: ([.[] | .detection.scores.divergence] | add / length),
            high_divergence: ([.[] | select(.detection.scores.divergence >= 0.30)] | length),
            high_confidence: ([.[] | select(.detection.scores.final_score >= 0.70)] | length)
        } |
        \"   Total Events:        \(.total)\n\" +
        \"   Avg Final Score:     \(.avg_final | . * 100 | round / 100)\n\" +
        \"   Avg Divergence:      \(.avg_divergence | . * 100 | round / 100)\n\" +
        \"   High Divergence:     \(.high_divergence) events (>= 0.30)\n\" +
        \"   High Confidence:     \(.high_confidence) events (>= 0.70)\"
        ' -r
    " 2>/dev/null || echo "   (Statistics calculation failed)"

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Success summary
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ✅ DAY 15 TEST COMPLETE                                   ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Test:        ${GREEN}${TEST_NAME}${NC}"
    echo -e "Duration:    ${DURATION} seconds"
    echo -e "Events:      ${YELLOW}${ARTIFACT_COUNT}${NC} (from artifacts)"
    echo ""
    echo -e "${YELLOW}📁 Output Files:${NC}"
    echo -e "   ${ARTIFACT_DIR}/"
    if vagrant ssh defender -c "test -f ${RAG_FILE}" 2>/dev/null; then
        JSONL_COUNT=$(vagrant ssh defender -c "wc -l < ${RAG_FILE} 2>/dev/null" 2>/dev/null | tr -d ' ')
        echo -e "   ${RAG_FILE} (${JSONL_COUNT} events)"
    fi
    echo ""
    echo -e "${YELLOW}🔍 Next Steps:${NC}"
    echo -e "   1. Analyze individual events:"
    echo -e "      ${CYAN}cat ${ARTIFACT_DIR}/event_*.json | jq '.detection'${NC}"
    echo -e "   2. Validate with CTU-13 ground truth"
    echo -e "   3. Consolidate: ${CYAN}make rag-consolidate${NC}"
    echo ""
    echo -e "${GREEN}✅ ML Defender lab still running for additional tests${NC}"
    echo -e "   Stop with: ${YELLOW}make kill-lab${NC}"
    echo ""

    exit 0  # Success

else
    # No artifacts found
    echo -e "${RED}❌ No RAG events generated!${NC}"
    echo ""
    echo -e "${YELLOW}Debugging information:${NC}"
    echo ""
    echo "ML-Detector last 20 lines:"
    vagrant ssh defender -c "tail -20 ${LOG_DIR}/detector.log" 2>/dev/null || echo "  (log not found)"
    echo ""
    echo "Checking artifact directory:"
    vagrant ssh defender -c "ls -la ${ARTIFACT_DIR} 2>/dev/null" || echo "  (directory not found)"
    echo ""
    echo "Detector still running?"
    vagrant ssh defender -c "pgrep -a ml-detector" 2>/dev/null || echo "  (detector dead)"
    echo ""
    exit 1
fi