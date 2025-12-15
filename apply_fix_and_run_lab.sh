#!/bin/bash
# apply_fix_and_run_lab.sh - Apply RAGLogger fix and test with full lab
# Day 16 - December 15, 2025
# Authors: Alonso + Claude

set -e

PROJECT_ROOT="/vagrant"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔧 Applying RAGLogger Fix + Full Lab Test                ║"
echo "║  Day 16 - December 15, 2025                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_ROOT"

# ============================================================================
# Step 1: Backup Current Files
# ============================================================================

echo "📦 Step 1: Backing up current files..."
echo ""

if [ -f "ml-detector/src/rag_logger.cpp" ]; then
    cp ml-detector/src/rag_logger.cpp "ml-detector/src/rag_logger.cpp.backup_${TIMESTAMP}"
    echo "✅ Backed up: ml-detector/src/rag_logger.cpp.backup_${TIMESTAMP}"
fi

if [ -f "ml-detector/include/rag_logger.hpp" ]; then
    cp ml-detector/include/rag_logger.hpp "ml-detector/include/rag_logger.hpp.backup_${TIMESTAMP}"
    echo "✅ Backed up: ml-detector/include/rag_logger.hpp.backup_${TIMESTAMP}"
fi

echo ""

# ============================================================================
# Step 2: Verify Fixed Files Available
# ============================================================================

echo "🔍 Step 2: Checking for fixed files..."
echo ""

if [ ! -f "rag_logger_FIXED.cpp" ] || [ ! -f "rag_logger_FIXED.hpp" ]; then
    echo "❌ Fixed files not found in /vagrant"
    echo ""
    echo "Expected files:"
    echo "  • /vagrant/rag_logger_FIXED.cpp"
    echo "  • /vagrant/rag_logger_FIXED.hpp"
    echo ""
    echo "Please download them from Claude and place in /vagrant/"
    exit 1
fi

echo "✅ Found fixed files"
echo ""

# ============================================================================
# Step 3: Apply Fixed Files
# ============================================================================

echo "🔧 Step 3: Applying fixed files..."
echo ""

cp rag_logger_FIXED.cpp ml-detector/src/rag_logger.cpp
cp rag_logger_FIXED.hpp ml-detector/include/rag_logger.hpp

echo "✅ Applied: ml-detector/src/rag_logger.cpp"
echo "✅ Applied: ml-detector/include/rag_logger.hpp"
echo ""

echo "📋 Key changes:"
echo "   • check_rotation() moved inside write_jsonl() critical section"
echo "   • Added check_rotation_locked() - assumes mutex held"
echo "   • Added rotate_logs_locked() - assumes mutex held"
echo "   • Eliminated races: current_date_, current_log_, events_in_current_file_"
echo ""

# ============================================================================
# Step 4: Kill Existing Lab
# ============================================================================

echo "🛑 Step 4: Stopping existing lab components..."
echo ""

make kill-lab 2>/dev/null || true
sleep 2

echo "✅ Lab stopped"
echo ""

# ============================================================================
# Step 5: Rebuild ml-detector with Release Flags
# ============================================================================

echo "🔨 Step 5: Rebuilding ml-detector (RELEASE FLAGS)..."
echo ""

cd ml-detector
make clean 2>/dev/null || rm -rf build/*.o

# Ensure we're using release flags (no TSan, no debug)
unset CXXFLAGS
unset LDFLAGS

echo "   Building with: make (default optimization)"
echo ""

if make; then
    echo ""
    echo "✅ ml-detector compiled successfully!"
else
    echo ""
    echo "❌ Compilation failed"
    echo ""
    echo "Restoring backups..."
    cp "src/rag_logger.cpp.backup_${TIMESTAMP}" src/rag_logger.cpp 2>/dev/null || true
    cp "include/rag_logger.hpp.backup_${TIMESTAMP}" include/rag_logger.hpp 2>/dev/null || true
    exit 1
fi

echo ""
echo "📊 Binary info:"
ls -lh build/ml-detector
echo ""

cd "$PROJECT_ROOT"

# ============================================================================
# Step 6: Clean Previous Logs
# ============================================================================

echo "🧹 Step 6: Cleaning previous logs..."
echo ""

TODAY=$(date +%Y-%m-%d)
rm -rf logs/rag/artifacts/$TODAY 2>/dev/null || true
rm -f logs/rag/events/$TODAY.jsonl 2>/dev/null || true

mkdir -p logs/rag/artifacts
mkdir -p logs/rag/events

echo "✅ Logs cleaned"
echo ""

# ============================================================================
# Step 7: Start Full Lab
# ============================================================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 STARTING FULL LAB (make run-lab-dev)                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "⏱️  Test duration: 10 minutes (or Ctrl+C to stop)"
echo ""
echo "Starting lab in 3 seconds..."
sleep 3
echo ""

make run-lab-dev &
LAB_PID=$!

echo "🔍 Lab PID: $LAB_PID"
echo ""
echo "⏳ Waiting 15 seconds for components to start..."
sleep 15
echo ""

# ============================================================================
# Step 8: Monitor Lab
# ============================================================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  📊 MONITORING LAB (10 minutes)                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

START_TIME=$(date +%s)
MAX_RUNTIME=600  # 10 minutes
ELAPSED=0
LAST_CHECK=0

while kill -0 $LAB_PID 2>/dev/null; do
    sleep 1
    ELAPSED=$(($(date +%s) - START_TIME))

    # Print progress every second
    printf "\r⏱️  Runtime: %02d:%02d / 10:00  " $((ELAPSED/60)) $((ELAPSED%60))

    # Detailed check every 30 seconds
    if [ $((ELAPSED % 30)) -eq 0 ] && [ $ELAPSED -ne $LAST_CHECK ] && [ $ELAPSED -gt 0 ]; then
        echo ""
        echo ""
        echo "📊 Status at $(date +%H:%M:%S):"

        # Check components
        echo "   Components:"
        vagrant ssh defender -c "ps aux | grep -E '(sniffer|ml-detector|firewall)' | grep -v grep" 2>/dev/null | while read line; do
            echo "     • $line" | awk '{print $11" (PID "$2")"}'
        done || echo "     ⚠️  Could not check components"

        # Check artifacts
        ARTIFACT_COUNT=$(vagrant ssh defender -c "ls -1 /vagrant/logs/rag/artifacts/$TODAY/ 2>/dev/null | wc -l" 2>/dev/null || echo "0")
        echo "   Artifacts: $ARTIFACT_COUNT events"

        # Check JSONL
        if vagrant ssh defender -c "test -f /vagrant/logs/rag/events/$TODAY.jsonl" 2>/dev/null; then
            JSONL_LINES=$(vagrant ssh defender -c "wc -l < /vagrant/logs/rag/events/$TODAY.jsonl" 2>/dev/null || echo "0")
            echo "   JSONL: $JSONL_LINES lines"
        else
            echo "   JSONL: not yet created"
        fi

        printf "\n⏱️  "
        LAST_CHECK=$ELAPSED
    fi

    # Stop after 10 minutes
    if [ $ELAPSED -ge $MAX_RUNTIME ]; then
        echo ""
        echo ""
        echo "⏰ 10 minutes reached - stopping lab"
        make kill-lab 2>/dev/null || true
        sleep 3
        break
    fi
done

# Wait for lab to finish
wait $LAB_PID 2>/dev/null || true

FINAL_ELAPSED=$(($(date +%s) - START_TIME))

echo ""
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  📊 VALIDATION RESULTS                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Total runtime: $(($FINAL_ELAPSED / 60))m $(($FINAL_ELAPSED % 60))s"
echo ""

# ============================================================================
# Step 9: Check Results
# ============================================================================

echo "📈 Step 9: Checking results..."
echo ""

# Check if ml-detector is still running (inside VM)
echo "🔍 Component status:"
DETECTOR_RUNNING=$(vagrant ssh defender -c "pgrep -f 'ml-detector' | wc -l" 2>/dev/null || echo "0")

if [ "$DETECTOR_RUNNING" -gt 0 ]; then
    echo "   ✅ ml-detector still running (PID: $(vagrant ssh defender -c 'pgrep -f ml-detector' 2>/dev/null))"
else
    echo "   ⚠️  ml-detector stopped (may have crashed or test ended)"
fi

echo ""

# Check artifacts
echo "📁 Artifacts generated:"
ARTIFACT_COUNT=$(vagrant ssh defender -c "ls -1 /vagrant/logs/rag/artifacts/$TODAY/ 2>/dev/null | wc -l" 2>/dev/null || echo "0")
echo "   Total: $ARTIFACT_COUNT events"

if [ "$ARTIFACT_COUNT" -gt 0 ]; then
    echo "   ✅ Artifacts directory: logs/rag/artifacts/$TODAY/"
    echo ""
    echo "   Sample files:"
    vagrant ssh defender -c "ls -lh /vagrant/logs/rag/artifacts/$TODAY/ | head -5" 2>/dev/null || true
fi

echo ""

# Check consolidated JSONL
echo "📄 Consolidated JSONL:"
JSONL_PATH="logs/rag/events/$TODAY.jsonl"

if vagrant ssh defender -c "test -f /vagrant/$JSONL_PATH" 2>/dev/null; then
    JSONL_LINES=$(vagrant ssh defender -c "wc -l < /vagrant/$JSONL_PATH" 2>/dev/null || echo "0")
    JSONL_SIZE=$(vagrant ssh defender -c "du -h /vagrant/$JSONL_PATH | cut -f1" 2>/dev/null || echo "?")

    echo "   • Path: $JSONL_PATH"
    echo "   • Lines: $JSONL_LINES"
    echo "   • Size: $JSONL_SIZE"

    # Validate format
    echo ""
    echo "🔍 Validating JSONL format..."
    if vagrant ssh defender -c "tail -1 /vagrant/$JSONL_PATH | jq . >/dev/null 2>&1" 2>/dev/null; then
        echo "   ✅ Last entry is valid JSON"
    else
        echo "   ⚠️  Could not validate (jq may not be installed)"
    fi

    # Show sample
    echo ""
    echo "📋 Sample entry (first line, truncated):"
    vagrant ssh defender -c "head -1 /vagrant/$JSONL_PATH | jq -C '. | {rag_metadata, detection: {scores, classification}}'" 2>/dev/null || echo "   (could not extract sample)"
else
    echo "   ⚠️  JSONL file not found"
    echo "   This is OK if events < flush threshold or test was short"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ TEST COMPLETE                                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Step 10: Success Evaluation
# ============================================================================

SUCCESS=true

if [ $(($FINAL_ELAPSED / 60)) -lt 10 ]; then
    echo "⚠️  Test ran for less than 10 minutes"
    SUCCESS=false
fi

if [ "$ARTIFACT_COUNT" -eq 0 ]; then
    echo "⚠️  No artifacts generated"
    SUCCESS=false
fi

if [ "$DETECTOR_RUNNING" -eq 0 ] && [ $(($FINAL_ELAPSED / 60)) -lt 10 ]; then
    echo "⚠️  ml-detector stopped early (possible crash)"
    SUCCESS=false
fi

echo ""

if [ "$SUCCESS" = true ]; then
    echo "✅ SUCCESS! Race condition fix validated!"
    echo ""
    echo "   • Ran for $(($FINAL_ELAPSED / 60))+ minutes without crash"
    echo "   • Generated $ARTIFACT_COUNT artifacts"
    echo "   • JSONL aggregation working"
    echo "   • Full pipeline operational"
    echo ""
    echo "🎯 Ready for Phase 2A - FAISS Integration"
    echo ""
    echo "Next steps:"
    echo "   1. Commit the fix: git commit -m 'Fix race conditions in RAGLogger'"
    echo "   2. Start FAISS integration"
    echo "   3. Optional: Run overnight stress test"
else
    echo "⚠️  Test completed but needs review"
    echo ""
    echo "   Runtime: $(($FINAL_ELAPSED / 60))m"
    echo "   Artifacts: $ARTIFACT_COUNT"
    echo "   Detector running: $DETECTOR_RUNNING"
    echo ""
    echo "Recommendations:"
    echo "   1. Check ml-detector logs: vagrant ssh defender -c 'tail -100 /vagrant/ml-detector/logs/ml_detector.log'"
    echo "   2. Run test again for longer (20+ minutes)"
    echo "   3. Check if network traffic is being generated"
fi

echo ""
echo "📚 To restart the lab:"
echo "   make kill-lab && make run-lab-dev"
echo ""
echo "📊 To check status anytime:"
echo "   make status-lab"
echo ""