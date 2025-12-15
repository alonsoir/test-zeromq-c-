#!/bin/bash
# tsan_investigation_inside_vm.sh - Run INSIDE the Vagrant VM
# Optimized for execution after "vagrant ssh defender"
# Day 16 - Race Condition Hunter

set -e

PROJECT_ROOT="/vagrant"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔬 ThreadSanitizer Race Condition Investigation           ║"
echo "║  Running INSIDE VM - No Vagrant Wrapper Needed             ║"
echo "║  Day 16 - December 15, 2025                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Step 0: Verify we're inside VM
# ============================================================================

echo "📋 Step 0: Environment check..."
echo ""

if [ ! -d "/vagrant" ]; then
    echo "❌ Error: Not in Vagrant VM (no /vagrant directory)"
    exit 1
fi

# Check we're NOT trying to run vagrant inside vagrant
if command -v vagrant &>/dev/null; then
    echo "⚠️  Warning: vagrant command available - you might be on host"
    echo "   This script should run INSIDE the VM after 'vagrant ssh'"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Running inside VM"
echo ""

# ============================================================================
# Step 1: Generate Protobuf (Direct, no vagrant)
# ============================================================================

echo "📝 Step 1: Generating protobuf files..."
echo ""

cd "$PROJECT_ROOT"

if [ -f "protos/network_security.proto" ]; then
    echo "   Found: protos/network_security.proto"

    # Generate directly
    protoc --cpp_out=include protos/network_security.proto

    if [ -f "include/network_security.pb.h" ]; then
        echo "✅ Protobuf generated:"
        ls -lh include/network_security.pb.*
    else
        echo "❌ Protobuf generation failed"
        exit 1
    fi
else
    echo "❌ Proto file not found: protos/network_security.proto"
    exit 1
fi

echo ""

# ============================================================================
# Step 2: Backup Current Build
# ============================================================================

echo "📦 Step 2: Backing up current build..."
echo ""

cd "$PROJECT_ROOT/ml-detector"

if [ -f "build/ml-detector" ]; then
    cp build/ml-detector "build/ml-detector.backup_${TIMESTAMP}"
    echo "✅ Backup: build/ml-detector.backup_${TIMESTAMP}"
else
    echo "ℹ️  No existing build to backup"
fi
echo ""

# ============================================================================
# Step 3: Compile with ThreadSanitizer
# ============================================================================

echo "🔨 Step 3: Compiling with ThreadSanitizer..."
echo ""

# Clean previous build
make clean 2>/dev/null || rm -rf build/*.o

# Set TSan flags
export CXXFLAGS="-fsanitize=thread -O1 -g -fno-omit-frame-pointer"
export LDFLAGS="-fsanitize=thread"

echo "   Compiler flags:"
echo "   CXXFLAGS=$CXXFLAGS"
echo "   LDFLAGS=$LDFLAGS"
echo ""

# Compile
if make; then
    echo ""
    echo "✅ Compilation successful!"
else
    echo ""
    echo "❌ Compilation failed - check errors above"
    exit 1
fi

echo ""
echo "📊 Binary info:"
ls -lh build/ml-detector
file build/ml-detector
echo ""

# Verify TSan
echo "🔍 Verifying ThreadSanitizer linkage:"
if ldd build/ml-detector | grep -i tsan; then
    echo "✅ TSan library linked"
else
    echo "⚠️  TSan library not visible in ldd (this is sometimes normal)"
    echo "   Will verify during execution"
fi

echo ""

# ============================================================================
# Step 4: Prepare Test Environment
# ============================================================================

echo "🧪 Step 4: Preparing test environment..."
echo ""

# Clean previous logs
TODAY=$(date +%Y-%m-%d)
rm -rf "$PROJECT_ROOT/logs/rag/artifacts/$TODAY" 2>/dev/null || true
rm -f "$PROJECT_ROOT/logs/rag/events/$TODAY.jsonl" 2>/dev/null || true

# Ensure directories exist
mkdir -p "$PROJECT_ROOT/logs/rag/artifacts"
mkdir -p "$PROJECT_ROOT/logs/rag/events"

echo "✅ Test environment ready"
echo ""

# ============================================================================
# Step 5: Run Test with ThreadSanitizer
# ============================================================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 STARTING THREADSANITIZER TEST                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

TSAN_LOG="$PROJECT_ROOT/tsan_report_${TIMESTAMP}.txt"
echo "📁 TSan report will be saved to:"
echo "   $TSAN_LOG"
echo ""

# Configure ThreadSanitizer
export TSAN_OPTIONS="log_path=${TSAN_LOG%.txt} second_deadlock_stack=1 history_size=7"

echo "⏱️  Test duration: 5 minutes (or until crash)"
echo "   You can press Ctrl+C to stop early"
echo ""
echo "Starting in 3 seconds..."
sleep 3
echo ""

# Run detector
START_TIME=$(date +%s)

./build/ml-detector config/ml_detector_config.json 2>&1 | tee -a "$TSAN_LOG" &
DETECTOR_PID=$!

echo "🔍 ml-detector PID: $DETECTOR_PID"
echo ""

# Monitor for 5 minutes or until crash
MAX_RUNTIME=300  # 5 minutes
ELAPSED=0
LAST_CHECK=0

while kill -0 $DETECTOR_PID 2>/dev/null; do
    sleep 1
    ELAPSED=$(($(date +%s) - START_TIME))

    # Update progress every second
    printf "\r⏱️  Runtime: %02d:%02d / 05:00  " $((ELAPSED/60)) $((ELAPSED%60))

    # Check for races every 10 seconds
    if [ $((ELAPSED % 10)) -eq 0 ] && [ $ELAPSED -ne $LAST_CHECK ]; then
        RACE_COUNT=$(find "$PROJECT_ROOT" -name "tsan_report_${TIMESTAMP}.*" -exec grep -c "WARNING: ThreadSanitizer" {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
        printf "| Races: %d" "$RACE_COUNT"
        LAST_CHECK=$ELAPSED
    fi

    # Stop after 5 minutes
    if [ $ELAPSED -ge $MAX_RUNTIME ]; then
        echo ""
        echo ""
        echo "⏰ Time limit reached (5 minutes) - stopping test"
        kill $DETECTOR_PID 2>/dev/null || true
        sleep 2
        break
    fi
done

# Wait for process to fully terminate
wait $DETECTOR_PID 2>/dev/null || CRASH_CODE=$?

FINAL_ELAPSED=$(($(date +%s) - START_TIME))

echo ""
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  📊 TEST COMPLETE                                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Step 6: Analyze Results
# ============================================================================

echo "📈 Step 6: Analyzing results..."
echo ""
echo "⏱️  Total runtime: $(($FINAL_ELAPSED / 60))m $(($FINAL_ELAPSED % 60))s"
echo ""

# Find all TSan report files
REPORT_FILES=$(find "$PROJECT_ROOT" -name "tsan_report_${TIMESTAMP}.*" 2>/dev/null)

if [ -z "$REPORT_FILES" ]; then
    echo "⚠️  No TSan report files found"
    echo ""
    echo "   Possible reasons:"
    echo "   • Detector crashed before TSan could write"
    echo "   • No race conditions triggered"
    echo "   • Test was too short"
    echo ""
    echo "   Check if detector created any output:"
    echo "   • Artifacts: ls -lh logs/rag/artifacts/$TODAY/ | wc -l"
    echo "   • Main log: tail logs/rag/events/$TODAY.jsonl"
else
    # Count total races
    TOTAL_RACES=0
    for file in $REPORT_FILES; do
        COUNT=$(grep -c "WARNING: ThreadSanitizer" "$file" 2>/dev/null || echo "0")
        TOTAL_RACES=$((TOTAL_RACES + COUNT))
    done

    if [ "$TOTAL_RACES" -gt 0 ]; then
        echo "🚨 ThreadSanitizer detected $TOTAL_RACES race condition(s)!"
        echo ""
        echo "📄 Report files:"
        for file in $REPORT_FILES; do
            echo "   • $file"
        done
        echo ""
        echo "🔍 Quick preview (first race):"
        echo "══════════════════════════════════════════════════════════"
        echo "$REPORT_FILES" | head -1 | xargs grep -A 30 "WARNING: ThreadSanitizer" 2>/dev/null | head -40 || echo "Could not extract preview"
        echo "══════════════════════════════════════════════════════════"
        echo ""
        echo "✅ SUCCESS! Race conditions captured!"
        echo ""
        echo "📋 Expected races (verify in full report):"
        echo "   1. current_date_ (std::string) - read/write race"
        echo "   2. current_log_ (std::ofstream) - close/write race"
        echo "   3. check_rotation() called without lock"
        echo ""
        echo "🔧 Next steps:"
        echo "   1. Review full report: cat $TSAN_LOG"
        echo "   2. Confirm races match predictions"
        echo "   3. Apply the fix to rag_logger.cpp/hpp"
        echo "   4. Recompile and retest"
    else
        echo "✅ No race conditions detected"
        echo ""
        echo "   This could mean:"
        echo "   • Race conditions didn't trigger (timing dependent)"
        echo "   • Test was too short"
        echo "   • Detector crashed before reporting"
        echo ""
        echo "   Recommendation: Try running again or for longer duration"
    fi
fi

# Check artifacts generated
ARTIFACT_COUNT=$(ls -1 "$PROJECT_ROOT/logs/rag/artifacts/$TODAY"/ 2>/dev/null | wc -l)
echo ""
echo "📁 Events captured: $ARTIFACT_COUNT artifacts"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🎯 INVESTIGATION COMPLETE                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📚 For detailed interpretation:"
echo "   cat TSAN_INTERPRETATION_GUIDE.md"
echo ""
echo "✅ Ready to apply fix!"