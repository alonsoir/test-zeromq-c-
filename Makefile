.PHONY: help status
.PHONY: up halt destroy ssh
.PHONY: lab-start lab-stop lab-restart lab-ps lab-logs lab-clean
.PHONY: proto proto-unified proto-verify sniffer detector firewall all rebuild
.PHONY: sniffer-build sniffer-clean sniffer-package sniffer-install
.PHONY: detector-build detector-clean
.PHONY: firewall-build firewall-clean
.PHONY: run-sniffer run-detector run-firewall
.PHONY: logs-sniffer logs-detector logs-firewall logs-lab
.PHONY: run-lab-dev kill-lab status-lab
.PHONY: run-lab-dev-day23 kill-lab-day23 status-lab-day23
.PHONY: kill-all check-ports restart
.PHONY: clean distclean test dev-setup schema-update
.PHONY: build-unified rebuild-unified create-verify-script quick-fix dev-setup-unified
.PHONY: check-libbpf verify-bpf-maps diagnose-bpf  # NUEVO
.PHONY: test-replay-small test-replay-neris test-replay-big
.PHONY: monitor-day13-tmux logs-dual-score logs-dual-score-live extract-dual-scores
.PHONY: test-integration-day13 test-integration-day13-tmux test-dual-score-quick
.PHONY: clean-day13-logs stats-dual-score
.PHONY: analyze-dual-scores test-analyze-workflow test-day13-full quick-analyze
.PHONY: rag-log-init rag-log-clean rag-log-status rag-log-analyze rag-log-tail rag-log-tail-live
.PHONY: test-rag-integration test-rag-quick rag-watch rag-validate test-rag-small test-rag-neris test-rag-big
.PHONY: quick-lab-test rag-consolidate detector-debug
.PHONY: etcd-client-build etcd-client-clean
.PHONY: verify-etcd-linkage verify-encryption verify-pipeline-config
.PHONY: monitor-day23-tmux
.PHONY: test-day23-full test-day23-stress verify-rag-logs-day23
.PHONY: day23
.PHONY: rag-ingester rag-ingester-build rag-ingester-clean

# ============================================================================
# ThreadSanitizer Build (Race Condition Detection)
# ============================================================================
# TSan requires specific compiler flags
TSAN_CXXFLAGS = -fsanitize=thread -O1 -g -fno-omit-frame-pointer
TSAN_LDFLAGS = -fsanitize=thread

.PHONY: detector-tsan
detector-tsan: CXXFLAGS += $(TSAN_CXXFLAGS)
detector-tsan: LDFLAGS += $(TSAN_LDFLAGS)
detector-tsan: clean ml-detector
	@echo "✅ ml-detector compiled with ThreadSanitizer"
	@echo "   Run with: TSAN_OPTIONS='log_path=tsan.log' ./build/ml-detector config.json"

# Quick test with TSan (5 minute run)
.PHONY: test-tsan
test-tsan: detector-tsan
	@echo "🔬 Running ThreadSanitizer test (5 minutes)..."
	@TSAN_OPTIONS="log_path=tsan_$(shell date +%Y%m%d_%H%M%S).txt" \
		timeout 300 ./build/ml-detector config/ml_detector_config.json || true
	@echo "✅ Test complete. Check tsan_*.txt for race reports."

# ============================================================================
# AddressSanitizer Build (Memory Leak Detection) - Day 30
# ============================================================================
ASAN_CXXFLAGS = -fsanitize=address -g -O1 -fno-omit-frame-pointer
ASAN_LDFLAGS = -fsanitize=address

.PHONY: detector-asan
detector-asan: proto etcd-client-build
	@echo "Building ml-detector with AddressSanitizer..."
	@echo "   Dependencies: proto + etcd-client"
	@vagrant ssh -c "mkdir -p /vagrant/ml-detector/build-asan/proto && \
		cp /vagrant/protobuf/network_security.pb.* /vagrant/ml-detector/build-asan/proto/ && \
		cd /vagrant/ml-detector/build-asan && \
		cmake -DCMAKE_CXX_FLAGS='-fsanitize=address -g -O1 -fno-omit-frame-pointer' \
		      -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address' \
		      -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && \
		make -j4"
	@echo "ml-detector (ASAN) built successfully"
	@echo "   Binary: /vagrant/ml-detector/build-asan/ml-detector"

.PHONY: run-detector-asan
run-detector-asan: detector-asan
	@echo "🔬 Running ml-detector with ASAN (leak detection enabled)..."
	@echo "⚠️  Requires: etcd-server + sniffer running"
	@echo ""
	@vagrant ssh -c "mkdir -p /tmp/asan-logs && \
		cd /vagrant/ml-detector/build-asan && \
		ASAN_OPTIONS='log_path=/tmp/asan-logs/asan_ml_detector.log:detect_leaks=1:leak_check_at_exit=1' \
		./ml-detector --config ../config/detector.json 2>&1 | tee /tmp/asan-logs/ml_detector_asan_output.txt"

.PHONY: monitor-asan-memory
monitor-asan-memory:
	@echo "📊 Monitoring ml-detector memory (ASAN build)..."
	@echo "   Sampling every 5 minutes for 1 hour (12 samples)"
	@echo "   Press Ctrl+C to stop early"
	@echo ""
	@vagrant ssh -c "mkdir -p /tmp/asan-logs && \
		for i in \$$(seq 1 12); do \
			if pgrep -f 'ml-detector.*config/detector.json' > /dev/null; then \
				MEM=\$$(ps -p \$$(pgrep -f 'ml-detector.*config/detector.json') -o rss= 2>/dev/null | awk '{print \$$1/1024}'); \
				echo \"\$$(date +%H:%M:%S) - Memory: \$${MEM} MB\" | tee -a /tmp/asan-logs/asan_memory_track.log; \
			else \
				echo \"\$$(date +%H:%M:%S) - ml-detector not running\" | tee -a /tmp/asan-logs/asan_memory_track.log; \
			fi; \
			sleep 300; \
		done"

.PHONY: analyze-asan-results
analyze-asan-results:
	@echo "════════════════════════════════════════════════════════════"
	@echo "🔍 ASAN LEAK REPORT ANALYSIS"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "=== LEAK SUMMARY ==="
	@vagrant ssh -c "if [ -f /tmp/asan-logs/ml_detector_asan_output.txt ]; then \
		grep -A 30 'LeakSanitizer' /tmp/asan-logs/ml_detector_asan_output.txt 2>/dev/null || echo 'No LeakSanitizer report found (process may still be running)'; \
	else \
		echo '❌ No ASAN output file found at /tmp/asan-logs/ml_detector_asan_output.txt'; \
	fi"
	@echo ""
	@echo "=== DIRECT LEAKS ==="
	@vagrant ssh -c "if [ -f /tmp/asan-logs/ml_detector_asan_output.txt ]; then \
		grep -A 30 'Direct leak' /tmp/asan-logs/ml_detector_asan_output.txt 2>/dev/null || echo 'No direct leaks found ✅'; \
	fi"
	@echo ""
	@echo "=== MEMORY GROWTH TRACKING ==="
	@vagrant ssh -c "if [ -f /tmp/asan-logs/asan_memory_track.log ]; then \
		cat /tmp/asan-logs/asan_memory_track.log; \
		echo ''; \
		echo 'Memory analysis:'; \
		FIRST=\$$(head -1 /tmp/asan-logs/asan_memory_track.log | awk '{print \$$4}'); \
		LAST=\$$(tail -1 /tmp/asan-logs/asan_memory_track.log | awk '{print \$$4}'); \
		if [ -n \"\$$FIRST\" ] && [ -n \"\$$LAST\" ]; then \
			GROWTH=\$$(echo \"\$$LAST - \$$FIRST\" | bc 2>/dev/null); \
			echo \"  Start: \$$FIRST MB\"; \
			echo \"  End: \$$LAST MB\"; \
			echo \"  Growth: \$$GROWTH MB\"; \
		fi; \
	else \
		echo '⚠️  No memory tracking data yet - run: make monitor-asan-memory'; \
	fi"
	@echo ""
	@echo "=== ASAN LOG FILES ==="
	@vagrant ssh -c "ls -lh /tmp/asan-logs/asan_ml_detector.log* 2>/dev/null || echo 'No ASAN log files generated yet'"
	@echo ""
	@echo "════════════════════════════════════════════════════════════"

.PHONY: clean-asan
clean-asan:
	@echo "🧹 Cleaning ASAN build and logs..."
	@vagrant ssh -c "rm -rf /vagrant/ml-detector/build-asan"
	@vagrant ssh -c "rm -rf /tmp/asan-logs"
	@echo "✅ ASAN artifacts cleaned"

# Quick ASAN test (30 min run)
.PHONY: test-asan-quick
test-asan-quick: detector-asan
	@echo "🔬 Running quick ASAN test (30 minutes)..."
	@echo ""
	@echo "Step 1: Ensure other components are running..."
	@vagrant ssh -c "pgrep -f etcd-server > /dev/null || (echo '❌ etcd-server not running. Start with: make etcd-server-start' && exit 1)"
	@vagrant ssh -c "pgrep -f sniffer > /dev/null || (echo '⚠️  Warning: sniffer not running' && sleep 2)"
	@echo ""
	@echo "Step 2: Starting ml-detector with ASAN (background)..."
	@vagrant ssh -c "mkdir -p /tmp/asan-logs && \
		cd /vagrant/ml-detector/build-asan && \
		ASAN_OPTIONS='log_path=/tmp/asan-logs/asan_ml_detector.log:detect_leaks=1:leak_check_at_exit=1' \
		nohup ./ml-detector --config ../config/detector.json > /tmp/asan-logs/ml_detector_asan_output.txt 2>&1 &"
	@sleep 3
	@vagrant ssh -c "pgrep -f 'ml-detector.*detector.json' && echo '✅ ml-detector started' || echo '❌ Failed to start'"
	@echo ""
	@echo "Step 3: Monitoring memory for 30 minutes (6 samples, 5 min each)..."
	@vagrant ssh -c "mkdir -p /tmp/asan-logs && \
		for i in 1 2 3 4 5 6; do \
			if pgrep -f 'ml-detector.*detector.json' > /dev/null; then \
				MEM=\$$(ps -p \$$(pgrep -f 'ml-detector.*detector.json') -o rss= 2>/dev/null | awk '{print \$$1/1024}'); \
				echo \"\$$(date +%H:%M:%S) - Sample \$$i/6 - Memory: \$${MEM} MB\" | tee -a /tmp/asan-logs/asan_memory_track.log; \
			else \
				echo \"\$$(date +%H:%M:%S) - Sample \$$i/6 - ml-detector crashed!\" | tee -a /tmp/asan-logs/asan_memory_track.log; \
				break; \
			fi; \
			sleep 300; \
		done"
	@echo ""
	@echo "Step 4: Stopping ml-detector..."
	@vagrant ssh -c "pkill -f 'ml-detector.*detector.json' && sleep 2"
	@echo ""
	@echo "Step 5: Analyzing results..."
	@$(MAKE) analyze-asan-results
	@echo ""
	@echo "✅ Quick ASAN test complete"

# ============================================================================
# ML Defender Pipeline - Host Makefile
# Run from macOS - Commands execute in VM via vagrant ssh -c
# ============================================================================

help:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ML Defender Pipeline - Development Makefile               ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "VM Management:"
	@echo "  make up              - Start VM"
	@echo "  make halt            - Stop VM"
	@echo "  make destroy         - Destroy VM"
	@echo "  make ssh             - SSH into VM"
	@echo "  make status          - Show VM status + libbpf version"
	@echo ""
	@echo "Docker Lab:"
	@echo "  make lab-start       - Start docker-compose lab"
	@echo "  make lab-stop        - Stop docker-compose lab"
	@echo "  make lab-ps          - Show lab containers"
	@echo "  make lab-logs        - Show lab logs"
	@echo "  make lab-clean       - Stop and remove lab"
	@echo ""
	@echo "Build:"
	@echo "  make all             - Build sniffer + detector + firewall"
	@echo "  make proto           - Regenerate protobuf schema (unified)"
	@echo "  make proto-unified   - Protobuf unified system"
	@echo "  make proto-verify    - Verify protobuf consistency"
	@echo "  make sniffer         - Build sniffer"
	@echo "  make detector        - Build ml-detector"
	@echo "  make firewall        - Build firewall-acl-agent"
	@echo "  make rebuild         - Clean + build all (unified)"
	@echo "  make build-unified   - Build with unified protobuf"
	@echo "  make rebuild-unified - Clean + unified build"
	@echo "  make rag-ingester    - Build RAG Ingester"
	@echo ""
	@echo "Sniffer Packaging:"
	@echo "  make sniffer-package - Create .deb package"
	@echo "  make sniffer-install - Install .deb in VM"
	@echo ""
	@echo "Run Components (individual):"
	@echo "  make run-firewall    - Run firewall (Terminal 1)"
	@echo "  make run-detector    - Run detector (Terminal 2)"
	@echo "  make run-sniffer     - Run sniffer (Terminal 3)"
	@echo ""
	@echo "Run Lab (integrated):"
	@echo "  make run-lab-dev     - 🚀 START FULL LAB (background + monitor)"
	@echo "  make kill-lab        - Stop full lab"
	@echo "  make status-lab      - Show lab status"
	@echo "  make logs-lab        - Combined logs (all 3 components)"
	@echo ""
	@echo "Logs (individual):"
	@echo "  make logs-firewall   - Show firewall logs"
	@echo "  make logs-detector   - Show detector logs"
	@echo "  make logs-sniffer    - Show sniffer logs"
	@echo ""
	@echo "Development:"
	@echo "  make dev-setup       - Full setup (up + lab + build)"
	@echo "  make dev-setup-unified - Setup with unified protobuf"
	@echo "  make test            - Check what's built"
	@echo "  make schema-update   - Update schema + rebuild"
	@echo "  make quick-fix       - Quick bug fix procedure"
	@echo ""
	@echo "Troubleshooting:"
	@echo "  make kill-all        - Kill all processes"
	@echo "  make check-ports     - Check if ports are in use"
	@echo "  make check-libbpf    - 🔥 Verify libbpf >= 1.2.0 (Day 8 fix)"
	@echo "  make verify-bpf-maps - 🔍 Verify BPF maps load correctly"
	@echo "  make diagnose-bpf    - 🔧 Full BPF diagnostics"
	@echo "  make clean           - Clean build artifacts"
	@echo ""
	@echo "Memory Debugging:"
	@echo "  make detector-asan       - Build ml-detector with AddressSanitizer"

# ============================================================================
# VM Management
# ============================================================================

up:
	@vagrant up

halt:
	@vagrant halt

destroy:
	@vagrant destroy -f

ssh:
	@vagrant ssh

status:
	@echo "════════════════════════════════════════════════════════════"
	@echo "VM Status:"
	@vagrant status
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "libbpf Status (Day 8 Fix):"
	@vagrant ssh -c "pkg-config --modversion libbpf 2>/dev/null || echo '❌ libbpf not found'" | \
		awk '{if ($$1 >= "1.2.0") print "✅ libbpf " $$1 " (BPF map bug FIXED)"; else print "⚠️  libbpf " $$1 " (needs upgrade to 1.2.0+)"}'
	@echo "════════════════════════════════════════════════════════════"

# ============================================================================
# Docker Lab
# ============================================================================

lab-start:
	@echo "🚀 Starting Docker Lab..."
	@vagrant ssh -c "cd /vagrant && docker-compose up -d"
	@make lab-ps

lab-stop:
	@echo "⏸️  Stopping Docker Lab..."
	@vagrant ssh -c "cd /vagrant && docker-compose stop"

lab-restart:
	@vagrant ssh -c "cd /vagrant && docker-compose restart"

lab-ps:
	@echo "📦 Lab Containers:"
	@vagrant ssh -c "cd /vagrant && docker-compose ps"

lab-logs:
	@echo "📋 Lab Logs:"
	@vagrant ssh -c "cd /vagrant && docker-compose logs --tail=50 -f"

lab-clean:
	@echo "🧹 Cleaning Docker Lab..."
	@vagrant ssh -c "cd /vagrant && docker-compose down -v"

# ----------------------------------------------------------------------------
# 1. Protobuf (base de todo)
# ----------------------------------------------------------------------------

PROTOBUF_VERIFY_SCRIPT := /vagrant/scripts/verify_protobuf.sh

proto-unified:
	@echo "🔨 Protobuf Unified System..."
	@vagrant ssh -c "cd /vagrant/protobuf && chmod +x generate.sh && ./generate.sh"

proto-verify:
	@echo "🔍 Verificando consistencia protobuf..."
	@vagrant ssh -c "cd /vagrant && bash scripts/verify_protobuf.sh"

proto: proto-unified
	@echo "✅ Protobuf unificado generado y distribuido"

# ============================================================================
# Build Targets - CORRECTED DEPENDENCY ORDER (Day 23)
# ============================================================================
# Orden de compilación:
#   1. proto-unified (protobuf)
#   2. etcd-client-build (librería compartida)
#   3. sniffer/detector/firewall (dependen de proto + etcd-client)
#   4. etcd-server-build (independiente)
# ============================================================================
# ----------------------------------------------------------------------------
# 3. Componentes (dependen de proto + etcd-client)
# ----------------------------------------------------------------------------
sniffer: proto etcd-client-build
	@echo "🔨 Building Sniffer..."
	@echo "   Dependencies: proto ✅  etcd-client ✅"
	@vagrant ssh -c "cd /vagrant/sniffer && make"

# ----------------------------------------------------------------------------
# 2a. crypto-transport (librería base - NUEVA)
# ----------------------------------------------------------------------------

crypto-transport-build:
	@echo "🔨 Building crypto-transport library..."
	@vagrant ssh -c "cd /vagrant/crypto-transport && \
		rm -rf build && \
		mkdir -p build && \
		cd build && \
		cmake .. && \
		make -j4"
	@echo "✅ crypto-transport library built"
	@echo "Installing system-wide..."
	@vagrant ssh -c "cd /vagrant/crypto-transport/build && sudo make install && sudo ldconfig"
	@echo "✅ crypto-transport installed to /usr/local/lib"
	@echo "Verifying library..."
	@vagrant ssh -c "sudo ldconfig -p | grep crypto_transport || ls -lh /usr/local/lib/libcrypto_transport.so*"

crypto-transport-clean:
	@echo "🧹 Cleaning crypto-transport..."
	@vagrant ssh -c "rm -rf /vagrant/crypto-transport/build"
	@vagrant ssh -c "sudo rm -f /usr/local/lib/libcrypto_transport.so*"
	@vagrant ssh -c "sudo rm -rf /usr/local/include/crypto_transport"
	@echo "✅ crypto-transport cleaned"

crypto-transport-test:
	@echo "🧪 Testing crypto-transport..."
	@vagrant ssh -c "cd /vagrant/crypto-transport/build && ctest --output-on-failure"

# ----------------------------------------------------------------------------
# 2b. etcd-client (depende de crypto-transport)
# ----------------------------------------------------------------------------

sniffer-build: sniffer

sniffer-clean:
	@echo "🧹 Cleaning Sniffer..."
	@vagrant ssh -c "cd /vagrant/sniffer && make clean"

sniffer-package:
	@echo "📦 Creating Sniffer .deb package..."
	@vagrant ssh -c "cd /vagrant/sniffer && make && ./scripts/create_deb.sh"
	@vagrant ssh -c "ls -lh /vagrant/sniffer/*.deb"

sniffer-install: sniffer-package
	@echo "📥 Installing Sniffer .deb..."
	@vagrant ssh -c "cd /vagrant/sniffer && sudo dpkg -i *.deb || sudo apt-get install -f -y"

detector: proto etcd-client-build
	@echo "🔨 Building ML Detector..."
	@echo "   Dependencies: proto ✅  etcd-client ✅"
	@vagrant ssh -c "mkdir -p /vagrant/ml-detector/build/proto && \
		cp /vagrant/protobuf/network_security.pb.* /vagrant/ml-detector/build/proto/ && \
		cd /vagrant/ml-detector/build && \
		cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS='-g -O0 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer' .. && \
		make clean && make -j4"

detector-debug: proto etcd-client-build
	@echo "🔨 Building ML Detector (DEBUG + SANITIZERS)..."
	@echo "   Dependencies: proto ✅  etcd-client ✅"
	@vagrant ssh -c "mkdir -p /vagrant/ml-detector/build/proto && \
		cp /vagrant/protobuf/network_security.pb.* /vagrant/ml-detector/build/proto/ && \
		cd /vagrant/ml-detector/build && \
		cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS='-g -O0 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer' .. && \
		make clean && make -j4"

detector-production: proto etcd-client-build
	@echo "🔨 Building ML Detector (PRODUCTION - Optimized)..."
	@echo "   Dependencies: proto ✅  etcd-client ✅"
	@vagrant ssh -c "mkdir -p /vagrant/ml-detector/build/proto && \
		cp /vagrant/protobuf/network_security.pb.* /vagrant/ml-detector/build/proto/ && \
		cd /vagrant/ml-detector/build && \
		cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-O3 -march=native -DNDEBUG' .. && \
		make -j4"
	@echo "⚠️  Production build requires hardware-specific tuning"

detector-build: detector

detector-clean:
	@echo "🧹 Cleaning ML Detector..."
	@vagrant ssh -c "rm -rf /vagrant/ml-detector/build/*"
firewall: proto etcd-client-build
	@echo "🔨 Building Firewall ACL Agent..."
	@echo "   Dependencies: proto ✅  etcd-client ✅"
	@vagrant ssh -c "mkdir -p /vagrant/firewall-acl-agent/build && \
		cd /vagrant/firewall-acl-agent/build && \
		cmake .. && make -j4"

firewall-build: firewall

firewall-clean:
	@echo "🧹 Cleaning Firewall ACL Agent..."
	@vagrant ssh -c "rm -rf /vagrant/firewall-acl-agent/build/*"

# ============================================================================
# RAG Ingester (Day 36)
# ============================================================================

rag-ingester: proto etcd-client-build crypto-transport-build
	@echo "🔨 Building RAG Ingester..."
	@echo "   Dependencies: proto ✅  etcd-client ✅  crypto-transport ✅"
	@vagrant ssh -c "mkdir -p /vagrant/rag-ingester/build/proto && \
		cp /vagrant/protobuf/network_security.pb.* /vagrant/rag-ingester/build/proto/ && \
		cd /vagrant/rag-ingester/build && \
		cmake -DCMAKE_BUILD_TYPE=Debug .. && \
		make -j4"

rag-ingester-build: rag-ingester

rag-ingester-clean:
	@echo "🧹 Cleaning RAG Ingester..."
	@vagrant ssh -c "rm -rf /vagrant/rag-ingester/build/*"

# ----------------------------------------------------------------------------
# 4. etcd-server (independiente)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 5. Build unificado (ORDEN CORRECTO)
# ----------------------------------------------------------------------------
build-unified: proto-unified etcd-client-build crypto-transport-build sniffer detector firewall rag-ingester
	@echo "🚀 Build completo con protobuf unificado y etcd-client"
	@$(MAKE) proto-verify
	@echo ""
	@echo "✅ Build order executed correctly:"
	@echo "   1. ✅ proto-unified"
	@echo "   2. ✅ etcd-client-build"
	@echo "   3. ✅ crypto-transport-build"
	@echo "   4. ✅ sniffer"
	@echo "   5. ✅ detector"
	@echo "   5. ✅ rag-ingester"

all: build-unified etcd-server-build
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ ALL COMPONENTS BUILT (Day 23 - Correct Order)         ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build Summary:"
	@echo "  ✅ Protobuf unified"
	@echo "  ✅ etcd-client library"
	@echo "  ✅ Sniffer"
	@echo "  ✅ ML Detector"
	@echo "  ✅ Firewall ACL Agent"
	@echo "  ✅ etcd-server"
	@echo ""
	@echo "Next step: make verify-pipeline-config"

rebuild-unified: clean build-unified
	@echo "✅ Rebuild completo con protobuf unificado y etcd-client"

rebuild: rebuild-unified etcd-server-build
	@echo "✅ Full rebuild complete (all components)"

clean: sniffer-clean detector-clean firewall-clean rag-ingester-clean crypto-transport-clean etcd-client-clean etcd-server-clean
	@echo "✅ Clean complete (including crypto ecosystem)"

distclean: clean
	@vagrant ssh -c "rm -f /vagrant/protobuf/network_security.pb.* /vagrant/protobuf/network_security_pb2.py"


# ============================================================================
# BPF Diagnostics (Day 8 Fix Verification) - NUEVO
# ============================================================================

check-libbpf:
	@echo "════════════════════════════════════════════════════════════"
	@echo "🔍 Checking libbpf installation (Day 8 Fix)"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "1️⃣  libbpf version:"
	@vagrant ssh -c "pkg-config --modversion libbpf 2>/dev/null || echo '❌ libbpf not found'"
	@echo ""
	@echo "2️⃣  libbpf CFLAGS:"
	@vagrant ssh -c "pkg-config --cflags libbpf 2>/dev/null || echo '❌ pkg-config failed'"
	@echo ""
	@echo "3️⃣  libbpf LDFLAGS:"
	@vagrant ssh -c "pkg-config --libs libbpf 2>/dev/null || echo '❌ pkg-config failed'"
	@echo ""
	@echo "4️⃣  libbpf library files:"
	@vagrant ssh -c "ls -lh /usr/lib64/libbpf.* 2>/dev/null | head -3 || ls -lh /usr/local/lib/libbpf.* 2>/dev/null | head -3 || echo '❌ Libraries not found'"
	@echo ""
	@echo "5️⃣  Verification:"
	@vagrant ssh -c "LIBBPF_VER=\$$(pkg-config --modversion libbpf 2>/dev/null); \
		if [ -z \"\$$LIBBPF_VER\" ]; then \
			echo '❌ libbpf NOT installed - run: vagrant provision'; \
		elif [ \"\$$(printf '%s\n' '1.2.0' \"\$$LIBBPF_VER\" | sort -V | head -n1)\" = '1.2.0' ]; then \
			echo \"✅ libbpf \$$LIBBPF_VER >= 1.2.0 (BPF map bug FIXED)\"; \
		else \
			echo \"⚠️  libbpf \$$LIBBPF_VER < 1.2.0 (BUG PRESENT - run: vagrant provision)\"; \
		fi"
	@echo "════════════════════════════════════════════════════════════"

verify-bpf-maps:
	@echo "════════════════════════════════════════════════════════════"
	@echo "🔍 Verifying BPF Maps Loading (Day 8 interface_configs)"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "1️⃣  Compiling sniffer..."
	@vagrant ssh -c "cd /vagrant/sniffer && make clean && make" > /dev/null 2>&1 && echo "   ✅ Compiled successfully" || echo "   ❌ Compilation failed"
	@echo ""
	@echo "2️⃣  Checking BPF object file:"
	@vagrant ssh -c "ls -lh /vagrant/sniffer/build/sniffer.bpf.o 2>/dev/null || echo '   ❌ BPF object not found'"
	@echo ""
	@echo "3️⃣  Searching for interface_configs in object:"
	@vagrant ssh -c "llvm-objdump -h /vagrant/sniffer/build/sniffer.bpf.o 2>/dev/null | grep -i maps && echo '   ✅ .maps section found' || echo '   ❌ .maps section not found'"
	@echo ""
	@echo "4️⃣  Checking BTF for interface_config type:"
	@vagrant ssh -c "bpftool btf dump file /vagrant/sniffer/build/sniffer.bpf.o 2>/dev/null | grep -A 5 'interface_config' | head -10 || echo '   ⚠️  interface_config not in BTF'"
	@echo ""
	@echo "5️⃣  Testing map load (requires root):"
	@vagrant ssh -c "cd /vagrant/sniffer/build && sudo timeout 5s ./sniffer --test-load 2>&1 | grep -E 'interface_configs|map.*load' || echo '   ℹ️  Run sniffer to test map loading'"
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "💡 TIP: If maps don't load, verify libbpf >= 1.2.0"
	@echo "    Run: make check-libbpf"
	@echo "════════════════════════════════════════════════════════════"

diagnose-bpf: check-libbpf verify-bpf-maps
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "🔧 BPF DIAGNOSTICS COMPLETE"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "If interface_configs map still fails to load:"
	@echo "  1. Verify libbpf >= 1.2.0: make check-libbpf"
	@echo "  2. Rebuild from scratch: make rebuild"
	@echo "  3. Check kernel compatibility: vagrant ssh -c 'uname -r'"
	@echo "  4. Enable debug: vagrant ssh -c 'cd /vagrant/sniffer && make DEBUG=1'"
	@echo ""

# ============================================================================
# Run Individual Components
# ============================================================================

run-firewall:
	@echo "🔥 Running Firewall ACL Agent..."
	@echo "⚠️  Requires: Detector running on tcp://localhost:5572"
	@vagrant ssh -c "cd /vagrant/firewall-acl-agent/build && sudo ./firewall-acl-agent -c ../config/firewall.json"

run-detector:
	@echo "🤖 Running ML Detector..."
	@echo "⚠️  Requires: Sniffer running on tcp://127.0.0.1:5571"
	@vagrant ssh -c "cd /vagrant/ml-detector/build && ./ml-detector -c config/ml_detector_config.json"

run-sniffer:
	@echo "📡 Running Sniffer..."
	@vagrant ssh -c "cd /vagrant/sniffer/build && sudo ./sniffer -c config/sniffer.json"

# ============================================================================
# Run Full Lab (Development Mode)
# ============================================================================

run-lab-dev-day23:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Starting ML Defender Lab - Day 23 (with etcd-server)  ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📋 Execution Order:"
	@echo "   0️⃣  etcd-server         (Config + Heartbeat supervisor)"
	@echo "   1️⃣  Firewall ACL Agent  (SUB tcp://localhost:5572)"
	@echo "   2️⃣  ML Detector         (PUB tcp://0.0.0.0:5572)"
	@echo "   3️⃣  Sniffer             (PUSH tcp://127.0.0.1:5571)"
	@echo ""
	@echo "Step 0: Starting etcd-server..."
	@$(MAKE) etcd-server-start
	@sleep 3
	@echo ""
	@echo "Step 1-3: Starting pipeline components..."
	@vagrant ssh -c "cd /vagrant && bash scripts/run_lab_dev.sh"
	@echo ""
	@echo "✅ Lab started with etcd ecosystem"


kill-lab-day23:
	@echo "💀 Stopping ML Defender Lab (including etcd-server)..."
	@echo ""
	@echo "Checking processes..."
	@vagrant ssh -c "pgrep -a -f etcd-server || echo '  etcd-server: ❌ Not running'"
	@vagrant ssh -c "pgrep -a -f firewall-acl-agent || echo '  Firewall: ❌ Not running'"
	@vagrant ssh -c "pgrep -a -f ml-detector || echo '  Detector: ❌ Not running'"
	@vagrant ssh -c "pgrep -a -f sniffer || echo '  Sniffer:  ❌ Not running'"
	@echo ""
	@echo "Killing processes..."
	-@vagrant ssh -c "pkill -9 -f etcd-server" 2>/dev/null || echo "  etcd-server already stopped"
	-@vagrant ssh -c "sudo pkill -9 -f firewall-acl-agent" 2>/dev/null || echo "  Firewall already stopped"
	-@vagrant ssh -c "pkill -9 -f ml-detector" 2>/dev/null || echo "  Detector already stopped"
	-@vagrant ssh -c "sudo pkill -9 -f sniffer" 2>/dev/null || echo "  Sniffer already stopped"
	@sleep 2
	@echo ""
	@echo "Verifying cleanup..."
	@vagrant ssh -c "pgrep -a -f 'etcd-server|firewall-acl-agent|ml-detector|sniffer' || echo '✅ All processes stopped'"

status-lab-day23:
	@echo "════════════════════════════════════════════════════════════"
	@echo "ML Defender Lab Status (Day 23 - with etcd-server):"
	@echo "════════════════════════════════════════════════════════════"
	@vagrant ssh -c "pgrep -a -f etcd-server && echo '✅ etcd-server: RUNNING' || echo '❌ etcd-server: STOPPED'"
	@vagrant ssh -c "pgrep -a -f firewall-acl-agent && echo '✅ Firewall: RUNNING' || echo '❌ Firewall: STOPPED'"
	@vagrant ssh -c "pgrep -a -f ml-detector && echo '✅ Detector: RUNNING' || echo '❌ Detector: STOPPED'"
	@vagrant ssh -c "pgrep -a -f 'sniffer.*-c' && echo '✅ Sniffer:  RUNNING' || echo '❌ Sniffer:  STOPPED'"
	@echo "════════════════════════════════════════════════════════════"
	@echo "Heartbeat status:"
	@vagrant ssh -c "ETCDCTL_API=3 etcdctl get --prefix '/components/' 2>/dev/null | grep -c heartbeat | xargs echo 'Active heartbeats:' || echo '⚠️  No heartbeats registered'"
	@echo "════════════════════════════════════════════════════════════"

check-ports:
	@vagrant ssh -c "sudo ss -tlnp | grep -E '5571|5572' && echo '⚠️  Ports in use' || echo '✅ Ports free'"

# ============================================================================
# Logs
# ============================================================================

logs-firewall:
	@vagrant ssh -c "tail -f /vagrant/firewall-acl-agent/build/logs/*.log 2>/dev/null || echo 'No firewall logs yet'"

logs-detector:
	@vagrant ssh -c "tail -f /vagrant/ml-detector/build/logs/*.log 2>/dev/null || echo 'No detector logs yet'"

logs-sniffer:
	@vagrant ssh -c "tail -f /vagrant/logs/lab/sniffer.log 2>/dev/null || echo 'No sniffer logs yet'"

logs-lab:
	@echo "📋 Combined Lab Logs (CTRL+C to exit)..."
	@vagrant ssh -c "cd /vagrant && bash scripts/monitor_lab.sh"

# ============================================================================
# Development Workflows
# ============================================================================

dev-setup: up lab-start build-unified
	@echo "✅ Development environment ready"

dev-setup-unified: up lab-start build-unified
	@echo "✅ Development environment ready (unified protobuf)"

test:
	@echo "Checking built components..."
	@vagrant ssh -c "ls -lh /vagrant/sniffer/build/sniffer 2>/dev/null && echo '✅ Sniffer built' || echo '❌ Sniffer not built'"
	@vagrant ssh -c "ls -lh /vagrant/ml-detector/build/ml-detector 2>/dev/null && echo '✅ Detector built' || echo '❌ Detector not built'"
	@vagrant ssh -c "ls -lh /vagrant/firewall-acl-agent/build/firewall-acl-agent 2>/dev/null && echo '✅ Firewall built' || echo '❌ Firewall not built'"

schema-update: proto rebuild
	@echo "✅ Schema updated and components rebuilt"

quick-fix:
	@echo "🔧 Quick bug fix procedure..."
	@$(MAKE) kill-lab
	@$(MAKE) rebuild
	@echo "✅ Ready to test fix"

kill-all:
	@echo "💀 Killing all ML Defender processes..."
	@$(MAKE) kill-lab
	@vagrant ssh -c "docker-compose down 2>/dev/null || true"
	@echo "✅ All processes terminated"

quick-lab-test:
	@echo "⚡ Quick Lab Test (smallFlows dataset)..."
	@$(MAKE) clean-day13-logs
	@$(MAKE) run-lab-dev
	@sleep 10
	@$(MAKE) status-lab
	@$(MAKE) test-replay-small
	@sleep 5
	@$(MAKE) extract-dual-scores
	@echo "✅ Quick test complete - check logs/dual_scores_*.txt"

lab-full-test:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 ML Defender - Full Lab Test (Integrated)              ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1: Clean previous logs..."
	@$(MAKE) clean-day13-logs
	@echo ""
	@echo "Step 2: Starting ML Defender lab..."
	@$(MAKE) run-lab-dev
	@echo ""
	@echo "⏳ Waiting for components to initialize (15s)..."
	@sleep 15
	@echo ""
	@echo "Step 3: Verify lab status..."
	@$(MAKE) status-lab
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "💡 OPTIONAL: Open tmux monitor in another terminal"
	@echo "   Run: make monitor-day13-tmux"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@read -p "Press ENTER when ready to start replay..." dummy
	@echo ""
	@echo "Step 4: Replaying CTU-13 Neris (this will take ~5-10 min)..."
	@$(MAKE) test-rag-neris
	@echo ""
	@echo "⏳ Waiting for pipeline to process events (10s)..."
	@sleep 10
	@echo ""
	@echo "Step 5: Extract dual-score logs..."
	@$(MAKE) extract-dual-scores
	@echo ""
	@echo "✅ Full lab test complete!"
	@echo ""
	@echo "📊 Next steps:"
	@echo "   - Analyze results: make analyze-dual-scores"
	@echo "   - Check RAG logs: make rag-log-status"
	@echo "   - View stats: make stats-dual-score"

# ============================================================================
# RAG Ecosystem Integration (RAG + etcd-server)
# ============================================================================

rag-build:
	@echo "🔨 Building RAG Security System..."
	@vagrant ssh -c "cd /vagrant/rag && make build"

rag-clean:
	@echo "🧹 Cleaning RAG..."
	@vagrant ssh -c "cd /vagrant/rag && make clean"

rag-start:
	@echo "🚀 Starting RAG Security System..."
	@vagrant ssh -c "mkdir -p /vagrant/logs"
	@vagrant ssh -c "if ! pgrep -f rag-security > /dev/null; then \
		cd /vagrant/rag/build && nohup ./rag-security -c ../config/rag-config.json > /vagrant/logs/rag.log 2>&1 & \
		sleep 2; \
		echo '✅ RAG started'; \
	else \
		echo '⚠️  RAG already running'; \
	fi"

rag-stop:
	@echo "🛑 Stopping RAG..."
	@vagrant ssh -c "pkill -f rag-security 2>/dev/null || true"

rag-status:
	@echo "🔍 RAG Status:"
	@vagrant ssh defender -c "pid=\$$(pgrep -f rag-security); if [ -n \"\$$pid\" ]; then echo \"✅ RAG running (PID: \$$pid)\"; else echo '❌ RAG stopped'; fi"

rag-logs:
	@echo "📋 RAG Logs:"
	@vagrant ssh -c "tail -20 /vagrant/logs/rag.log 2>/dev/null || echo 'No logs found'"

rag-download-model:
	@echo "📥 Downloading LLM model for RAG..."
	@vagrant ssh -c "cd /vagrant/rag && \
		if [ ! -f models/default.gguf ]; then \
			mkdir -p models && cd models && \
			wget -q --show-progress https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf && \
			ln -sf tinyllama-1.1b-chat-v1.0.Q4_0.gguf default.gguf && \
			echo '✅ Model downloaded'; \
		else \
			echo '✅ Model already exists'; \
		fi"
# ----------------------------------------------------------------------------

# Initialize RAG directories
rag-log-init:
	@echo "🎯 Initializing RAG directories..."
	mkdir -p /vagrant/logs/rag/events
	mkdir -p /vagrant/logs/rag/artifacts
	mkdir -p /vagrant/logs/rag/stats
	@echo "✅ RAG directories created"
	@ls -la /vagrant/logs/rag/

# Clean RAG logs (CAREFUL!)
rag-log-clean:
	@echo "⚠️  WARNING: This will delete all RAG logs!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "🗑️  Cleaning RAG logs..."; \
		rm -rf /vagrant/logs/rag/events/*; \
		rm -rf /vagrant/logs/rag/artifacts/*; \
		rm -rf /vagrant/logs/rag/stats/*; \
		echo "✅ RAG logs cleaned"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Show RAG status
rag-log-status:
	@echo "=" | tr '=' '=' | head -c 80; echo
	@echo "📊 RAG LOGGER STATUS"
	@echo "=" | tr '=' '=' | head -c 80; echo
	@echo
	@echo "📂 Directory sizes:"
	@du -sh /vagrant/logs/rag/events 2>/dev/null || echo "  events: (empty)"
	@du -sh /vagrant/logs/rag/artifacts 2>/dev/null || echo "  artifacts: (empty)"
	@du -sh /vagrant/logs/rag/stats 2>/dev/null || echo "  stats: (empty)"
	@echo
	@echo "📄 Event files:"
	@ls -lh /vagrant/logs/rag/events/ 2>/dev/null | tail -n +2 || echo "  (no files)"
	@echo
	@echo "📊 Event counts:"
	@for file in /vagrant/logs/rag/events/*.jsonl; do \
		[ -f "$$file" ] || continue; \
		count=$$(wc -l < "$$file"); \
		echo "  $$(basename $$file): $$count events"; \
	done
	@echo
	@echo "📦 Artifact counts:"
	@for dir in /vagrant/logs/rag/artifacts/*/; do \
		[ -d "$$dir" ] || continue; \
		pb_count=$$(find "$$dir" -name "*.pb" | wc -l); \
		json_count=$$(find "$$dir" -name "*.json" | wc -l); \
		echo "  $$(basename $$dir): $$pb_count .pb files, $$json_count .json files"; \
	done

# Analyze today's RAG log
rag-log-analyze:
	@TODAY=$$(date +%Y-%m-%d); \
	LOG_FILE="/vagrant/logs/rag/events/$$TODAY.jsonl"; \
	if [ -f "$$LOG_FILE" ]; then \
		echo "📊 Analyzing $$LOG_FILE..."; \
		python3 /vagrant/scripts/analyze_rag_logs.py "$$LOG_FILE"; \
	else \
		echo "❌ No log file found for today: $$LOG_FILE"; \
	fi

# Analyze specific date
rag-log-analyze-date:
	@read -p "Enter date (YYYY-MM-DD): " date; \
	LOG_FILE="/vagrant/logs/rag/events/$$date.jsonl"; \
	if [ -f "$$LOG_FILE" ]; then \
		echo "📊 Analyzing $$LOG_FILE..."; \
		python3 /vagrant/scripts/analyze_rag_logs.py "$$LOG_FILE"; \
	else \
		echo "❌ No log file found: $$LOG_FILE"; \
	fi

# Tail RAG logs (follow mode)
rag-log-tail:
	@TODAY=$$(date +%Y-%m-%d); \
	LOG_FILE="/vagrant/logs/rag/events/$$TODAY.jsonl"; \
	if [ -f "$$LOG_FILE" ]; then \
		echo "📜 Tailing $$LOG_FILE (Ctrl+C to stop)..."; \
		tail -f "$$LOG_FILE"; \
	else \
		echo "❌ No log file found for today: $$LOG_FILE"; \
		echo "Waiting for file to be created..."; \
		tail -f "$$LOG_FILE" 2>/dev/null || echo "File not created yet"; \
	fi

# Tail RAG logs with pretty-printing
rag-log-tail-live:
	@TODAY=$$(date +%Y-%m-%d); \
	LOG_FILE="/vagrant/logs/rag/events/$$TODAY.jsonl"; \
	echo "📜 Live RAG log (pretty-printed, Ctrl+C to stop)..."; \
	tail -f "$$LOG_FILE" 2>/dev/null | while read -r line; do \
		echo "$$line" | jq -C '.' 2>/dev/null || echo "$$line"; \
	done

# View specific event artifact
rag-log-view-event:
	@read -p "Enter event ID: " event_id; \
	TODAY=$$(date +%Y-%m-%d); \
	ARTIFACT="/vagrant/logs/rag/artifacts/$$TODAY/event_$$event_id.json"; \
	if [ -f "$$ARTIFACT" ]; then \
		echo "📄 Viewing $$ARTIFACT:"; \
		jq -C '.' "$$ARTIFACT" | less -R; \
	else \
		echo "❌ Artifact not found: $$ARTIFACT"; \
		echo "Searching in other dates..."; \
		find /vagrant/logs/rag/artifacts -name "event_$$event_id.json" -exec cat {} \; | jq -C '.' | less -R; \
	fi

# Export RAG logs to CSV (for Excel analysis)
rag-log-export-csv:
	@TODAY=$$(date +%Y-%m-%d); \
	LOG_FILE="/vagrant/logs/rag/events/$$TODAY.jsonl"; \
	OUTPUT="/vagrant/logs/rag/stats/$$TODAY.csv"; \
	if [ -f "$$LOG_FILE" ]; then \
		echo "📊 Exporting to CSV: $$OUTPUT"; \
		python3 /vagrant/scripts/export_rag_to_csv.py "$$LOG_FILE" "$$OUTPUT"; \
		echo "✅ Exported to: $$OUTPUT"; \
	else \
		echo "❌ No log file found for today: $$LOG_FILE"; \
	fi

# Compress old logs (> 7 days)
rag-log-compress-old:
	@echo "🗜️  Compressing logs older than 7 days..."
	@find /vagrant/logs/rag/events -name "*.jsonl" -mtime +7 -exec gzip {} \;
	@find /vagrant/logs/rag/artifacts -type d -mtime +7 -exec tar -czf {}.tar.gz {} \; -exec rm -rf {} \;
	@echo "✅ Old logs compressed"

# Show RAG statistics summary
rag-log-stats-summary:
	@echo "=" | tr '=' '=' | head -c 80; echo
	@echo "📊 RAG STATISTICS SUMMARY"
	@echo "=" | tr '=' '=' | head -c 80; echo
	@echo
	@echo "Total events logged (all time):"
	@find /vagrant/logs/rag/events -name "*.jsonl" -exec wc -l {} + | tail -1 | awk '{print "  " $$1 " events"}'
	@echo
	@echo "Total artifacts saved (all time):"
	@find /vagrant/logs/rag/artifacts -name "*.pb" | wc -l | awk '{print "  " $$1 " protobuf files"}'
	@find /vagrant/logs/rag/artifacts -name "*.json" | wc -l | awk '{print "  " $$1 " json files"}'
	@echo
	@echo "Disk usage:"
	@du -sh /vagrant/logs/rag | awk '{print "  Total: " $$1}'
	@echo
	@echo "Latest 5 event files:"
	@ls -lt /vagrant/logs/rag/events/*.jsonl 2>/dev/null | head -5 | awk '{print "  " $$9 " (" $$5 " bytes)"}'


etcd-server-build:
	@echo "🔨 Building custom etcd-server..."
	@vagrant ssh -c "cd /vagrant/etcd-server && make build"

etcd-server-clean:
	@echo "🧹 Cleaning etcd-server..."
	@vagrant ssh -c "cd /vagrant/etcd-server && make clean"

etcd-server-start:
	@echo "🚀 Starting etcd-server..."
	@vagrant ssh -c "mkdir -p /vagrant/logs && cd /vagrant/etcd-server/build && nohup ./etcd-server > /vagrant/logs/etcd-server.log 2>&1 &"
	@echo "✅ etcd-server started (logs: /vagrant/logs/etcd-server.log)"

etcd-server-stop:
	@echo "🛑 Stopping etcd-server..."
	@vagrant ssh -c "pkill -f etcd-server 2>/dev/null || true"

etcd-server-status:
	@echo "🔍 etcd-server Status:"
	@vagrant ssh -c "pid=\$$(pgrep -f etcd-server); if [ -n \"\$$pid\" ]; then echo \"✅ etcd-server running (PID: \$$pid)\"; else echo \"❌ etcd-server stopped\"; fi"

etcd-server-logs:
	@echo "📋 etcd-server Logs:"
	@vagrant ssh -c "tail -20 /vagrant/logs/etcd-server.log 2>/dev/null || echo 'No logs found'"

etcd-server-health:
	@echo "🩺 Checking etcd-server health..."
	@vagrant ssh -c "curl -s http://localhost:2379/health 2>/dev/null | grep -i healthy || echo '⚠️  etcd-server health check failed'"
# ----------------------------------------------------------------------------

rag-etcd-build: rag-build etcd-server-build
	@echo "✅ RAG ecosystem built"

rag-etcd-start: etcd-server-start rag-start
	@echo "✅ RAG ecosystem started (etcd-server + RAG)"
	@echo "   etcd-server: http://localhost:2379"
	@echo "   RAG CLI: cd /vagrant/rag/build && ./rag-security"

rag-etcd-stop: rag-stop etcd-server-stop
	@echo "✅ RAG ecosystem stopped"

rag-etcd-status: etcd-server-status rag-status
	@echo "✅ RAG ecosystem status checked"

rag-etcd-logs:
	@echo "📋 Combined RAG ecosystem logs:"
	@echo "=== etcd-server (last 10 lines) ==="
	@vagrant ssh -c "tail -10 /vagrant/logs/etcd-server.log 2>/dev/null || echo 'No etcd-server logs'"
	@echo -e "\n=== RAG (last 10 lines) ==="
	@vagrant ssh -c "tail -10 /vagrant/logs/rag.log 2>/dev/null || echo 'No RAG logs'"

# ============================================================================
# Full System Integration (ML Defender + RAG Ecosystem)
# ============================================================================

# Build everything including RAG ecosystem
all-with-rag: build-unified rag-etcd-build
	@echo "✅ All components built including RAG ecosystem"

# Start full system
start-all: rag-etcd-start
	@echo "⏳ Waiting for RAG ecosystem to initialize..."
	@sleep 3
	@make run-lab-dev
	@echo "✅ Full system started (RAG ecosystem + ML Defender lab)"

# Stop full system
stop-all: rag-etcd-stop
	@make kill-lab
	@echo "✅ Full system stopped"

# Status of everything
status-all:
	@echo "════════════════════════════════════════════════════════════"
	@echo "ML Defender Full System Status"
	@echo "════════════════════════════════════════════════════════════"
	@make status-lab
	@echo "════════════════════════════════════════════════════════════"
	@echo "RAG Ecosystem Status"
	@echo "════════════════════════════════════════════════════════════"
	@make rag-etcd-status
	@echo "════════════════════════════════════════════════════════════"

# Clean everything
clean-all: clean rag-clean etcd-server-clean
	@echo "✅ All components cleaned including RAG ecosystem"

# ============================================================================
# Quick Start/Test targets
# ============================================================================

test-rag-etcd: rag-etcd-build rag-etcd-start
	@echo "✅ RAG ecosystem built and started"
	@echo "Testing communication..."
	@vagrant ssh -c "sleep 2 && curl -s http://localhost:2379/health || echo 'etcd-server health check failed'"
	@echo "✅ RAG ecosystem test complete"

quick-rag: rag-build rag-start
	@echo "✅ RAG started quickly (assuming etcd-server already running)"

# ============================================================================
# Day 13 - Dual-Score Architecture Testing (CTU-13 Dataset)
# ============================================================================

# Dataset paths
CTU13_SMALL := /vagrant/datasets/ctu13/smallFlows.pcap
CTU13_NERIS := /vagrant/datasets/ctu13/botnet-capture-20110810-neris.pcap
CTU13_BIG := /vagrant/datasets/ctu13/bigFlows.pcap


# Replay targets with logging
test-rag-small:
	@./scripts/test_rag_logger.sh datasets/ctu13/smallFlows.pcap

test-rag-neris:
	@./scripts/test_rag_logger.sh datasets/ctu13/botnet-capture-20110810-neris.pcap

rag-consolidate:
	@echo "📦 Consolidating artifacts into .jsonl..."
	@TODAY=$$(date +%Y-%m-%d); \
	vagrant ssh defender -c "find /vagrant/logs/rag/artifacts/$$TODAY -name 'event_*.json' -exec cat {} \; | jq -c '.' > /vagrant/logs/rag/events/$$TODAY.jsonl && echo '✅ Consolidated: /vagrant/logs/rag/events/$$TODAY.jsonl'"

test-rag-big:
	@./scripts/test_rag_logger.sh datasets/ctu13/bigFlows.pcap

test-replay-small:
	@echo "🧪 Replaying CTU-13 smallFlows.pcap..."
	@vagrant ssh client -c "mkdir -p /vagrant/logs/lab && \
		sudo tcpreplay -i eth1 --mbps=10 --stats=2 $(CTU13_SMALL) 2>&1 | tee /vagrant/logs/lab/tcpreplay.log"
	@echo "✅ Replay complete"

test-replay-neris:
	@echo "🧪 Replaying CTU-13 Neris botnet (492K events)..."
	@echo "⚠️  This will take ~5-10 minutes"
	@vagrant ssh client -c "mkdir -p /vagrant/logs/lab && \
		sudo tcpreplay -i eth1 --mbps=10 --stats=5 $(CTU13_NERIS) 2>&1 | tee /vagrant/logs/lab/tcpreplay.log"
	@echo "✅ Replay complete"

test-replay-big:
	@echo "🧪 Replaying CTU-13 bigFlows.pcap (352M)..."
	@echo "⚠️  This will take ~30-60 minutes"
	@vagrant ssh client -c "mkdir -p /vagrant/logs/lab && \
		sudo tcpreplay -i eth1 --mbps=10 --stats=10 $(CTU13_BIG) 2>&1 | tee /vagrant/logs/lab/tcpreplay.log"
	@echo "✅ Replay complete"

# Monitor targets
monitor-day13-tmux:
	@echo "🚀 Starting Day 13 tmux multi-panel monitor..."
	@echo "   Layout: 4 panels (tcpreplay + logs + stats)"
	@vagrant ssh defender -c "cd /vagrant && bash scripts/monitor_day13_test.sh"

logs-dual-score:
	@echo "📊 Monitoring Dual-Score logs (CTRL+C to exit)..."
	@vagrant ssh defender -c "tail -f /tmp/ml-detector.log | grep -E 'DUAL-SCORE|⚠️'"

logs-dual-score-live:
	@echo "📊 Live Dual-Score analysis with highlighting..."
	@vagrant ssh defender -c "tail -f /tmp/ml-detector.log | grep --line-buffered 'DUAL-SCORE' | \
		awk '{print \$$0; divergence=\$$NF; if (divergence+0 > 0.30) print \"  ⚠️  HIGH DIVERGENCE DETECTED\"}'"

# Extract logs for F1-score calculation
extract-dual-scores:
	@echo "📥 Extracting Dual-Score logs..."
	@vagrant ssh defender -c "grep 'DUAL-SCORE' /vagrant/logs/lab/detector.log > /vagrant/logs/dual_scores_$(shell date +%Y%m%d_%H%M%S).txt" || true
	@echo "✅ Logs extracted to /vagrant/logs/"
	@ls -lh logs/dual_scores_*.txt 2>/dev/null | tail -1 || echo "No logs found yet"

# Integration test with tmux monitoring
test-integration-day13-tmux:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Day 13 Integration Test - tmux Multi-Panel            ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "This will open a 4-panel tmux session showing:"
	@echo "  Panel 1: tcpreplay progress"
	@echo "  Panel 2: Dual-Score logs"
	@echo "  Panel 3: Sniffer activity"
	@echo "  Panel 4: Live statistics"
	@echo ""
	@echo "Step 1: Starting ML Defender lab..."
	@$(MAKE) run-lab-dev
	@echo ""
	@echo "⏳ Waiting for components to initialize (10s)..."
	@sleep 10
	@echo ""
	@echo "Step 2: Verify lab status..."
	@$(MAKE) status-lab
	@echo ""
	@echo "Step 3: Open tmux monitor in a new terminal"
	@echo "   Run: make monitor-day13-tmux"
	@echo ""
	@echo "Step 4: Start replay from another terminal"
	@echo "   Run: make test-replay-small"
	@echo ""
	@echo "💡 TIP: Use 'tmux detach' (Ctrl+B, D) to keep monitor running"

# Original simple integration test (non-tmux)
test-integration-day13:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Day 13 Integration Test - Dual-Score Architecture      ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1: Verify protobuf sync..."
	@$(MAKE) proto-verify
	@echo ""
	@echo "Step 2: Start ML Defender lab..."
	@$(MAKE) run-lab-dev
	@echo ""
	@echo "⏳ Waiting for components to initialize (10s)..."
	@sleep 10
	@echo ""
	@echo "Step 3: Verify lab status..."
	@$(MAKE) status-lab
	@echo ""
	@echo "Step 4: Replay CTU-13 smallFlows.pcap..."
	@$(MAKE) test-replay-small
	@echo ""
	@echo "Step 5: Extract Dual-Score logs..."
	@$(MAKE) extract-dual-scores
	@echo ""
	@echo "✅ Integration test complete!"
	@echo "📊 Check logs/dual_scores_*.txt for F1-score analysis"

# Quick test (fast validation)
test-dual-score-quick:
	@echo "⚡ Quick Dual-Score Test..."
	@echo "Starting components in background..."
	@$(MAKE) run-lab-dev > /dev/null 2>&1 &
	@sleep 8
	@echo "Replaying small dataset..."
	@$(MAKE) test-replay-small
	@sleep 2
	@echo "Checking for Dual-Score logs..."
	@vagrant ssh defender -c "grep -c 'DUAL-SCORE' /tmp/ml-detector.log && echo '✅ Dual-Score logging working' || echo '❌ No Dual-Score logs found'"

# Show Day 13 statistics
stats-dual-score:
	@echo "════════════════════════════════════════════════════════════"
	@echo "Day 13 Dual-Score Statistics"
	@echo "════════════════════════════════════════════════════════════"
	@vagrant ssh defender -c "if [ -f /vagrant/logs/lab/detector.log ]; then \
		echo 'Total Dual-Score events:'; \
		grep -c 'DUAL-SCORE' /vagrant/logs/lab/detector.log; \
		echo ''; \
		echo 'Authoritative Sources:'; \
		grep 'DUAL-SCORE' /vagrant/logs/lab/detector.log | grep -oP 'source=\K[A-Z_]+' | sort | uniq -c; \
		echo ''; \
		echo 'High Divergence (>0.30):'; \
		grep 'Score divergence' /vagrant/logs/lab/detector.log | wc -l; \
	else \
		echo 'No logs found. Run a test first.'; \
	fi"
	@echo "════════════════════════════════════════════════════════════"

# Clean Day 13 logs
clean-day13-logs:
	@echo "🧹 Cleaning Day 13 logs..."
	@vagrant ssh defender -c "sudo truncate -s 0 /vagrant/logs/lab/detector.log"
	@vagrant ssh defender -c "sudo truncate -s 0 /vagrant/logs/lab/sniffer.log"
	@vagrant ssh defender -c "sudo truncate -s 0 /vagrant/logs/lab/firewall.log"
	@vagrant ssh client -c "sudo rm -f /vagrant/logs/lab/tcpreplay.log"
	@rm -f logs/dual_scores_*.txt
	@echo "✅ Logs cleaned"

# ============================================================================
# Day 13 - Analysis Tools
# ============================================================================

# Analyze dual-score logs
analyze-dual-scores:
	@echo "📊 Analyzing Dual-Score logs..."
	@if [ -f logs/dual_scores_manual.txt ]; then \
		python3 scripts/analyze_dual_scores.py logs/dual_scores_manual.txt; \
	else \
		echo "❌ No dual_scores_manual.txt found. Run: make extract-dual-scores"; \
	fi

# Full analysis workflow
test-analyze-workflow:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  Day 13 - Complete Analysis Workflow                      ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1: Extract dual-score logs..."
	@$(MAKE) extract-dual-scores
	@echo ""
	@echo "Step 2: Analyze results..."
	@$(MAKE) analyze-dual-scores
	@echo ""
	@echo "✅ Analysis complete"

# Full Day 13 test cycle (clean → test → analyze)
test-day13-full:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  Day 13 - Full Test Cycle (Clean → Test → Analyze)        ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1: Clean previous logs..."
	@$(MAKE) clean-day13-logs
	@echo ""
	@echo "Step 2: Start lab..."
	@$(MAKE) run-lab-dev
	@sleep 10
	@echo ""
	@echo "Step 3: Verify lab status..."
	@$(MAKE) status-lab
	@echo ""
	@echo "Step 4: Replay dataset (open monitor in another terminal)..."
	@echo "   Terminal 2: make monitor-day13-tmux"
	@echo "   Terminal 3: make test-replay-small (or test-replay-neris)"
	@read -p "Press ENTER when replay is complete..." dummy
	@echo ""
	@echo "Step 5: Extract and analyze..."
	@$(MAKE) test-analyze-workflow
	@echo ""
	@echo "✅ Full test cycle complete"

# Quick analysis of current logs (no extraction)
quick-analyze:
	@echo "⚡ Quick analysis of current detector logs..."
	@vagrant ssh defender -c "grep 'DUAL-SCORE' /vagrant/logs/lab/detector.log" > logs/dual_scores_quick.txt
	@python3 scripts/analyze_dual_scores.py logs/dual_scores_quick.txt
	@rm -f logs/dual_scores_quick.txt

# Full integration test with RAG
test-rag-integration: rag-log-init
	@echo "🧪 Running RAG integration test..."
	@echo
	@echo "1️⃣  Starting lab..."
	@$(MAKE) run-lab-dev
	@sleep 5
	@echo
	@echo "2️⃣  Starting monitoring..."
	@$(MAKE) monitor-day13-tmux &
	@sleep 2
	@echo
	@echo "3️⃣  Replaying test dataset..."
	@$(MAKE) test-replay-small
	@echo
	@echo "4️⃣  Waiting for processing..."
	@sleep 10
	@echo
	@echo "5️⃣  Analyzing results..."
	@$(MAKE) rag-status
	@$(MAKE) rag-analyze
	@echo
	@echo "✅ Integration test complete"

# Quick RAG test (no tmux)
test-rag-quick: rag-clean rag-log-init
	@echo "⚡ Quick RAG test..."
	@$(MAKE) test-replay-small
	@sleep 5
	@$(MAKE) rag-stats-summary
	@$(MAKE) rag-analyze

rag-watch:
	@echo "📺 Live RAG monitoring (Ctrl+C to stop)..."
	@vagrant ssh defender -c "watch -n 5 'echo \"=== RAG Stats ===\";\
	  cat /vagrant/logs/rag/events/*.jsonl 2>/dev/null | wc -l | xargs echo \"Total events:\";\
	  ps aux | grep ml-detector | grep -v grep | awk \"{print \\\"CPU: \\\"\\\$$3\\\"%\\\"}\"'"

rag-validate:
	@echo "🔍 Validating RAG logs..."
	@vagrant ssh defender -c "cat /vagrant/logs/rag/events/*.jsonl | jq empty && echo '✅ All logs valid JSON' || echo '❌ Invalid JSON found'"

# ============================================================================
# Help updates
# ============================================================================

help-rag:
	@echo "RAG Ecosystem Commands:"
	@echo "  make rag-build           - Build RAG Security System"
	@echo "  make rag-start           - Start RAG"
	@echo "  make rag-stop            - Stop RAG"
	@echo "  make rag-status          - Check RAG status"
	@echo "  make rag-logs            - Show RAG logs"
	@echo ""
	@echo "  make etcd-server-build   - Build custom etcd-server"
	@echo "  make etcd-server-start   - Start etcd-server"
	@echo "  make etcd-server-stop    - Stop etcd-server"
	@echo "  make etcd-server-status  - Check etcd-server status"
	@echo "  make etcd-server-logs    - Show etcd-server logs"
	@echo ""
	@echo "  make rag-etcd-build      - Build both RAG and etcd-server"
	@echo "  make rag-etcd-start      - Start RAG ecosystem"
	@echo "  make rag-etcd-stop       - Stop RAG ecosystem"
	@echo "  make rag-etcd-status     - Check RAG ecosystem status"
	@echo "  make rag-etcd-logs       - Show combined logs"
	@echo ""
	@echo "  make all-with-rag        - Build everything including RAG"
	@echo "  make start-all           - Start full system"
	@echo "  make stop-all            - Stop full system"
	@echo "  make status-all          - Check everything"
	@echo "  make clean-all           - Clean everything"
	@echo ""
	@echo "  make test-rag-etcd       - Quick test of RAG ecosystem"
	@echo "  make quick-rag           - Quick start RAG (needs etcd-server)"
	@echo "RAG Logger Management Commands:"

help-day13:
	@echo "Day 13 - Dual-Score Testing (tmux):"
	@echo "  make test-integration-day13-tmux - 🚀 Full test with tmux monitor"
	@echo "  make monitor-day13-tmux      - 📊 Open 4-panel tmux monitor"
	@echo "  make test-dual-score-quick   - ⚡ Quick validation test"
	@echo "  make test-replay-small       - Replay smallFlows.pcap"
	@echo "  make test-replay-neris       - Replay Neris botnet"
	@echo "  make logs-dual-score         - Monitor Dual-Score logs"
	@echo "  make extract-dual-scores     - Extract logs for F1-calculation"
	@echo "  make stats-dual-score        - Show Dual-Score statistics"
	@echo "  make clean-day13-logs        - Clean Day 13 logs"

# ----------------------------------------------------------------------------
# 2. etcd-client (librería compartida - ANTES de componentes)
# ----------------------------------------------------------------------------
etcd-client-build: proto-unified crypto-transport-build
	@echo "🔨 Building etcd-client library..."
	@echo "   Dependencies: proto ✅  crypto-transport ✅"
	@vagrant ssh -c "cd /vagrant/etcd-client && \
		rm -rf build && \
		mkdir -p build && \
		cd build && \
		cmake .. && \
		make -j4"
	@echo "✅ etcd-client library built"
	@echo "Installing system-wide..."
	@vagrant ssh -c "cd /vagrant/etcd-client/build && sudo make install && sudo ldconfig"
	@echo "✅ etcd-client installed to /usr/local/lib"
	@echo "Verifying library..."
	@vagrant ssh -c "ls -lh /vagrant/etcd-client/build/libetcd_client.so"

etcd-client-clean:
	@echo "🧹 Cleaning etcd-client..."
	@vagrant ssh -c "rm -rf /vagrant/etcd-client/build"
	@vagrant ssh -c "sudo rm -f /usr/local/lib/libetcd_client.so*"
	@vagrant ssh -c "sudo rm -rf /usr/local/include/etcd_client"
	@echo "✅ etcd-client cleaned"

etcd-client-test:
	@echo "🧪 Testing etcd-client..."
	@vagrant ssh -c "cd /vagrant/etcd-client/build && ctest --output-on-failure"

# Verify that components are linked with etcd-client
# ============================================================================
# Verificación (también corregido)
# ============================================================================

verify-etcd-linkage:
	@echo "════════════════════════════════════════════════════════════"
	@echo "🔍 Verifying etcd-client linkage in components"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "0️⃣  etcd-client library:"
	@vagrant ssh -c "if [ -f /vagrant/etcd-client/build/libetcd_client.so ]; then \
		echo '   ✅ Library exists'; \
		ls -lh /vagrant/etcd-client/build/libetcd_client.so; \
	else \
		echo '   ❌ Library NOT FOUND - run: make etcd-client-build'; \
	fi"
	@echo ""
	@echo "1️⃣  Sniffer:"
	@vagrant ssh -c "if [ -f /vagrant/sniffer/build/sniffer ]; then \
		ldd /vagrant/sniffer/build/sniffer 2>/dev/null | grep etcd_client && echo '   ✅ Linked' || echo '   ❌ NOT linked'; \
	else \
		echo '   ⚠️  Binary not found - not built yet'; \
	fi"
	@echo ""
	@echo "2️⃣  ML Detector:"
	@vagrant ssh -c "if [ -f /vagrant/ml-detector/build/ml-detector ]; then \
		ldd /vagrant/ml-detector/build/ml-detector 2>/dev/null | grep etcd_client && echo '   ✅ Linked' || echo '   ❌ NOT linked'; \
	else \
		echo '   ⚠️  Binary not found - not built yet'; \
	fi"
	@echo ""
	@echo "3️⃣  Firewall ACL Agent:"
	@vagrant ssh -c "if [ -f /vagrant/firewall-acl-agent/build/firewall-acl-agent ]; then \
		ldd /vagrant/firewall-acl-agent/build/firewall-acl-agent 2>/dev/null | grep etcd_client && echo '   ✅ Linked' || echo '   ❌ NOT linked'; \
	else \
		echo '   ⚠️  Binary not found - not built yet'; \
	fi"
	@echo ""
	@echo "════════════════════════════════════════════════════════════"

# ============================================================================
# NOTAS IMPORTANTES
# ============================================================================
#
# ORDEN DE DEPENDENCIAS (CRÍTICO):
#
# proto-unified
#     └── etcd-client-build (depende de proto)
#             ├── sniffer (depende de proto + etcd-client)
#             ├── detector (depende de proto + etcd-client)
#             └── firewall (depende de proto + etcd-client)
#
# etcd-server-build (independiente, se puede compilar en paralelo)
#
# ============================================================================
#
# PROBLEMA PREVIO:
# - all: build-unified etcd-client-build
#   → Compilaba componentes ANTES de etcd-client
#   → Funcionaba solo porque libetcd_client.so existía de compilaciones previas
#
# SOLUCIÓN:
# - etcd-client-build se compila ANTES de cualquier componente
# - Cada componente declara explícitamente: proto + etcd-client-build
#
# ============================================================================

# Verify encryption/compression is enabled in configs
verify-encryption:
	@echo "════════════════════════════════════════════════════════════"
	@echo "🔐 Verifying Encryption/Compression Configuration"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "1️⃣  Sniffer config:"
	@vagrant ssh -c "jq '.encryption_enabled, .compression_enabled' /vagrant/sniffer/config/sniffer.json 2>/dev/null || echo '   ⚠️  Config not found'"
	@echo ""
	@echo "2️⃣  ML Detector config:"
	@vagrant ssh -c "jq '.encryption_enabled, .compression_enabled' /vagrant/ml-detector/config/ml_detector_config.json 2>/dev/null || echo '   ⚠️  Config not found'"
	@echo ""
	@echo "3️⃣  Firewall config:"
	@vagrant ssh -c "jq '.encryption_enabled, .compression_enabled' /vagrant/firewall-acl-agent/config/firewall.json 2>/dev/null || echo '   ⚠️  Config not found'"
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "Expected: Both should be 'true' for full pipeline"
	@echo "════════════════════════════════════════════════════════════"

# Verify full pipeline configuration
verify-pipeline-config:
	@echo "════════════════════════════════════════════════════════════"
	@echo "🔧 Day 23 Pipeline Configuration Status"
	@echo "════════════════════════════════════════════════════════════"
	@$(MAKE) verify-etcd-linkage
	@echo ""
	@$(MAKE) verify-encryption
	@echo ""
	@echo "🔍 etcd-server connectivity:"
	@vagrant ssh -c "curl -s http://localhost:2379/health 2>/dev/null | jq . || echo '   ❌ etcd-server not responding'"
	@echo ""
	@echo "════════════════════════════════════════════════════════════"

monitor-day23-tmux:
	@echo "🚀 Starting Day 23 tmux monitor (5 panels + etcd-server)..."
	@vagrant ssh -c "cd /vagrant && bash scripts/monitor_day23.sh"

test-day23-full:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Day 23 - Full Integration Test (etcd + encryption)    ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Phase 1: Build verification..."
	@$(MAKE) all
	@echo ""
	@echo "Phase 2: Configuration verification..."
	@$(MAKE) verify-pipeline-config
	@echo ""
	@echo "Phase 3: Clean previous logs..."
	@$(MAKE) clean-day13-logs
	@$(MAKE) rag-log-clean
	@echo ""
	@echo "Phase 4: Start full lab..."
	@$(MAKE) run-lab-dev-day23
	@echo ""
	@echo "⏳ Waiting for components to initialize (15s)..."
	@sleep 15
	@echo ""
	@echo "Phase 5: Verify lab status..."
	@$(MAKE) status-lab-day23
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "💡 Open monitor in another terminal: make monitor-day23-tmux"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@read -p "Press ENTER to start traffic replay..." dummy
	@echo ""
	@echo "Phase 6: Replay test traffic..."
	@$(MAKE) test-rag-neris
	@echo ""
	@echo "⏳ Waiting for pipeline processing (30s)..."
	@sleep 30
	@echo ""
	@echo "Phase 7: Verify RAG logs..."
	@$(MAKE) verify-rag-logs-day23
	@echo ""
	@echo "✅ Day 23 full test complete!"

test-day23-stress:
	@echo "⚡ Day 23 Stress Test (10+ minutes continuous operation)..."
	@echo ""
	@echo "Starting lab..."
	@$(MAKE) run-lab-dev-day23
	@sleep 15
	@echo ""
	@echo "Status before stress:"
	@$(MAKE) status-lab-day23
	@echo ""
	@echo "Starting continuous traffic replay (600 seconds = 10 minutes)..."
	@vagrant ssh client -c "sudo timeout 600 tcpreplay -i eth1 --mbps=10 --loop=10 /vagrant/datasets/ctu13/botnet-capture-20110810-neris.pcap 2>&1 | tee /vagrant/logs/lab/stress_test.log"
	@echo ""
	@echo "Status after stress:"
	@$(MAKE) status-lab-day23
	@echo ""
	@echo "Checking for crashes or errors..."
	@vagrant ssh -c "grep -i 'error\|crash\|segfault' /vagrant/logs/*.log || echo '✅ No crashes detected'"
	@echo ""
	@echo "✅ Stress test complete - system stable for 10+ minutes"

verify-rag-logs-day23:
	@echo "🔍 Verifying RAG logs for Day 23..."
	@echo ""
	@TODAY=$$(date +%Y-%m-%d); \
	LOG_FILE="/vagrant/logs/rag/events/$$TODAY.jsonl"; \
	vagrant ssh -c "if [ -f $$LOG_FILE ]; then \
		echo '✅ RAG log file exists: $$LOG_FILE'; \
		wc -l $$LOG_FILE | awk '{print \"   Events logged: \" \$$1}'; \
		echo ''; \
		echo 'Sample events (first 3):'; \
		head -3 $$LOG_FILE | jq -c '.timestamp, .event_type, .score' 2>/dev/null || head -3 $$LOG_FILE; \
	else \
		echo '❌ No RAG log file found for today'; \
		echo '   Expected: $$LOG_FILE'; \
	fi"

day23:
	@echo "🚀 Day 23 Quick Workflow"
	@echo "1. Build all: make all"
	@echo "2. Verify: make verify-pipeline-config"
	@echo "3. Start: make run-lab-dev-day23"
	@echo "4. Status: make status-lab-day23"
	@echo "5. Monitor: make monitor-day23-tmux (in new terminal)"
	@echo "6. Test: make test-day23-stress"
	@echo ""
	@read -p "Run full Day 23 test? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) test-day23-full; \
	fi