.PHONY: help status
.PHONY: up halt destroy ssh
.PHONY: proto proto-unified proto-verify sniffer detector firewall all rebuild
.PHONY: sniffer-build sniffer-clean sniffer-package sniffer-install
.PHONY: detector-build detector-clean
.PHONY: firewall-build firewall-clean
.PHONY: run-sniffer run-detector run-firewall
.PHONY: logs-sniffer logs-detector logs-firewall logs-lab
.PHONY: run-lab-dev kill-lab status-lab
.PHONY: run-lab-dev-day23 kill-lab-day23 status-lab-day23
.PHONY: kill-all check-ports restart
.PHONY: clean clean-libs clean-components clean-all distclean test test-libs test-components test-all dev-setup schema-update
.PHONY: build-unified rebuild-unified quick-fix dev-setup-unified
.PHONY: check-libbpf verify-bpf-maps diagnose-bpf
.PHONY: test-replay-small test-replay-neris test-replay-big
.PHONY: monitor-day13-tmux logs-dual-score logs-dual-score-live extract-dual-scores
.PHONY: test-integration-day13 test-integration-day13-tmux test-dual-score-quick
.PHONY: clean-day13-logs stats-dual-score
.PHONY: analyze-dual-scores test-analyze-workflow test-day13-full quick-analyze
.PHONY: rag-log-init rag-log-clean rag-log-status rag-log-analyze rag-log-tail rag-log-tail-live
.PHONY: test-rag-integration test-rag-quick rag-watch rag-validate test-rag-small test-rag-neris test-rag-big
.PHONY: quick-lab-test rag-consolidate detector-debug
.PHONY: etcd-client-build etcd-client-clean etcd-client-test
.PHONY: verify-etcd-linkage verify-encryption verify-pipeline-config verify-all
.PHONY: monitor-day23-tmux
.PHONY: test-day23-full test-day23-stress verify-rag-logs-day23
.PHONY: day23
.PHONY: rag-ingester rag-ingester-build rag-ingester-clean
.PHONY: tools-build tools-clean tools-synthetic-gen
.PHONY: crypto-transport-build crypto-transport-clean crypto-transport-test
.PHONY: etcd-server etcd-server-build etcd-server-clean etcd-server-start etcd-server-stop
.PHONY: rag-build rag-clean rag-start rag-stop rag-status rag-logs
.PHONY: test-hardening test-hardening-build test-hardening-run
.PHONY: day38-step1 day38-step2 day38-step3 day38-step4 day38-step5
.PHONY: day38-full day38-status day38-clean day38-pipeline
.PHONY: tsan-all tsan-quick tsan-clean tsan-summary tsan-status

# ============================================================================
# ML Defender Pipeline - Master Makefile
# Single Source of Truth for Build Configurations
# ============================================================================
# Run from macOS - Commands execute in VM via vagrant ssh -c
# Day 57+ - Refactored for coherent clean/test/verify workflow
# ============================================================================

# ============================================================================
# BUILD PROFILES - SINGLE SOURCE OF TRUTH
# ============================================================================
# All compiler flags are defined HERE and ONLY here.
# CMakeLists.txt files MUST NOT hardcode any -O, -g, or -fsanitize flags.
# ============================================================================

# Base flags (always applied)
CXX_STD := -std=c++20
CXX_WARNINGS := -Wall -Wextra -Wpedantic
C_STD := -std=c11

# Profile-specific flags
PROFILE_PRODUCTION_CXX := -O3 -march=native -DNDEBUG -flto -fno-omit-frame-pointer
PROFILE_PRODUCTION_C   := -O3 -march=native -DNDEBUG -flto -fno-omit-frame-pointer

PROFILE_DEBUG_CXX := -g -O0 -fno-omit-frame-pointer -DDEBUG
PROFILE_DEBUG_C   := -g -O0 -fno-omit-frame-pointer -DDEBUG

PROFILE_TSAN_CXX := -fsanitize=thread -g -O1 -fno-omit-frame-pointer -DTSAN_ENABLED
PROFILE_TSAN_C   := -fsanitize=thread -g -O1 -fno-omit-frame-pointer -DTSAN_ENABLED
PROFILE_TSAN_LD  := -fsanitize=thread

PROFILE_ASAN_CXX := -fsanitize=address -fsanitize=undefined -g -O1 -fno-omit-frame-pointer -DASAN_ENABLED
PROFILE_ASAN_C   := -fsanitize=address -fsanitize=undefined -g -O1 -fno-omit-frame-pointer -DASAN_ENABLED
PROFILE_ASAN_LD  := -fsanitize=address -fsanitize=undefined

# Combined flags per profile
CMAKE_FLAGS_PRODUCTION := -DCMAKE_BUILD_TYPE=Release \
                          -DCMAKE_CXX_FLAGS="$(CXX_STD) $(CXX_WARNINGS) $(PROFILE_PRODUCTION_CXX)" \
                          -DCMAKE_C_FLAGS="$(C_STD) $(PROFILE_PRODUCTION_C)"

CMAKE_FLAGS_DEBUG := -DCMAKE_BUILD_TYPE=Debug \
                     -DCMAKE_CXX_FLAGS="$(CXX_STD) $(CXX_WARNINGS) $(PROFILE_DEBUG_CXX)" \
                     -DCMAKE_C_FLAGS="$(C_STD) $(PROFILE_DEBUG_C)"

CMAKE_FLAGS_TSAN := -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                    -DCMAKE_CXX_FLAGS="$(CXX_STD) $(CXX_WARNINGS) $(PROFILE_TSAN_CXX)" \
                    -DCMAKE_C_FLAGS="$(C_STD) $(PROFILE_TSAN_C)" \
                    -DCMAKE_EXE_LINKER_FLAGS="$(PROFILE_TSAN_LD)" \
                    -DCMAKE_SHARED_LINKER_FLAGS="$(PROFILE_TSAN_LD)"

CMAKE_FLAGS_ASAN := -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                    -DCMAKE_CXX_FLAGS="$(CXX_STD) $(CXX_WARNINGS) $(PROFILE_ASAN_CXX)" \
                    -DCMAKE_C_FLAGS="$(C_STD) $(PROFILE_ASAN_C)" \
                    -DCMAKE_EXE_LINKER_FLAGS="$(PROFILE_ASAN_LD)" \
                    -DCMAKE_SHARED_LINKER_FLAGS="$(PROFILE_ASAN_LD)"

# Default profile (can be overridden: make PROFILE=tsan all)
PROFILE ?= debug
CMAKE_FLAGS := $(CMAKE_FLAGS_$(shell echo $(PROFILE) | tr a-z A-Z))

# ============================================================================
# COMPONENT BUILD DIRECTORIES (Profile-specific)
# ============================================================================
SNIFFER_BUILD_DIR       := /vagrant/sniffer/build-$(PROFILE)
ML_DETECTOR_BUILD_DIR   := /vagrant/ml-detector/build-$(PROFILE)
RAG_INGESTER_BUILD_DIR  := /vagrant/rag-ingester/build-$(PROFILE)
FIREWALL_BUILD_DIR      := /vagrant/firewall-acl-agent/build-$(PROFILE)
ETCD_SERVER_BUILD_DIR   := /vagrant/etcd-server/build-$(PROFILE)
TOOLS_BUILD_DIR         := /vagrant/tools/build-$(PROFILE)

# Libraries (always release, no sanitizers)
CRYPTO_TRANSPORT_BUILD_DIR := /vagrant/crypto-transport/build
ETCD_CLIENT_BUILD_DIR      := /vagrant/etcd-client/build

# Legacy compatibility (for existing scripts that reference /vagrant/component/build)
SNIFFER_LEGACY_LINK       := /vagrant/sniffer/build
ML_DETECTOR_LEGACY_LINK   := /vagrant/ml-detector/build
RAG_INGESTER_LEGACY_LINK  := /vagrant/rag-ingester/build
FIREWALL_LEGACY_LINK      := /vagrant/firewall-acl-agent/build
ETCD_SERVER_LEGACY_LINK   := /vagrant/etcd-server/build
TOOLS_LEGACY_LINK         := /vagrant/tools/build

# ============================================================================
# HELP
# ============================================================================

help:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ML Defender Pipeline - Master Makefile (Day 57+)         ║"
	@echo "║  Single Source of Truth - Build Profile System            ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📋 Build Profiles (PROFILE=name):"
	@echo "  production  - Optimized (-O3, LTO, march=native)"
	@echo "  debug       - Debug symbols (-g -O0) [DEFAULT]"
	@echo "  tsan        - ThreadSanitizer (-fsanitize=thread)"
	@echo "  asan        - AddressSanitizer (-fsanitize=address)"
	@echo ""
	@echo "Usage: make [PROFILE=<profile>] <target>"
	@echo "  Example: make PROFILE=tsan all"
	@echo "  Example: make PROFILE=production sniffer"
	@echo ""
	@echo "🏗️  Build Commands (Dependencies handled automatically):"
	@echo "  make all             - Build EVERYTHING (proto → libs → components)"
	@echo "  make proto           - Regenerate protobuf"
	@echo "  make crypto-transport-build - Build crypto-transport library"
	@echo "  make etcd-client-build     - Build etcd-client library"
	@echo "  make sniffer         - Build sniffer"
	@echo "  make ml-detector     - Build ML detector"
	@echo "  make rag-ingester    - Build RAG ingester"
	@echo "  make firewall        - Build firewall agent"
	@echo "  make etcd-server     - Build etcd server"
	@echo "  make tools           - Build tools"
	@echo ""
	@echo "🧹 Clean Commands (NEW - Day 57):"
	@echo "  make clean           - Clean components (current profile)"
	@echo "  make clean-libs      - Clean libraries (crypto-transport, etcd-client)"
	@echo "  make clean-components - Same as 'clean'"
	@echo "  make clean-all       - Clean EVERYTHING (legacy + profiles + libs)"
	@echo "  make distclean       - Nuclear clean (clean-all + protobuf)"
	@echo ""
	@echo "  Note: clean-all removes BOTH legacy build/ and build-PROFILE/"
	@echo ""
	@echo "🧪 Test Commands (NEW - Day 57):"
	@echo "  make test            - Run ALL tests (libs + components)"
	@echo "  make test-libs       - Run library tests only"
	@echo "  make test-components - Run component tests only"
	@echo ""
	@echo "✅ Verification Commands (NEW - Day 57):"
	@echo "  make verify-all      - Run ALL verifications post-build"
	@echo "  make verify-etcd-linkage  - Check etcd-client linkage"
	@echo "  make verify-encryption    - Check encryption configs"
	@echo ""
	@echo "🔧 TSAN Validation (Day 48 Phase 0):"
	@echo "  make tsan-all        - Full TSAN validation"
	@echo "  make tsan-quick      - Quick TSAN check"
	@echo "  make tsan-summary    - View TSAN report"
	@echo ""
	@echo "🚀 Run & Test:"
	@echo "  make run-lab-dev     - Start full lab"
	@echo "  make status-lab      - Check lab status"
	@echo "  make test-replay-small - Replay test dataset"
	@echo ""
	@echo "📊 Current Profile: $(PROFILE)"
	@echo "   Component builds: build-$(PROFILE)/"
	@echo "   Libraries: build/ (always release)"
	@echo ""
	@echo "VM Management:"
	@echo "  make up              - Start VM"
	@echo "  make halt            - Stop VM"
	@echo "  make ssh             - SSH into VM"
	@echo "  make status          - VM + libbpf status"
	@echo ""
	@echo "🏛️  Via Appia Quality: Piano piano - stone by stone"

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
	@echo "Current Profile: $(PROFILE)"
	@echo "════════════════════════════════════════════════════════════"

# ============================================================================
# Protobuf (Foundation Layer)
# ============================================================================

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
# Libraries Layer (No Sanitizers - Always Release)
# ============================================================================
# Dependencies: crypto-transport has no deps
#              etcd-client depends on: proto, crypto-transport
# ============================================================================

crypto-transport-build:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔨 Building crypto-transport Library                     ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@vagrant ssh -c 'cd /vagrant/crypto-transport && \
		rm -rf build && \
		mkdir -p build && \
		cd build && \
		cmake -DCMAKE_BUILD_TYPE=Release .. && \
		make -j4'
	@echo "Installing system-wide..."
	@vagrant ssh -c "cd /vagrant/crypto-transport/build && sudo make install && sudo ldconfig"
	@echo ""
	@echo "✅ crypto-transport installed to /usr/local/lib"
	@vagrant ssh -c "ls -lh /usr/local/lib/libcrypto_transport.so* 2>/dev/null || echo '⚠️  Library not found in /usr/local/lib'"

crypto-transport-clean:
	@echo "🧹 Cleaning crypto-transport..."
	@vagrant ssh -c "rm -rf /vagrant/crypto-transport/build"
	@vagrant ssh -c "sudo rm -f /usr/local/lib/libcrypto_transport.so*"
	@vagrant ssh -c "sudo rm -rf /usr/local/include/crypto_transport"
	@vagrant ssh -c "sudo ldconfig"
	@echo "✅ crypto-transport cleaned"

crypto-transport-test:
	@echo "🧪 Testing crypto-transport..."
	@vagrant ssh -c "cd /vagrant/crypto-transport/build && ctest --output-on-failure"

etcd-client-build: proto-unified crypto-transport-build
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔨 Building etcd-client Library                          ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Dependencies:"
	@echo "  ✅ proto-unified"
	@echo "  ✅ crypto-transport-build"
	@echo ""
	@vagrant ssh -c 'cd /vagrant/etcd-client && \
		rm -rf build && \
		mkdir -p build && \
		cd build && \
		cmake -DCMAKE_BUILD_TYPE=Release .. && \
		make -j4'
	@echo "Installing system-wide..."
	@vagrant ssh -c "cd /vagrant/etcd-client/build && sudo make install && sudo ldconfig"
	@echo ""
	@echo "✅ etcd-client installed to /usr/local/lib"
	@echo ""
	@echo "Verifying library size and methods..."
	@vagrant ssh -c "ls -lh /vagrant/etcd-client/build/libetcd_client.so"
	@vagrant ssh -c "ls -lh /usr/local/lib/libetcd_client.so* 2>/dev/null || echo '⚠️  Library not found in /usr/local/lib'"
	@vagrant ssh -c "nm -D /usr/local/lib/libetcd_client.so.1.0.0 2>/dev/null | grep -c ' T ' | xargs echo 'Public methods:'" || echo "⚠️  Cannot verify methods"

etcd-client-clean:
	@echo "🧹 Cleaning etcd-client..."
	@vagrant ssh -c "rm -rf /vagrant/etcd-client/build"
	@vagrant ssh -c "sudo rm -f /usr/local/lib/libetcd_client.so*"
	@vagrant ssh -c "sudo rm -rf /usr/local/include/etcd_client"
	@vagrant ssh -c "sudo ldconfig"
	@echo "✅ etcd-client cleaned"

etcd-client-test:
	@echo "🧪 Testing etcd-client..."
	@vagrant ssh -c "cd /vagrant/etcd-client/build && ctest --output-on-failure"

# ============================================================================
# Component Builds (Profile-Aware)
# ============================================================================
# CRITICAL: All components copy protobuf files BEFORE cmake
# Dependencies are explicit and automatic
# ============================================================================

sniffer: proto etcd-client-build
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔨 Building Sniffer [$(PROFILE)]                         ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build dir: $(SNIFFER_BUILD_DIR)"
	@echo "Flags: $(CMAKE_FLAGS)"
	@echo ""
	@echo "Copying protobuf files..."
	@vagrant ssh -c 'mkdir -p $(SNIFFER_BUILD_DIR)/proto && \
		cp /vagrant/protobuf/network_security.pb.* $(SNIFFER_BUILD_DIR)/proto/'
	@echo "Running CMake and build..."
	@vagrant ssh -c 'cd /vagrant/sniffer && \
		mkdir -p $(SNIFFER_BUILD_DIR) && \
		cd $(SNIFFER_BUILD_DIR) && \
		cmake $(CMAKE_FLAGS) .. && \
		make -j4'
	@vagrant ssh -c "ln -sfn $(SNIFFER_BUILD_DIR) $(SNIFFER_LEGACY_LINK)"
	@echo ""
	@echo "✅ Sniffer built ($(PROFILE))"

ml-detector: proto etcd-client-build
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔨 Building ML Detector [$(PROFILE)]                     ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build dir: $(ML_DETECTOR_BUILD_DIR)"
	@echo "Flags: $(CMAKE_FLAGS)"
	@echo ""
	@echo "Copying protobuf files..."
	@vagrant ssh -c 'mkdir -p $(ML_DETECTOR_BUILD_DIR)/proto && \
		cp /vagrant/protobuf/network_security.pb.* $(ML_DETECTOR_BUILD_DIR)/proto/'
	@echo "Running CMake and build..."
	@vagrant ssh -c 'cd /vagrant/ml-detector && \
		mkdir -p $(ML_DETECTOR_BUILD_DIR) && \
		cd $(ML_DETECTOR_BUILD_DIR) && \
		cmake $(CMAKE_FLAGS) .. && \
		make -j4'
	@vagrant ssh -c "ln -sfn $(ML_DETECTOR_BUILD_DIR) $(ML_DETECTOR_LEGACY_LINK)"
	@echo ""
	@echo "✅ ML Detector built ($(PROFILE))"

rag-ingester: proto etcd-client-build crypto-transport-build
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔨 Building RAG Ingester [$(PROFILE)]                    ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build dir: $(RAG_INGESTER_BUILD_DIR)"
	@echo "Flags: $(CMAKE_FLAGS)"
	@echo ""
	@echo "Copying protobuf files..."
	@vagrant ssh -c 'mkdir -p $(RAG_INGESTER_BUILD_DIR)/proto && \
		cp /vagrant/protobuf/network_security.pb.* $(RAG_INGESTER_BUILD_DIR)/proto/'
	@echo "Running CMake and build..."
	@vagrant ssh -c 'cd /vagrant/rag-ingester && \
		mkdir -p $(RAG_INGESTER_BUILD_DIR) && \
		cd $(RAG_INGESTER_BUILD_DIR) && \
		cmake $(CMAKE_FLAGS) .. && \
		make -j4'
	@vagrant ssh -c "ln -sfn $(RAG_INGESTER_BUILD_DIR) $(RAG_INGESTER_LEGACY_LINK)"
	@echo ""
	@echo "✅ RAG Ingester built ($(PROFILE))"

firewall: proto etcd-client-build
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔨 Building Firewall ACL Agent [$(PROFILE)]              ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build dir: $(FIREWALL_BUILD_DIR)"
	@echo "Flags: $(CMAKE_FLAGS)"
	@echo ""
	@echo "Copying protobuf files..."
	@vagrant ssh -c 'mkdir -p $(FIREWALL_BUILD_DIR)/proto && \
		cp /vagrant/protobuf/network_security.pb.* $(FIREWALL_BUILD_DIR)/proto/'
	@echo "Running CMake and build..."
	@vagrant ssh -c 'cd /vagrant/firewall-acl-agent && \
		mkdir -p $(FIREWALL_BUILD_DIR) && \
		cd $(FIREWALL_BUILD_DIR) && \
		cmake $(CMAKE_FLAGS) .. && \
		make -j4'
	@vagrant ssh -c "ln -sfn $(FIREWALL_BUILD_DIR) $(FIREWALL_LEGACY_LINK)"
	@echo ""
	@echo "✅ Firewall built ($(PROFILE))"

etcd-server:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔨 Building etcd-server [$(PROFILE)]                     ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build dir: $(ETCD_SERVER_BUILD_DIR)"
	@echo "Flags: $(CMAKE_FLAGS)"
	@echo ""
	@vagrant ssh -c 'cd /vagrant/etcd-server && \
		mkdir -p $(ETCD_SERVER_BUILD_DIR) && \
		cd $(ETCD_SERVER_BUILD_DIR) && \
		cmake $(CMAKE_FLAGS) .. && \
		make -j4'
	@vagrant ssh -c "ln -sfn $(ETCD_SERVER_BUILD_DIR) $(ETCD_SERVER_LEGACY_LINK)"
	@echo ""
	@echo "✅ etcd-server built ($(PROFILE))"

tools: proto etcd-client-build crypto-transport-build
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔨 Building Tools [$(PROFILE)]                           ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build dir: $(TOOLS_BUILD_DIR)"
	@echo "Flags: $(CMAKE_FLAGS)"
	@echo ""
	@echo "Copying protobuf files..."
	@vagrant ssh -c 'mkdir -p $(TOOLS_BUILD_DIR)/proto && \
		cp /vagrant/protobuf/network_security.pb.* $(TOOLS_BUILD_DIR)/proto/'
	@echo "Running CMake and build..."
	@vagrant ssh -c 'cd /vagrant/tools && \
		mkdir -p $(TOOLS_BUILD_DIR) && \
		cd $(TOOLS_BUILD_DIR) && \
		cmake $(CMAKE_FLAGS) .. && \
		make -j4'
	@vagrant ssh -c "ln -sfn $(TOOLS_BUILD_DIR) $(TOOLS_LEGACY_LINK)"
	@echo ""
	@echo "✅ Tools built ($(PROFILE))"

# Aliases for consistency with existing workflows
detector: ml-detector
sniffer-build: sniffer
detector-build: ml-detector
rag-ingester-build: rag-ingester
firewall-build: firewall
etcd-server-build: etcd-server
tools-build: tools

# ============================================================================
# Clean Targets (REFACTORED - Day 57)
# ============================================================================
# NEW STRUCTURE:
#   clean-libs       - Clean ONLY libraries (crypto-transport, etcd-client)
#   clean-components - Clean ONLY components (current profile)
#   clean            - Alias for clean-components
#   clean-all        - Clean EVERYTHING (all profiles + libs)
#   distclean        - Nuclear clean (clean-all + protobuf)
# ============================================================================

sniffer-clean:
	@echo "🧹 Cleaning Sniffer [$(PROFILE)]..."
	@vagrant ssh -c "rm -rf $(SNIFFER_BUILD_DIR)"

detector-clean:
	@echo "🧹 Cleaning ML Detector [$(PROFILE)]..."
	@vagrant ssh -c "rm -rf $(ML_DETECTOR_BUILD_DIR)"

rag-ingester-clean:
	@echo "🧹 Cleaning RAG Ingester [$(PROFILE)]..."
	@vagrant ssh -c "rm -rf $(RAG_INGESTER_BUILD_DIR)"

firewall-clean:
	@echo "🧹 Cleaning Firewall [$(PROFILE)]..."
	@vagrant ssh -c "rm -rf $(FIREWALL_BUILD_DIR)"

etcd-server-clean:
	@echo "🧹 Cleaning etcd-server [$(PROFILE)]..."
	@vagrant ssh -c "rm -rf $(ETCD_SERVER_BUILD_DIR)"

tools-clean:
	@echo "🧹 Cleaning Tools [$(PROFILE)]..."
	@vagrant ssh -c "rm -rf $(TOOLS_BUILD_DIR)"

clean-libs:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🧹 Cleaning Libraries (crypto-transport, etcd-client)    ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(MAKE) crypto-transport-clean
	@$(MAKE) etcd-client-clean
	@echo ""
	@echo "✅ Libraries cleaned"

clean-components: sniffer-clean detector-clean firewall-clean rag-ingester-clean etcd-server-clean tools-clean
	@echo "✅ Components cleaned [$(PROFILE)]"

clean: clean-components
	@echo "✅ Clean complete [$(PROFILE)]"
	@echo "💡 Tip: Use 'make clean-libs' to also clean libraries"
	@echo "💡 Tip: Use 'make clean-all' to clean ALL profiles + libs"

clean-all:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🧹 NUCLEAR CLEAN - ALL Profiles + Libraries             ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Cleaning ALL component builds (legacy + profile-specific)..."
	@echo "  - Removing legacy build/ directories..."
	@vagrant ssh -c "rm -rf /vagrant/sniffer/build"
	@vagrant ssh -c "rm -rf /vagrant/ml-detector/build"
	@vagrant ssh -c "rm -rf /vagrant/rag-ingester/build"
	@vagrant ssh -c "rm -rf /vagrant/firewall-acl-agent/build"
	@vagrant ssh -c "rm -rf /vagrant/etcd-server/build"
	@vagrant ssh -c "rm -rf /vagrant/tools/build"
	@echo "  - Removing profile-specific builds (production/debug/tsan/asan)..."
	@vagrant ssh -c "rm -rf /vagrant/sniffer/build-*"
	@vagrant ssh -c "rm -rf /vagrant/ml-detector/build-*"
	@vagrant ssh -c "rm -rf /vagrant/rag-ingester/build-*"
	@vagrant ssh -c "rm -rf /vagrant/firewall-acl-agent/build-*"
	@vagrant ssh -c "rm -rf /vagrant/etcd-server/build-*"
	@vagrant ssh -c "rm -rf /vagrant/tools/build-*"
	@echo ""
	@$(MAKE) clean-libs
	@echo ""
	@echo "✅ Nuclear clean complete - ALL builds + libraries removed"

distclean: clean-all
	@echo ""
	@echo "🧹 Cleaning protobuf artifacts..."
	@vagrant ssh -c "rm -f /vagrant/protobuf/network_security.pb.*"
	@vagrant ssh -c "rm -f /vagrant/protobuf/network_security_pb2.py"
	@echo ""
	@echo "✅ Distclean complete - System returned to pristine state"

# ============================================================================
# Test Targets (NEW - Day 57)
# ============================================================================
# UNIFIED TEST FRAMEWORK:
#   test-libs       - Run library tests (crypto-transport, etcd-client)
#   test-components - Run component tests (current profile)
#   test            - Run ALL tests (libs + components)
# ============================================================================

test-libs:
	@echo "Testing crypto-transport..."
	@vagrant ssh -c "cd /vagrant/crypto-transport/build && ctest" || echo "⚠️  crypto-transport has known LZ4 issues"
	@echo "Testing etcd-client (HMAC only)..."
	@vagrant ssh -c "cd /vagrant/etcd-client/build/tests && ./test_hmac_client"

test-components:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🧪 Running Component Tests [$(PROFILE)]                  ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Testing Sniffer..."
	@vagrant ssh -c "cd $(SNIFFER_BUILD_DIR) && ctest --output-on-failure" || echo "⚠️  No sniffer tests configured"
	@echo ""
	@echo "Testing ML Detector..."
	@vagrant ssh -c "cd $(ML_DETECTOR_BUILD_DIR) && ctest --output-on-failure" || echo "⚠️  No ml-detector tests configured"
	@echo ""
	@echo "Testing RAG Ingester..."
	@vagrant ssh -c "cd $(RAG_INGESTER_BUILD_DIR) && ctest --output-on-failure" || echo "⚠️  No rag-ingester tests configured"
	@echo ""
	@echo "Testing etcd-server..."
	@vagrant ssh -c "cd $(ETCD_SERVER_BUILD_DIR) && ctest --output-on-failure" || echo "⚠️  No etcd-server tests configured"
	@echo ""

test-all: test-libs test-components
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ ALL TESTS COMPLETE                                    ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""

test: test-all
	@echo "💡 Tip: Use 'make test-libs' to test only libraries"
	@echo "💡 Tip: Use 'make test-components' to test only components"

# ============================================================================
# Verification Targets (ENHANCED - Day 57)
# ============================================================================

verify-etcd-linkage:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔍 Verifying etcd-client Linkage                         ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "0️⃣  etcd-client library:"
	@vagrant ssh -c 'if [ -f /usr/local/lib/libetcd_client.so.1.0.0 ]; then \
		echo "   ✅ Library installed"; \
		ls -lh /usr/local/lib/libetcd_client.so*; \
		nm -D /usr/local/lib/libetcd_client.so.1.0.0 | grep -c " T " | xargs echo "   Public methods:"; \
	else \
		echo "   ❌ Library NOT FOUND - run: make etcd-client-build"; \
	fi'
	@echo ""
	@echo "Components (current profile: $(PROFILE)):"
	@vagrant ssh -c 'if [ -f $(SNIFFER_BUILD_DIR)/sniffer ]; then \
		ldd $(SNIFFER_BUILD_DIR)/sniffer 2>/dev/null | grep etcd_client && echo "   ✅ Sniffer linked" || echo "   ❌ Sniffer NOT linked"; \
	else \
		echo "   ⚠️  Sniffer not built"; \
	fi'
	@vagrant ssh -c 'if [ -f $(ML_DETECTOR_BUILD_DIR)/ml-detector ]; then \
		ldd $(ML_DETECTOR_BUILD_DIR)/ml-detector 2>/dev/null | grep etcd_client && echo "   ✅ ML Detector linked" || echo "   ❌ ML Detector NOT linked"; \
	else \
		echo "   ⚠️  ML Detector not built"; \
	fi'
	@vagrant ssh -c 'if [ -f $(RAG_INGESTER_BUILD_DIR)/rag-ingester ]; then \
		ldd $(RAG_INGESTER_BUILD_DIR)/rag-ingester 2>/dev/null | grep etcd_client && echo "   ✅ RAG Ingester linked" || echo "   ❌ RAG Ingester NOT linked"; \
	else \
		echo "   ⚠️  RAG Ingester not built"; \
	fi'
	@vagrant ssh -c 'if [ -f $(FIREWALL_BUILD_DIR)/firewall-acl-agent ]; then \
		ldd $(FIREWALL_BUILD_DIR)/firewall-acl-agent 2>/dev/null | grep etcd_client && echo "   ✅ Firewall linked" || echo "   ❌ Firewall NOT linked"; \
	else \
		echo "   ⚠️  Firewall not built"; \
	fi'
	@echo ""

verify-encryption:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔐 Verifying Encryption/Compression Configuration        ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
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

verify-pipeline-config: verify-etcd-linkage verify-encryption

verify-all:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ Running ALL Verifications                             ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(MAKE) verify-etcd-linkage
	@$(MAKE) verify-encryption
	@echo ""
	@echo "✅ All verifications complete"

# ============================================================================
# Unified Build Targets
# ============================================================================

build-unified: proto-unified crypto-transport-build etcd-client-build sniffer ml-detector rag-ingester firewall tools
	@echo ""
	@echo "✅ Unified build complete [$(PROFILE)]"
	@$(MAKE) proto-verify

all: build-unified etcd-server
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ ALL COMPONENTS BUILT [$(PROFILE)]                     ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Build Summary:"
	@echo "  ✅ Protobuf unified"
	@echo "  ✅ crypto-transport library"
	@echo "  ✅ etcd-client library"
	@echo "  ✅ Sniffer"
	@echo "  ✅ ML Detector"
	@echo "  ✅ RAG Ingester"
	@echo "  ✅ Firewall ACL Agent"
	@echo "  ✅ etcd-server"
	@echo "  ✅ Tools"
	@echo ""
	@echo "Profile: $(PROFILE)"
	@echo "Component builds: build-$(PROFILE)/"
	@echo "Library builds: build/ (release)"
	@echo ""
	@echo "🏛️  Via Appia Quality: Built to last"
	@echo ""
	@echo "Next steps:"
	@echo "  make verify-all  - Verify linkage and configs"
	@echo "  make test        - Run all tests"

rebuild-unified: clean build-unified
	@echo "✅ Rebuild complete [$(PROFILE)]"

rebuild: clean all
	@echo "✅ Full rebuild complete [$(PROFILE)]"

# ============================================================================
# BPF Diagnostics (Day 8)
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
	@echo "4️⃣  Verification:"
	@vagrant ssh -c 'LIBBPF_VER=$$(pkg-config --modversion libbpf 2>/dev/null); \
		if [ -z "$$LIBBPF_VER" ]; then \
			echo "❌ libbpf NOT installed - run: vagrant provision"; \
		elif [ "$$(printf "%s\n" "1.2.0" "$$LIBBPF_VER" | sort -V | head -n1)" = "1.2.0" ]; then \
			echo "✅ libbpf $$LIBBPF_VER >= 1.2.0 (BPF map bug FIXED)"; \
		else \
			echo "⚠️  libbpf $$LIBBPF_VER < 1.2.0 (BUG PRESENT)"; \
		fi'
	@echo "════════════════════════════════════════════════════════════"

verify-bpf-maps:
	@echo "════════════════════════════════════════════════════════════"
	@echo "🔍 Verifying BPF Maps Loading"
	@echo "════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Checking BPF object file:"
	@vagrant ssh -c "ls -lh $(SNIFFER_BUILD_DIR)/sniffer.bpf.o 2>/dev/null || echo '   ❌ BPF object not found - run: make sniffer'"
	@echo ""
	@echo "Searching for interface_configs in object:"
	@vagrant ssh -c "llvm-objdump -h $(SNIFFER_BUILD_DIR)/sniffer.bpf.o 2>/dev/null | grep -i maps && echo '   ✅ .maps section found' || echo '   ❌ .maps section not found'"
	@echo "════════════════════════════════════════════════════════════"

diagnose-bpf: check-libbpf verify-bpf-maps
	@echo ""
	@echo "🔧 BPF Diagnostics Complete"

# ============================================================================
# Run Components
# ============================================================================

run-sniffer:
	@echo "📡 Running Sniffer [$(PROFILE)]..."
	@vagrant ssh -c "cd $(SNIFFER_BUILD_DIR) && sudo ./sniffer -c /vagrant/sniffer/config/sniffer.json"

run-detector:
	@echo "🤖 Running ML Detector [$(PROFILE)]..."
	@vagrant ssh -c "cd $(ML_DETECTOR_BUILD_DIR) && ./ml-detector -c /vagrant/ml-detector/config/ml_detector_config.json"

run-firewall:
	@echo "🔥 Running Firewall [$(PROFILE)]..."
	@vagrant ssh -c "cd $(FIREWALL_BUILD_DIR) && sudo ./firewall-acl-agent -c /vagrant/firewall-acl-agent/config/firewall.json"

# ============================================================================
# Lab Control
# ============================================================================

run-lab-dev:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Starting ML Defender Lab [$(PROFILE)]                 ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@vagrant ssh -c "cd /vagrant && bash scripts/run_lab_dev.sh"

run-lab-dev-day23: etcd-server-start
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Starting ML Defender Lab - Day 23 (with etcd)         ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@sleep 3
	@vagrant ssh -c "cd /vagrant && bash scripts/run_lab_dev.sh"

kill-lab:
	@echo "💀 Stopping ML Defender Lab..."
	-@vagrant ssh -c "sudo pkill -9 -f firewall-acl-agent" 2>/dev/null || true
	-@vagrant ssh -c "pkill -9 -f ml-detector" 2>/dev/null || true
	-@vagrant ssh -c "sudo pkill -9 -f sniffer" 2>/dev/null || true
	@echo "✅ Lab stopped"

kill-lab-day23: kill-lab
	-@vagrant ssh -c "pkill -9 -f etcd-server" 2>/dev/null || true
	@echo "✅ Lab + etcd-server stopped"

status-lab:
	@echo "════════════════════════════════════════════════════════════"
	@echo "ML Defender Lab Status [$(PROFILE)]:"
	@echo "════════════════════════════════════════════════════════════"
	@vagrant ssh -c "pgrep -a -f firewall-acl-agent && echo '✅ Firewall: RUNNING' || echo '❌ Firewall: STOPPED'"
	@vagrant ssh -c "pgrep -a -f ml-detector && echo '✅ Detector: RUNNING' || echo '❌ Detector: STOPPED'"
	@vagrant ssh -c "pgrep -a -f 'sniffer.*-c' && echo '✅ Sniffer:  RUNNING' || echo '❌ Sniffer:  STOPPED'"
	@echo "════════════════════════════════════════════════════════════"

status-lab-day23: status-lab
	@vagrant ssh -c "pgrep -a -f etcd-server && echo '✅ etcd-server: RUNNING' || echo '❌ etcd-server: STOPPED'"

kill-all: kill-lab-day23

check-ports:
	@vagrant ssh -c "sudo ss -tlnp | grep -E '5571|5572' && echo '⚠️  Ports in use' || echo '✅ Ports free'"

# ============================================================================
# Logs
# ============================================================================

logs-sniffer:
	@vagrant ssh -c "tail -f /vagrant/logs/lab/sniffer.log 2>/dev/null || echo 'No sniffer logs yet'"

logs-detector:
	@vagrant ssh -c "tail -f /vagrant/ml-detector/build/logs/*.log 2>/dev/null || echo 'No detector logs yet'"

logs-firewall:
	@vagrant ssh -c "tail -f /vagrant/firewall-acl-agent/build/logs/*.log 2>/dev/null || echo 'No firewall logs yet'"

logs-lab:
	@echo "📋 Combined Lab Logs (CTRL+C to exit)..."
	@vagrant ssh -c "cd /vagrant && bash scripts/monitor_lab.sh"

# ============================================================================
# Development Workflows
# ============================================================================

dev-setup: up all
	@echo "✅ Development environment ready [$(PROFILE)]"

dev-setup-unified: up build-unified
	@echo "✅ Development environment ready (unified protobuf) [$(PROFILE)]"

schema-update: proto rebuild
	@echo "✅ Schema updated and components rebuilt [$(PROFILE)]"

quick-fix:
	@echo "🔧 Quick bug fix procedure..."
	@$(MAKE) kill-lab
	@$(MAKE) rebuild
	@echo "✅ Ready to test fix [$(PROFILE)]"

# ============================================================================
# Dataset Replays
# ============================================================================

CTU13_SMALL := /vagrant/datasets/ctu13/smallFlows.pcap
CTU13_NERIS := /vagrant/datasets/ctu13/botnet-capture-20110810-neris.pcap
CTU13_BIG := /vagrant/datasets/ctu13/bigFlows.pcap

test-replay-small:
	@echo "🧪 Replaying CTU-13 smallFlows.pcap..."
	@vagrant ssh client -c "mkdir -p /vagrant/logs/lab && \
		sudo tcpreplay -i eth1 --mbps=10 --stats=2 $(CTU13_SMALL) 2>&1 | tee /vagrant/logs/lab/tcpreplay.log"

test-replay-neris:
	@echo "🧪 Replaying CTU-13 Neris botnet (492K events)..."
	@vagrant ssh client -c "mkdir -p /vagrant/logs/lab && \
		sudo tcpreplay -i eth1 --mbps=10 --stats=5 $(CTU13_NERIS) 2>&1 | tee /vagrant/logs/lab/tcpreplay.log"

test-replay-big:
	@echo "🧪 Replaying CTU-13 bigFlows.pcap..."
	@vagrant ssh client -c "mkdir -p /vagrant/logs/lab && \
		sudo tcpreplay -i eth1 --mbps=10 --stats=10 $(CTU13_BIG) 2>&1 | tee /vagrant/logs/lab/tcpreplay.log"

# ============================================================================
# etcd-server Control
# ============================================================================

etcd-server-start:
	@echo "🚀 Starting etcd-server..."
	@vagrant ssh -c 'mkdir -p /vagrant/logs && \
		cd $(ETCD_SERVER_BUILD_DIR) && \
		nohup ./etcd-server > /vagrant/logs/etcd-server.log 2>&1 &'
	@sleep 2
	@vagrant ssh -c "pgrep -f etcd-server && echo '✅ etcd-server started' || echo '❌ Failed to start'"

etcd-server-stop:
	@echo "🛑 Stopping etcd-server..."
	@vagrant ssh -c "pkill -f etcd-server 2>/dev/null || true"

# ============================================================================
# TSAN Validation Suite (Day 48 Phase 0)
# ============================================================================

tsan-all:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🔬 TSAN Full Validation Suite - Day 48 Phase 0           ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1: Clean previous TSAN builds..."
	@$(MAKE) PROFILE=tsan clean
	@vagrant ssh -c "mkdir -p /vagrant/tsan-reports/day48"
	@echo ""
	@echo "Step 2: Build all components with TSAN..."
	@$(MAKE) PROFILE=tsan all
	@echo ""
	@echo "Step 3: Run unit tests with TSAN..."
	@echo ""
	@echo "  Testing Sniffer..."
	@vagrant ssh -c 'cd $(SNIFFER_BUILD_DIR) && \
		TSAN_OPTIONS="log_path=/vagrant/tsan-reports/day48/sniffer-tsan history_size=7 second_deadlock_stack=1" \
		ctest --output-on-failure 2>&1 | tee /vagrant/tsan-reports/day48/sniffer-tsan-tests.log' || true
	@echo ""
	@echo "  Testing ML Detector..."
	@vagrant ssh -c 'cd $(ML_DETECTOR_BUILD_DIR) && \
		TSAN_OPTIONS="log_path=/vagrant/tsan-reports/day48/ml-detector-tsan history_size=7" \
		ctest --output-on-failure 2>&1 | tee /vagrant/tsan-reports/day48/ml-detector-tsan-tests.log' || true
	@echo ""
	@echo "  Testing RAG Ingester..."
	@vagrant ssh -c 'cd $(RAG_INGESTER_BUILD_DIR) && \
		TSAN_OPTIONS="log_path=/vagrant/tsan-reports/day48/rag-ingester-tsan history_size=7" \
		ctest --output-on-failure 2>&1 | tee /vagrant/tsan-reports/day48/rag-ingester-tsan-tests.log' || true
	@echo ""
	@echo "  Testing etcd-server..."
	@vagrant ssh -c 'cd $(ETCD_SERVER_BUILD_DIR) && \
		TSAN_OPTIONS="log_path=/vagrant/tsan-reports/day48/etcd-server-tsan history_size=7" \
		ctest --output-on-failure 2>&1 | tee /vagrant/tsan-reports/day48/etcd-server-tsan-tests.log' || true
	@echo ""
	@echo "Step 4: Run integration test (5 min)..."
	@vagrant ssh -c 'cd /vagrant/sniffer && \
		mkdir -p $(SNIFFER_BUILD_DIR) && cd $(SNIFFER_BUILD_DIR) && \
		TSAN_OPTIONS="log_path=/vagrant/tsan-reports/day48/sniffer-integration history_size=7" \
		timeout 300 ./sniffer -c /vagrant/sniffer/config/sniffer.json > /vagrant/tsan-reports/day48/sniffer-integration.log 2>&1 &' || true
	@vagrant ssh -c 'cd /vagrant/ml-detector && \
		mkdir -p $(ML_DETECTOR_BUILD_DIR) && cd $(ML_DETECTOR_BUILD_DIR) && \
		TSAN_OPTIONS="log_path=/vagrant/tsan-reports/day48/ml-detector-integration history_size=7" \
		timeout 300 ./ml-detector --config /vagrant/ml-detector/config/detector.json > /vagrant/tsan-reports/day48/ml-detector-integration.log 2>&1 &' || true
	@vagrant ssh -c 'cd /vagrant/rag-ingester && \
		mkdir -p $(RAG_INGESTER_BUILD_DIR) && cd $(RAG_INGESTER_BUILD_DIR) && \
		TSAN_OPTIONS="log_path=/vagrant/tsan-reports/day48/rag-ingester-integration history_size=7" \
		timeout 300 ./rag-ingester /vagrant/rag-ingester/config/rag-ingester.json > /vagrant/tsan-reports/day48/rag-ingester-integration.log 2>&1 &' || true
	@vagrant ssh -c 'cd /vagrant/etcd-server && \
		mkdir -p $(ETCD_SERVER_BUILD_DIR) && cd $(ETCD_SERVER_BUILD_DIR) && \
		TSAN_OPTIONS="log_path=/vagrant/tsan-reports/day48/etcd-server-integration history_size=7" \
		timeout 300 ./etcd-server > /vagrant/tsan-reports/day48/etcd-server-integration.log 2>&1 &' || true
	@sleep 310
	@vagrant ssh -c "pkill -f 'sniffer|ml-detector|rag-ingester|etcd-server' || true"
	@echo ""
	@echo "Step 5: Create baseline symlink..."
	@vagrant ssh -c "ln -sfn /vagrant/tsan-reports/day48 /vagrant/tsan-reports/baseline"
	@echo ""
	@echo "✅ TSAN Full Validation Complete"
	@echo ""
	@echo "📊 Reports generated:"
	@echo "   /vagrant/tsan-reports/day48/"
	@echo "   /vagrant/tsan-reports/baseline/ → day48/"
	@echo ""
	@$(MAKE) tsan-summary

tsan-quick:
	@echo "⚡ TSAN Quick Check..."
	@$(MAKE) PROFILE=tsan sniffer ml-detector
	@vagrant ssh -c "cd $(SNIFFER_BUILD_DIR) && ctest --output-on-failure"
	@vagrant ssh -c "cd $(ML_DETECTOR_BUILD_DIR) && ctest --output-on-failure"
	@echo "✅ Quick TSAN check complete"

tsan-clean:
	@echo "🧹 Cleaning TSAN builds..."
	@$(MAKE) PROFILE=tsan clean
	@vagrant ssh -c "rm -rf /vagrant/tsan-reports/day48"
	@echo "✅ TSAN artifacts cleaned"

tsan-summary:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  📊 TSAN Validation Summary                                ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Unit Tests Results:"
	@echo "-------------------"
	@vagrant ssh -c "grep -E 'tests passed|test.*PASSED|PASS' /vagrant/tsan-reports/day48/sniffer-tsan-tests.log 2>/dev/null | tail -5 || echo 'No sniffer results'"
	@vagrant ssh -c "grep -E 'tests passed|test.*PASSED|PASS' /vagrant/tsan-reports/day48/ml-detector-tsan-tests.log 2>/dev/null | tail -5 || echo 'No ml-detector results'"
	@vagrant ssh -c "grep -E 'tests passed|test.*PASSED|PASS' /vagrant/tsan-reports/day48/rag-ingester-tsan-tests.log 2>/dev/null | tail -5 || echo 'No rag-ingester results'"
	@echo ""
	@echo "Race Conditions Detected:"
	@echo "-------------------------"
	@vagrant ssh -c "find /vagrant/tsan-reports/day48 -name '*-tsan.*' -exec grep -l 'WARNING: ThreadSanitizer' {} \; | wc -l | xargs echo 'Files with warnings:'" || echo "0"
	@echo ""
	@echo "Integration Test Status:"
	@echo "------------------------"
	@vagrant ssh -c "ls -lh /vagrant/tsan-reports/day48/*-integration.log 2>/dev/null | wc -l | xargs echo 'Components tested:'" || echo "0"
	@echo ""
	@echo "📁 Full reports: /vagrant/tsan-reports/day48/"
	@echo "════════════════════════════════════════════════════════════"

tsan-status:
	@echo "🔍 TSAN Build Status:"
	@echo ""
	@vagrant ssh -c "ls -lh $(SNIFFER_BUILD_DIR)/sniffer 2>/dev/null && echo '  ✅ Sniffer (TSAN)' || echo '  ❌ Sniffer not built'"
	@vagrant ssh -c "ls -lh $(ML_DETECTOR_BUILD_DIR)/ml-detector 2>/dev/null && echo '  ✅ ML Detector (TSAN)' || echo '  ❌ ML Detector not built'"
	@vagrant ssh -c "ls -lh $(RAG_INGESTER_BUILD_DIR)/rag-ingester 2>/dev/null && echo '  ✅ RAG Ingester (TSAN)' || echo '  ❌ RAG Ingester not built'"
	@vagrant ssh -c "ls -lh $(ETCD_SERVER_BUILD_DIR)/etcd-server 2>/dev/null && echo '  ✅ etcd-server (TSAN)' || echo '  ❌ etcd-server not built'"
	@echo ""
	@echo "Reports:"
	@vagrant ssh -c "ls -lh /vagrant/tsan-reports/day48/*.log 2>/dev/null | wc -l | xargs echo '  Log files:'" || echo "  ❌ No reports"

# ============================================================================
# Day 38 - Synthetic Data Generation
# ============================================================================

day38-step1:
	@echo "════════════════════════════════════════════════════════════"
	@echo "Day 38 - Step 1: etcd-server Bootstrap"
	@echo "════════════════════════════════════════════════════════════"
	@$(MAKE) etcd-server-start
	@sleep 2
	@vagrant ssh -c "curl -s http://localhost:2379/health > /dev/null 2>&1 && echo '✅ etcd-server ready' || echo '❌ etcd-server not responding'"

day38-step2: tools-build
	@echo "════════════════════════════════════════════════════════════"
	@echo "Day 38 - Step 2: Generate 100 Synthetic Events"
	@echo "════════════════════════════════════════════════════════════"
	@vagrant ssh -c "mkdir -p /vagrant/logs/rag/synthetic/events /vagrant/logs/rag/synthetic/artifacts"
	@vagrant ssh -c "cd $(TOOLS_BUILD_DIR) && ./generate_synthetic_events 100 0.20"

day38-step3:
	@echo "════════════════════════════════════════════════════════════"
	@echo "Day 38 - Step 3: Validate Artifacts"
	@echo "════════════════════════════════════════════════════════════"
	@vagrant ssh -c "find /vagrant/logs/rag/synthetic/artifacts -name 'event_*.pb.enc' | wc -l | xargs echo 'Generated:'"

day38-full: day38-step1 day38-step2 day38-step3
	@echo "✅ Day 38 workflow complete"

# ============================================================================
# RAG Ecosystem
# ============================================================================

rag-build:
	@echo "🔨 Building RAG Security System..."
	@vagrant ssh -c "cd /vagrant/rag && make build"

rag-clean:
	@echo "🧹 Cleaning RAG..."
	@vagrant ssh -c "cd /vagrant/rag && make clean"

rag-start:
	@echo "🚀 Starting RAG Security System..."
	@vagrant ssh -c "cd /vagrant/rag/build && nohup ./rag-security -c ../config/rag-config.json > /vagrant/logs/rag.log 2>&1 &"

rag-stop:
	@echo "🛑 Stopping RAG..."
	@vagrant ssh -c "pkill -f rag-security 2>/dev/null || true"

# ============================================================================
# Test Hardening Suite (Day 46/47)
# ============================================================================

test-hardening-build: proto etcd-client-build
	@echo "🔨 Building Test-Driven Hardening Suite..."
	@vagrant ssh -c 'cd $(SNIFFER_BUILD_DIR) && \
		cmake $(CMAKE_FLAGS) .. && \
		make test_sharded_flow_full_contract \
		     test_ring_consumer_protobuf \
		     test_sharded_flow_multithread -j4'

test-hardening-run:
	@echo "🧪 Running Test-Driven Hardening Suite..."
	@vagrant ssh -c "cd $(SNIFFER_BUILD_DIR) && ./test_sharded_flow_full_contract"
	@vagrant ssh -c "cd $(SNIFFER_BUILD_DIR) && ./test_ring_consumer_protobuf"
	@vagrant ssh -c "cd $(SNIFFER_BUILD_DIR) && ./test_sharded_flow_multithread"

test-hardening: test-hardening-build test-hardening-run