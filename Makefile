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
.PHONY: kill-all check-ports restart
.PHONY: clean distclean test dev-setup schema-update
.PHONY: build-unified rebuild-unified create-verify-script quick-fix dev-setup-unified
.PHONY: check-libbpf verify-bpf-maps diagnose-bpf  # NUEVO

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

# ============================================================================
# Protobuf Schema - UNIFIED SYSTEM
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
# Build Targets - UPDATED FOR UNIFIED PROTOBUF
# ============================================================================

sniffer: proto
	@echo "🔨 Building Sniffer..."
	@vagrant ssh -c "cd /vagrant/sniffer && make"

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

detector: proto
	@echo "🔨 Building ML Detector..."
	@vagrant ssh -c "mkdir -p /vagrant/ml-detector/build && cd /vagrant/ml-detector/build && cmake .. && make -j4"

detector-build: detector

detector-clean:
	@echo "🧹 Cleaning ML Detector..."
	@vagrant ssh -c "rm -rf /vagrant/ml-detector/build/*"

firewall: proto
	@echo "🔨 Building Firewall ACL Agent..."
	@vagrant ssh -c "mkdir -p /vagrant/firewall-acl-agent/build && cd /vagrant/firewall-acl-agent/build && cmake .. && make -j4"

firewall-build: firewall

firewall-clean:
	@echo "🧹 Cleaning Firewall ACL Agent..."
	@vagrant ssh -c "rm -rf /vagrant/firewall-acl-agent/build/*"

# Build con protobuf unificado
build-unified: proto-unified sniffer detector firewall
	@echo "🚀 Build completo con protobuf unificado"
	@$(MAKE) proto-verify

all: build-unified
	@echo "✅ All components built con protobuf unificado"

rebuild-unified: clean build-unified
	@echo "✅ Rebuild completo con protobuf unificado"

rebuild: rebuild-unified
	@echo "✅ Full rebuild complete con protobuf unificado"

clean: sniffer-clean detector-clean firewall-clean
	@echo "✅ Clean complete"

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

run-lab-dev:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Starting ML Defender Lab (Development Mode)            ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📋 Execution Order:"
	@echo "   1️⃣  Firewall ACL Agent  (SUB tcp://localhost:5572)"
	@echo "   2️⃣  ML Detector         (PUB tcp://0.0.0.0:5572)"
	@echo "   3️⃣  Sniffer             (PUSH tcp://127.0.0.1:5571)"
	@echo ""
	@vagrant ssh -c "cd /vagrant && bash scripts/run_lab_dev.sh"

	@vagrant ssh -c "sudo pkill -f -9 firewall-acl-agent || true"
	@vagrant ssh -c "pkill -f -9 ml-detector || true"
	@vagrant ssh -c "sudo pkill -f -9 sniffer || true"
	@sleep 2
	@echo "✅ Lab stopped"


kill-lab:
	@echo "💀 Stopping ML Defender Lab..."
	@echo ""
	@echo "Checking processes..."
	@vagrant ssh -c "pgrep -a -f firewall-acl-agent || echo '  Firewall: ❌ Not running'"
	@vagrant ssh -c "pgrep -a -f ml-detector || echo '  Detector: ❌ Not running'"
	@vagrant ssh -c "pgrep -a -f sniffer || echo '  Sniffer:  ❌ Not running'"
	@echo ""
	@echo "Killing processes..."
	-@vagrant ssh -c "sudo pkill -9 -f firewall-acl-agent" 2>/dev/null || echo "  Firewall already stopped"
	-@vagrant ssh -c "pkill -9 -f ml-detector" 2>/dev/null || echo "  Detector already stopped"
	-@vagrant ssh -c "sudo pkill -9 -f sniffer" 2>/dev/null || echo "  Sniffer already stopped"
	@sleep 2
	@echo ""
	@echo "Verifying cleanup..."
	@vagrant ssh -c "pgrep -a -f 'firewall-acl-agent|ml-detector|sniffer' || echo '✅ All processes stopped'"

status-lab:
	@echo "════════════════════════════════════════════════════════════"
	@echo "ML Defender Lab Status:"
	@echo "════════════════════════════════════════════════════════════"
	@vagrant ssh -c "pgrep -a -f firewall-acl-agent && echo '✅ Firewall: RUNNING' || echo '❌ Firewall: STOPPED'"
	@vagrant ssh -c "pgrep -a -f ml-detector && echo '✅ Detector: RUNNING' || echo '❌ Detector: STOPPED'"
	@vagrant ssh -c "pgrep -a -f 'sniffer.*-c' && echo '✅ Sniffer:  RUNNING' || echo '❌ Sniffer:  STOPPED'"
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