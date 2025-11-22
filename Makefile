.PHONY: help status
.PHONY: up halt destroy ssh
.PHONY: lab-start lab-stop lab-restart lab-ps lab-logs lab-clean
.PHONY: proto sniffer detector firewall all rebuild
.PHONY: sniffer-build sniffer-clean sniffer-package sniffer-install
.PHONY: detector-build detector-clean
.PHONY: firewall-build firewall-clean
.PHONY: run-sniffer run-detector run-firewall
.PHONY: logs-sniffer logs-detector logs-firewall logs-lab
.PHONY: run-lab-dev kill-lab status-lab
.PHONY: kill-all check-ports restart
.PHONY: clean distclean test dev-setup schema-update

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
	@echo "  make status          - Show VM status"
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
	@echo "  make proto           - Regenerate protobuf schema"
	@echo "  make sniffer         - Build sniffer"
	@echo "  make detector        - Build ml-detector"
	@echo "  make firewall        - Build firewall-acl-agent"
	@echo "  make rebuild         - Clean + build all"
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
	@echo "  make test            - Check what's built"
	@echo "  make schema-update   - Update schema + rebuild"
	@echo ""
	@echo "Troubleshooting:"
	@echo "  make kill-all        - Kill all processes"
	@echo "  make check-ports     - Check if ports are in use"
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
	@vagrant status

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
# Protobuf Schema
# ============================================================================

proto:
	@echo "📦 Regenerating protobuf schema..."
	@vagrant ssh -c "cd /vagrant/protobuf && ./generate.sh"
	@echo "📋 Copying protobuf to components..."
	@vagrant ssh -c "mkdir -p /vagrant/firewall-acl-agent/proto"
	@vagrant ssh -c "cp /vagrant/protobuf/network_security.pb.cc /vagrant/firewall-acl-agent/proto/"
	@vagrant ssh -c "cp /vagrant/protobuf/network_security.pb.h /vagrant/firewall-acl-agent/proto/"
	@echo "✅ Protobuf synchronized across all components"

# ============================================================================
# Build Targets
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
	@vagrant ssh -c "mkdir -p /vagrant/ml-detector/src/protobuf && cp /vagrant/protobuf/network_security.pb.cc /vagrant/ml-detector/src/protobuf/ && cp /vagrant/protobuf/network_security.pb.h /vagrant/ml-detector/src/protobuf/ && mkdir -p /vagrant/ml-detector/build && cd /vagrant/ml-detector/build && cmake .. && make -j4"

detector-build: detector

detector-clean:
	@echo "🧹 Cleaning ML Detector..."
	@vagrant ssh -c "rm -rf /vagrant/ml-detector/build/*"
	@vagrant ssh -c "rm -f /vagrant/ml-detector/src/protobuf/network_security.pb.*"

firewall: proto
	@echo "🔨 Building Firewall ACL Agent..."
	@vagrant ssh -c "mkdir -p /vagrant/firewall-acl-agent/build && cd /vagrant/firewall-acl-agent/build && cmake .. && make -j4"

firewall-build: firewall

firewall-clean:
	@echo "🧹 Cleaning Firewall ACL Agent..."
	@vagrant ssh -c "rm -rf /vagrant/firewall-acl-agent/build/*"
	@vagrant ssh -c "rm -f /vagrant/firewall-acl-agent/proto/network_security.pb.*"

all: sniffer detector firewall
	@echo "✅ All components built (Sniffer + Detector + Firewall)"

rebuild: clean all
	@echo "✅ Full rebuild complete"

clean: sniffer-clean detector-clean firewall-clean
	@echo "✅ Clean complete"

distclean: clean
	@vagrant ssh -c "rm -f /vagrant/protobuf/network_security.pb.* /vagrant/protobuf/network_security_pb2.py"

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

kill-lab:
	@echo "💀 Stopping ML Defender Lab..."
	@vagrant ssh -c "sudo pkill -9 firewall-acl-agent || true"
	@vagrant ssh -c "pkill -9 ml-detector || true"
	@vagrant ssh -c "sudo pkill -9 sniffer || true"
	@sleep 2
	@echo "✅ Lab stopped"

status-lab:
	@echo "📊 ML Defender Lab Status:"
	@echo ""
	@vagrant ssh -c "pgrep -a firewall-acl-agent | head -1 | awk '{print \"Firewall: ✅ PID\", \$$1}' || echo 'Firewall: ❌ Not running'"
	@vagrant ssh -c "pgrep -a ml-detector | head -1 | awk '{print \"Detector: ✅ PID\", \$$1}' || echo 'Detector: ❌ Not running'"
	@vagrant ssh -c "pgrep -a sniffer | head -1 | awk '{print \"Sniffer:  ✅ PID\", \$$1}' || echo 'Sniffer:  ❌ Not running'"
	@echo ""
	@vagrant ssh -c "sudo ss -tlnp | grep -E '5571|5572' || echo 'Ports: ⚠️  Not listening'"

logs-lab:
	@echo "📋 Combined Logs (all components):"
	@vagrant ssh -c "cd /vagrant && bash scripts/monitor_lab.sh"

# ============================================================================
# Individual Logs
# ============================================================================

logs-firewall:
	@vagrant ssh -c "tail -50 /vagrant/firewall-acl-agent/build/logs/*.log 2>/dev/null || tail -50 /var/log/ml-defender/firewall-acl-agent.log 2>/dev/null || echo 'No logs yet'"

logs-detector:
	@vagrant ssh -c "tail -50 /vagrant/ml-detector/build/logs/*.log 2>/dev/null || echo 'No logs yet'"

logs-sniffer:
	@vagrant ssh -c "tail -50 /vagrant/sniffer/build/logs/*.log 2>/dev/null || echo 'No logs yet'"

# ============================================================================
# Development Helpers
# ============================================================================

dev-setup: up lab-start all
	@echo ""
	@echo "✅ Development environment ready"
	@echo ""
	@echo "Next steps:"
	@echo "  make run-lab-dev     - Start full integrated lab"
	@echo "  make ssh             - SSH into VM for debugging"
	@echo ""

test:
	@echo "🧪 Testing build artifacts..."
	@vagrant ssh -c "echo 'Sniffer:  \$$([ -f /vagrant/sniffer/build/sniffer ] && echo ✅ || echo ❌)' && echo 'Detector: \$$([ -f /vagrant/ml-detector/build/ml-detector ] && echo ✅ || echo ❌)' && echo 'Firewall: \$$([ -f /vagrant/firewall-acl-agent/build/firewall-acl-agent ] && echo ✅ || echo ❌)' && echo 'Protobuf: \$$([ -f /vagrant/protobuf/network_security.pb.cc ] && echo ✅ || echo ❌)'"

schema-update: proto rebuild

check-ports:
	@vagrant ssh -c "sudo ss -tlnp | grep -E '5571|5572' && echo '⚠️  Ports in use' || echo '✅ Ports free'"

kill-all: kill-lab

restart: kill-lab
	@echo "♻️  Restart with: make run-lab-dev"

.DEFAULT_GOAL := help