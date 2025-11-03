.PHONY: help status
.PHONY: up halt destroy ssh
.PHONY: lab-start lab-stop lab-restart lab-ps lab-logs lab-clean
.PHONY: proto sniffer detector all rebuild
.PHONY: sniffer-build sniffer-clean sniffer-package sniffer-install
.PHONY: detector-build detector-clean
.PHONY: run-sniffer run-detector logs-sniffer logs-detector
.PHONY: kill-all check-ports restart
.PHONY: clean distclean test dev-setup schema-update

# ============================================================================
# ML Detector Pipeline - Host Makefile
# Run from macOS - Commands execute in VM via vagrant ssh -c
# ============================================================================

help:
		@echo ""
		@echo "╔════════════════════════════════════════════════════════════╗"
		@echo "║  ML Detector Pipeline - Development Makefile               ║"
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
		@echo "  make all             - Build sniffer + detector"
		@echo "  make proto           - Regenerate protobuf schema"
		@echo "  make sniffer         - Build sniffer"
		@echo "  make detector        - Build ml-detector"
		@echo "  make rebuild         - Clean + build all"
		@echo ""
		@echo "Sniffer Packaging:"
		@echo "  make sniffer-package - Create .deb package"
		@echo "  make sniffer-install - Install .deb in VM"
		@echo ""
		@echo "Run (foreground):"
		@echo "  make run-sniffer     - Run sniffer (Terminal 1)"
		@echo "  make run-detector    - Run detector (Terminal 2)"
		@echo ""
		@echo "Development:"
		@echo "  make dev-setup       - Full setup (up + lab + build)"
		@echo "  make test            - Check what's built"
		  @echo "  make schema-update   - Update schema + rebuild"
		  @echo "  make logs-sniffer    - Show sniffer logs"
		  @echo "  make logs-detector   - Show detector logs"
		  @echo ""
		  @echo "Troubleshooting:"
		  @echo "  make kill-all        - Kill all processes"
		  @echo "  make check-ports     - Check if ports are in use"
		  @echo "  make clean           - Clean build artifacts"
		  @echo ""

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

proto:
	  @echo "📦 Regenerating protobuf schema..."
	  @vagrant ssh -c "cd /vagrant/protobuf && ./generate.sh"

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

all: sniffer detector
	  @echo "✅ All components built"

rebuild: clean all
	  @echo "✅ Full rebuild complete"

run-sniffer:
	  @echo "📡 Running Sniffer..."
	  @vagrant ssh -c "cd /vagrant/sniffer/build && sudo ./sniffer -c config/sniffer.json --verbose"

run-detector:
	  @echo "🤖 Running ML Detector..."
	  @vagrant ssh -c "cd /vagrant/ml-detector/build && ./ml-detector --verbose"

logs-sniffer:
	  @vagrant ssh -c "tail -50 /vagrant/sniffer/build/logs/*.log 2>/dev/null || echo 'No logs yet'"

logs-detector:
	  @vagrant ssh -c "tail -50 /vagrant/ml-detector/build/logs/*.log 2>/dev/null || echo 'No logs yet'"

dev-setup: up lab-start all
	  @echo ""
	  @echo "✅ Development environment ready"
	  @echo ""
	  @echo "Next steps:"
	  @echo "  Terminal 1: make run-sniffer"
	  @echo "  Terminal 2: make run-detector"
	  @echo "  Terminal 3: make ssh"
	  @echo ""

test:
	  @echo "🧪 Testing build artifacts..."
	  @vagrant ssh -c "echo 'Sniffer:     \$$([ -f /vagrant/sniffer/build/sniffer ] && echo ✅ || echo ❌)' && echo 'ML Detector: \$$([ -f /vagrant/ml-detector/build/ml-detector ] && echo ✅ || echo ❌)' && echo 'Protobuf:    \$$([ -f /vagrant/protobuf/network_security.pb.cc ] && echo ✅ || echo ❌)'"

schema-update: proto rebuild

clean: sniffer-clean detector-clean
	  @echo "✅ Clean complete"

distclean: clean
	  @vagrant ssh -c "rm -f /vagrant/protobuf/network_security.pb.* /vagrant/protobuf/network_security_pb2.py"

check-ports:
	  @vagrant ssh -c "sudo ss -tlnp | grep -E '5571|5572' && echo '⚠️  Ports in use' || echo '✅ Ports free'"

kill-all:
	  @echo "💀 Killing processes..."
	  @vagrant ssh -c "sudo pkill -9 sniffer || true && pkill -9 ml-detector || true && sleep 2"

restart: kill-all
	  @echo "♻️  Restart with: make run-sniffer / make run-detector"

.DEFAULT_GOAL := help
