# DDOS Pipeline Laboratory Makefile
# Enhanced para sniffer eBPF - Actualizado para mejor integración

# Colores para output
GREEN = \033[0;32m
BLUE = \033[0;34m
YELLOW = \033[1;33m
RED = \033[0;31m
PURPLE = \033[0;35m
NC = \033[0m

.PHONY: all lab-start lab-stop clean help check-deps status lab-logs lab-test lab-debug \
        sniffer-build sniffer-start sniffer-stop sniffer-status sniffer-clean \
        service3-build service3-start service3-stop service3-logs sniffer-docs \
        sniffer-test sniffer-install sniffer-package lab-full-stack

# Target por defecto
all: lab-start

help: ## Mostrar ayuda completa
	@echo "$(GREEN)DDOS Pipeline Laboratory v3.1$(NC)"
	@echo "====================================="
	@echo ""
	@echo "$(BLUE)🚀 Comando principal:$(NC)"
	@echo "  $(YELLOW)lab-start$(NC)         - Iniciar laboratorio completo (etcd + servicios)"
	@echo "  $(YELLOW)lab-full-stack$(NC)    - Iniciar stack completo + sniffer eBPF"
	@echo ""
	@echo "$(BLUE)🕷️ eBPF Sniffer (Kernel-space capture):$(NC)"
	@echo "  $(YELLOW)sniffer-build$(NC)     - Compilar sniffer eBPF con dependencias"
	@echo "  $(YELLOW)sniffer-start$(NC)     - Iniciar sniffer con detección automática"
	@echo "  $(YELLOW)sniffer-stop$(NC)      - Parar sniffer y limpiar eBPF programs"
	@echo "  $(YELLOW)sniffer-status$(NC)    - Ver estado eBPF y interfaces de red"
	@echo "  $(YELLOW)sniffer-test$(NC)      - Ejecutar test suite del sniffer"
	@echo "  $(YELLOW)sniffer-clean$(NC)     - Limpiar build artifacts del sniffer"
	@echo "  $(YELLOW)sniffer-install$(NC)   - Instalar sniffer en el sistema"
	@echo "  $(YELLOW)sniffer-docs$(NC)      - Generar documentación del sniffer"
	@echo ""
	@echo "$(BLUE)🔧 Pipeline Services:$(NC)"
	@echo "  $(YELLOW)service3-build$(NC)    - Compilar service3 (receptor ZeroMQ)"
	@echo "  $(YELLOW)service3-start$(NC)    - Iniciar service3"
	@echo "  $(YELLOW)service3-stop$(NC)     - Parar service3"
	@echo "  $(YELLOW)service3-logs$(NC)     - Ver logs de service3"
	@echo ""
	@echo "$(BLUE)📊 Monitoreo y debug:$(NC)"
	@echo "  $(YELLOW)status$(NC)            - Ver estado completo del pipeline"
	@echo "  $(YELLOW)lab-logs$(NC)          - Ver logs en tiempo real"
	@echo "  $(YELLOW)lab-test$(NC)          - Ejecutar tests de comunicación"
	@echo "  $(YELLOW)lab-debug$(NC)         - Modo debug con etcd-browser"
	@echo ""
	@echo "$(BLUE)🧹 Gestión:$(NC)"
	@echo "  $(YELLOW)lab-stop$(NC)          - Parar laboratorio completo"
	@echo "  $(YELLOW)clean$(NC)             - Limpiar todo (VM + contenedores + build)"
	@echo "  $(YELLOW)help$(NC)              - Esta ayuda"
	@echo ""
	@echo "$(PURPLE)🎯 Flujo típico:$(NC)"
	@echo "  1. make lab-start        # Iniciar pipeline básico"
	@echo "  2. make sniffer-build    # Compilar sniffer eBPF"
	@echo "  3. make sniffer-start    # Captura de paquetes en kernel"
	@echo "  4. make status           # Verificar todo funcionando"

check-deps: ## Verificar dependencias del pipeline completo
	@echo "$(BLUE)Verificando dependencias del pipeline...$(NC)"
	@command -v vagrant >/dev/null 2>&1 || (echo "$(RED)Error: Vagrant no instalado$(NC)" && exit 1)
	@command -v VBoxManage >/dev/null 2>&1 || (echo "$(RED)Error: VirtualBox no instalado$(NC)" && exit 1)
	@test -f Vagrantfile || (echo "$(RED)Error: Vagrantfile no encontrado$(NC)" && exit 1)
	@test -f docker-compose.yaml || test -f docker-compose.yml || (echo "$(RED)Error: docker-compose.yaml no encontrado$(NC)" && exit 1)
	@test -d protobuf || (echo "$(RED)Error: directorio protobuf/ no encontrado$(NC)" && exit 1)
	@test -d service1 || (echo "$(RED)Error: directorio service1/ no encontrado$(NC)" && exit 1)
	@test -d service2 || (echo "$(RED)Error: directorio service2/ no encontrado$(NC)" && exit 1)
	@test -d common || (echo "$(RED)Error: directorio common/ no encontrado$(NC)" && exit 1)
	@test -d sniffer || (echo "$(RED)Error: directorio sniffer/ no encontrado$(NC)" && exit 1)
	@test -d scripts || (echo "$(RED)Error: directorio scripts/ no encontrado$(NC)" && exit 1)
	@test -f common/EtcdServiceRegistry.h || (echo "$(RED)Error: EtcdServiceRegistry.h no encontrado$(NC)" && exit 1)
	@test -f common/EtcdServiceRegistry.cpp || (echo "$(RED)Error: EtcdServiceRegistry.cpp no encontrado$(NC)" && exit 1)
	@test -f protobuf/network_security.proto || (echo "$(RED)Error: network_security.proto no encontrado$(NC)" && exit 1)
	@test -f sniffer/CMakeLists.txt || (echo "$(RED)Error: sniffer/CMakeLists.txt no encontrado$(NC)" && exit 1)
	@test -f scripts/run_sniffer_with_iface.sh || (echo "$(RED)Error: run_sniffer_with_iface.sh no encontrado$(NC)" && exit 1)
	@test -d sniffer/src/kernel || (echo "$(RED)Error: sniffer/src/kernel/ no encontrado$(NC)" && exit 1)
	@test -d sniffer/src/userspace || (echo "$(RED)Error: sniffer/src/userspace/ no encontrado$(NC)" && exit 1)
	@test -f sniffer/src/kernel/sniffer.bpf.c || (echo "$(RED)Error: sniffer.bpf.c no encontrado$(NC)" && exit 1)
	@echo "$(GREEN)Todas las dependencias del pipeline OK$(NC)"

sniffer-deps: ## Verificar dependencias específicas del sniffer eBPF
	@echo "$(BLUE)Verificando dependencias del sniffer eBPF...$(NC)"
	@vagrant ssh -c "command -v clang >/dev/null 2>&1" || (echo "$(RED)Error: clang no instalado en VM$(NC)" && exit 1)
	@vagrant ssh -c "command -v bpftool >/dev/null 2>&1" || (echo "$(RED)Error: bpftool no instalado en VM$(NC)" && exit 1)
	@vagrant ssh -c "command -v cmake >/dev/null 2>&1" || (echo "$(RED)Error: cmake no instalado en VM$(NC)" && exit 1)
	@vagrant ssh -c "pkg-config --exists libbpf" || (echo "$(RED)Error: libbpf-dev no instalado$(NC)" && exit 1)
	@vagrant ssh -c "pkg-config --exists libzmq" || (echo "$(RED)Error: libzmq-dev no instalado$(NC)" && exit 1)
	@vagrant ssh -c "pkg-config --exists protobuf" || (echo "$(RED)Error: protobuf-dev no instalado$(NC)" && exit 1)
	@echo "$(GREEN)Dependencias eBPF OK$(NC)"

lab-start: check-deps ## Iniciar laboratorio básico (sin sniffer)
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)🚀 DDOS Pipeline Laboratory$(NC)"
	@echo "$(GREEN)   etcd + ZeroMQ + Protobuf + Docker$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(BLUE)Paso 1: Iniciando VM Debian 12...$(NC)"
	@vagrant up
	@echo ""
	@echo "$(BLUE)Paso 2: Esperando que VM esté lista...$(NC)"
	@sleep 8
	@echo ""
	@echo "$(BLUE)Paso 3: Construyendo imágenes Docker...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose build --parallel"
	@echo ""
	@echo "$(BLUE)Paso 4: Iniciando pipeline distribuido...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose up -d etcd service1 service2 service3"
	@echo ""
	@echo "$(BLUE)Paso 5: Esperando inicialización de servicios...$(NC)"
	@sleep 15
	@echo ""
	@echo "$(BLUE)Paso 6: Verificando estado del pipeline...$(NC)"
	@$(MAKE) status
	@echo ""
	@echo "$(GREEN)✅ Pipeline DDOS iniciado exitosamente$(NC)"
	@echo ""
	@echo "$(PURPLE)🎯 Servicios disponibles:$(NC)"
	@echo "  - etcd:               http://192.168.56.20:2379"
	@echo "  - Service Discovery:  Automático vía etcd"
	@echo "  - ZeroMQ Pipeline:    service1 → service2"
	@echo "  - Service3:           Receptor protobuf (puerto 5571)"
	@echo "  - etcd Browser:       http://192.168.56.20:8082 (con lab-debug)"
	@echo ""
	@echo "$(YELLOW)Siguiente paso:$(NC)"
	@echo "  make sniffer-build && make sniffer-start  # Para captura eBPF"

lab-full-stack: check-deps sniffer-deps ## Iniciar stack completo (pipeline + sniffer eBPF)
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)🚀 FULL STACK DDOS PIPELINE$(NC)"
	@echo "$(GREEN)   Kernel eBPF + Userspace + Docker$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@$(MAKE) lab-start
	@echo ""
	@echo "$(BLUE)Paso 7: Compilando sniffer eBPF...$(NC)"
	@$(MAKE) sniffer-build
	@echo ""
	@echo "$(BLUE)Paso 8: Iniciando captura eBPF...$(NC)"
	@$(MAKE) sniffer-start
	@echo ""
	@echo "$(GREEN)🎉 FULL STACK OPERATIVO$(NC)"
	@echo ""
	@echo "$(PURPLE)🔥 Flujo de datos activo:$(NC)"
	@echo "  Kernel eBPF → Ring Buffer → Userspace → ZeroMQ → Service3"
	@echo "  Packets captured in kernel space → Protobuf messages"

sniffer-build: sniffer-deps ## Compilar sniffer eBPF con verificación completa
	@echo "$(BLUE)🔨 Compilando sniffer eBPF...$(NC)"
	@echo ""
	@echo "$(BLUE)Verificando capacidades eBPF del kernel...$(NC)"
	@vagrant ssh -c "uname -r && sudo sysctl kernel.bpf_jit_enable || echo 'JIT not available'"
	@vagrant ssh -c "ls /sys/fs/bpf/ >/dev/null 2>&1 && echo '✅ BPF filesystem mounted' || echo '⚠️ BPF filesystem not mounted'"
	@echo ""
	@echo "$(BLUE)Preparando entorno de compilación...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer && mkdir -p build"
	@echo ""
	@echo "$(BLUE)Ejecutando CMake configure...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && cmake .. -DCMAKE_BUILD_TYPE=Release"
	@echo ""
	@echo "$(BLUE)Compilando con make -j4...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && make -j4"
	@echo ""
	@echo "$(BLUE)Verificando artefactos generados...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && ls -la sniffer sniffer.bpf.o 2>/dev/null || echo 'Error: binarios no generados'"
	@echo ""
	@echo "$(GREEN)✅ Sniffer eBPF compilado exitosamente$(NC)"
	@echo "$(YELLOW)Archivos generados:$(NC)"
	@echo "  - sniffer.bpf.o (programa eBPF para kernel)"
	@echo "  - sniffer (aplicación userspace)"

sniffer-start: ## Iniciar sniffer eBPF con detección automática
	@echo "$(BLUE)🕷️ Iniciando sniffer eBPF...$(NC)"
	@echo ""
	@echo "$(YELLOW)Verificando que el sniffer esté compilado...$(NC)"
	@vagrant ssh -c "test -f /vagrant/sniffer/build/sniffer" || (echo "$(RED)Error: Sniffer no compilado. Ejecuta 'make sniffer-build' primero$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)Configurando límites del sistema para eBPF...$(NC)"
	@vagrant ssh -c "sudo sysctl -w kernel.unprivileged_bpf_disabled=0 2>/dev/null || echo 'Usando modo privilegiado'"
	@vagrant ssh -c "ulimit -l unlimited 2>/dev/null || echo 'Configurando límites de memoria...'"
	@echo ""
	@echo "$(YELLOW)Detección automática de interfaz y configuración...$(NC)"
	@vagrant ssh -c "cd /vagrant && chmod +x scripts/run_sniffer_with_iface.sh"
	@echo ""
	@echo "$(BLUE)🚀 Iniciando captura de paquetes en kernel space...$(NC)"
	@vagrant ssh -c "cd /vagrant && ./scripts/run_sniffer_with_iface.sh"

sniffer-stop: ## Parar sniffer y limpiar eBPF programs completamente
	@echo "$(YELLOW)🛑 Parando sniffer eBPF...$(NC)"
	@echo ""
	@echo "$(BLUE)Terminando procesos del sniffer...$(NC)"
	@vagrant ssh -c "sudo pkill -f sniffer || echo 'No hay procesos sniffer ejecutándose'"
	@echo ""
	@echo "$(BLUE)Limpiando programas XDP de todas las interfaces...$(NC)"
	@vagrant ssh -c "for iface in \$(ip -o link show | awk -F': ' '{print \$2}' | grep -v lo); do sudo bpftool net detach xdp dev \$iface 2>/dev/null || true; done"
	@echo ""
	@echo "$(BLUE)Limpiando objetos eBPF del filesystem...$(NC)"
	@vagrant ssh -c "sudo rm -f /sys/fs/bpf/xdp_sniffer_simple 2>/dev/null || true"
	@vagrant ssh -c "sudo find /sys/fs/bpf/ -name '*sniffer*' -delete 2>/dev/null || true"
	@echo ""
	@echo "$(BLUE)Verificando limpieza...$(NC)"
	@vagrant ssh -c "sudo bpftool prog show type xdp | grep -i sniffer || echo '✅ No hay programas XDP del sniffer cargados'"
	@echo ""
	@echo "$(GREEN)✅ Sniffer parado y eBPF programs limpiados completamente$(NC)"

sniffer-status: ## Ver estado detallado del sniffer eBPF
	@echo "$(BLUE)📊 Estado completo del sniffer eBPF:$(NC)"
	@echo ""
	@echo "$(BLUE)1. 🖥️ Estado del sistema:$(NC)"
	@vagrant ssh -c "uname -r | sed 's/^/  Kernel: /'"
	@vagrant ssh -c "grep MemAvailable /proc/meminfo | sed 's/^/  /'" || echo "  Memoria: No disponible"
	@vagrant ssh -c "uptime | sed 's/^/  Load: /'"
	@echo ""
	@echo "$(BLUE)2. 🔧 Capacidades eBPF:$(NC)"
	@vagrant ssh -c "sudo sysctl kernel.bpf_jit_enable 2>/dev/null | sed 's/^/  JIT: /' || echo '  JIT: No disponible'"
	@vagrant ssh -c "ls -d /sys/fs/bpf >/dev/null 2>&1 && echo '  ✅ BPF filesystem: /sys/fs/bpf' || echo '  ❌ BPF filesystem: No montado'"
	@vagrant ssh -c "command -v bpftool >/dev/null && bpftool version | head -1 | sed 's/^/  /' || echo '  ❌ bpftool: No disponible'"
	@echo ""
	@echo "$(BLUE)3. 📊 Programas eBPF cargados:$(NC)"
	@vagrant ssh -c "sudo bpftool prog show 2>/dev/null | grep -A3 -B1 xdp || echo '  📝 No hay programas XDP cargados'"
	@echo ""
	@echo "$(BLUE)4. 🌐 Attachments de red:$(NC)"
	@vagrant ssh -c "sudo bpftool net show 2>/dev/null || echo '  📝 No hay attachments XDP activos'"
	@echo ""
	@echo "$(BLUE)5. 🔌 Interfaces de red disponibles:$(NC)"
	@vagrant ssh -c "ip -o link show | grep -v lo | sed 's/^/  /' | cut -d: -f1-2" || echo "  ❌ No se pueden obtener interfaces"
	@echo ""
	@echo "$(BLUE)6. 🏃 Procesos del sniffer:$(NC)"
	@vagrant ssh -c "pgrep -f sniffer >/dev/null 2>&1 && echo '  ✅ Sniffer ejecutándose (PID: '$(pgrep -f sniffer)')' || echo '  ⏹️ Sniffer no está ejecutándose'"
	@echo ""
	@echo "$(BLUE)7. 📁 Artefactos de build:$(NC)"
	@vagrant ssh -c "ls -lh /vagrant/sniffer/build/sniffer 2>/dev/null | sed 's|/vagrant/sniffer/build/||' | sed 's/^/  ✅ Binary: /' || echo '  ❌ Binary: No compilado'"
	@vagrant ssh -c "ls -lh /vagrant/sniffer/build/sniffer.bpf.o 2>/dev/null | sed 's|/vagrant/sniffer/build/||' | sed 's/^/  ✅ eBPF: /' || echo '  ❌ eBPF: No compilado'"

sniffer-test: ## Ejecutar test suite del sniffer
	@echo "$(BLUE)🧪 Ejecutando test suite del sniffer eBPF...$(NC)"
	@echo ""
	@echo "$(BLUE)Test 1: Verificar compilación...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && test -x sniffer && echo '✅ Binary ejecutable encontrado' || (echo '❌ Binary no encontrado' && exit 1)"
	@echo ""
	@echo "$(BLUE)Test 2: Verificar programa eBPF...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && test -f sniffer.bpf.o && echo '✅ Programa eBPF encontrado' || (echo '❌ Programa eBPF no encontrado' && exit 1)"
	@echo ""
	@echo "$(BLUE)Test 3: Test de configuración...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && sudo ./sniffer --test-config --config=../config/sniffer.json && echo '✅ Configuración válida' || echo '❌ Error en configuración'"
	@echo ""
	@echo "$(BLUE)Test 4: Test de carga de programa eBPF (sin attach)...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && timeout 5 sudo ./sniffer --config=../config/sniffer.json 2>/dev/null || echo '✅ Programa puede cargar eBPF (timeout esperado)'"
	@echo ""
	@echo "$(BLUE)Test 5: Verificar dependencias runtime...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && ldd sniffer | grep -E '(zmq|protobuf|bpf)' | sed 's/^/  ✅ /'"
	@echo ""
	@echo "$(GREEN)🧪 Test suite completado$(NC)"

sniffer-clean: ## Limpiar build artifacts del sniffer
	@echo "$(YELLOW)🧹 Limpiando build artifacts del sniffer...$(NC)"
	@vagrant ssh -c "rm -rf /vagrant/sniffer/build/*" || echo "$(YELLOW)Build directory ya estaba limpio$(NC)"
	@echo "$(GREEN)✅ Build artifacts limpiados$(NC)"

sniffer-install: sniffer-build ## Instalar sniffer en el sistema de la VM
	@echo "$(BLUE)📦 Instalando sniffer en sistema...$(NC)"
	@vagrant ssh -c "cd /vagrant/sniffer/build && sudo cp sniffer /usr/local/bin/"
	@vagrant ssh -c "cd /vagrant/sniffer/build && sudo cp sniffer.bpf.o /usr/local/lib/"
	@vagrant ssh -c "cd /vagrant/sniffer && sudo cp config/sniffer.json /etc/"
	@echo "$(GREEN)✅ Sniffer instalado en:$(NC)"
	@echo "  - Binary: /usr/local/bin/sniffer"
	@echo "  - eBPF program: /usr/local/lib/sniffer.bpf.o"
	@echo "  - Config: /etc/sniffer.json"

sniffer-docs: ## Generar documentación del sniffer
	@echo "$(BLUE)📚 Generando documentación del sniffer...$(NC)"
	@echo "$(GREEN)Documentación disponible en:$(NC)"
	@echo "  - sniffer/docs/BUILD.md - Guía de compilación"
	@echo "  - sniffer/docs/BARE_METAL.md - Instalación en hardware físico"
	@echo "  - sniffer/README.md - Documentación principal"

# Resto de targets existentes...
lab-stop: ## Parar laboratorio completo
	@echo "$(YELLOW)🛑 Parando pipeline DDOS completo...$(NC)"
	@$(MAKE) sniffer-stop 2>/dev/null || true
	@vagrant ssh -c "cd /vagrant && docker-compose down --remove-orphans --volumes" 2>/dev/null || true
	@vagrant halt
	@echo "$(GREEN)✅ Pipeline parado completamente$(NC)"

service3-build: ## Compilar service3
	@echo "$(BLUE)🔨 Construyendo Service3...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose build service3"

service3-start: service3-build ## Iniciar service3
	@echo "$(BLUE)🚀 Iniciando Service3...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose up -d service3"

service3-stop: ## Parar service3
	@echo "$(BLUE)🛑 Deteniendo Service3...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose stop service3"

service3-logs: ## Ver logs de service3
	@echo "$(BLUE)📋 Logs de Service3...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose logs -f service3"