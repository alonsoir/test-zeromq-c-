# DDOS Pipeline Laboratory Makefile
# Actualizado para etcd + ZeroMQ + Protobuf

# Colores para output
GREEN = \033[0;32m
BLUE = \033[0;34m
YELLOW = \033[1;33m
RED = \033[0;31m
PURPLE = \033[0;35m
NC = \033[0m

.PHONY: all lab-start lab-stop clean help check-deps status lab-logs lab-test lab-debug

# Target por defecto
all: lab-start

help: ## Mostrar ayuda
	@echo "$(GREEN)DDOS Pipeline Laboratory$(NC)"
	@echo "========================="
	@echo ""
	@echo "$(BLUE)Comando principal:$(NC)"
	@echo "  $(YELLOW)lab-start$(NC)    - Iniciar laboratorio completo (etcd + servicios)"
	@echo ""
	@echo "$(BLUE)Monitoreo y debug:$(NC)"
	@echo "  $(YELLOW)status$(NC)       - Ver estado completo"
	@echo "  $(YELLOW)lab-logs$(NC)     - Ver logs en tiempo real"
	@echo "  $(YELLOW)lab-test$(NC)     - Ejecutar tests de comunicación"
	@echo "  $(YELLOW)lab-debug$(NC)    - Modo debug con etcd-browser"
	@echo ""
	@echo "$(BLUE)Gestión:$(NC)"
	@echo "  $(YELLOW)lab-stop$(NC)     - Parar laboratorio"
	@echo "  $(YELLOW)clean$(NC)        - Limpiar todo"
	@echo "  $(YELLOW)help$(NC)         - Esta ayuda"

check-deps: ## Verificar dependencias
	@echo "$(BLUE)Verificando dependencias...$(NC)"
	@command -v vagrant >/dev/null 2>&1 || (echo "$(RED)Error: Vagrant no instalado$(NC)" && exit 1)
	@command -v VBoxManage >/dev/null 2>&1 || (echo "$(RED)Error: VirtualBox no instalado$(NC)" && exit 1)
	@test -f Vagrantfile || (echo "$(RED)Error: Vagrantfile no encontrado$(NC)" && exit 1)
	@test -f docker-compose.yaml || test -f docker-compose.yml || (echo "$(RED)Error: docker-compose.yaml no encontrado$(NC)" && exit 1)
	@test -d protobuf || (echo "$(RED)Error: directorio protobuf/ no encontrado$(NC)" && exit 1)
	@test -d service1 || (echo "$(RED)Error: directorio service1/ no encontrado$(NC)" && exit 1)
	@test -d service2 || (echo "$(RED)Error: directorio service2/ no encontrado$(NC)" && exit 1)
	@test -d common || (echo "$(RED)Error: directorio common/ no encontrado$(NC)" && exit 1)
	@test -f common/EtcdServiceRegistry.h || (echo "$(RED)Error: EtcdServiceRegistry.h no encontrado$(NC)" && exit 1)
	@test -f common/EtcdServiceRegistry.cpp || (echo "$(RED)Error: EtcdServiceRegistry.cpp no encontrado$(NC)" && exit 1)
	@test -f protobuf/network_security.proto || (echo "$(RED)Error: network_security.proto no encontrado$(NC)" && exit 1)
	@echo "$(GREEN)Todas las dependencias OK$(NC)"

lab-start: check-deps ## Iniciar laboratorio completo (comando principal)
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)🚀 DDOS Pipeline Laboratory$(NC)"
	@echo "$(GREEN)   etcd + ZeroMQ + Protobuf + ML$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(BLUE)Paso 1: Iniciando VM Ubuntu 22.04...$(NC)"
	@vagrant up
	@echo ""
	@echo "$(BLUE)Paso 2: Esperando que VM esté lista...$(NC)"
	@sleep 8
	@echo ""
	@echo "$(BLUE)Paso 3: Construyendo imágenes Docker...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose build --parallel"
	@echo ""
	@echo "$(BLUE)Paso 4: Iniciando pipeline distribuido...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose up -d"
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
	@echo "  - etcd:               http://192.168.56.10:2379"
	@echo "  - Service Discovery:  Automático vía etcd"
	@echo "  - ZeroMQ Pipeline:    service1 → service2"
	@echo "  - etcd Browser:       http://192.168.56.10:8082 (con lab-debug)"
	@echo ""
	@echo "$(YELLOW)Comandos útiles:$(NC)"
	@echo "  make lab-logs     # Ver logs en tiempo real"
	@echo "  make lab-test     # Test de comunicación"
	@echo "  make lab-debug    # Modo debug"
	@echo "  make status       # Estado actual"

lab-stop: ## Parar laboratorio completo
	@echo "$(YELLOW)🛑 Parando pipeline DDOS...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose down --remove-orphans --volumes" 2>/dev/null || true
	@vagrant halt
	@echo "$(GREEN)✅ Pipeline parado$(NC)"

lab-debug: check-deps ## Iniciar en modo debug con etcd-browser
	@echo "$(PURPLE)🐛 Iniciando modo DEBUG...$(NC)"
	@vagrant up
	@sleep 5
	@vagrant ssh -c "cd /vagrant && docker-compose build"
	@vagrant ssh -c "cd /vagrant && docker-compose --profile debug up -d"
	@sleep 10
	@echo ""
	@echo "$(GREEN)🐛 Modo DEBUG activo$(NC)"
	@echo ""
	@echo "$(PURPLE)Interfaces disponibles:$(NC)"
	@echo "  - etcd Browser: http://192.168.56.10:8082"
	@echo "  - etcd API:     http://192.168.56.10:2379"
	@echo ""
	@echo "$(YELLOW)Usa 'make lab-logs' para ver actividad$(NC)"

lab-logs: ## Ver logs en tiempo real
	@echo "$(BLUE)📋 Logs del pipeline (Ctrl+C para salir)...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose logs -f --tail=50"

lab-test: ## Ejecutar tests de comunicación
	@echo "$(BLUE)🧪 Ejecutando tests de comunicación...$(NC)"
	@echo ""
	@echo "$(BLUE)Test 1: Verificar que etcd responde...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 endpoint health" || (echo "$(RED)❌ etcd no responde$(NC)" && exit 1)
	@echo "$(GREEN)✅ etcd funcionando$(NC)"
	@echo ""
	@echo "$(BLUE)Test 2: Verificar servicios registrados...$(NC)"
	@sleep 3
	@vagrant ssh -c "cd /vagrant && docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get --prefix /services/heartbeat/" || echo "$(YELLOW)⚠️ Servicios aún registrándose...$(NC)"
	@echo ""
	@echo "$(BLUE)Test 3: Verificar comunicación ZeroMQ...$(NC)"
	@vagrant ssh -c "cd /vagrant && timeout 10 docker-compose logs service2 | grep -i 'evento\|sospechoso\|processed'" || echo "$(YELLOW)⚠️ Aún no hay tráfico ZeroMQ$(NC)"
	@echo ""
	@echo "$(BLUE)Test 4: Estado de contenedores...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose ps"
	@echo ""
	@echo "$(GREEN)🧪 Tests completados$(NC)"

status: ## Ver estado del laboratorio
	@echo "$(GREEN)📊 Estado del Pipeline DDOS:$(NC)"
	@echo ""
	@echo "$(BLUE)1. Vagrant VM:$(NC)"
	@vagrant status 2>/dev/null || echo "  $(RED)VM no disponible$(NC)"
	@echo ""
	@echo "$(BLUE)2. Docker Containers:$(NC)"
	@vagrant ssh -c "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || echo "  $(RED)No se puede conectar a VM$(NC)"
	@echo ""
	@echo "$(BLUE)3. etcd Health:$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 endpoint health" 2>/dev/null || echo "  $(RED)etcd no disponible$(NC)"
	@echo ""
	@echo "$(BLUE)4. Servicios Registrados:$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get --prefix /services/heartbeat/ --keys-only" 2>/dev/null | sed 's|/services/heartbeat/|  - |' || echo "  $(RED)No se pueden obtener servicios$(NC)"
	@echo ""
	@echo "$(BLUE)5. Estadísticas de Servicios:$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get /services/config/packet_sniffer/stats 2>/dev/null" | tail -1 || echo "  packet_sniffer: $(YELLOW)No disponible$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get /services/config/feature_processor/stats 2>/dev/null" | tail -1 || echo "  feature_processor: $(YELLOW)No disponible$(NC)"

clean: ## Limpiar todo (VM, contenedores, imágenes)
	@echo "$(YELLOW)🧹 Limpiando pipeline completo...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose down --remove-orphans --volumes && docker system prune -af && docker volume prune -f" 2>/dev/null || true
	@vagrant destroy -f 2>/dev/null || true
	@echo "$(GREEN)✅ Limpieza completada$(NC)"

# Targets adicionales para desarrollo avanzado
lab-shell: ## Abrir shell en VM
	@vagrant ssh

lab-service1-logs: ## Ver logs solo de service1
	@vagrant ssh -c "cd /vagrant && docker-compose logs -f service1"

lab-service2-logs: ## Ver logs solo de service2
	@vagrant ssh -c "cd /vagrant && docker-compose logs -f service2"

lab-etcd-logs: ## Ver logs solo de etcd
	@vagrant ssh -c "cd /vagrant && docker-compose logs -f etcd"

lab-stats: ## Ver estadísticas detalladas
	@echo "$(BLUE)📊 Estadísticas detalladas del pipeline:$(NC)"
	@echo ""
	@vagrant ssh -c "cd /vagrant && docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get --prefix /services/config/" || echo "$(RED)No se pueden obtener estadísticas$(NC)"

lab-restart: ## Reiniciar solo los servicios (mantener VM)
	@echo "$(YELLOW)🔄 Reiniciando servicios...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose restart"
	@sleep 5
	@$(MAKE) status

# Target de emergencia para casos problemáticos
lab-emergency-restart: ## Reinicio completo forzado
	@echo "$(RED)🚨 Reinicio de emergencia...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose kill" 2>/dev/null || true
	@vagrant ssh -c "cd /vagrant && docker system prune -f" 2>/dev/null || true
	@vagrant reload
	@sleep 10
	@$(MAKE) lab-start