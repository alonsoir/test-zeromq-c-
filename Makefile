# ZeroMQ + Protobuf Laboratory Makefile
# Minimalista: Un comando para lanzar todo el laboratorio

# Colores para output
GREEN = \033[0;32m
BLUE = \033[0;34m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m

.PHONY: all lab-start lab-stop clean help check-deps status

# Target por defecto
all: lab-start

help: ## Mostrar ayuda
	@echo "$(GREEN)ZeroMQ + Protobuf Laboratory$(NC)"
	@echo "============================"
	@echo ""
	@echo "$(BLUE)Comando principal:$(NC)"
	@echo "  $(YELLOW)lab-start$(NC)    - Iniciar laboratorio completo (hace todo automáticamente)"
	@echo ""
	@echo "$(BLUE)Otros comandos:$(NC)"
	@echo "  $(YELLOW)lab-stop$(NC)     - Parar laboratorio"
	@echo "  $(YELLOW)status$(NC)       - Ver estado"
	@echo "  $(YELLOW)clean$(NC)        - Limpiar todo"
	@echo "  $(YELLOW)help$(NC)         - Esta ayuda"

check-deps: ## Verificar dependencias
	@echo "$(BLUE)Verificando dependencias...$(NC)"
	@command -v vagrant >/dev/null 2>&1 || (echo "$(RED)Error: Vagrant no instalado$(NC)" && exit 1)
	@command -v VBoxManage >/dev/null 2>&1 || (echo "$(RED)Error: VirtualBox no instalado$(NC)" && exit 1)
	@test -f Vagrantfile || (echo "$(RED)Error: Vagrantfile no encontrado$(NC)" && exit 1)
	@test -f docker-compose.yml || test -f docker-compose.yaml || (echo "$(RED)Error: docker-compose.yml no encontrado$(NC)" && exit 1)
	@test -f build_and_run.sh || (echo "$(RED)Error: build_and_run.sh no encontrado$(NC)" && exit 1)
	@test -d protobuf || (echo "$(RED)Error: directorio protobuf/ no encontrado$(NC)" && exit 1)
	@test -d service1 || (echo "$(RED)Error: directorio service1/ no encontrado$(NC)" && exit 1)
	@test -d service2 || (echo "$(RED)Error: directorio service2/ no encontrado$(NC)" && exit 1)
	@echo "$(GREEN)Todas las dependencias OK$(NC)"

lab-start: check-deps ## Iniciar laboratorio completo (comando principal)
	@echo "$(GREEN)================================$(NC)"
	@echo "$(GREEN)🚀 ZeroMQ + Protobuf Laboratory$(NC)"
	@echo "$(GREEN)================================$(NC)"
	@echo ""
	@echo "$(BLUE)Paso 1: Iniciando VM Ubuntu 22.04...$(NC)"
	@vagrant up
	@echo ""
	@echo "$(BLUE)Paso 2: Esperando que VM esté lista...$(NC)"
	@sleep 5
	@echo ""
	@echo "$(BLUE)Paso 3: Ejecutando demo ZeroMQ + Protobuf...$(NC)"
	@chmod +x build_and_run.sh
	@vagrant ssh -c "cd /vagrant && chmod +x build_and_run.sh && ./build_and_run.sh"
	@echo ""
	@echo "$(GREEN)✅ Laboratorio completado exitosamente$(NC)"
	@echo ""
	@echo "$(YELLOW)Usa 'make status' para ver el estado$(NC)"
	@echo "$(YELLOW)Usa 'make lab-stop' para parar todo$(NC)"

lab-stop: ## Parar laboratorio completo
	@echo "$(YELLOW)🛑 Parando laboratorio...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose down --remove-orphans" 2>/dev/null || true
	@vagrant halt
	@echo "$(GREEN)✅ Laboratorio parado$(NC)"

status: ## Ver estado del laboratorio
	@echo "$(GREEN)📊 Estado del laboratorio:$(NC)"
	@echo ""
	@echo "$(BLUE)Vagrant VM:$(NC)"
	@vagrant status 2>/dev/null || echo "  $(RED)VM no disponible$(NC)"
	@echo ""
	@echo "$(BLUE)Docker (en VM):$(NC)"
	@vagrant ssh -c "docker ps --format 'table {{.Names}}\t{{.Status}}'" 2>/dev/null || echo "  $(RED)No se puede conectar a VM$(NC)"

clean: ## Limpiar todo (VM, contenedores, imágenes)
	@echo "$(YELLOW)🧹 Limpiando laboratorio completo...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose down --remove-orphans && docker system prune -af" 2>/dev/null || true
	@vagrant destroy -f 2>/dev/null || true
	@echo "$(GREEN)✅ Limpieza completada$(NC)"