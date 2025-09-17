# ZeroMQ Docker Lab Makefile
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
LDFLAGS = -lzmq

# Directorios y archivos
SRC1 = service1/main.cpp
SRC2 = service2/main.cpp
BIN_DIR = bin
BIN1 = $(BIN_DIR)/service1_exe
BIN2 = $(BIN_DIR)/service2_exe

# Colores para output
GREEN = \033[0;32m
BLUE = \033[0;34m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: all run clean docker-build docker-run docker-stop docker-logs \
        vagrant-up vagrant-down vagrant-ssh lab-start lab-stop lab-restart \
        help native-build native-run status

# Default target
all: help

help: ## Mostrar ayuda
	@echo "$(GREEN)ZeroMQ Docker Laboratory$(NC)"
	@echo "========================="
	@echo ""
	@echo "$(BLUE)Comandos principales:$(NC)"
	@echo "  $(YELLOW)lab-start$(NC)    - Iniciar laboratorio completo (Vagrant + Docker)"
	@echo "  $(YELLOW)lab-stop$(NC)     - Parar laboratorio completo"
	@echo "  $(YELLOW)lab-restart$(NC)  - Reiniciar laboratorio"
	@echo "  $(YELLOW)status$(NC)       - Ver estado del laboratorio"
	@echo ""
	@echo "$(BLUE)Comandos Docker:$(NC)"
	@echo "  $(YELLOW)docker-build$(NC) - Construir imágenes Docker"
	@echo "  $(YELLOW)docker-run$(NC)   - Ejecutar contenedores"
	@echo "  $(YELLOW)docker-stop$(NC)  - Parar contenedores"
	@echo "  $(YELLOW)docker-logs$(NC)  - Ver logs de contenedores"
	@echo ""
	@echo "$(BLUE)Comandos Vagrant:$(NC)"
	@echo "  $(YELLOW)vagrant-up$(NC)   - Levantar VM"
	@echo "  $(YELLOW)vagrant-down$(NC) - Apagar VM"
	@echo "  $(YELLOW)vagrant-ssh$(NC)  - Conectar a VM"
	@echo ""
	@echo "$(BLUE)Desarrollo local:$(NC)"
	@echo "  $(YELLOW)native-build$(NC) - Compilar localmente"
	@echo "  $(YELLOW)native-run$(NC)   - Ejecutar localmente"
	@echo "  $(YELLOW)clean$(NC)        - Limpiar archivos compilados"

# ==== COMANDOS PRINCIPALES ====

lab-start: ## Iniciar laboratorio completo
	@echo "$(GREEN)🚀 Iniciando laboratorio ZeroMQ...$(NC)"
	@$(MAKE) vagrant-up
	@echo "$(BLUE)⏳ Esperando que VM esté lista...$(NC)"
	@sleep 10
	@echo "$(GREEN)🐳 Ejecutando POC en Docker...$(NC)"
	@vagrant ssh -c "cd /vagrant && ./run_zeroMQ_poc.sh"

lab-stop: ## Parar laboratorio completo
	@echo "$(YELLOW)🛑 Parando laboratorio...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose down" 2>/dev/null || true
	@$(MAKE) vagrant-down

lab-restart: ## Reiniciar laboratorio
	@echo "$(BLUE)🔄 Reiniciando laboratorio...$(NC)"
	@$(MAKE) lab-stop
	@sleep 2
	@$(MAKE) lab-start

status: ## Ver estado del laboratorio
	@echo "$(GREEN)📊 Estado del laboratorio:$(NC)"
	@echo ""
	@echo "$(BLUE)Vagrant VM:$(NC)"
	@vagrant status 2>/dev/null || echo "  $(RED)❌ Vagrant no disponible$(NC)"
	@echo ""
	@echo "$(BLUE)Docker (en VM):$(NC)"
	@vagrant ssh -c "docker ps 2>/dev/null" 2>/dev/null || echo "  $(RED)❌ No se puede conectar a VM$(NC)"

# ==== COMANDOS DOCKER ====

docker-build: ## Construir imágenes Docker (dentro de VM)
	@echo "$(GREEN)🔨 Construyendo imágenes Docker...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose build"

docker-run: ## Ejecutar contenedores (dentro de VM)
	@echo "$(GREEN)▶️  Ejecutando contenedores...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose up -d"
	@sleep 3
	@$(MAKE) docker-logs

docker-stop: ## Parar contenedores
	@echo "$(YELLOW)⏹️  Parando contenedores...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose down"

docker-logs: ## Ver logs de contenedores
	@echo "$(BLUE)📋 Logs de contenedores:$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose logs --tail=20"

# ==== COMANDOS VAGRANT ====

vagrant-up: ## Levantar VM
	@echo "$(GREEN)🖥️  Levantando VM...$(NC)"
	@vagrant up

vagrant-down: ## Apagar VM
	@echo "$(YELLOW)🖥️  Apagando VM...$(NC)"
	@vagrant halt

vagrant-ssh: ## Conectar a VM
	@vagrant ssh

# ==== DESARROLLO LOCAL ====

native-build: $(BIN_DIR) $(BIN1) $(BIN2) ## Compilar localmente

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(BIN1): $(SRC1)
	@echo "$(GREEN)🔨 Compilando service1...$(NC)"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BIN2): $(SRC2)
	@echo "$(GREEN)🔨 Compilando service2...$(NC)"
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

native-run: native-build ## Ejecutar localmente
	@echo "$(GREEN)🚀 Ejecutando service1 en background...$(NC)"
	@./$(BIN1) & \
	SERVER_PID=$$!; \
	sleep 1; \
	echo "$(GREEN)📤 Ejecutando service2...$(NC)"; \
	./$(BIN2); \
	echo "$(YELLOW)🛑 Parando service1...$(NC)"; \
	kill $$SERVER_PID

clean: ## Limpiar archivos compilados
	@echo "$(YELLOW)🧹 Limpiando...$(NC)"
	@rm -rf $(BIN_DIR)
	@vagrant ssh -c "cd /vagrant && docker system prune -f" 2>/dev/null || true

# ==== COMANDOS DE UTILIDAD ====

quick-test: ## Test rápido en VM existente
	@echo "$(GREEN)⚡ Test rápido...$(NC)"
	@vagrant ssh -c "cd /vagrant && docker-compose down && docker-compose up --build -d && sleep 3 && docker-compose logs && docker-compose down"

rebuild: ## Reconstruir todo desde cero
	@echo "$(GREEN)🔄 Reconstrucción completa...$(NC)"
	@$(MAKE) clean
	@$(MAKE) lab-stop
	@vagrant destroy -f
	@$(MAKE) lab-start

logs: ## Ver todos los logs
	@echo "$(BLUE)📋 Logs del sistema:$(NC)"
	@echo ""
	@echo "$(YELLOW)=== Vagrant Status ====$(NC)"
	@vagrant status
	@echo ""
	@echo "$(YELLOW)=== Docker Containers ====$(NC)"
	@vagrant ssh -c "docker ps -a" 2>/dev/null || echo "VM no disponible"
	@echo ""
	@echo "$(YELLOW)=== Docker Images ====$(NC)"
	@vagrant ssh -c "docker images" 2>/dev/null || echo "VM no disponible"

# Verificar dependencias
check-deps: ## Verificar dependencias
	@echo "$(GREEN)🔍 Verificando dependencias...$(NC)"
	@command -v vagrant >/dev/null 2>&1 || (echo "$(RED)❌ Vagrant no instalado$(NC)" && exit 1)
	@command -v VBoxManage >/dev/null 2>&1 || (echo "$(RED)❌ VirtualBox no instalado$(NC)" && exit 1)
	@test -f Vagrantfile || (echo "$(RED)❌ Vagrantfile no encontrado$(NC)" && exit 1)
	@test -f docker-compose.yaml || (echo "$(RED)❌ docker-compose.yaml no encontrado$(NC)" && exit 1)
	@echo "$(GREEN)✅ Todas las dependencias OK$(NC)"