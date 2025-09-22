#!/bin/bash
# build_and_run.sh - Script actualizado para DDOS Pipeline con etcd

set -e  # Exit on any error

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 DDOS Pipeline - Build & Run${NC}"
echo -e "${GREEN}   etcd + ZeroMQ + Protobuf + ML${NC}"
echo -e "${GREEN}========================================${NC}"

# Función para mostrar progreso
show_progress() {
    echo -e "${BLUE}$1${NC}"
}

# Función para mostrar éxito
show_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Función para mostrar advertencia
show_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Función para mostrar error
show_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "docker-compose.yaml" ] && [ ! -f "docker-compose.yml" ]; then
    show_error "docker-compose.yaml no encontrado. ¿Estás en el directorio correcto?"
    exit 1
fi

# Verificar archivos críticos
show_progress "Verificando estructura del proyecto..."
required_files=(
    "protobuf/network_security.proto"
    "common/EtcdServiceRegistry.h"
    "common/EtcdServiceRegistry.cpp"
    "service1/main.cpp"
    "service1/main.h"
    "service2/main.cpp"
    "service2/main.h"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    show_error "Archivos faltantes:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    show_warning "Asegúrate de que todos los archivos estén en su lugar antes de continuar"
    exit 1
fi

show_success "Estructura del proyecto verificada"

# Limpiar ejecución anterior si existe
show_progress "Limpiando ejecución anterior..."
docker-compose down --remove-orphans --volumes 2>/dev/null || true
docker system prune -f 2>/dev/null || true

# Construir imágenes Docker
show_progress "Construyendo imágenes Docker..."
echo "  - Esto puede tardar varios minutos la primera vez"
echo "  - Se instalarán: gRPC, etcd-cpp-apiv3, ZeroMQ, Protobuf"

if docker-compose build --parallel; then
    show_success "Imágenes construidas exitosamente"
else
    show_error "Error construyendo imágenes Docker"
    exit 1
fi

# Levantar servicios
show_progress "Iniciando pipeline distribuido..."
if docker-compose up -d; then
    show_success "Servicios iniciados"
else
    show_error "Error iniciando servicios"
    exit 1
fi

# Esperar a que los servicios estén listos
show_progress "Esperando inicialización de servicios..."
sleep 15

# Verificar estado de etcd
show_progress "Verificando etcd..."
max_retries=10
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 endpoint health >/dev/null 2>&1; then
        show_success "etcd está funcionando"
        break
    else
        retry_count=$((retry_count + 1))
        echo "  Intento $retry_count/$max_retries..."
        sleep 2
    fi
done

if [ $retry_count -eq $max_retries ]; then
    show_error "etcd no responde después de $max_retries intentos"
    echo ""
    show_warning "Logs de etcd:"
    docker-compose logs etcd | tail -20
    exit 1
fi

# Verificar servicios registrados
show_progress "Verificando registro de servicios..."
sleep 5

services_registered=0
if docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get --prefix /services/heartbeat/ >/dev/null 2>&1; then
    services_count=$(docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get --prefix /services/heartbeat/ --keys-only 2>/dev/null | wc -l)
    show_success "Servicios registrados: $services_count"
    services_registered=1
else
    show_warning "Servicios aún registrándose..."
fi

# Verificar comunicación ZeroMQ
show_progress "Verificando comunicación ZeroMQ..."
sleep 3

if timeout 5 docker-compose logs service2 2>/dev/null | grep -q -i "eventos procesados\|evento sospechoso\|processed"; then
    show_success "Comunicación ZeroMQ funcionando"
else
    show_warning "Comunicación ZeroMQ aún estableciéndose..."
fi

# Mostrar estado final
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 DDOS Pipeline Iniciado${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Mostrar información de servicios
echo -e "${PURPLE}📊 Estado de servicios:${NC}"
docker-compose ps

echo ""
echo -e "${PURPLE}🌐 Servicios disponibles:${NC}"
echo "  - etcd API:           http://$(hostname -I | awk '{print $1}'):2379"
echo "  - Service Discovery:  Automático vía etcd"
echo "  - ZeroMQ Pipeline:    service1 → service2"

echo ""
echo -e "${PURPLE}🔧 Comandos útiles:${NC}"
echo "  docker-compose logs -f           # Ver todos los logs"
echo "  docker-compose logs -f service1  # Logs de packet sniffer"
echo "  docker-compose logs -f service2  # Logs de feature processor"
echo "  docker-compose logs -f etcd      # Logs de etcd"
echo ""
echo "  # Ver servicios registrados en etcd:"
echo "  docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get --prefix /services/"
echo ""
echo "  # Ver estadísticas de servicios:"
echo "  docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get /services/config/packet_sniffer/stats"
echo "  docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get /services/config/feature_processor/stats"

# Función para mostrar actividad en tiempo real
show_activity() {
    echo ""
    echo -e "${BLUE}📋 Mostrando actividad del pipeline (Ctrl+C para salir)...${NC}"
    echo ""
    docker-compose logs -f --tail=20
}

# Verificar si hay argumentos para mostrar logs
if [ "$1" = "--logs" ] || [ "$1" = "-l" ]; then
    show_activity
elif [ "$1" = "--status" ] || [ "$1" = "-s" ]; then
    echo ""
    echo -e "${BLUE}📊 Estado detallado:${NC}"
    echo ""
    echo -e "${YELLOW}Containers:${NC}"
    docker-compose ps
    echo ""
    echo -e "${YELLOW}etcd Health:${NC}"
    docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 endpoint health
    echo ""
    echo -e "${YELLOW}Servicios Registrados:${NC}"
    docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get --prefix /services/heartbeat/
elif [ "$1" = "--test" ] || [ "$1" = "-t" ]; then
    echo ""
    echo -e "${BLUE}🧪 Ejecutando tests de comunicación...${NC}"
    sleep 10  # Dar más tiempo para que los servicios se estabilicen

    echo ""
    echo -e "${YELLOW}Test 1: Verificar eventos generados por service1${NC}"
    if timeout 5 docker-compose logs service1 | grep -q "Eventos enviados"; then
        echo -e "${GREEN}✅ Service1 está generando eventos${NC}"
    else
        echo -e "${RED}❌ Service1 no está generando eventos${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Test 2: Verificar procesamiento en service2${NC}"
    if timeout 5 docker-compose logs service2 | grep -q -i "procesado\|sospechoso"; then
        echo -e "${GREEN}✅ Service2 está procesando eventos${NC}"
    else
        echo -e "${RED}❌ Service2 no está procesando eventos${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Test 3: Verificar detección de eventos sospechosos${NC}"
    if timeout 10 docker-compose logs service2 | grep -q "EVENTO SOSPECHOSO"; then
        echo -e "${GREEN}✅ Sistema está detectando eventos sospechosos${NC}"
    else
        echo -e "${YELLOW}⚠️  No se han detectado eventos sospechosos aún${NC}"
    fi

    echo ""
    echo -e "${BLUE}📊 Estadísticas actuales:${NC}"
    docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get /services/config/packet_sniffer/stats 2>/dev/null | tail -1 || echo "Stats de packet_sniffer no disponibles"
    docker-compose exec etcd /usr/local/bin/etcdctl --endpoints=http://localhost:2379 get /services/config/feature_processor/stats 2>/dev/null | tail -1 || echo "Stats de feature_processor no disponibles"
else
    echo ""
    echo -e "${YELLOW}💡 Opciones adicionales:${NC}"
    echo "  ./build_and_run.sh --logs     # Ver logs en tiempo real"
    echo "  ./build_and_run.sh --status   # Ver estado detallado"
    echo "  ./build_and_run.sh --test     # Ejecutar tests de comunicación"
fi

echo ""
echo -e "${GREEN}✅ Pipeline DDOS listo para detectar amenazas!${NC}"