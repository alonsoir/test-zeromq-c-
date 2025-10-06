#!/bin/bash
# pre-merge-validation.sh
# Script de validación completa antes de merge a main
# Verifica: Vagrant, interfaces de red, eBPF, compilación, y captura de tráfico

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Contadores
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Funciones de utilidad
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

run_test() {
    ((TESTS_TOTAL++))
    local test_name="$1"
    local test_cmd="$2"

    log_info "Ejecutando: $test_name"

    if eval "$test_cmd" > /tmp/test_output.log 2>&1; then
        log_success "$test_name"
        return 0
    else
        log_error "$test_name"
        log_warning "Output del comando:"
        cat /tmp/test_output.log
        return 1
    fi
}

separator() {
    echo ""
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

# Inicio de validación
separator "VALIDACIÓN PRE-MERGE - BRANCH: feature/eth0-vagrantfile"

# 1. VERIFICAR QUE ESTAMOS EN LA BRANCH CORRECTA
separator "1. Verificación de Branch y Git"

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "feature/eth0-vagrantfile" ]; then
    log_error "No estás en la branch feature/eth0-vagrantfile (actual: $current_branch)"
    exit 1
fi
log_success "Branch correcta: $current_branch"

# Verificar estado git
if ! git diff-index --quiet HEAD --; then
    log_warning "Hay cambios sin commitear"
    git status --short
else
    log_success "Working directory limpio"
fi

# 2. VERIFICAR ESTADO DE VAGRANT
separator "2. Verificación de Vagrant"

log_info "Verificando estado de Vagrant..."
VAGRANT_STATUS=$(vagrant status | grep "zeromq-etcd-lab-debian" | awk '{print $2}')

if [ "$VAGRANT_STATUS" != "running" ]; then
    log_warning "Vagrant no está corriendo. Iniciando VM..."
    vagrant up
    sleep 5
else
    log_success "Vagrant está corriendo"
fi

# 3. VERIFICAR INTERFACES DE RED
separator "3. Verificación de Interfaces de Red"

log_info "Detectando interfaces de red en la VM..."

# Obtener información de interfaces
vagrant ssh -c "ip -4 addr show" > /tmp/interfaces.txt

# Verificar eth0 (NAT)
if grep -q "eth0" /tmp/interfaces.txt; then
    ETH0_IP=$(vagrant ssh -c "ip -4 addr show eth0 | grep inet | awk '{print \$2}' | cut -d'/' -f1")
    log_success "eth0 (NAT) detectada: $ETH0_IP"
else
    log_error "eth0 (NAT) no detectada"
fi

# Verificar eth1 (Private Network)
if grep -q "eth1" /tmp/interfaces.txt; then
    ETH1_IP=$(vagrant ssh -c "ip -4 addr show eth1 | grep inet | awk '{print \$2}' | cut -d'/' -f1")
    if [ "$ETH1_IP" == "192.168.56.20" ]; then
        log_success "eth1 (Private) correcta: $ETH1_IP"
    else
        log_error "eth1 tiene IP incorrecta: $ETH1_IP (esperada: 192.168.56.20)"
    fi
else
    log_error "eth1 (Private Network) no detectada"
fi

# Verificar eth2 (Bridged)
if grep -q "eth2" /tmp/interfaces.txt; then
    ETH2_IP=$(vagrant ssh -c "ip -4 addr show eth2 | grep inet | awk '{print \$2}' | cut -d'/' -f1" 2>/dev/null || echo "")
    if [ -n "$ETH2_IP" ]; then
        log_success "eth2 (Bridged) detectada: $ETH2_IP"

        # Verificar que eth2 tiene acceso a la red
        log_info "Probando conectividad desde eth2..."
        if vagrant ssh -c "ping -c 2 -I eth2 8.8.8.8" > /dev/null 2>&1; then
            log_success "eth2 tiene conectividad a Internet"
        else
            log_warning "eth2 no tiene conectividad a Internet (puede ser normal según firewall)"
        fi
    else
        log_error "eth2 existe pero no tiene IP asignada"
    fi
else
    log_error "eth2 (Bridged) no detectada - CRÍTICO para el sniffer"
fi

# Mostrar tabla de routing
log_info "Tabla de routing:"
vagrant ssh -c "ip route" | while read line; do
    echo "  $line"
done

# 4. VERIFICAR CONFIGURACIÓN eBPF
separator "4. Verificación de Configuración eBPF"

run_test "BPF JIT habilitado" \
    "vagrant ssh -c 'cat /proc/sys/net/core/bpf_jit_enable' | grep -q '1'"

run_test "BPF filesystem montado" \
    "vagrant ssh -c 'mountpoint -q /sys/fs/bpf'"

run_test "BPF en fstab (persistencia)" \
    "vagrant ssh -c 'grep -q \"/sys/fs/bpf\" /etc/fstab'"

# Ejecutar make verify-bpf si existe
if vagrant ssh -c "cd /vagrant && make -n verify-bpf" > /dev/null 2>&1; then
    run_test "Make verify-bpf" \
        "vagrant ssh -c 'cd /vagrant && make verify-bpf'"
fi

# 5. VERIFICAR DOCKER Y DEPENDENCIAS
separator "5. Verificación de Docker y Dependencias"

run_test "Docker instalado" \
    "vagrant ssh -c 'command -v docker'"

run_test "Docker en ejecución" \
    "vagrant ssh -c 'sudo systemctl is-active docker'"

run_test "Docker Compose instalado" \
    "vagrant ssh -c 'command -v docker-compose'"

run_test "Usuario vagrant en grupo docker" \
    "vagrant ssh -c 'groups vagrant | grep -q docker'"

# 6. COMPILACIÓN DEL SNIFFER
separator "6. Compilación del Sniffer"

log_info "Limpiando compilación anterior..."
vagrant ssh -c "cd /vagrant && make sniffer-clean" > /dev/null 2>&1 || true

run_test "Compilación del sniffer" \
    "vagrant ssh -c 'cd /vagrant && make sniffer-build-local'"

run_test "Binario del sniffer existe" \
    "vagrant ssh -c 'test -f /vagrant/sniffer/build/sniffer'"

# Verificar que el sniffer puede leer la configuración
run_test "Sniffer puede leer configuración" \
    "vagrant ssh -c 'cd /vagrant && timeout 2 sudo ./sniffer/build/sniffer --help || true' | grep -q 'eBPF'"

# 7. VERIFICAR ARCHIVOS DE CONFIGURACIÓN
separator "7. Verificación de Archivos de Configuración"

run_test "sniffer.json existe" \
    "vagrant ssh -c 'test -f /vagrant/sniffer/config/sniffer.json'"

run_test "sniffer.json es JSON válido" \
    "vagrant ssh -c 'python3 -m json.tool /vagrant/sniffer/config/sniffer.json' > /dev/null"

run_test "sniffer-proposal.json existe" \
    "vagrant ssh -c 'test -f /vagrant/sniffer/config/sniffer-proposal.json'"

# 8. TEST DE CAPTURA EN ETH2
separator "8. Test de Captura de Tráfico en eth2"

if [ -n "$ETH2_IP" ]; then
    log_info "Iniciando captura de prueba en eth2 (10 segundos)..."

    # Crear directorio de capturas si no existe
    vagrant ssh -c "mkdir -p /tmp/zeromq_captures"

    # Iniciar captura en background
    log_info "Ejecutando tcpdump en eth2..."
    vagrant ssh -c "sudo timeout 10 tcpdump -i eth2 -c 10 -w /tmp/test_capture.pcap" > /tmp/tcpdump_output.log 2>&1 &
    TCPDUMP_PID=$!

    # Generar tráfico de prueba
    sleep 2
    log_info "Generando tráfico de prueba desde eth2..."
    vagrant ssh -c "ping -c 5 -I eth2 8.8.8.8" > /dev/null 2>&1 || log_warning "No se pudo generar tráfico ICMP"

    # Esperar a que termine tcpdump
    wait $TCPDUMP_PID 2>/dev/null || true

    # Verificar captura
    if vagrant ssh -c "test -f /tmp/test_capture.pcap && test -s /tmp/test_capture.pcap"; then
        PACKET_COUNT=$(vagrant ssh -c "sudo tcpdump -r /tmp/test_capture.pcap 2>/dev/null | wc -l")
        if [ "$PACKET_COUNT" -gt 0 ]; then
            log_success "Captura en eth2 exitosa: $PACKET_COUNT paquetes capturados"
        else
            log_error "No se capturaron paquetes en eth2"
        fi
    else
        log_error "Archivo de captura vacío o no existe"
    fi

    # Limpiar
    vagrant ssh -c "sudo rm -f /tmp/test_capture.pcap"
else
    log_error "No se puede probar captura: eth2 no tiene IP"
fi

# 9. VERIFICAR SCRIPTS
separator "9. Verificación de Scripts"

REQUIRED_SCRIPTS=(
    "scripts/network_diagnostics.sh"
    "scripts/capture_zeromq_traffic.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    run_test "Script existe: $script" \
        "vagrant ssh -c 'test -f /vagrant/$script'"

    run_test "Script es ejecutable: $script" \
        "vagrant ssh -c 'test -x /vagrant/$script'"
done

# 10. TEST DEL PIPELINE COMPLETO (OPCIONAL)
separator "10. Test del Pipeline ZeroMQ (Opcional)"

read -p "¿Ejecutar test del pipeline completo? (s/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    log_info "Iniciando pipeline ZeroMQ..."

    if vagrant ssh -c "cd /vagrant && make lab-start" > /tmp/lab_output.log 2>&1; then
        sleep 10

        # Verificar contenedores
        RUNNING_CONTAINERS=$(vagrant ssh -c "docker ps --format '{{.Names}}'" | wc -l)
        if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
            log_success "Pipeline iniciado: $RUNNING_CONTAINERS contenedores corriendo"
            vagrant ssh -c "docker ps --format 'table {{.Names}}\t{{.Status}}'"
        else
            log_error "No hay contenedores corriendo"
        fi

        # Detener pipeline
        log_info "Deteniendo pipeline..."
        vagrant ssh -c "cd /vagrant && make lab-stop" > /dev/null 2>&1
    else
        log_error "Fallo al iniciar pipeline"
        cat /tmp/lab_output.log
    fi
else
    log_info "Test de pipeline omitido"
fi

# RESUMEN FINAL
separator "RESUMEN DE VALIDACIÓN"

echo ""
echo "Total de tests ejecutados: $TESTS_TOTAL"
echo -e "${GREEN}Tests exitosos: $TESTS_PASSED${NC}"
echo -e "${RED}Tests fallidos: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    log_success "¡TODAS LAS VALIDACIONES PASARON!"
    echo ""
    echo "Próximos pasos para merge a main:"
    echo "  1. git checkout main"
    echo "  2. git merge feature/eth0-vagrantfile"
    echo "  3. git tag -a v3.2.0 -m 'Version 3.2.0 - eth2 bridged network support'"
    echo "  4. git push origin main"
    echo "  5. git push origin v3.2.0"
    echo ""
    exit 0
else
    log_error "ALGUNAS VALIDACIONES FALLARON"
    echo ""
    echo "Por favor, revisa los errores antes de hacer el merge."
    echo ""
    exit 1
fi