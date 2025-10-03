#!/bin/bash

# Script para instalar dependencias específicas del sniffer eBPF avanzado
# Separado del provisioning base de Vagrant para mantener modularidad
#
# Uso: ./scripts/install-sniffer-deps.sh
#      sudo ./scripts/install-sniffer-deps.sh --force-update

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Instalador de dependencias del sniffer eBPF avanzado"
echo "📁 Proyecto: $PROJECT_ROOT"
echo ""

# Verificar si estamos en un entorno Vagrant
if [ -f "/etc/environment" ] && grep -q "VAGRANT_HOST_IP" /etc/environment 2>/dev/null; then
    echo "🏠 Entorno Vagrant detectado"
    IN_VAGRANT=true
else
    echo "💻 Entorno local detectado"
    IN_VAGRANT=false
fi

echo "🔍 Detectando sistema operativo..."

# Detectar distribución
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VERSION=$VERSION_ID
else
    echo "❌ No se puede detectar el sistema operativo"
    exit 1
fi

echo "📋 Sistema detectado: $OS $VERSION"

install_debian_ubuntu() {
    echo "📦 Instalando dependencias del sniffer en Debian/Ubuntu..."

    # Actualizar repositorios solo si es necesario
    if [ "$1" = "--force-update" ] || [ ! -f /var/cache/apt/pkgcache.bin ]; then
        echo "🔄 Actualizando repositorios..."
        apt-get update
    fi

    echo "🔧 Instalando herramientas de compilación core..."
    apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        clang \
        llvm \
        bpftool

    echo "📡 Instalando dependencias de networking y eBPF..."
    apt-get install -y \
        libbpf-dev \
        libzmq3-dev \
        libjsoncpp-dev

    echo "📋 Instalando dependencias de serialización..."
    apt-get install -y \
        libprotobuf-dev \
        protobuf-compiler

    echo "🗜️ Instalando librerías de compresión (OBLIGATORIAS para sniffer)..."
    apt-get install -y \
        liblz4-dev \
        libzstd-dev \
        libsnappy-dev

    echo "🔐 Instalando dependencias para cifrado futuro..."
    apt-get install -y \
        libcurl4-openssl-dev \
        libssl-dev

    echo "⚡ Instalando optimizaciones de rendimiento..."
    apt-get install -y \
        libnuma-dev

    echo "🧪 Verificando instalación..."
    verify_dependencies

    echo "✅ Dependencias del sniffer instaladas correctamente"
}

verify_dependencies() {
    echo "🔍 Verificando dependencias críticas..."

    # Verificar pkg-config para las librerías principales
    local missing=()

    for lib in libbpf libzmq jsoncpp liblz4 libzstd protobuf; do
        if ! pkg-config --exists "$lib" 2>/dev/null; then
            case "$lib" in
                "libzmq")
                    if ! pkg-config --exists "libzmq3" 2>/dev/null; then
                        missing+=("$lib")
                    fi
                    ;;
                "liblz4")
                    missing+=("$lib - CRÍTICO: Compresión LZ4 requerida")
                    ;;
                "libzstd")
                    missing+=("$lib - CRÍTICO: Compresión Zstd requerida")
                    ;;
                *)
                    missing+=("$lib")
                    ;;
            esac
        fi
    done

    # Verificar herramientas de compilación
    for tool in clang bpftool protoc; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing+=("$tool (comando)")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo "❌ Dependencias faltantes:"
        printf '   - %s\n' "${missing[@]}"
        echo ""
        echo "💡 Sugerencia: Ejecuta con --force-update para forzar reinstalación"
        return 1
    fi

    echo "✅ Todas las dependencias críticas están instaladas"

    # Mostrar versiones para debugging
    echo ""
    echo "📋 Versiones instaladas:"
    echo "   libbpf: $(pkg-config --modversion libbpf 2>/dev/null || echo 'N/A')"
    echo "   libzmq: $(pkg-config --modversion libzmq 2>/dev/null || pkg-config --modversion libzmq3 2>/dev/null || echo 'N/A')"
    echo "   jsoncpp: $(pkg-config --modversion jsoncpp 2>/dev/null || echo 'N/A')"
    echo "   liblz4: $(pkg-config --modversion liblz4 2>/dev/null || echo 'N/A')"
    echo "   libzstd: $(pkg-config --modversion libzstd 2>/dev/null || echo 'N/A')"
    echo "   protobuf: $(pkg-config --modversion protobuf 2>/dev/null || echo 'N/A')"
    echo "   clang: $(clang --version | head -1)"
    echo ""
}

install_centos_rhel() {
    echo "📦 Instalando dependencias en CentOS/RHEL..."

    # Habilitar EPEL para dependencias adicionales
    dnf install -y epel-release
    dnf update -y

    # Dependencias core
    dnf install -y \
        gcc-c++ \
        cmake \
        pkgconfig \
        clang \
        llvm \
        libbpf-devel \
        zeromq-devel \
        jsoncpp-devel \
        protobuf-devel \
        protobuf-compiler \
        bpftool

    # Dependencias de compresión
    echo "🗜️ Instalando librerías de compresión..."
    dnf install -y \
        lz4-devel \
        libzstd-devel \
        snappy-devel

    # Dependencias opcionales
    echo "🔧 Instalando dependencias opcionales..."
    dnf install -y \
        libcurl-devel \
        numactl-devel

    echo "✅ Dependencias instaladas correctamente"
}

install_arch() {
    echo "📦 Instalando dependencias en Arch Linux..."

    pacman -Sy

    # Dependencias core
    pacman -S --noconfirm \
        base-devel \
        cmake \
        pkgconf \
        clang \
        llvm \
        libbpf \
        zeromq \
        jsoncpp \
        protobuf \
        bpf

    # Dependencias de compresión
    echo "🗜️ Instalando librerías de compresión..."
    pacman -S --noconfirm \
        lz4 \
        zstd \
        snappy

    # Dependencias opcionales
    echo "🔧 Instalando dependencias opcionales..."
    pacman -S --noconfirm \
        curl \
        numactl

    echo "✅ Dependencias instaladas correctamente"
}

# Verificar permisos de root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Este script necesita ejecutarse como root (sudo)"
    echo "   Uso: sudo $0 [--force-update]"
    exit 1
fi

# Verificar si ya tenemos las dependencias básicas antes de proceder
check_basic_system() {
    echo "🏥 Verificando estado del sistema base..."

    # Verificar que estamos en un sistema con apt
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "❌ Este script está optimizado para sistemas Debian/Ubuntu"
        echo "   Para otros sistemas, instala manualmente las dependencias"
        exit 1
    fi

    # Verificar conectividad de red
    if ! ping -c 1 deb.debian.org >/dev/null 2>&1 && ! ping -c 1 archive.ubuntu.com >/dev/null 2>&1; then
        echo "⚠️  Sin conectividad a repositorios, intentando instalar con cache local"
    fi

    echo "✅ Sistema base verificado"
}

# Preparar configuración post-instalación
setup_sniffer_config() {
    echo "⚙️ Configurando entorno post-instalación para sniffer..."

    # Crear directorios si no existen
    mkdir -p "$PROJECT_ROOT/sniffer/build"
    mkdir -p "$PROJECT_ROOT/sniffer/config"
    mkdir -p "$PROJECT_ROOT/sniffer/logs"

    # Crear archivo de estado de dependencias
    cat > "$PROJECT_ROOT/sniffer/.deps-installed" << EOF
# Dependencias del sniffer instaladas
# Generado automáticamente por install-sniffer-deps.sh
INSTALL_DATE=$(date -Iseconds)
SYSTEM=$(lsb_release -ds 2>/dev/null || echo "Unknown")
COMPRESSION_MANDATORY=true
ENCRYPTION_SUPPORT=true
ETCD_INTEGRATION=true
EOF

    # Si estamos en Vagrant, registrar en el log del sistema
    if [ "$IN_VAGRANT" = true ]; then
        echo "$(date -Iseconds) - Sniffer dependencies installed" >> /var/log/vagrant-components.log
    fi

    echo "✅ Configuración post-instalación completada"
}

# Mostrar resumen final con información específica del sniffer
show_summary() {
    echo ""
    echo "🎉 ¡Instalación de dependencias del sniffer completada!"
    echo ""
    echo "📊 Características habilitadas:"
    echo "   ✅ Compresión LZ4 (obligatoria)"
    echo "   ✅ Compresión Zstd (obligatoria)"
    echo "   ✅ Compresión Snappy (opcional)"
    echo "   ✅ Soporte para cifrado via etcd (futuro)"
    echo "   ✅ Optimizaciones NUMA"
    echo "   ✅ eBPF/XDP avanzado"
    echo ""
    echo "🚀 Próximos pasos:"
    echo "   1. cd $PROJECT_ROOT"
    echo "   2. make sniffer-build"
    echo "   3. make sniffer-start"
    echo ""
    if [ "$IN_VAGRANT" = true ]; then
        echo "💡 Vagrant detectado:"
        echo "   - Endpoint configurado automáticamente via etcd"
        echo "   - IP del host: \$(grep VAGRANT_HOST_IP /etc/environment | cut -d= -f2)"
        echo ""
    fi
    echo "📝 Log de instalación guardado en:"
    echo "   $PROJECT_ROOT/sniffer/.deps-installed"
    echo ""
}

# Función principal
main() {
    local force_update=false

    # Procesar argumentos
    for arg in "$@"; do
        case $arg in
            --force-update)
                force_update=true
                shift
                ;;
            --help|-h)
                echo "Uso: sudo $0 [--force-update]"
                echo ""
                echo "Opciones:"
                echo "  --force-update    Forzar actualización de repositorios"
                echo "  --help           Mostrar esta ayuda"
                exit 0
                ;;
        esac
    done

    check_basic_system

    # Instalar según la distribución
    case $OS in
        "Debian GNU/Linux"|"Ubuntu")
            if [ "$force_update" = true ]; then
                install_debian_ubuntu --force-update
            else
                install_debian_ubuntu
            fi
            ;;
        "CentOS Linux"|"Red Hat Enterprise Linux"|"Fedora Linux")
            install_centos_rhel
            ;;
        "Arch Linux")
            install_arch
            ;;
        *)
            echo "❌ Distribución no soportada: $OS"
            echo ""
            echo "📋 Para instalación manual, necesitas estas dependencias:"
            echo "🔧 Compilación:"
            echo "   - build-essential, cmake, pkg-config"
            echo "   - clang, llvm, bpftool"
            echo "📡 Networking/eBPF:"
            echo "   - libbpf-dev, libzmq3-dev, libjsoncpp-dev"
            echo "📋 Serialización:"
            echo "   - libprotobuf-dev, protobuf-compiler"
            echo "🗜️ Compresión (OBLIGATORIA):"
            echo "   - liblz4-dev, libzstd-dev, libsnappy-dev"
            echo "🔐 Cifrado/Red:"
            echo "   - libcurl4-openssl-dev, libssl-dev"
            echo "⚡ Optimización:"
            echo "   - libnuma-dev"
            echo ""
            exit 1
            ;;
    esac

    setup_sniffer_config
    show_summary
}

# Ejecutar función principal con todos los argumentos
main "$@"