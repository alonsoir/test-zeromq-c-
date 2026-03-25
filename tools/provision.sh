#!/usr/bin/env bash
# =============================================================================
# tools/provision.sh
# ML Defender — Cryptographic Provisioning Tool
#
# Genera keypairs Ed25519 + seeds ChaCha20 para todos los componentes
# del pipeline y sus plugins declarados en los JSONs de contrato.
#
# REQUIERE: sudo (escribe en /etc/ml-defender/ con chmod 0600)
# COMPATIBLE: AppArmor desde el primer día — paths fijos, nunca dinámicos
#             (ADR-019: /etc/ml-defender/{component}/ es el path canónico)
#
# PHASE 1 (DAY 95): keypairs Ed25519 + seed aleatorio en disco (chmod 0600)
# PHASE 2 (futuro): seed.enc cifrado con clave pública del receptor (seed-client)
#                   La HMAC key NO estará nunca en el JSON — se deriva del
#                   intercambio de keypairs en provisioning (ADR-013)
#
# USO:
#   sudo bash tools/provision.sh full                    # primer arranque
#   sudo bash tools/provision.sh status                  # tabla de estado
#   sudo bash tools/provision.sh verify                  # verifica todo (CI/CD)
#   sudo bash tools/provision.sh reprovision sniffer     # re-provisiona uno
#
# Authors: Alonso Isidoro Roman + Claude (Anthropic)
# DAY 97 — 25 marzo 2026
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURACIÓN — SINGLE SOURCE OF TRUTH
# Estos paths son los que AppArmor permitirá (ADR-019).
# Si cambias un path aquí, debes actualizar los perfiles AppArmor también.
# =============================================================================

readonly KEYS_ROOT="/etc/ml-defender"
readonly SEED_BYTES=32           # ChaCha20 requiere exactamente 32 bytes
readonly KEY_ALGORITHM="ed25519" # Ed25519 para keypairs de identidad

# Componentes del pipeline en orden de dependencia
# etcd-server primero — es la autoridad de semillas
readonly COMPONENTS=(
    "etcd-server"
    "sniffer"
    "ml-detector"
    "firewall-acl-agent"
    "rag-ingester"
    "rag-security"
)

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m' # No Color

# =============================================================================
# UTILIDADES
# =============================================================================

log_info()    { echo -e "${GREEN}  ✅${NC} $*"; }
log_warn()    { echo -e "${YELLOW}  ⚠️ ${NC} $*"; }
log_error()   { echo -e "${RED}  ❌${NC} $*" >&2; }
log_section() { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}"; }
log_item()    { echo -e "     ${CYAN}→${NC} $*"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "provision.sh debe ejecutarse con sudo"
        echo "  Uso: sudo bash /vagrant/tools/provision.sh <modo>"
        exit 1
    fi
}

check_dependencies() {
    local missing=()
    command -v openssl    >/dev/null 2>&1 || missing+=("openssl")
    command -v jq         >/dev/null 2>&1 || missing+=("jq")
    command -v wget       >/dev/null 2>&1 || missing+=("wget")
    command -v pkg-config >/dev/null 2>&1 || missing+=("pkg-config")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Dependencias faltantes: ${missing[*]}"
        echo "  Instalar: apt-get install -y openssl jq wget pkg-config"
        exit 1
    fi

    # Verificar que OpenSSL soporta Ed25519
    if ! openssl genpkey -algorithm ed25519 -out /dev/null 2>/dev/null; then
        log_error "OpenSSL no soporta Ed25519. Requiere OpenSSL >= 1.1.1"
        openssl version
        exit 1
    fi
}

# =============================================================================
# ENTROPÍA — DEBT-CRYPTO-003b
# Garantiza suficiente entropía antes de generar material criptográfico.
# Si el pool es bajo, instala haveged como fuente de entropía adicional.
# =============================================================================

check_entropy() {
    log_section "Verificación de entropía del sistema"

    local avail
    avail=$(cat /proc/sys/kernel/random/entropy_avail 2>/dev/null || echo 999)
    log_item "Entropía disponible: ${avail} bits"

    if [[ "$avail" -lt 256 ]]; then
        log_warn "Entropía baja (${avail} < 256 bits) — instalando haveged"
        apt-get install -y --quiet haveged >/dev/null 2>&1
        systemctl enable haveged >/dev/null 2>&1
        systemctl start  haveged >/dev/null 2>&1
        sleep 1
        avail=$(cat /proc/sys/kernel/random/entropy_avail)
        log_info "Entropía tras haveged: ${avail} bits"
    else
        log_info "Entropía suficiente (${avail} bits) — no se requiere haveged"
    fi
}

# =============================================================================
# LIBSODIUM 1.0.19 — Compilación desde fuente
#
# Debian Bookworm solo distribuye libsodium 1.0.18.
# crypto_kdf_hkdf_sha256_* (HKDF nativo) requiere >= 1.0.19 (ADR-013 PHASE 2).
#
# Función idempotente: salta si la versión correcta ya está instalada.
# Verificación SHA-256 obligatoria antes de compilar — supply-chain safety.
#
# SHA-256 verificado el 2026-03-25 contra:
#   https://download.libsodium.org/libsodium/releases/libsodium-1.0.19.tar.gz
# =============================================================================

readonly SODIUM_REQUIRED="1.0.19"
readonly SODIUM_URL="https://download.libsodium.org/libsodium/releases/libsodium-${SODIUM_REQUIRED}.tar.gz"
readonly SODIUM_SHA256="018d79fe0a045cca07331d37bd0cb57b2e838c51bc48fd837a1472e50068bbea"

install_libsodium_1019() {
    log_section "libsodium ${SODIUM_REQUIRED}"

    # ── Idempotencia: comprobar versión instalada ──────────────────────────
    local installed
    installed=$(pkg-config --modversion libsodium 2>/dev/null || echo "none")

    if [[ "$installed" == "$SODIUM_REQUIRED" ]]; then
        # Verificar además que crypto_kdf_hkdf_sha256 está en los headers
        if grep -r "crypto_kdf_hkdf_sha256_extract" /usr/local/include/sodium/ \
                >/dev/null 2>&1; then
            log_info "libsodium ${SODIUM_REQUIRED} ya instalada con HKDF nativo — saltando"
            return 0
        fi
        log_warn "libsodium ${SODIUM_REQUIRED} instalada pero HKDF headers ausentes — reinstalando"
    else
        log_item "Versión instalada: ${installed}  →  compilando ${SODIUM_REQUIRED} desde fuente"
    fi

    # ── Dependencias de compilación ────────────────────────────────────────
    log_item "Instalando dependencias de compilación"
    apt-get install -y --quiet build-essential wget pkg-config >/dev/null 2>&1

    # ── Directorio temporal (limpiado al salir de la función) ──────────────
    local tmpdir
    tmpdir=$(mktemp -d /tmp/sodium_build_XXXXXX)
    # shellcheck disable=SC2064
    trap "rm -rf ${tmpdir}" RETURN

    # ── Descarga ───────────────────────────────────────────────────────────
    log_item "Descargando libsodium-${SODIUM_REQUIRED}.tar.gz"
    if ! wget -q "${SODIUM_URL}" -O "${tmpdir}/libsodium.tar.gz"; then
        log_error "Descarga fallida: ${SODIUM_URL}"
        log_error "Comprueba conectividad: wget -q ${SODIUM_URL} -O /tmp/test.tar.gz"
        exit 1
    fi

    # ── Verificación SHA-256 — obligatoria, no salteable ──────────────────
    log_item "Verificando integridad SHA-256"
    local actual_sha
    actual_sha=$(sha256sum "${tmpdir}/libsodium.tar.gz" | awk '{print $1}')

    if [[ "$actual_sha" != "$SODIUM_SHA256" ]]; then
        log_error "SHA-256 no coincide — ABORTANDO"
        log_error "  Esperado: ${SODIUM_SHA256}"
        log_error "  Obtenido: ${actual_sha}"
        log_error "NO se instalará libsodium — posible supply-chain attack o descarga corrupta"
        rm -f "${tmpdir}/libsodium.tar.gz"
        exit 1
    fi
    log_info "SHA-256 verificado: ${actual_sha:0:16}…"

    # ── Compilación ────────────────────────────────────────────────────────
    log_item "Compilando libsodium ${SODIUM_REQUIRED} (~2 min en primera ejecución)"
    tar xzf "${tmpdir}/libsodium.tar.gz" -C "${tmpdir}"

    # Verificar que el directorio se extrajo correctamente
    local src_dir="${tmpdir}/libsodium-stable"
    if [[ ! -d "$src_dir" ]]; then
        log_error "tar no extrajo el directorio esperado: ${src_dir}"
        log_error "Contenido de ${tmpdir}: $(ls ${tmpdir})"
        exit 1
    fi

    cd "$src_dir"
    ./configure --prefix=/usr/local --quiet 2>/dev/null
    make -j"$(nproc)" --quiet
    make install --quiet
    cd /tmp

    ldconfig

    # ── Reinstalar dependencias que apt pudo eliminar al quitar libsodium23 ──
    # apt-get remove libsodium23 arrastra libzmq5, libzmq3-dev, cppzmq-dev
    # porque Bookworm los construyó contra libsodium.so.23. Los reinstalamos
    # explícitamente para no romper el resto del pipeline.
    log_item "Verificando dependencias del pipeline (ZeroMQ, vim...)"
    apt-get install -y --quiet         libzmq5 libzmq3-dev cppzmq-dev vim >/dev/null 2>&1
    log_info "Dependencias del pipeline verificadas"

    # ── Verificación post-instalación ─────────────────────────────────────
    local installed_now
    installed_now=$(pkg-config --modversion libsodium 2>/dev/null || echo "none")

    if [[ "$installed_now" != "$SODIUM_REQUIRED" ]]; then
        log_error "Instalación fallida — pkg-config reporta: ${installed_now}"
        log_error "Comprueba que /usr/local/lib está en ldconfig: ldconfig -p | grep sodium"
        exit 1
    fi

    # Verificar que el símbolo HKDF está presente
    if ! grep -r "crypto_kdf_hkdf_sha256_extract" /usr/local/include/sodium/ \
            >/dev/null 2>&1; then
        log_error "crypto_kdf_hkdf_sha256_* no encontrada tras instalación"
        log_error "Headers instalados en: $(ls /usr/local/include/sodium/ 2>/dev/null | head -5)"
        exit 1
    fi

    log_info "libsodium ${installed_now} instalada correctamente"
    log_item "Headers: /usr/local/include/sodium/"
    log_item "Libs:    /usr/local/lib/libsodium.so.${SODIUM_REQUIRED}"
    log_item "HKDF:    crypto_kdf_hkdf_sha256_extract/expand disponibles ✅"
}

# =============================================================================
# OPERACIONES DE CLAVES
# =============================================================================

# Crea el directorio para un componente con permisos correctos (AppArmor-safe)
create_component_dir() {
    local component="$1"
    local dir="${KEYS_ROOT}/${component}"

    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        log_item "Directorio creado: $dir"
    fi

    # Permisos: solo root puede leer el directorio en sí
    chmod 700 "$dir"
    # root:root — los binarios del pipeline corren con sudo donde necesitan claves
    chown root:root "$dir"
}

# Genera keypair Ed25519 para un componente
# Salida: private.pem (0600) + public.pem (0644)
generate_keypair() {
    local component="$1"
    local dir="${KEYS_ROOT}/${component}"
    local private_key="${dir}/private.pem"
    local public_key="${dir}/public.pem"

    if [[ -f "$private_key" ]] && [[ -f "$public_key" ]]; then
        log_warn "Keypair ya existe para ${component} (usa 'reprovision' para regenerar)"
        return 0
    fi

    # Generar clave privada Ed25519
    openssl genpkey -algorithm ed25519 -out "$private_key" 2>/dev/null
    chmod 600 "$private_key"
    chown root:root "$private_key"

    # Extraer clave pública
    openssl pkey -in "$private_key" -pubout -out "$public_key" 2>/dev/null
    chmod 644 "$public_key"
    chown root:root "$public_key"

    log_item "Keypair Ed25519 generado para ${component}"
}

# Genera seed ChaCha20 (32 bytes de entropía del SO)
# PHASE 1: seed en claro con chmod 0600
# PHASE 2: seed cifrado con clave pública del receptor (seed-client, ADR-013)
generate_seed() {
    local component="$1"
    local dir="${KEYS_ROOT}/${component}"
    local seed_file="${dir}/seed.bin"

    if [[ -f "$seed_file" ]]; then
        log_warn "Seed ya existe para ${component} (usa 'reprovision' para regenerar)"
        return 0
    fi

    # 32 bytes de /dev/urandom — fuente de entropía del SO
    openssl rand -out "$seed_file" ${SEED_BYTES}
    chmod 600 "$seed_file"
    chown root:root "$seed_file"

    # Verificación de integridad: el seed debe tener exactamente SEED_BYTES
    local actual_size
    actual_size=$(stat -c%s "$seed_file" 2>/dev/null || stat -f%z "$seed_file" 2>/dev/null)
    if [[ "$actual_size" -ne "$SEED_BYTES" ]]; then
        log_error "Seed incorrecto para ${component}: ${actual_size} bytes (esperado: ${SEED_BYTES})"
        rm -f "$seed_file"
        exit 1
    fi

    # Guardar también en hex para debugging (chmod 600)
    # Solo en PHASE 1 — en PHASE 2 el hex también irá cifrado
    openssl rand -hex ${SEED_BYTES} > "${dir}/seed.hex"
    chmod 600 "${dir}/seed.hex"
    chown root:root "${dir}/seed.hex"

    log_item "Seed ChaCha20 (${SEED_BYTES}B) generado para ${component}"
}

# Genera un fingerprint SHA256 de la clave pública (para verificación rápida)
generate_fingerprint() {
    local component="$1"
    local dir="${KEYS_ROOT}/${component}"
    local public_key="${dir}/public.pem"
    local fingerprint_file="${dir}/fingerprint.txt"

    if [[ ! -f "$public_key" ]]; then
        return 1
    fi

    openssl pkey -in "$public_key" -pubin -outform DER 2>/dev/null \
        | openssl dgst -sha256 -hex \
        | awk '{print $2}' > "$fingerprint_file"

    chmod 644 "$fingerprint_file"
    chown root:root "$fingerprint_file"
}

# Genera metadatos de provisioning (timestamp + version)
generate_metadata() {
    local component="$1"
    local dir="${KEYS_ROOT}/${component}"
    local metadata_file="${dir}/provision_meta.json"

    cat > "$metadata_file" << EOF
{
  "component_id": "${component}",
  "provisioned_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "provisioned_by": "tools/provision.sh",
  "phase": "PHASE_1",
  "phase_note": "seed en claro (chmod 0600). PHASE 2: seed.enc via seed-client (ADR-013)",
  "key_algorithm": "${KEY_ALGORITHM}",
  "seed_bytes": ${SEED_BYTES},
  "keys_dir": "${dir}",
  "libsodium_version": "${SODIUM_REQUIRED}",
  "apparmor_compatible": true,
  "adr_refs": ["ADR-013", "ADR-019", "ADR-020"]
}
EOF
    chmod 644 "$metadata_file"
    chown root:root "$metadata_file"
}

# =============================================================================
# PROVISIONING DE COMPONENTE
# =============================================================================

provision_component() {
    local component="$1"
    echo -e "\n  ${BOLD}${component}${NC}"

    create_component_dir    "$component"
    generate_keypair        "$component"
    generate_seed           "$component"
    generate_fingerprint    "$component"
    generate_metadata       "$component"

    log_info "${component} provisionado correctamente"
}

# Re-provisiona forzando regeneración (borra claves existentes)
reprovision_component() {
    local component="$1"
    local dir="${KEYS_ROOT}/${component}"

    log_warn "Re-provisionando ${component} — borrando claves existentes"

    # Backup antes de borrar (nunca borres sin backup)
    if [[ -d "$dir" ]]; then
        local backup_dir="${dir}.bak.$(date +%Y%m%d_%H%M%S)"
        cp -r "$dir" "$backup_dir"
        chmod 700 "$backup_dir"
        log_item "Backup creado: $backup_dir"

        rm -f "${dir}/private.pem" \
              "${dir}/public.pem" \
              "${dir}/seed.bin" \
              "${dir}/seed.hex" \
              "${dir}/fingerprint.txt" \
              "${dir}/provision_meta.json"
    fi

    provision_component "$component"
    log_warn "Re-provisioning completo. Los demás componentes que intercambiaban claves con ${component} pueden necesitar re-provisioning también."
}

# =============================================================================
# PLUGINS USERSPACE (ADR-017)
# Lee los JSONs de componentes y provisiona los plugins declarados
# =============================================================================

provision_plugins() {
    local vagrant_root="/vagrant"
    local plugins_provisioned=0

    log_section "Plugins Userspace (ADR-017)"

    # Buscar plugins declarados en los JSONs de componentes
    local component_configs=(
        "${vagrant_root}/sniffer/config/sniffer.json"
        "${vagrant_root}/ml-detector/config/ml_detector_config.json"
        "${vagrant_root}/firewall-acl-agent/config/firewall.json"
        "${vagrant_root}/rag-ingester/config/rag-ingester.json"
        "${vagrant_root}/rag/config/rag-config.json"
        "${vagrant_root}/etcd-server/config/etcd-server.json"
    )

    for config in "${component_configs[@]}"; do
        if [[ ! -f "$config" ]]; then
            continue
        fi

        local plugins
        plugins=$(jq -r '.identity.plugins[]? // empty' "$config" 2>/dev/null || true)

        if [[ -n "$plugins" ]]; then
            while IFS= read -r plugin_id; do
                echo -e "\n  ${BOLD}plugin: ${plugin_id}${NC}"
                create_component_dir "plugins/${plugin_id}"
                generate_keypair     "plugins/${plugin_id}"
                generate_seed        "plugins/${plugin_id}"
                generate_fingerprint "plugins/${plugin_id}"
                generate_metadata    "plugins/${plugin_id}"
                log_info "Plugin ${plugin_id} provisionado"
                ((plugins_provisioned++))
            done <<< "$plugins"
        fi
    done

    if [[ $plugins_provisioned -eq 0 ]]; then
        log_item "No hay plugins declarados en los JSONs (PHASE 2 pendiente)"
        log_item "Directorio /etc/ml-defender/plugins/ creado y listo"
        mkdir -p "${KEYS_ROOT}/plugins"
        chmod 755 "${KEYS_ROOT}/plugins"
    fi
}

# =============================================================================
# PLUGINS eBPF (ADR-018)
# =============================================================================

provision_ebpf_plugins() {
    local ebpf_config="/vagrant/sniffer/config/kernel_telemetry.json"

    log_section "Plugins eBPF (ADR-018)"

    mkdir -p "${KEYS_ROOT}/ebpf-plugins"
    chmod 755 "${KEYS_ROOT}/ebpf-plugins"

    if [[ ! -f "$ebpf_config" ]]; then
        log_item "kernel_telemetry.json no encontrado (ADR-018 pendiente)"
        log_item "Directorio /etc/ml-defender/ebpf-plugins/ creado y listo"
        return 0
    fi

    local ebpf_programs
    ebpf_programs=$(jq -r '.programs[]?.id // empty' "$ebpf_config" 2>/dev/null || true)

    if [[ -z "$ebpf_programs" ]]; then
        log_item "No hay programas eBPF declarados en kernel_telemetry.json"
        return 0
    fi

    while IFS= read -r prog_id; do
        echo -e "\n  ${BOLD}ebpf-plugin: ${prog_id}${NC}"
        create_component_dir "ebpf-plugins/${prog_id}"
        generate_keypair     "ebpf-plugins/${prog_id}"
        generate_seed        "ebpf-plugins/${prog_id}"
        generate_fingerprint "ebpf-plugins/${prog_id}"
        generate_metadata    "ebpf-plugins/${prog_id}"
        log_info "eBPF plugin ${prog_id} provisionado"
    done <<< "$ebpf_programs"
}

# =============================================================================
# VERIFICACIÓN
# =============================================================================

verify_component() {
    local component="$1"
    local dir="${KEYS_ROOT}/${component}"
    local ok=true

    local private_key="${dir}/private.pem"
    local public_key="${dir}/public.pem"
    local seed_file="${dir}/seed.bin"

    [[ -f "$private_key" ]] || { log_error "${component}: private.pem ausente";  ok=false; }
    [[ -f "$public_key"  ]] || { log_error "${component}: public.pem ausente";   ok=false; }
    [[ -f "$seed_file"   ]] || { log_error "${component}: seed.bin ausente";     ok=false; }

    if [[ "$ok" == "false" ]]; then
        return 1
    fi

    # Verificar permisos
    local priv_perms
    priv_perms=$(stat -c "%a" "$private_key" 2>/dev/null || stat -f "%OLp" "$private_key" 2>/dev/null)
    if [[ "$priv_perms" != "600" ]]; then
        log_error "${component}: private.pem permisos incorrectos (${priv_perms}, esperado 600)"
        ok=false
    fi

    # Verificar tamaño del seed
    local seed_size
    seed_size=$(stat -c%s "$seed_file" 2>/dev/null || stat -f%z "$seed_file" 2>/dev/null)
    if [[ "$seed_size" -ne "$SEED_BYTES" ]]; then
        log_error "${component}: seed.bin tamaño incorrecto (${seed_size}B, esperado ${SEED_BYTES}B)"
        ok=false
    fi

    # Verificar que el keypair es válido
    local pub_from_priv
    pub_from_priv=$(openssl pkey -in "$private_key" -pubout 2>/dev/null | openssl dgst -sha256 | awk '{print $2}')
    local pub_stored
    pub_stored=$(openssl pkey -in "$public_key" -pubin -pubout 2>/dev/null | openssl dgst -sha256 | awk '{print $2}')

    if [[ "$pub_from_priv" != "$pub_stored" ]]; then
        log_error "${component}: keypair inconsistente"
        ok=false
    fi

    [[ "$ok" == "true" ]]
}

verify_all() {
    log_section "Verificación de integridad"
    local all_ok=true

    for component in "${COMPONENTS[@]}"; do
        if verify_component "$component"; then
            log_info "${component}: OK"
        else
            all_ok=false
        fi
    done

    if [[ "$all_ok" == "true" ]]; then
        echo -e "\n${GREEN}${BOLD}  ✅ Todas las claves verificadas correctamente${NC}"
        return 0
    else
        echo -e "\n${RED}${BOLD}  ❌ Verificación fallida — ejecuta: sudo bash tools/provision.sh full${NC}"
        return 1
    fi
}

# =============================================================================
# STATUS — Tabla visual de estado
# =============================================================================

status_all() {
    echo -e "\n${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  ML Defender — Estado de Provisioning Criptográfico           ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # libsodium version
    local sodium_installed
    sodium_installed=$(pkg-config --modversion libsodium 2>/dev/null || echo "none")
    if [[ "$sodium_installed" == "$SODIUM_REQUIRED" ]]; then
        echo -e "  ${GREEN}✅${NC} libsodium: ${sodium_installed} (HKDF nativo disponible)"
    else
        echo -e "  ${RED}❌${NC} libsodium: ${sodium_installed} (requiere ${SODIUM_REQUIRED} — ejecuta 'full')"
    fi
    echo ""

    printf "  %-25s %-12s %-12s %-12s %-20s\n" \
        "COMPONENTE" "PRIVATE KEY" "PUBLIC KEY" "SEED" "FINGERPRINT"
    printf "  %-25s %-12s %-12s %-12s %-20s\n" \
        "─────────────────────────" "───────────" "──────────" "────" "──────────────────"

    for component in "${COMPONENTS[@]}"; do
        local dir="${KEYS_ROOT}/${component}"

        local priv_status pub_status seed_status fingerprint

        [[ -f "${dir}/private.pem" ]] && priv_status="${GREEN}✅ OK${NC}" || priv_status="${RED}❌ FALTA${NC}"
        [[ -f "${dir}/public.pem"  ]] && pub_status="${GREEN}✅ OK${NC}"  || pub_status="${RED}❌ FALTA${NC}"
        [[ -f "${dir}/seed.bin"    ]] && seed_status="${GREEN}✅ OK${NC}" || seed_status="${RED}❌ FALTA${NC}"

        if [[ -f "${dir}/fingerprint.txt" ]]; then
            fingerprint=$(head -c 16 "${dir}/fingerprint.txt")...
        else
            fingerprint="${RED}❌ FALTA${NC}"
        fi

        local prov_date=""
        if [[ -f "${dir}/provision_meta.json" ]]; then
            prov_date=$(jq -r '.provisioned_at // "?"' "${dir}/provision_meta.json" 2>/dev/null | cut -c1-10)
        fi

        printf "  %-25s " "$component"
        echo -e "${priv_status}       ${pub_status}      ${seed_status}   ${fingerprint}  ${prov_date}"
    done

    echo ""
    local plugin_count=0
    [[ -d "${KEYS_ROOT}/plugins" ]] && \
        plugin_count=$(find "${KEYS_ROOT}/plugins" -name "private.pem" 2>/dev/null | wc -l)
    local ebpf_count=0
    [[ -d "${KEYS_ROOT}/ebpf-plugins" ]] && \
        ebpf_count=$(find "${KEYS_ROOT}/ebpf-plugins" -name "private.pem" 2>/dev/null | wc -l)

    echo -e "  ${CYAN}Plugins userspace provisionados:${NC} ${plugin_count}"
    echo -e "  ${CYAN}Plugins eBPF provisionados:${NC}      ${ebpf_count}"
    echo ""
    echo -e "  ${CYAN}Keys root:${NC} ${KEYS_ROOT}/"
    echo -e "  ${CYAN}Phase:${NC}     PHASE 1 (seed en claro, chmod 0600)"
    echo -e "  ${YELLOW}  → PHASE 2:${NC} seed.enc via seed-client (ADR-013)"
    echo ""
}

# =============================================================================
# MODO FULL — Provisioning completo
# =============================================================================

provision_full() {
    echo -e "\n${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║  ML Defender — Provisioning Criptográfico PHASE 1            ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${CYAN}Keys root:${NC}  ${KEYS_ROOT}/"
    echo -e "  ${CYAN}Algorithm:${NC}  Ed25519 keypairs + ChaCha20 seeds (${SEED_BYTES}B)"
    echo -e "  ${CYAN}libsodium:${NC}  ${SODIUM_REQUIRED} (HKDF nativo, ADR-013)"
    echo -e "  ${CYAN}AppArmor:${NC}   Compatible (paths fijos, ADR-019)"
    echo ""

    check_root
    check_dependencies
    check_entropy
    install_libsodium_1019

    # Crear directorio raíz
    if [[ ! -d "$KEYS_ROOT" ]]; then
        mkdir -p "$KEYS_ROOT"
        chmod 755 "$KEYS_ROOT"
        chown root:root "$KEYS_ROOT"
        log_info "Directorio raíz creado: ${KEYS_ROOT}"
    fi

    # Provisionar los 6 componentes del pipeline
    log_section "Componentes del pipeline (6)"
    for component in "${COMPONENTS[@]}"; do
        provision_component "$component"
    done

    # Provisionar plugins
    provision_plugins
    provision_ebpf_plugins

    # Verificación final
    log_section "Verificación post-provisioning"
    if verify_all; then
        echo -e "\n${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}${BOLD}║  ✅ PROVISIONING COMPLETADO CORRECTAMENTE                    ║${NC}"
        echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  ${CYAN}libsodium:${NC}     ${SODIUM_REQUIRED} con HKDF nativo"
        echo -e "  ${CYAN}Siguiente:${NC}     make pipeline-start"
        echo -e "  ${CYAN}Verificar:${NC}     make provision-status"
        echo ""
    else
        echo -e "\n${RED}${BOLD}  ❌ Provisioning fallido — revisa los errores anteriores${NC}"
        exit 1
    fi
}

# =============================================================================
# ENTRY POINT
# =============================================================================

MODE="${1:-full}"
COMPONENT="${2:-}"

case "$MODE" in
    full)
        provision_full
        ;;

    status)
        check_root
        status_all
        ;;

    verify)
        check_root
        verify_all
        ;;

    reprovision)
        check_root
        check_dependencies
        if [[ -z "$COMPONENT" ]]; then
            log_error "reprovision requiere un nombre de componente"
            echo "  Uso: sudo bash tools/provision.sh reprovision <componente>"
            echo "  Componentes válidos: ${COMPONENTS[*]}"
            exit 1
        fi
        valid=false
        for c in "${COMPONENTS[@]}"; do
            [[ "$c" == "$COMPONENT" ]] && valid=true && break
        done
        if [[ "$valid" == "false" ]]; then
            log_error "Componente desconocido: ${COMPONENT}"
            echo "  Componentes válidos: ${COMPONENTS[*]}"
            exit 1
        fi
        reprovision_component "$COMPONENT"
        verify_component "$COMPONENT" && log_info "Verificación post-reprovision: OK"
        ;;

    help|--help|-h)
        echo ""
        echo "  ML Defender — tools/provision.sh"
        echo ""
        echo "  FASE:  PHASE 1 (DAY 97) — Ed25519 keypairs + ChaCha20 seeds + libsodium 1.0.19"
        echo ""
        echo "  USO:"
        echo "    sudo bash tools/provision.sh full                 # Provisiona todo"
        echo "    sudo bash tools/provision.sh status               # Tabla de estado"
        echo "    sudo bash tools/provision.sh verify               # Verifica integridad"
        echo "    sudo bash tools/provision.sh reprovision <name>   # Re-provisiona uno"
        echo ""
        echo "  COMPONENTES: ${COMPONENTS[*]}"
        echo ""
        echo "  PATHS (AppArmor-safe, ADR-019):"
        echo "    ${KEYS_ROOT}/{componente}/private.pem   chmod 600"
        echo "    ${KEYS_ROOT}/{componente}/public.pem    chmod 644"
        echo "    ${KEYS_ROOT}/{componente}/seed.bin      chmod 600"
        echo ""
        echo "  LIBSODIUM:"
        echo "    Versión requerida: ${SODIUM_REQUIRED}"
        echo "    SHA-256:           ${SODIUM_SHA256}"
        echo "    HKDF nativo:       crypto_kdf_hkdf_sha256_extract/expand"
        echo ""
        ;;

    *)
        log_error "Modo desconocido: ${MODE}"
        echo "  Modos válidos: full | status | verify | reprovision | help"
        exit 1
        ;;
esac