#!/usr/bin/env bash
# vagrant/hardened-x86/scripts/setup-falco.sh
# Instala Falco y carga las reglas específicas de aRGus.
# Falco complementa AppArmor: AA previene, Falco detecta comportamiento anómalo.
# Se ejecuta DENTRO de la hardened VM como root.
#
# DAY 133 — aRGus NDR — ADR-030 Variant A
set -euo pipefail

FALCO_RULES=/etc/falco/rules.d/argus.yaml

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Falco — runtime security monitoring para aRGus NDR      ║"
echo "║  AppArmor previene. Falco detecta. Dos capas.            ║"
echo "╚════════════════════════════════════════════════════════════╝"

# ── Instalar Falco ────────────────────────────────────────────────────────────
echo ""
echo "── Installing Falco ──"
if command -v falco &>/dev/null; then
    echo "  ✅ Falco ya instalado: $(falco --version 2>/dev/null | head -1)"
else
    # Repositorio oficial de Falco
    curl -fsSL https://falco.org/repo/falcosecurity-packages.asc | \
        gpg --dearmor -o /usr/share/keyrings/falco-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/falco-archive-keyring.gpg] \
https://download.falco.org/packages/deb stable main" | \
        tee /etc/apt/sources.list.d/falcosecurity.list
    apt-get update -qq
    # Instalar sin módulo de kernel (usamos eBPF probe o modern_ebpf)
    FALCO_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        falco linux-headers-$(uname -r) 2>/dev/null || \
    apt-get install -y --no-install-recommends falco 2>/dev/null
    echo "  ✅ Falco instalado"
fi

# ── Configurar Falco para usar modern_ebpf (sin módulo de kernel) ─────────────
echo ""
echo "── Configuring Falco (modern_ebpf driver) ──"
if [ -f /etc/falco/falco.yaml ]; then
    # Preferir modern_ebpf sobre kmod para compatibilidad con VirtualBox
    sed -i 's/^driver:.*$/driver: modern_ebpf/' /etc/falco/falco.yaml 2>/dev/null || true
    sed -i 's/^engine:.*$/engine: modern_ebpf/' /etc/falco/falco.yaml 2>/dev/null || true
    echo "  ✅ Driver: modern_ebpf configurado"
fi

# ── Reglas específicas de aRGus ───────────────────────────────────────────────
echo ""
echo "── Installing aRGus Falco rules ──"
mkdir -p /etc/falco/rules.d

cat > "${FALCO_RULES}" << 'FALCO_RULES_EOF'
# aRGus NDR — Falco rules (DAY 133 — ADR-030 Variant A)
# Principio: AppArmor previene accesos prohibidos.
#            Falco detecta comportamiento anómalo aunque AA lo permita.
#
# Categorías:
#   1. Acceso a ficheros fuera del patrón esperado
#   2. Exec inesperado desde componentes del pipeline
#   3. Acceso a red fuera del patrón esperado
#   4. Cambios en ficheros de configuración o binarios
#   5. Escalada de privilegios

# ── Macros ────────────────────────────────────────────────────────────────────

- macro: argus_processes
  condition: proc.name in (etcd-server, sniffer, ml-detector, firewall-acl-agent, rag-ingester, rag-security)

- macro: argus_allowed_write_paths
  condition: >
    fd.name startswith /var/log/argus/ or
    fd.name startswith /etc/ml-defender/ or
    fd.name startswith /opt/argus/

- macro: argus_allowed_read_paths
  condition: >
    fd.name startswith /opt/argus/ or
    fd.name startswith /etc/ml-defender/ or
    fd.name startswith /var/log/argus/ or
    fd.name startswith /proc/ or
    fd.name startswith /sys/fs/bpf/ or
    fd.name startswith /usr/lib/ or
    fd.name startswith /lib/ or
    fd.name = /etc/ld.so.cache or
    fd.name = /proc/cpuinfo

# ── Regla 1: Acceso a ficheros fuera del patrón ───────────────────────────────

- rule: argus_unexpected_file_open
  desc: Un componente de aRGus accede a una ruta fuera de su patrón esperado
  condition: >
    argus_processes and
    open_write and
    not argus_allowed_write_paths
  output: >
    aRGus component writing to unexpected path
    (proc=%proc.name pid=%proc.pid path=%fd.name user=%user.name container=%container.name)
  priority: WARNING
  tags: [argus, filesystem, adr-030]

# ── Regla 2: Exec inesperado ──────────────────────────────────────────────────

- rule: argus_unexpected_exec
  desc: Un componente de aRGus ejecuta un binario inesperado
  condition: >
    spawned_process and
    proc.pname in (etcd-server, sniffer, ml-detector, rag-ingester, rag-security) and
    not proc.name in (sh, iptables, ipset)
  output: >
    aRGus component spawned unexpected process
    (parent=%proc.pname child=%proc.name cmd=%proc.cmdline pid=%proc.pid)
  priority: CRITICAL
  tags: [argus, exec, cwe-78, adr-030]

- rule: argus_firewall_unexpected_exec
  desc: firewall-acl-agent ejecuta algo distinto de iptables/ipset
  condition: >
    spawned_process and
    proc.pname = firewall-acl-agent and
    not proc.name in (iptables, iptables-save, ipset)
  output: >
    aRGus firewall-acl-agent spawned unexpected process
    (child=%proc.name cmd=%proc.cmdline) — posible CWE-78
  priority: CRITICAL
  tags: [argus, firewall, cwe-78, adr-030]

# ── Regla 3: Shell desde cualquier componente ─────────────────────────────────

- rule: argus_shell_spawn
  desc: Un componente de aRGus invoca una shell
  condition: >
    spawned_process and
    proc.pname in (etcd-server, sniffer, ml-detector, firewall-acl-agent, rag-ingester, rag-security) and
    proc.name in (bash, sh, zsh, dash, ksh)
  output: >
    aRGus component spawned shell — CRITICAL
    (parent=%proc.pname shell=%proc.name cmd=%proc.cmdline pid=%proc.pid)
  priority: CRITICAL
  tags: [argus, shell, critical, adr-030]

# ── Regla 4: Modificación de binarios o configuración en runtime ──────────────

- rule: argus_binary_modified
  desc: Un binario del pipeline aRGus ha sido modificado en runtime
  condition: >
    open_write and
    fd.name startswith /opt/argus/bin/ and
    not proc.name in (deploy-hardened, install)
  output: >
    aRGus binary modified at runtime — CRITICAL
    (proc=%proc.name path=%fd.name pid=%proc.pid user=%user.name)
  priority: CRITICAL
  tags: [argus, integrity, adr-039, bsr]

- rule: argus_seed_accessed_by_wrong_process
  desc: Un seed.bin es accedido por un proceso que no es el propietario del componente
  condition: >
    open_read and
    fd.name glob "/etc/ml-defender/*/seed.bin" and
    not argus_processes and
    not proc.name in (provision, setup-filesystem)
  output: >
    aRGus seed.bin accessed by unexpected process
    (proc=%proc.name path=%fd.name pid=%proc.pid user=%user.name)
  priority: CRITICAL
  tags: [argus, crypto, seeds, adr-025]

# ── Regla 5: Acceso a /etc/shadow o /etc/passwd ──────────────────────────────

- rule: argus_sensitive_file_access
  desc: Un componente de aRGus accede a ficheros sensibles del sistema
  condition: >
    argus_processes and
    open_read and
    fd.name in (/etc/shadow, /etc/gshadow, /etc/sudoers)
  output: >
    aRGus component accessing sensitive system file
    (proc=%proc.name path=%fd.name pid=%proc.pid)
  priority: CRITICAL
  tags: [argus, privilege-escalation, adr-030]

# ── Regla 6: Network inesperada (solo sniffer/firewall deberían abrir raw) ────

- rule: argus_unexpected_raw_socket
  desc: Componente no-sniffer abre un raw socket
  condition: >
    evt.type = socket and
    evt.arg.domain in (AF_PACKET, PF_PACKET) and
    proc.name in (ml-detector, rag-ingester, rag-security, etcd-server) and
    argus_processes
  output: >
    aRGus non-sniffer component opened raw socket
    (proc=%proc.name pid=%proc.pid)
  priority: WARNING
  tags: [argus, network, adr-030]

FALCO_RULES_EOF

echo "  ✅ ${FALCO_RULES}: $(wc -l < ${FALCO_RULES}) líneas"

# ── Validar sintaxis de las reglas ────────────────────────────────────────────
echo ""
echo "── Validating Falco rules ──"
if falco --validate "${FALCO_RULES}" 2>/dev/null; then
    echo "  ✅ Sintaxis OK"
else
    echo "  ⚠️  Validación falló — puede ser versión de Falco incompatible"
    echo "     Las reglas se instalan de todas formas para revisión manual"
fi

# ── Habilitar y arrancar Falco ────────────────────────────────────────────────
echo ""
echo "── Enabling Falco service ──"
systemctl enable falco 2>/dev/null || true
systemctl start  falco 2>/dev/null || \
    echo "  ⚠️  falco no arrancó — puede necesitar reboot o modern_ebpf driver"

sleep 2
if systemctl is-active falco &>/dev/null; then
    echo "  ✅ Falco: RUNNING"
    echo "  Reglas cargadas: $(falco --list 2>/dev/null | grep -c 'argus_' || echo 'desconocido')"
else
    echo "  ⚠️  Falco no está corriendo — revisar logs: journalctl -u falco"
fi

echo ""
echo "── Falco monitoring paths ──"
echo "  Logs: journalctl -u falco -f"
echo "  Rules: ${FALCO_RULES}"
echo "  Test: falco --list | grep argus"
echo ""
echo "✅ setup-falco completado"