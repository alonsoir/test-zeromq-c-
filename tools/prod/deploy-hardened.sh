#!/usr/bin/env bash
# tools/prod/deploy-hardened.sh
# Instala los binarios compilados en /opt/argus/ en la hardened VM.
# Aplica permisos estrictos y capabilities mínimas (ADR-030).
# Se ejecuta DENTRO de la hardened VM como root (sudo).
#
# DAY 133 — aRGus NDR — ADR-030 Variant A
set -euo pipefail

DIST=/vagrant/dist/x86
OPT_ARGUS=/opt/argus
ARGUS_USER=argus
ARGUS_GROUP=argus

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  aRGus NDR — Deploy to hardened-x86                      ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Verificar que dist/x86/ existe y tiene contenido
if [ ! -d "${DIST}/bin" ]; then
    echo "FAIL: ${DIST}/bin no existe"
    echo "      Ejecuta: make prod-build-x86 primero"
    exit 1
fi

# Verificar SHA256SUMS
if [ -f "${DIST}/SHA256SUMS" ]; then
    echo "── Verifying SHA256SUMS ──"
    cd "${DIST}"
    sha256sum -c SHA256SUMS --quiet && echo "  ✅ SHA256SUMS OK" || \
        (echo "FAIL: SHA256SUMS mismatch — binarios comprometidos"; exit 1)
else
    echo "  ⚠️  SHA256SUMS no encontrado — saltando verificación"
fi

# Crear estructura de directorios
echo ""
echo "── Creating /opt/argus/ structure ──"
install -d -o ${ARGUS_USER} -g ${ARGUS_GROUP} -m 0755 \
    ${OPT_ARGUS} \
    ${OPT_ARGUS}/bin \
    ${OPT_ARGUS}/lib \
    ${OPT_ARGUS}/plugins \
    ${OPT_ARGUS}/models

# Instalar binarios
echo ""
echo "── Installing binaries ──"
for binary in "${DIST}/bin/"*; do
    [ -f "${binary}" ] || continue
    [[ "${binary}" == *.sig ]] && continue
    name=$(basename "${binary}")
    install -o ${ARGUS_USER} -g ${ARGUS_GROUP} -m 0550 "${binary}" "${OPT_ARGUS}/bin/${name}"
    # Copiar firma si existe
    [ -f "${binary}.sig" ] && \
        install -o ${ARGUS_USER} -g ${ARGUS_GROUP} -m 0440 "${binary}.sig" "${OPT_ARGUS}/bin/${name}.sig"
    echo "  ✅ ${name} (0550 ${ARGUS_USER}:${ARGUS_GROUP})"
done

# Instalar librerías propias
echo ""
echo "── Installing runtime libraries ──"
for lib in "${DIST}/lib/"*; do
    [ -f "${lib}" ] || continue
    name=$(basename "${lib}")
    install -o root -g ${ARGUS_GROUP} -m 0550 "${lib}" "${OPT_ARGUS}/lib/${name}"
    echo "  ✅ ${name}"
done
ldconfig "${OPT_ARGUS}/lib"

# Instalar plugins
echo ""
echo "── Installing plugins ──"
for plugin in "${DIST}/plugins/"*.so; do
    [ -f "${plugin}" ] || continue
    name=$(basename "${plugin}")
    install -o root -g ${ARGUS_GROUP} -m 0550 "${plugin}" "${OPT_ARGUS}/plugins/${name}"
    [ -f "${plugin}.sig" ] && \
        install -o root -g ${ARGUS_GROUP} -m 0440 "${plugin}.sig" "${OPT_ARGUS}/plugins/${name}.sig"
    echo "  ✅ ${name}"
done

# Aplicar Linux Capabilities (BSR: en vez de SUID root)
echo ""
echo "── Setting Linux Capabilities (no SUID root) ──"

# sniffer: XDP/eBPF necesita net_admin + net_raw + sys_admin (eBPF load)
setcap cap_net_admin,cap_net_raw,cap_sys_admin+eip "${OPT_ARGUS}/bin/sniffer"
echo "  ✅ sniffer: cap_net_admin,cap_net_raw,cap_sys_admin+eip"

# firewall-acl-agent: iptables/ipset necesita net_admin
setcap cap_net_admin+eip "${OPT_ARGUS}/bin/firewall-acl-agent"
echo "  ✅ firewall-acl-agent: cap_net_admin+eip"

# etcd-server: ipc_lock para mlock() del seed en memoria
setcap cap_ipc_lock+eip "${OPT_ARGUS}/bin/etcd-server"
echo "  ✅ etcd-server: cap_ipc_lock+eip"

# Resto de componentes: sin capabilities especiales (no-root real)
for comp in ml-detector rag-ingester rag-security; do
    echo "  ✅ ${comp}: no capabilities (runs as ${ARGUS_USER})"
done

echo ""
echo "── Deployment summary ──"
echo "  Binaries:  $(ls ${OPT_ARGUS}/bin/ | grep -v '\.sig' | wc -l) installed"
echo "  Libraries: $(ls ${OPT_ARGUS}/lib/ | wc -l) installed"
echo "  Plugins:   $(ls ${OPT_ARGUS}/plugins/*.so 2>/dev/null | wc -l) installed"
echo ""
echo "✅ deploy-hardened completado"