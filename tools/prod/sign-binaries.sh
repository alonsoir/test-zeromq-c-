#!/usr/bin/env bash
# tools/prod/sign-binaries.sh
# Firma binarios y plugins con Ed25519 (ADR-025).
# Reutiliza el keypair PEM de provision.sh — formato canónico.
# Se ejecuta desde la dev VM via vagrant ssh.
set -euo pipefail

SK=/etc/ml-defender/plugins/plugin_signing.sk
DIST_BIN=/vagrant/dist/x86/bin
DIST_PLUGINS=/vagrant/dist/x86/plugins

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Ed25519 signing — dist/x86/ binaries + plugins          ║"
echo "╚════════════════════════════════════════════════════════════╝"

if [ ! -f "${SK}" ]; then
    echo "FAIL: plugin_signing.sk no encontrada en ${SK}"
    echo "  Ejecuta: sudo bash /vagrant/tools/provision.sh full"
    exit 1
fi

sign_file() {
    local file="$1"
    local sig="${file}.sig"
    openssl pkeyutl -sign \
        -inkey "${SK}" \
        -rawin \
        -in  "${file}" \
        -out "${sig}" 2>/dev/null
    local sig_size
    sig_size=$(stat -c%s "${sig}" 2>/dev/null || stat -f%z "${sig}" 2>/dev/null)
    if [ "${sig_size}" -ne 64 ]; then
        echo "  ❌ FAIL: ${file} — sig inesperado ${sig_size} bytes"
        exit 1
    fi
    echo "  ✅ $(basename ${file}) → $(basename ${sig}) (${sig_size} bytes)"
}

echo "── Signing binaries ──"
while IFS= read -r bin; do
    sign_file "${bin}"
done < <(find "${DIST_BIN}" -maxdepth 1 -type f ! -name "*.sig")

echo "── Signing plugins ──"
while IFS= read -r plugin; do
    sign_file "${plugin}"
done < <(find "${DIST_PLUGINS}" -maxdepth 1 -type f -name "*.so")

echo ""
echo "✅ prod-sign completado"
