#!/usr/bin/env bash
# tools/prod/sign-binaries.sh
# Firma Ed25519 todos los binarios y plugins en dist/x86/.
# Reutiliza el keypair de ADR-025 (plugin_signing.sk).
# Se ejecuta DENTRO de la dev VM.
#
# DAY 133 — aRGus NDR — ADR-025 + ADR-039
set -euo pipefail

DIST=/vagrant/dist/x86
SK=/etc/ml-defender/plugins/plugin_signing.sk
PK=/etc/ml-defender/plugins/plugin_signing.pk

# Verificar que la clave existe
if [ ! -f "${SK}" ]; then
    echo "FAIL: plugin_signing.sk no encontrada en ${SK}"
    echo "      Ejecuta: make provision"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Ed25519 signing — dist/x86/ binaries + plugins          ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Función de firma usando libsodium (mismo mecanismo que ADR-025)
sign_file() {
    local file=$1
    local sigfile="${file}.sig"

    # Leer clave privada (64 bytes hex → binario)
    local sk_hex
    sk_hex=$(sudo cat "${SK}" 2>/dev/null | tr -d '[:space:]')

    # Usar python3 + libsodium para firmar
    python3 - <<PYEOF
import sys, os, binascii
from ctypes import CDLL, c_ubyte, c_ulonglong, byref, create_string_buffer, cast, POINTER

lib = CDLL('/usr/local/lib/libsodium.so.26')

sk_hex = '${sk_hex}'
sk_bytes = bytes.fromhex(sk_hex)

with open('${file}', 'rb') as f:
    msg = f.read()

sig = create_string_buffer(64)
sig_len = c_ulonglong(0)
msg_buf = (c_ubyte * len(msg))(*msg)
sk_buf  = (c_ubyte * 64)(*sk_bytes)

ret = lib.crypto_sign_ed25519_detached(
    sig, byref(sig_len),
    msg_buf, len(msg),
    sk_buf
)
if ret != 0:
    print(f'FAIL: signing failed for ${file}', file=sys.stderr)
    sys.exit(1)

with open('${sigfile}', 'wb') as f:
    f.write(bytes(sig))

print(f'  ✅ $(basename ${file}) → .sig ({len(bytes(sig))} bytes)')
PYEOF
}

# Firmar binarios
echo ""
echo "── Signing binaries ──"
for binary in "${DIST}"/bin/*; do
    [ -f "${binary}" ] || continue
    [[ "${binary}" == *.sig ]] && continue
    sign_file "${binary}"
done

# Firmar plugins
echo ""
echo "── Signing plugins ──"
for plugin in "${DIST}"/plugins/*.so; do
    [ -f "${plugin}" ] || continue
    sign_file "${plugin}"
done

# Mostrar pubkey activa (para registrar en HARDWARE-REQUIREMENTS.md)
echo ""
PK_HEX=$(sudo cat "${PK}" 2>/dev/null | tr -d '[:space:]' || echo "ERROR")
echo "── Active pubkey (ADR-025) ──"
echo "  MLD_PLUGIN_PUBKEY_HEX=${PK_HEX}"
echo ""
echo "✅ sign-binaries completado"