#!/bin/bash
# Extract Ed25519 pubkey hex from PEM — ADR-025 DEBT-PUBKEY-RUNTIME-001
# Output: 64-char hex string (32 bytes raw pubkey)
PK_FILE="/etc/ml-defender/plugins/plugin_signing.pk"
if [ ! -f "$PK_FILE" ]; then
    echo "ERROR: $PK_FILE not found" >&2
    exit 1
fi
openssl pkey -in "$PK_FILE" -pubin -outform DER 2>/dev/null | tail -c 32 | xxd -p -c 32 | tr -d '\n'
