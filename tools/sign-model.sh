#!/bin/bash
# sign-model.sh — firma un modelo con el keypair Ed25519 ADR-025
# Uso: sudo bash sign-model.sh <model_path>
# Genera: <model_path>.sig (64 bytes Ed25519)
set -euo pipefail

MODEL_PATH="${1:-}"
PRIVATE_KEY="/etc/ml-defender/plugins/plugin_signing.sk"

if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ Uso: $0 <model_path>" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ Modelo no encontrado: $MODEL_PATH" >&2
    exit 1
fi

if [[ ! -f "$PRIVATE_KEY" ]]; then
    echo "❌ Clave privada no encontrada: $PRIVATE_KEY" >&2
    echo "   Ejecuta: sudo bash tools/provision.sh full" >&2
    exit 1
fi

SIG_PATH="${MODEL_PATH}.sig"

openssl pkeyutl -sign \
    -inkey "$PRIVATE_KEY" \
    -rawin \
    -in  "$MODEL_PATH" \
    -out "$SIG_PATH" 2>/dev/null

if [[ $? -ne 0 ]]; then
    echo "❌ Firma fallida para $MODEL_PATH" >&2
    exit 1
fi

SIG_SIZE=$(wc -c < "$SIG_PATH")
echo "   → Modelo firmado: $(basename $MODEL_PATH) → $(basename $SIG_PATH) ($SIG_SIZE bytes)"
