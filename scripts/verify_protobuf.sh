#!/bin/bash
echo "🔍 Verificando consistencia protobuf..."
echo "========================================"
echo ""

# Directorios a verificar
declare -a DIRS=(
    "/vagrant/protobuf"
    "/vagrant/sniffer/build/proto"
    "/vagrant/ml-detector/build/proto"
    "/vagrant/firewall-acl-agent/build/proto"
)

# Archivos a verificar
declare -a FILES=("network_security.pb.cc" "network_security.pb.h")

# Variable para rastrear si hay errores
ERRORS=0

# Verificar que los directorios existan
for DIR in "${DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "❌ Directorio no encontrado: $DIR"
        ERRORS=$((ERRORS + 1))
    fi
done

# Verificar que los archivos existan en cada directorio
for FILE in "${FILES[@]}"; do
    echo ""
    echo "📊 Verificando $FILE:"
    for DIR in "${DIRS[@]}"; do
        if [ -f "$DIR/$FILE" ]; then
            echo "   ✅ $DIR/$FILE"
        else
            echo "   ❌ $DIR/$FILE (no existe)"
            ERRORS=$((ERRORS + 1))
        fi
    done
done

# Verificar checksums
echo ""
echo "🔢 Comparando checksums:"
for FILE in "${FILES[@]}"; do
    echo ""
    echo "📄 $FILE:"
    PREV_CHECKSUM=""
    for DIR in "${DIRS[@]}"; do
        if [ -f "$DIR/$FILE" ]; then
            CURRENT_CHECKSUM=$(sha256sum "$DIR/$FILE" | cut -d ' ' -f1)
            if [ -z "$PREV_CHECKSUM" ]; then
                PREV_CHECKSUM=$CURRENT_CHECKSUM
                echo "   ✅ $DIR: $CURRENT_CHECKSUM"
            else
                if [ "$PREV_CHECKSUM" == "$CURRENT_CHECKSUM" ]; then
                    echo "   ✅ $DIR: $CURRENT_CHECKSUM"
                else
                    echo "   ❌ $DIR: $CURRENT_CHECKSUM (NO COINCIDE)"
                    ERRORS=$((ERRORS + 1))
                fi
            fi
        else
            echo "   ❌ $DIR: Archivo no encontrado"
        fi
    done
done

echo ""
echo "========================================"
if [ $ERRORS -eq 0 ]; then
    echo "✅ Todos los checksums son idénticos. Protobuf unificado correcto."
    exit 0
else
    echo "❌ Se encontraron $ERRORS erro(es). Revisar la consistencia de los archivos protobuf."
    exit 1
fi