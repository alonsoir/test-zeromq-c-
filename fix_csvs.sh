#!/bin/bash
# Fix CSV decimal separators (European comma → standard period)

METRICS_DIR="performance_metrics"

echo "🔧 Arreglando CSVs en $METRICS_DIR..."

for csv in "$METRICS_DIR"/*.csv; do
    if [ -f "$csv" ]; then
        echo "   Procesando: $(basename $csv)"
        
        # Backup
        cp "$csv" "${csv}.bak"
        
        # Fix: Solo cambiar comas que están entre números (decimales)
        # NO cambiar las comas separadoras de campos
        sed -i 's/\([0-9]\),\([0-9]\)/\1.\2/g' "$csv"
    fi
done

echo "✅ CSVs arreglados (backups en *.bak)"
