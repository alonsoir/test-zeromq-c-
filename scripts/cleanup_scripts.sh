#!/bin/bash
# cleanup_scripts.sh - Consolidar y organizar scripts

set -e

echo "🧹 Limpieza y Consolidación de Scripts"
echo "======================================="
echo ""

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPTS_DIR"

# Backup antes de hacer cambios
BACKUP_DIR="./archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creando backup en: $BACKUP_DIR"

# === RENOMBRAR SCRIPTS BUENOS ===
echo ""
echo "✏️  Renombrando scripts finales..."

# Plot metrics (el bueno)
if [ -f "plot_metrics_fixed.py" ]; then
    echo "  plot_metrics_fixed.py → plot_metrics.py"
    cp plot_metrics_fixed.py "$BACKUP_DIR/"
    mv plot_metrics_fixed.py plot_metrics.py
fi

# CSV fixer (el bueno)
if [ -f "fix_cpu_csv.py" ]; then
    echo "  fix_cpu_csv.py → fix_csv_decimals.py"
    cp fix_cpu_csv.py "$BACKUP_DIR/"
    mv fix_cpu_csv.py fix_csv_decimals.py
fi

# Monitor stability (si hay versión fixed)
if [ -f "monitor_stability_fixed.sh" ] && [ -f "monitor_stability.sh" ]; then
    echo "  monitor_stability.sh → monitor_stability.sh.old"
    echo "  monitor_stability_fixed.sh → monitor_stability.sh"
    mv monitor_stability.sh "$BACKUP_DIR/monitor_stability.sh.old"
    mv monitor_stability_fixed.sh monitor_stability.sh
fi

# === MOVER SCRIPTS ROTOS A ARCHIVO ===
echo ""
echo "📂 Moviendo scripts obsoletos a archive..."

scripts_to_archive=(
    "fix_csvs.sh"
    "fix_csvs_python.py"
    "fix_csvs_smart.py"
    "fix_csvs_pandas.py"
    "fix_csvs_macos.sh"
    "performance_monitor.sh.old"
)

for script in "${scripts_to_archive[@]}"; do
    if [ -f "$script" ]; then
        echo "  → $script"
        mv "$script" "$BACKUP_DIR/"
    fi
done

# === LIMPIAR BACKUPS DE CSVs ===
echo ""
echo "🗑️  Limpiando backups de CSVs..."

if [ -d "performance_metrics" ]; then
    bak_count=$(find performance_metrics -name "*.bak" -type f | wc -l)
    if [ "$bak_count" -gt 0 ]; then
        echo "  Moviendo $bak_count archivos .bak"
        find performance_metrics -name "*.bak" -type f -exec mv {} "$BACKUP_DIR/" \;
    else
        echo "  No hay archivos .bak"
    fi
fi

# === RESUMEN ===
echo ""
echo "="*60
echo "✅ Limpieza completada"
echo "="*60
echo ""
echo "📊 Scripts finales activos:"
ls -lh performance_monitor*.sh plot_metrics.py fix_csv_decimals.py monitor_stability*.sh stability_test_v2.sh 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "📦 Backup guardado en: $BACKUP_DIR"
echo "   Contiene $(ls -1 "$BACKUP_DIR" | wc -l) archivos"

echo ""
echo "📋 Próximos pasos:"
echo "  1. Arreglar performance_monitor.sh (system_cpu con 'us.')"
echo "  2. Refactorizar JSONs de configuración"
echo "  3. Integrar en Makefile"
echo ""
echo "🏛️  Via Appia quality - fundaciones sólidas"