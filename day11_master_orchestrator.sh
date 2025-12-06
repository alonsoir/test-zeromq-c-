#!/bin/bash
# day11_master_orchestrator.sh
# Integración: Qwen scripts + DeepSeek automation

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ML DEFENDER - DÍA 11: VALIDACIÓN HOSPITALARIA          ║"
echo "║  Integración: Qwen (test suite) + DeepSeek (automation) ║"
echo "╚══════════════════════════════════════════════════════════╝"

# 1. PREFLIGHT (Qwen)
echo "🔍 Ejecutando preflight check de Qwen..."
cd day11_hospital_benchmark
./preflight/preflight_check.sh || exit 1

# 2. INICIAR DASHBOARD EN SEGUNDO PLANO (Qwen + DeepSeek)
echo "📊 Iniciando dashboards combinados..."
./monitoring/gateway_pulse.sh &
DASHBOARD_PID=$!

# Dashboard web de DeepSeek en paralelo
python3 ../day11_integration/realtime_dashboard.py &
WEB_DASHBOARD_PID=$!

# 3. EJECUTAR SUITE HOSPITALARIA COMPLETA (Qwen)
echo "🏥 Ejecutando suite hospitalaria de Qwen..."
./run_hospital_stress.sh

# 4. ANÁLISIS AUTOMÁTICO (Qwen + DeepSeek)
echo "📈 Analizando resultados..."
./analysis/validate_results.sh
python3 ../day11_integration/analyze_comprehensive.py

# 5. GENERAR REPORTE (DeepSeek)
echo "📄 Generando reporte integrado..."
python3 ../day11_integration/generate_performance_report.py

# 6. CERRAR DASHBOARDS
kill $DASHBOARD_PID $WEB_DASHBOARD_PID 2>/dev/null

echo "✅ DÍA 11 COMPLETADO - Resultados en: reports/day11_full/"