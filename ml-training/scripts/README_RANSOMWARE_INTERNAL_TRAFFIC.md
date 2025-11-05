(.venv) aironman@MacBook-Pro-de-Alonso scripts % python3 model_analyzer.py
🚀 ANALIZADOR DE MODELOS ML - VERSIÓN FINAL
============================================================
🔍 BUSCANDO MODELOS...
📁 Encontrados 17 modelos principales

🎯 ANALIZANDO MODELOS...
============================================================
🔍 Analizando: ransomware_detector_xgboost/ransomware_detector_xgboost
🎯 XGBoost - Calidad: 100.0/100
📦 📋 PARCIAL (50.0%)
💡 🎯 EXCELENTE - Listo para producción
🔍 Analizando: ransmap_ransomware_xgboost/ransmap_ransomware_xgboost
🎯 XGBoost - Calidad: 47.9/100
📦 📋 PARCIAL (50.0%)
💡 ❌ DESCARTAR - Baja calidad
🔍 Analizando: ransomware_detector_rpi/ransomware_detector_rpi
🎯 XGBoost - Calidad: 90.0/100
📦 📋 PARCIAL (50.0%)
💡 🎯 EXCELENTE - Listo para producción
🔍 Analizando: ransomware_network_detector_proto_aligned/ransomware_network_detector_proto_aligned
🎯 XGBoost - Calidad: 100.0/100
📦 📋 PARCIAL (50.0%)
💡 🎯 EXCELENTE - Listo para producción
🔍 Analizando: level2_ransomware_xgboost/level2_ransomware_xgboost
🎯 XGBoost - Calidad: 43.3/100
📦 📋 PARCIAL (50.0%)
💡 ❌ DESCARTAR - Baja calidad
🔍 Analizando: ransomware_anomaly_detector/ransomware_anomaly_detector
❌ Error cargando ransomware_anomaly_detector.pkl: invalid load key, '\x0b'.
🔍 Analizando: internal_traffic_detector_xgboost/internal_traffic_detector_xgboost
🎯 XGBoost - Calidad: 90.0/100
📦 📋 PARCIAL (50.0%)
💡 🎯 EXCELENTE - Listo para producción
🔍 Analizando: ransomware_detector_optimized/ransomware_detector_optimized
❌ Error cargando ransomware_detector_optimized.pkl: invalid load key, '\x0b'.
🔍 Analizando: internal_traffic_detector_onnx_ready/internal_traffic_detector_onnx_ready
🎯 XGBoost - Calidad: 90.0/100
📦 📋 PARCIAL (50.0%)
💡 🎯 EXCELENTE - Listo para producción
🔍 Analizando: internal_traffic_cpp_compatible/internal_traffic_cpp_compatible
❌ Error cargando internal_traffic_cpp_compatible.pkl: invalid load key, '\x09'.
🔍 Analizando: ransomware_detector_adapted/ransomware_detector_adapted
🎯 XGBoost - Calidad: 56.9/100
📦 📋 PARCIAL (50.0%)
💡 🔧 MEJORABLE - Reentrenar
🔍 Analizando: proto_ransomware_xgboost/proto_ransomware_xgboost
🎯 XGBoost - Calidad: 0.0/100
📦 📋 PARCIAL (50.0%)
💡 ❌ DESCARTAR - Baja calidad
🔍 Analizando: ransomware_xgboost_production/ransomware_xgboost_production
🎯 XGBoost - Calidad: 100.0/100
📦 📋 PARCIAL (50.0%)
💡 🎯 EXCELENTE - Listo para producción
🔍 Analizando: ransomware_xgboost_production_v2/ransomware_xgboost_production_v2
🎯 XGBoost - Calidad: 100.0/100
📦 📋 PARCIAL (50.0%)
💡 🎯 EXCELENTE - Listo para producción
🔍 Analizando: ransomware_cpp_compatible/ransomware_cpp_compatible
❌ Error cargando ransomware_cpp_compatible.pkl: invalid load key, '\x09'.
🔍 Analizando: models/level2_ddos_binary_detector
🎯 RandomForest - Calidad: 0.0/100
📦 ❌ INCOMPLETO (25.0%)
💡 ❌ DESCARTAR - Baja calidad
🔍 Analizando: models/level1_attack_detector
🎯 RandomForest - Calidad: 0.0/100
📦 ❌ INCOMPLETO (25.0%)
💡 ❌ DESCARTAR - Baja calidad

📊 GENERANDO REPORTE...

================================================================================
🎯 RESUMEN FINAL - MODELOS APROVECHABLES
================================================================================

🚀 **TOP 7 MODELOS PARA PRODUCCIÓN:**
1. ransomware_detector_xgboost
   🔧 XGBoost | 📊 100.0/100 | 📦 50.0%
   📍 ransomware_detector_xgboost/

2. ransomware_network_detector_proto_aligned
   🔧 XGBoost | 📊 100.0/100 | 📦 50.0%
   📍 ransomware_network_detector_proto_aligned/

3. ransomware_xgboost_production_v2
   🔧 XGBoost | 📊 100.0/100 | 📦 50.0%
   📍 ransomware_xgboost_production_v2/

4. ransomware_xgboost_production
   🔧 XGBoost | 📊 100.0/100 | 📦 50.0%
   📍 ransomware_xgboost_production/

5. ransomware_detector_rpi
   🔧 XGBoost | 📊 90.0/100 | 📦 50.0%
   📍 ransomware_detector_rpi/

6. internal_traffic_detector_onnx_ready
   🔧 XGBoost | 📊 90.0/100 | 📦 50.0%
   📍 internal_traffic_detector_onnx_ready/

7. internal_traffic_detector_xgboost
   🔧 XGBoost | 📊 90.0/100 | 📦 50.0%
   📍 internal_traffic_detector_xgboost/

📁 Reporte completo en: /Users/aironman/CLionProjects/test-zeromq-docker/ml-training/outputs/model_analysis_report_final

🎉 ANÁLISIS COMPLETADO!
(.venv) aironman@MacBook-Pro-de-Alonso scripts % python3 CLEAN_HOUSE_MODELS.py
🧹 LIMPIADOR DE MODELOS DE BAJA CALIDAD
============================================================
ADVERTENCIA: Esta acción ELIMINARÁ permanentemente modelos
Se creará un backup automáticamente
============================================================
¿Continuar con la limpieza? (sí/no): si
🚀 INICIANDO LIMPIEZA DE MODELOS...
============================================================
📊 CLASIFICANDO MODELOS...
✅ Modelos para mantener: 7
🗑️  Modelos para eliminar: 6

💾 CREANDO BACKUP...
✅ Backup creado en: /Users/aironman/CLionProjects/test-zeromq-docker/ml-training/outputs/model_backup_before_cleanup

🗑️  ELIMINANDO 6 MODELOS...
============================================================

1/6 Eliminando: ransmap_ransomware_xgboost/ransmap_ransomware_xgboost
📊 Calidad: 47.9/100, Completitud: 50.0%
✅ Eliminados: 3 archivos
- ransmap_ransomware_xgboost_metadata.json
- ransmap_ransomware_xgboost_scaler.pkl
- ransmap_ransomware_xgboost.pkl

2/6 Eliminando: level2_ransomware_xgboost/level2_ransomware_xgboost
📊 Calidad: 43.3/100, Completitud: 50.0%
✅ Eliminados: 3 archivos
- level2_ransomware_xgboost_scaler.pkl
- level2_ransomware_xgboost_metadata.json
- level2_ransomware_xgboost.pkl

3/6 Eliminando: ransomware_detector_adapted/ransomware_detector_adapted
📊 Calidad: 56.9/100, Completitud: 50.0%
✅ Eliminados: 3 archivos
- ransomware_detector_adapted_scaler.pkl
- ransomware_detector_adapted_metadata.json
- ransomware_detector_adapted.pkl

4/6 Eliminando: proto_ransomware_xgboost/proto_ransomware_xgboost
📊 Calidad: 0.0/100, Completitud: 50.0%
✅ Eliminados: 3 archivos
- proto_ransomware_xgboost_proto_meta.json
- proto_ransomware_xgboost_scaler.pkl
- proto_ransomware_xgboost.pkl

5/6 Eliminando: models/level2_ddos_binary_detector
📊 Calidad: 0.0/100, Completitud: 25.0%
⚠️  No se encontraron archivos para eliminar

6/6 Eliminando: models/level1_attack_detector
📊 Calidad: 0.0/100, Completitud: 25.0%
⚠️  No se encontraron archivos para eliminar

🧹 LIMPIANDO DIRECTORIOS VACÍOS...
✅ Directorios vacíos eliminados: 4
- ransmap_ransomware_xgboost
- level2_ransomware_xgboost
- ransomware_detector_adapted
- proto_ransomware_xgboost

📋 GENERANDO REPORTE DE LIMPIEZA...
✅ Reporte de limpieza guardado en: /Users/aironman/CLionProjects/test-zeromq-docker/ml-training/outputs/cleanup_report.json

🎯 ESTADO FINAL DESPUÉS DE LA LIMPIEZA
============================================================
📁 Modelos restantes: 11

🏆 MODELOS DE ALTA CALIDAD CONSERVADOS:
1. ransomware_detector_xgboost/ransomware_detector_xgboost
2. ransomware_detector_rpi/ransomware_detector_rpi
3. ransomware_network_detector_proto_aligned/ransomware_network_detector_proto_aligned
4. ransomware_anomaly_detector/ransomware_anomaly_detector
5. internal_traffic_detector_xgboost/internal_traffic_detector_xgboost
6. ransomware_detector_optimized/ransomware_detector_optimized
7. internal_traffic_detector_onnx_ready/internal_traffic_detector_onnx_ready
8. internal_traffic_cpp_compatible/internal_traffic_cpp_compatible
9. ransomware_xgboost_production/ransomware_xgboost_production
10. ransomware_xgboost_production_v2/ransomware_xgboost_production_v2
11. ransomware_cpp_compatible/ransomware_cpp_compatible

💾 Backup disponible en: /Users/aironman/CLionProjects/test-zeromq-docker/ml-training/outputs/model_backup_before_cleanup
🔧 Puedes restaurar desde el backup si es necesario

🎉 LIMPIEZA COMPLETADA!
🗑️  4 modelos eliminados
📄 12 archivos liberados
💾 Backup guardado en: /Users/aironman/CLionProjects/test-zeromq-docker/ml-training/outputs/model_backup_before_cleanup
(.venv) aironman@MacBook-Pro-de-Alonso scripts % 