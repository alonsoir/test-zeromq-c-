# 🧹 REPORTE DE LIMPIEZA DE MODELOS ML

**Fecha de limpieza:** 2025-11-05 13:26:24

## 📊 RESUMEN

| Métrica | Valor |
|---------|-------|
| 🗑️  Modelos eliminados | 4 |
| 📄 Archivos eliminados | 12 |
| 💾 Modelos conservados | 7 |
| 💽 Backup | /Users/aironman/CLionProjects/test-zeromq-docker/ml-training/outputs/model_backup_before_cleanup |

## 🚀 MODELOS CONSERVADOS (Producción)

| Modelo | Calidad |
|--------|---------|
| ransomware_detector_xgboost | 100.0/100 |
| ransomware_detector_rpi | 90.0/100 |
| ransomware_network_detector_proto_aligned | 100.0/100 |
| internal_traffic_detector_xgboost | 90.0/100 |
| internal_traffic_detector_onnx_ready | 90.0/100 |
| ransomware_xgboost_production | 100.0/100 |
| ransomware_xgboost_production_v2 | 100.0/100 |

## 🗑️ MODELOS ELIMINADOS

### ❌ ransmap_ransomware_xgboost/ransmap_ransomware_xgboost
- **Calidad:** 47.9/100
- **Archivos eliminados:** 3
- **Archivos:**
  - `ransmap_ransomware_xgboost_metadata.json`
  - `ransmap_ransomware_xgboost_scaler.pkl`
  - `ransmap_ransomware_xgboost.pkl`

### ❌ level2_ransomware_xgboost/level2_ransomware_xgboost
- **Calidad:** 43.3/100
- **Archivos eliminados:** 3
- **Archivos:**
  - `level2_ransomware_xgboost_scaler.pkl`
  - `level2_ransomware_xgboost_metadata.json`
  - `level2_ransomware_xgboost.pkl`

### ❌ ransomware_detector_adapted/ransomware_detector_adapted
- **Calidad:** 56.9/100
- **Archivos eliminados:** 3
- **Archivos:**
  - `ransomware_detector_adapted_scaler.pkl`
  - `ransomware_detector_adapted_metadata.json`
  - `ransomware_detector_adapted.pkl`

### ❌ proto_ransomware_xgboost/proto_ransomware_xgboost
- **Calidad:** 0.0/100
- **Archivos eliminados:** 3
- **Archivos:**
  - `proto_ransomware_xgboost_proto_meta.json`
  - `proto_ransomware_xgboost_scaler.pkl`
  - `proto_ransomware_xgboost.pkl`

---
*Limpieza automática ejecutada por Model Cleaner*