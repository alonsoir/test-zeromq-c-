# Level 2 DDoS Detector - Training Guide

## 🎯 Objetivo

Entrenar modelo binario **BENIGN vs DDOS** usando CIC-DDoS-2019 dataset.

## 📋 Prerequisites

### 1. Dataset
```bash
# Verificar que tienes el dataset descargado
ls -lh datasets/CIC-DDoS-2019/

# Debe mostrar carpetas con CSVs:
# - DrDoS_DNS/
# - DrDoS_LDAP/
# - DrDoS_MSSQL/
# - DrDoS_NetBIOS/
# - DrDoS_NTP/
# - DrDoS_SNMP/
# - DrDoS_SSDP/
# - DrDoS_UDP/
# - Syn/
# - UDPLag/
# - WebDDoS/
```

### 2. Python Environment
```bash
# Crear virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn
pip install onnx onnxruntime skl2onnx imbalanced-learn joblib
```

### 3. Estructura de Directorios
```bash
ml-training/
├── datasets/
│   └── CIC-DDoS-2019/          # Dataset descargado
├── scripts/
│   ├── train_level2_ddos_binary.py
│   ├── convert_level2_ddos_to_onnx.py
│   └── level2_ddos_feature_mapping.json
└── outputs/
    ├── models/                  # Se creará automáticamente
    ├── onnx/                    # Se creará automáticamente
    ├── metadata/                # Se creará automáticamente
    └── plots/                   # Se creará automáticamente
```

## 🚀 Proceso de Entrenamiento

### Paso 1: Entrenar Modelo (2-30 min dependiendo del sample size)

```bash
cd ml-training
python scripts/train_level2_ddos_binary.py
```

**Salida esperada:**
```
================================================================================
🎯 LEVEL 2 DDOS DETECTOR - BINARY CLASSIFICATION
================================================================================

Dataset: CIC-DDoS-2019
Classes: BENIGN (0) vs DDOS (1)
Model: Random Forest
Features: ~70 numeric from NetworkFeatures

================================================================================
📊 CARGANDO CIC-DDoS-2019 DATASET
================================================================================

✅ Archivos encontrados: 12
  Cargando DrDoS_DNS.csv... ✅ 50,063 flows
  Cargando DrDoS_LDAP.csv... ✅ 45,897 flows
  ...

✅ Dataset completo: 1,234,567 flows, 88 columnas

================================================================================
🧹 PREPROCESAMIENTO
================================================================================

Columna de labels: ' Label'

📊 Distribución original:
  BENIGN                        :    250,000 ( 20.25%)
  DrDoS_DNS                     :    100,000 (  8.10%)
  DrDoS_LDAP                    :    100,000 (  8.10%)
  ...

📊 Distribución binaria:
  BENIGN (0):    250,000 ( 20.25%)
  DDOS   (1):    984,567 ( 79.75%)

🔍 Seleccionando features...
  Features disponibles: 70/83

🧹 Limpiando datos...
  Valores infinitos: 12,345
  Valores nulos: 5,678

✅ Datos limpios: 1,234,567 samples, 70 features

================================================================================
🌲 ENTRENAMIENTO RANDOM FOREST
================================================================================

📊 Split:
  Train: 987,653 samples
  Test:  246,914 samples

🔄 Aplicando SMOTE para balancear clases...
  Antes: BENIGN=200,000, DDOS=787,653
  Después: BENIGN=787,653, DDOS=787,653

🌲 Entrenando Random Forest...
  Parámetros:
    n_estimators: 150
    max_depth: 25
    min_samples_split: 10
    min_samples_leaf: 4
    class_weight: balanced

[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   12.5s
[Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed:   45.2s finished

✅ Modelo entrenado

🔄 Validación cruzada (5-fold)...
  F1 scores: [0.9812 0.9823 0.9807 0.9819 0.9815]
  F1 mean: 0.9815 (+/- 0.0012)

================================================================================
📊 EVALUACIÓN DEL MODELO
================================================================================

📈 Train Metrics:
  Accuracy:  0.9876
  Precision: 0.9834
  Recall:    0.9912
  F1-Score:  0.9873

📈 Test Metrics:
  Accuracy:  0.9812
  Precision: 0.9789
  Recall:    0.9856
  F1-Score:  0.9822
  ROC AUC:   0.9945

📊 Confusion Matrix:
[[47893  2107]
 [ 1421 195493]]

✅ Confusion matrix: outputs/plots/level2_ddos_binary_detector_confusion_matrix.png
✅ ROC curve: outputs/plots/level2_ddos_binary_detector_roc_curve.png
✅ Feature importance: outputs/plots/level2_ddos_binary_detector_feature_importance.png

================================================================================
💾 GUARDANDO MODELO
================================================================================

✅ Modelo: outputs/models/level2_ddos_binary_detector.joblib
✅ Metadata: outputs/metadata/level2_ddos_binary_detector_metadata.json

================================================================================
✅ LEVEL 2 DDOS BINARY MODEL TRAINING COMPLETADO
================================================================================

📊 Métricas finales (Test):
  Accuracy:  98.12%
  Precision: 97.89%
  Recall:    98.56%
  F1-Score:  98.22%
  ROC AUC:   0.9945

🎯 Siguiente paso:
  python scripts/convert_level2_ddos_to_onnx.py
```

### Paso 2: Convertir a ONNX (10-30 seg)

```bash
python scripts/convert_level2_ddos_to_onnx.py
```

**Salida esperada:**
```
================================================================================
🔄 CONVERSIÓN LEVEL 2 DDOS A ONNX
================================================================================

📦 Cargando modelo: outputs/models/level2_ddos_binary_detector.joblib
📋 Cargando metadata: outputs/metadata/level2_ddos_binary_detector_metadata.json

✅ Modelo cargado:
  Tipo: RandomForest
  Features: 70
  Classes: ['BENIGN', 'DDOS']
  F1 Score: 0.9822
  ROC AUC: 0.9945

================================================================================
🔄 CONVERSIÓN A ONNX
================================================================================

  Features: 70
  Opset: 12

🔄 Convirtiendo...

✅ Modelo ONNX guardado:
  Path: outputs/onnx/level2_ddos_binary_detector.onnx
  Size: 5.43 MB

================================================================================
🔍 VERIFICANDO MODELO ONNX
================================================================================

✅ Modelo ONNX válido

📊 Información del modelo:
  Opset: 12

  Inputs:
    - float_input: ['None', 70]

  Outputs:
    - label: ['None']
    - probabilities: ['None', 2]

================================================================================
🧪 VALIDANDO CONVERSIÓN
================================================================================

🔢 Generando 1000 samples de prueba...
🔮 Predicción sklearn...
🔮 Predicción ONNX...

📊 Comparación de labels:
  Match: True

📊 Comparación de probabilidades:
  Close (rtol=1e-4): True

✅ VALIDACIÓN EXITOSA
  sklearn y ONNX producen resultados idénticos

================================================================================
📝 CREANDO METADATA ONNX
================================================================================

✅ Metadata ONNX guardada: outputs/metadata/level2_ddos_binary_detector_onnx_metadata.json

================================================================================
✅ CONVERSIÓN A ONNX COMPLETADA
================================================================================

📦 Modelo ONNX:
  Path: outputs/onnx/level2_ddos_binary_detector.onnx
  Size: 5.43 MB
  Features: 70
  Validación: ✅ PASSED

🎯 Para integrar en ml-detector:
  1. Copiar a: ../ml-detector/models/production/level2/
  2. Actualizar: ../ml-detector/config/ml_detector_config.json
  3. Código C++: Cargar con ONNX Runtime
  4. Input: std::vector<float> con 70 valores
  5. Output: label (0/1) + probabilities [2 valores]
```

## 📁 Archivos Generados

Después de entrenar y convertir, tendrás:

```
outputs/
├── models/
│   └── level2_ddos_binary_detector.joblib        # Modelo sklearn
├── onnx/
│   └── level2_ddos_binary_detector.onnx          # Modelo ONNX para C++
├── metadata/
│   ├── level2_ddos_binary_detector_metadata.json
│   └── level2_ddos_binary_detector_onnx_metadata.json
└── plots/
    ├── level2_ddos_binary_detector_confusion_matrix.png
    ├── level2_ddos_binary_detector_roc_curve.png
    └── level2_ddos_binary_detector_feature_importance.png
```

## 🔧 Configuración Avanzada

### Entrenar con Sample Reducido (Para Pruebas)

```python
# En train_level2_ddos_binary.py, línea ~500
df = load_ddos_dataset(sample_size=100000)  # Solo 100k samples
```

### Ajustar Hiperparámetros

```python
# En train_level2_ddos_binary.py, línea ~350
rf = RandomForestClassifier(
    n_estimators=200,      # Aumentar árboles (más lento, mejor)
    max_depth=30,          # Mayor profundidad
    min_samples_split=5,   # Menos samples para split
    # ...
)
```

### Desactivar SMOTE

```python
# En train_level2_ddos_binary.py, línea ~500
model, X_train, X_test, y_train, y_test = train_model(X, y, use_smote=False)
```

## 🎯 Integración en ml-detector C++

### 1. Copiar Modelo

```bash
# Desde ml-training/
cp outputs/onnx/level2_ddos_binary_detector.onnx \
   ../ml-detector/models/production/level2/
```

### 2. Actualizar Configuración

```json
// ml-detector/config/ml_detector_config.json
{
  "models": {
    "level2_ddos": {
      "path": "models/production/level2/level2_ddos_binary_detector.onnx",
      "enabled": true,
      "threshold": 0.85,
      "n_features": 70
    }
  }
}
```

### 3. Código C++ (Ejemplo)

```cpp
// ml-detector/src/ml_predictor.cpp
#include <onnxruntime_cxx_api.h>

class Level2DDoSPredictor {
private:
    Ort::Env env_;
    Ort::Session session_;
    
public:
    Level2DDoSPredictor(const std::string& model_path) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        session_ = Ort::Session(env_, model_path.c_str(), opts);
    }
    
    struct Prediction {
        int label;           // 0=BENIGN, 1=DDOS
        float prob_benign;
        float prob_ddos;
    };
    
    Prediction predict(const protobuf::NetworkFeatures& nf) {
        // Extraer features (ver level2_ddos_feature_mapping.json)
        std::vector<float> features = extract_level2_features(nf);
        
        // Inferencia ONNX
        auto input_tensor = create_tensor(features);
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), 2
        );
        
        // Parsear resultado
        Prediction pred;
        pred.label = output_tensors[0].GetTensorData<int64_t>()[0];
        const float* probs = output_tensors[1].GetTensorData<float>();
        pred.prob_benign = probs[0];
        pred.prob_ddos = probs[1];
        
        return pred;
    }
};
```

## 📊 Métricas Esperadas

| Métrica | Objetivo | Típico |
|---------|----------|--------|
| Accuracy | >98% | 98-99% |
| Precision | >97% | 97-98% |
| Recall | >96% | 98-99% |
| F1-Score | >97% | 98% |
| ROC AUC | >0.99 | 0.994-0.996 |
| Inference Time | <5ms | 1-3ms (CPU) |

## 🐛 Troubleshooting

### Error: Dataset no encontrado
```bash
# Verificar path
ls -lh datasets/CIC-DDoS-2019/

# Si no existe, descargar:
# https://www.unb.ca/cic/datasets/ddos-2019.html
```

### Error: Out of Memory
```python
# Reducir sample size
df = load_ddos_dataset(sample_size=100000)  # En vez de None
```

### Error: Conversion ONNX falla
```bash
# Verificar versiones
pip list | grep -E "onnx|skl2onnx|scikit"

# Reinstalar si es necesario
pip install --upgrade onnx onnxruntime skl2onnx
```

### Warnings durante entrenamiento
```
# Warnings de sklearn son normales (deprecation, etc.)
# Se pueden ignorar si el modelo entrena correctamente
```

## 🎯 Próximos Pasos

1. ✅ **Entrenar modelo binario BENIGN/DDOS** (este documento)
2. ⏭️ **Modelo multi-clase**: Distinguir tipos de DDoS (SYN, UDP, HTTP, etc.)
3. ⏭️ **Nivel 2 Ransomware**: Detectar ransomware
4. ⏭️ **Nivel 3 Anomalías**: Detección de anomalías (4 features)

## 📚 Referencias

- Dataset: [CIC-DDoS-2019](https://www.unb.ca/cic/datasets/ddos-2019.html)
- Paper: "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy"
- ONNX Runtime: [https://onnxruntime.ai/](https://onnxruntime.ai/)

---

**¿Problemas?** Revisa logs en `outputs/` o contacta al equipo.
