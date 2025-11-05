#!/usr/bin/env python3
"""
ENTRENADOR DE MODELOS COMPATIBLES CON C++/ONNX
Crea modelos simples que sean fáciles de usar en C++ nativo
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
from datetime import datetime
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# =====================================================================
# CONFIG - MODELOS COMPATIBLES C++
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
OUTPUT_PATH = BASE_PATH / "outputs"
MODELS_PATH = OUTPUT_PATH / "models"
ONNX_PATH = OUTPUT_PATH / "onnx_models"

def create_cpp_compatible_internal_model():
    """Crea modelo de tráfico interno compatible con C++"""
    print("🎯 CREANDO MODELO INTERNO COMPATIBLE C++")

    # Cargar datos internos
    internal_path = BASE_PATH / "datasets" / "internal-normal" / "internal_traffic_dataset.csv"
    df = pd.read_csv(internal_path)

    # Features simples y compatibles
    features = ['sbytes', 'dbytes', 'spkts', 'dpkts']
    X = df[features].fillna(0)
    y = np.ones(len(X))  # Todo es tráfico interno

    # Crear datos externos sintéticos
    n_external = len(X)
    external_data = []
    for i in range(n_external):
        sample = {
            'sbytes': np.random.exponential(5000),
            'dbytes': np.random.exponential(1000),
            'spkts': np.random.poisson(100),
            'dpkts': np.random.poisson(20)
        }
        external_data.append(sample)

    X_ext = pd.DataFrame(external_data)
    y_ext = np.zeros(len(X_ext))

    # Combinar
    X_combined = pd.concat([X, X_ext], ignore_index=True)
    y_combined = np.concatenate([y, y_ext])

    print(f"📊 Dataset: {len(X_combined)} muestras, {len(features)} features")

    # Entrenar modelo simple (RandomForest es más compatible que XGBoost)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo RandomForest (máxima compatibilidad)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluar
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"✅ Modelo entrenado - Accuracy: {acc:.4f}, Recall: {rec:.4f}")

    # Guardar modelo sklearn
    model_name = "internal_traffic_cpp_compatible"
    model_dir = MODELS_PATH / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / f"{model_name}.pkl")
    joblib.dump(scaler, model_dir / f"{model_name}_scaler.pkl")

    # Guardar features
    with open(model_dir / f"{model_name}_features.json", 'w') as f:
        json.dump(features, f, indent=2)

    # Convertir DIRECTAMENTE a ONNX (sin problemas de XGBoost)
    print("🔄 Convirtiendo a ONNX...")
    initial_type = [('float_input', FloatTensorType([None, len(features)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    ONNX_PATH.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, ONNX_PATH / f"{model_name}.onnx")

    # Verificar que funciona
    ort_session = ort.InferenceSession(str(ONNX_PATH / f"{model_name}.onnx"))
    test_input = X_test_scaled[:1].astype(np.float32)
    outputs = ort_session.run(None, {'float_input': test_input})

    print(f"✅ ONNX verificado - Output: {outputs[0]}")

    # Metadata
    metadata = {
        'model_name': model_name,
        'model_type': 'INTERNAL_TRAFFIC_RANDOM_FOREST',
        'features': features,
        'metrics': {
            'accuracy': float(acc),
            'recall': float(rec)
        },
        'cpp_compatible': True,
        'onnx_verified': True,
        'created': datetime.now().isoformat()
    }

    with open(model_dir / f"{model_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Modelo guardado en: {model_dir}")
    print(f"📁 ONNX guardado en: {ONNX_PATH / f'{model_name}.onnx'}")

    return model_dir

def create_cpp_compatible_ransomware_model():
    """Crea modelo de ransomware compatible con C++"""
    print("\n🎯 CREANDO MODELO RANSOMWARE COMPATIBLE C++")

    # Cargar datos de CIC-IDS-2017 para tráfico normal
    cic_path = BASE_PATH / "datasets" / "CIC-IDS-2017"
    csv_files = list(cic_path.rglob("*.csv"))[:1]

    normal_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, low_memory=False)
        # Tomar muestra
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
        normal_data.append(df)

    if not normal_data:
        raise ValueError("No se encontraron datos CIC-IDS-2017")

    df_normal = pd.concat(normal_data, ignore_index=True)

    # Features compatibles
    features = [
        ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
        ' Flow Bytes/s', ' Flow Packets/s'
    ]

    # Filtrar features disponibles
    available_features = [f for f in features if f in df_normal.columns]
    X_normal = df_normal[available_features].fillna(0).replace([np.inf, -np.inf], 0)

    # Crear datos de ransomware sintéticos
    n_ransomware = len(X_normal)
    ransomware_data = []

    for i in range(n_ransomware):
        sample = {}
        for feature in available_features:
            if 'Duration' in feature:
                sample[feature] = np.random.exponential(1000000)
            elif 'Fwd Packets' in feature:
                sample[feature] = np.random.poisson(1000)
            elif 'Bytes/s' in feature:
                sample[feature] = np.random.exponential(500000)
            elif 'Packets/s' in feature:
                sample[feature] = np.random.exponential(200)
            else:
                sample[feature] = np.random.exponential(1000)
        ransomware_data.append(sample)

    X_ransomware = pd.DataFrame(ransomware_data)

    # Combinar
    X_combined = pd.concat([X_normal, X_ransomware], ignore_index=True)
    y_combined = np.array([0] * len(X_normal) + [1] * len(X_ransomware))

    print(f"📊 Dataset Ransomware: {len(X_combined)} muestras, {len(available_features)} features")

    # Entrenar
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"✅ Modelo ransomware entrenado - Accuracy: {acc:.4f}, Recall: {rec:.4f}")

    # Guardar
    model_name = "ransomware_cpp_compatible"
    model_dir = MODELS_PATH / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / f"{model_name}.pkl")
    joblib.dump(scaler, model_dir / f"{model_name}_scaler.pkl")

    with open(model_dir / f"{model_name}_features.json", 'w') as f:
        json.dump(available_features, f, indent=2)

    # Convertir a ONNX
    print("🔄 Convirtiendo a ONNX...")
    initial_type = [('float_input', FloatTensorType([None, len(available_features)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx.save(onnx_model, ONNX_PATH / f"{model_name}.onnx")

    # Verificar
    ort_session = ort.InferenceSession(str(ONNX_PATH / f"{model_name}.onnx"))
    test_input = X_test_scaled[:1].astype(np.float32)
    outputs = ort_session.run(None, {'float_input': test_input})

    print(f"✅ ONNX verificado - Output: {outputs[0]}")

    metadata = {
        'model_name': model_name,
        'model_type': 'RANSOMWARE_RANDOM_FOREST',
        'features': available_features,
        'metrics': {
            'accuracy': float(acc),
            'recall': float(rec)
        },
        'cpp_compatible': True,
        'onnx_verified': True,
        'created': datetime.now().isoformat()
    }

    with open(model_dir / f"{model_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Modelo guardado en: {model_dir}")
    print(f"📁 ONNX guardado en: {ONNX_PATH / f'{model_name}.onnx'}")

    return model_dir

if __name__ == "__main__":
    print("🚀 CREANDO MODELOS COMPATIBLES CON C++/ONNX")
    print("=" * 60)

    # Crear ambos modelos compatibles
    internal_dir = create_cpp_compatible_internal_model()
    ransomware_dir = create_cpp_compatible_ransomware_model()

    print(f"\n🎉 MODELOS CREADOS EXITOSAMENTE!")
    print("📁 ARCHIVOS GENERADOS:")
    print(f"   - Modelos sklearn: {MODELS_PATH}/")
    print(f"   - Modelos ONNX: {ONNX_PATH}/")
    print("\n🚀 LISTOS PARA USO EN C++ NATIVO")