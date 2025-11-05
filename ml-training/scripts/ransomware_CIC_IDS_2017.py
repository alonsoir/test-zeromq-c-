#!/usr/bin/env python3
"""
RANSOMWARE DETECTOR - DETECCIÓN DE ANOMALÍAS EN RED
Usa Isolation Forest para detectar comportamiento anómalo sin necesidad de labels de ataques
ransomware_CIC_IDS_2017.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG - DETECCIÓN DE ANOMALÍAS
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "datasets" / "CIC-IDS-2017"
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_NAME = "ransomware_anomaly_detector"

# Parámetros Isolation Forest optimizados para detección de ransomware
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.1,  # Esperamos ~10% de anomalías (ataques)
    'max_samples': 10000,   # Muestras por árbol
    'max_features': 1.0,    # Usar todas las features
    'bootstrap': False,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 0
}

def load_and_sample_cic_data(n_samples=50000):
    """Carga y muestrea datos del CIC-IDS-2017"""
    print("📥 CARGANDO Y MUESTREANDO DATOS CIC-IDS-2017")

    csv_files = list(DATASET_PATH.rglob("*.csv"))
    print(f"   Encontrados {len(csv_files)} archivos CSV")

    all_samples = []

    for csv_file in csv_files[:3]:  # Usar solo 3 archivos para variedad
        print(f"   📖 Muestreando {csv_file.name}...", end=" ")
        try:
            # Leer una muestra de cada archivo
            df = pd.read_csv(csv_file, low_memory=False)

            # Tomar muestra representativa
            sample_size = min(len(df), n_samples // len(csv_files))
            df_sample = df.sample(n=sample_size, random_state=42)

            all_samples.append(df_sample)
            print(f"{len(df_sample):,} muestras")

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    if not all_samples:
        raise ValueError("❌ No se pudieron cargar archivos")

    combined_data = pd.concat(all_samples, ignore_index=True)
    print(f"✅ Dataset combinado: {combined_data.shape}")

    return combined_data

def select_ransomware_features(df):
    """Selecciona features relevantes para detección de ransomware"""
    print("\n🎯 SELECCIONANDO FEATURES PARA RANSOMWARE")

    # Features específicas para detectar patrones de ransomware
    ransomware_pattern_features = [
        # Comunicación C&C (alto volumen, múltiples conexiones)
        ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        ' Flow Bytes/s', ' Flow Packets/s',

        # Comportamiento anómalo (timing irregular)
        ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
        ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
        ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',

        # Patrones de escaneo y exploración
        ' Destination Port', ' Fwd Packet Length Max', ' Fwd Packet Length Std',
        ' Bwd Packet Length Max', ' Bwd Packet Length Std',

        # Actividad de protocolo sospechosa
        'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
        ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
        ' Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',

        # Patrones de transferencia (exfiltración)
        ' Fwd Packets/s', ' Bwd Packets/s', ' Average Packet Size',
        ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
        ' Fwd Header Length', ' Bwd Header Length',

        # Comportamiento de sesión
        ' Active Mean', ' Active Std', ' Active Max', ' Active Min',
        ' Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
    ]

    # Buscar features disponibles
    available_features = []
    for feature in ransomware_pattern_features:
        if feature in df.columns:
            available_features.append(feature)

    print(f"   ✅ {len(available_features)} features seleccionadas")

    # Si no encontramos suficientes, usar las numéricas más importantes
    if len(available_features) < 15:
        print("   🔍 Complementando con features numéricas...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Excluir posibles columnas de label
        numeric_cols = [col for col in numeric_cols if 'Label' not in col and 'label' not in col]
        additional_features = [col for col in numeric_cols if col not in available_features]
        available_features.extend(additional_features[:20])  # Agregar hasta 20 adicionales

    print(f"   Features finales: {len(available_features)}")
    print(f"   Ejemplos: {available_features[:8]}")

    return available_features

def create_synthetic_ransomware_data(X_normal, n_anomalies=1000):
    """Crea datos sintéticos que simulan patrones de ransomware"""
    print(f"\n🔧 CREANDO PATRONES SINTÉTICOS DE RANSOMWARE")

    # Patrones típicos de ransomware en tráfico de red
    ransomware_patterns = []

    for i in range(n_anomalies):
        pattern = {}

        # Patrón 1: Comunicación C&C (alto volumen DNS/HTTP)
        if i < n_anomalies // 3:
            pattern.update({
                ' Flow Duration': np.random.exponential(500000),  # Conexiones largas
                ' Total Fwd Packets': np.random.poisson(1000),    # Muchos paquetes forward
                ' Flow Bytes/s': np.random.exponential(1000000),  # Alto throughput
                ' Flow IAT Std': np.random.exponential(100000),   # Timing irregular
                ' SYN Flag Count': np.random.poisson(50),         # Muchas SYNs
            })

        # Patrón 2: Exfiltración de datos (upload masivo)
        elif i < 2 * n_anomalies // 3:
            pattern.update({
                'Total Length of Fwd Packets': np.random.exponential(1000000),  # Muchos datos forward
                'Total Length of Bwd Packets': np.random.exponential(10000),    # Pocos datos backward
                ' Fwd Packets/s': np.random.exponential(1000),                  # Alta tasa forward
                ' Bwd Packets/s': np.random.exponential(10),                    # Baja tasa backward
                ' Average Packet Size': np.random.exponential(1500),            # Paquetes grandes
            })

        # Patrón 3: Escaneo y descubrimiento
        else:
            pattern.update({
                ' Destination Port': np.random.randint(1000, 65535),           # Puertos no estándar
                ' Flow IAT Mean': np.random.exponential(1000),                 # Timing rápido
                ' RST Flag Count': np.random.poisson(20),                      # Muchos RSTs
                ' Fwd Packet Length Std': np.random.exponential(500),          # Tamaño variable
                ' Active Std': np.random.exponential(100000),                  # Actividad irregular
            })

        ransomware_patterns.append(pattern)

    # Convertir a DataFrame y alinear con features existentes
    df_ransomware = pd.DataFrame(ransomware_patterns)

    # Asegurar que tenemos todas las columnas necesarias
    for col in X_normal.columns:
        if col not in df_ransomware.columns:
            # Para columnas faltantes, usar valores extremos del dataset normal
            if X_normal[col].dtype in [np.int64, np.float64]:
                df_ransomware[col] = np.random.choice([
                    X_normal[col].max() * 1.5,
                    X_normal[col].min() * 0.5,
                    X_normal[col].mean() * 3
                ], size=len(df_ransomware))

    # Reordenar columnas para que coincidan
    df_ransomware = df_ransomware[X_normal.columns]

    print(f"   ✅ {len(df_ransomware)} patrones de ransomware creados")
    return df_ransomware

def train_anomaly_detection_model():
    """Entrena modelo de detección de anomalías para ransomware"""
    print("🚀 ENTRENANDO DETECTOR DE ANOMALÍAS (RANSOMWARE)")
    print("=" * 60)

    # 1. Cargar datos de tráfico normal
    normal_data = load_and_sample_cic_data(n_samples=30000)

    # 2. Seleccionar features para ransomware
    feature_columns = select_ransomware_features(normal_data)

    # 3. Preparar datos normales
    X_normal = normal_data[feature_columns].fillna(0)

    # Limpiar datos infinitos y normalizar
    X_normal = X_normal.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4. Crear datos anómalos (ransomware sintético)
    X_ransomware = create_synthetic_ransomware_data(X_normal, n_anomalies=3000)

    # 5. Combinar datos para entrenamiento
    X_combined = pd.concat([X_normal, X_ransomware], ignore_index=True)
    y_combined = np.array([0] * len(X_normal) + [1] * len(X_ransomware))

    print(f"\n📊 DATASET DE ENTRENAMIENTO:")
    print(f"   Tráfico normal: {len(X_normal):,} muestras")
    print(f"   Patrones ransomware: {len(X_ransomware):,} muestras")
    print(f"   Total: {len(X_combined):,} muestras")
    print(f"   Features: {len(X_combined.columns)}")

    # 6. Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # 7. Split de datos
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_combined, test_size=0.3, stratify=y_combined, random_state=42
    )

    print(f"\n🔧 ENTRENANDO ISOLATION FOREST...")
    print(f"   Entrenamiento: {X_train.shape[0]:,} muestras")
    print(f"   Prueba: {X_test.shape[0]:,} muestras")

    # 8. Entrenar Isolation Forest
    model = IsolationForest(**ISOLATION_FOREST_PARAMS)
    model.fit(X_train)

    # 9. Evaluar modelo
    print(f"\n📈 EVALUANDO MODELO...")

    # Predecir anomalías (-1 para anomalías, 1 para normal)
    y_pred = model.predict(X_test)
    # Convertir a 0/1 (0=normal, 1=anomalía)
    y_pred_binary = np.where(y_pred == -1, 1, 0)

    # Métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary)
    rec = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    # Score de anomalía (cuanto más negativo, más anómalo)
    anomaly_scores = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, -anomaly_scores)  # Invertir porque IsolationForest devuelve scores negativos para anomalías

    print(f"🎯 MÉTRICAS DE DETECCIÓN DE RANSOMWARE:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_binary)
    print(f"\n📊 MATRIZ DE CONFUSIÓN:")
    print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}]")
    print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")

    # 10. Guardar modelo
    print(f"\n💾 GUARDANDO MODELO...")
    model_dir = OUTPUT_PATH / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / f"{MODEL_NAME}.pkl")
    joblib.dump(scaler, model_dir / f"{MODEL_NAME}_scaler.pkl")

    # Guardar lista de features
    with open(model_dir / f"{MODEL_NAME}_features.json", 'w') as f:
        json.dump(feature_columns, f, indent=2)

    # 11. Metadata
    metadata = {
        'model_name': MODEL_NAME,
        'model_type': 'RANSOMWARE_ANOMALY_DETECTOR',
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'algorithm': 'IsolationForest',
        'features_used': feature_columns,
        'feature_count': len(feature_columns),
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'training_params': ISOLATION_FOREST_PARAMS,
        'dataset_info': {
            'normal_samples': len(X_normal),
            'ransomware_samples': len(X_ransomware),
            'total_samples': len(X_combined),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'data_source': 'CIC-IDS-2017 + Synthetic Ransomware Patterns'
        },
        'detection_capability': 'Detección de patrones anómalos típicos de ransomware en tráfico de red',
        'usage_note': 'El modelo devuelve -1 para anomalías (posible ransomware) y 1 para tráfico normal'
    }

    with open(model_dir / f"{MODEL_NAME}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ MODELO GUARDADO EN: {model_dir}")
    print(f"🎯 CAPACIDAD DE DETECCIÓN:")
    print(f"   - Comunicación C&C sospechosa")
    print(f"   - Patrones de exfiltración de datos")
    print(f"   - Escaneo y descubrimiento de red")
    print(f"   - Comportamiento anómalo en timing y volumen")

    return model_dir, metadata

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 RANSOMWARE DETECTOR - DETECCIÓN DE ANOMALÍAS")
    print("=" * 60)
    print("Estrategia: Aprende tráfico normal + detecta desviaciones")
    print("Perfecto para ransomware y ataques desconocidos")
    print("=" * 60)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        model_dir, metadata = train_anomaly_detection_model()

        print(f"\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print(f"✅ Modelo de detección de anomalías entrenado")
        print(f"✅ Recall: {metadata['metrics']['recall']:.4f} (detección de ransomware)")
        print(f"✅ ROC-AUC: {metadata['metrics']['roc_auc']:.4f}")
        print(f"✅ Dataset: {metadata['dataset_info']['total_samples']:,} muestras")
        print(f"✅ Features: {metadata['feature_count']} de red")
        print(f"💾 Guardado en: {model_dir}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()