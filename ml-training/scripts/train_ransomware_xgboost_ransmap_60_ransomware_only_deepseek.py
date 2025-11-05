#!/usr/bin/env python3
"""
RANSOMWARE-ONLY DETECTOR - XGBoost - MEMORY OPTIMIZED
Carga el 60% de datasets grandes y 100% de datasets pequeños
"""

import pandas as pd
import numpy as np
import tarfile
import zipfile
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json
from datetime import datetime
import warnings
import gc
import sys

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG - RANSOMWARE ONLY - MEMORY OPTIMIZED
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "datasets"
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_NAME = "ransomware_only_detector_60pct"

# MEMORY OPTIMIZATION SETTINGS
MEMORY_LIMIT_PCT = 0.60  # 60% de datasets grandes
LARGE_DATASET_THRESHOLD = 100000  # Considerar "grande" > 100K muestras

# Parámetros XGBoost optimizados
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# =====================================================================
# FUNCIONES OPTIMIZADAS PARA MEMORIA
# =====================================================================
def print_section(title):
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def load_ransomware_data_optimized():
    """Carga datos de ransomware optimizados para memoria"""
    print_section("LOADING RANSOMWARE DATA (OPTIMIZED - 60% LARGE DATASETS)")

    ransomware_sources = [
        ("RANsMAP", "RANsMAP"),
        ("Ransomware Dataset 2024", "Ransomware Dataset 2024"),
        ("UGRansome", "UGRansome")
    ]

    all_ransomware_data = []
    total_samples = 0

    for dataset_name, path_name in ransomware_sources:
        dataset_path = DATASET_PATH / path_name
        if not dataset_path.exists():
            print(f"  ⚠️  {dataset_name} no encontrado")
            continue

        print(f"  📂 {dataset_name}:")
        csv_files = list(dataset_path.rglob("*.csv"))

        for f in csv_files:
            print(f"    📖 {f.name}...", end=" ")
            try:
                # PRIMERO verificar el tamaño del archivo
                file_size = f.stat().st_size / (1024 * 1024)  # MB
                print(f"({file_size:.1f} MB)", end=" ")

                # Leer solo las primeras filas para estimar tamaño
                sample_df = pd.read_csv(f, nrows=1000, low_memory=False)
                estimated_rows = (file_size * 1024 * 1024) // (sample_df.memory_usage(deep=True).sum() / len(sample_df))

                print(f"~{estimated_rows:,.0f} filas estimadas...", end=" ")

                # ESTRATEGIA DE CARGA OPTIMIZADA
                if estimated_rows > LARGE_DATASET_THRESHOLD:
                    # Dataset grande: cargar solo el 60%
                    rows_to_load = int(estimated_rows * MEMORY_LIMIT_PCT)
                    print(f"GRANDE → cargando {rows_to_load:,} filas (60%)...", end=" ")

                    # Leer muestra representativa
                    df = pd.read_csv(f, low_memory=False)

                    # Si es demasiado grande, muestrear
                    if len(df) > rows_to_load:
                        df = df.sample(n=rows_to_load, random_state=42)

                else:
                    # Dataset pequeño: cargar 100%
                    print(f"pequeño → cargando 100%...", end=" ")
                    df = pd.read_csv(f, low_memory=False)

                if df is None or len(df) == 0:
                    print("vacío")
                    continue

                # Marcar como ransomware
                df['is_ransomware'] = 1
                df['dataset_source'] = f"{dataset_name}/{f.name}"

                all_ransomware_data.append(df)
                total_samples += len(df)
                print(f"{len(df):,} muestras")

                # Limpiar memoria periódicamente
                if total_samples > 500000:
                    gc.collect()

            except Exception as e:
                print(f"error: {e}")
                continue

    if not all_ransomware_data:
        raise ValueError("❌ No se pudieron cargar datasets de ransomware")

    # Combinar todos los datos
    combined_data = pd.concat(all_ransomware_data, ignore_index=True)

    print(f"\n  📊 Resumen Ransomware (OPTIMIZADO):")
    print(f"    Total archivos procesados: {len(all_ransomware_data)}")
    print(f"    Total muestras: {total_samples:,}")
    print(f"    Dataset combinado: {combined_data.shape}")
    print(f"    Memoria usada: {combined_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    return combined_data

def extract_ransomware_features_optimized(df, dataset_name, file_name):
    """Extrae características optimizadas para memoria"""
    print(f"    🛠️  Extrayendo features de {file_name}...")

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # LIMITAR el número de muestras si es muy grande
    max_samples_per_file = 50000
    if len(df) > max_samples_per_file:
        df = df.sample(n=max_samples_per_file, random_state=42)
        print(f"      (muestreado a {max_samples_per_file:,} muestras)")

    features_list = []

    # Procesar en lotes para ahorrar memoria
    batch_size = 1000
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        batch_features = []
        for idx, row in batch.iterrows():
            features = {}

            # FEATURES ESPECÍFICAS POR TIPO DE DATASET
            if dataset_name == "RANsMAP":
                # RANsMAP: datos de memoria/host
                numeric_cols = [col for col in row.index if pd.api.types.is_numeric_dtype(type(row[col]))]

                if len(numeric_cols) >= 3:
                    val1 = float(row[numeric_cols[0]]) if pd.notna(row[numeric_cols[0]]) else 0
                    val2 = float(row[numeric_cols[1]]) if pd.notna(row[numeric_cols[1]]) else 0
                    val3 = float(row[numeric_cols[2]]) if pd.notna(row[numeric_cols[2]]) else 0

                    features['memory_activity'] = val1 + val2 + val3
                    features['memory_intensity'] = abs(val1 * val2) / 1e6
                    features['access_pattern'] = val3 / max(abs(val1), 1)

            elif dataset_name == "Ransomware Dataset 2024":
                # Dataset de características de archivos PE
                if 'registry_total' in row.index and pd.notna(row['registry_total']):
                    features['registry_activity'] = float(row['registry_total'])
                if 'network_connections' in row.index and pd.notna(row['network_connections']):
                    features['network_activity'] = float(row['network_connections'])
                if 'files_malicious' in row.index and pd.notna(row['files_malicious']):
                    features['malicious_files'] = float(row['files_malicious'])
                if 'dlls_calls' in row.index and pd.notna(row['dlls_calls']):
                    features['dll_activity'] = float(row['dlls_calls'])

                if 'file_extension' in row.index and pd.notna(row['file_extension']):
                    features['is_executable'] = 1 if str(row['file_extension']).lower() in ['exe', 'dll', 'bin'] else 0

            elif dataset_name == "UGRansome":
                # UGRansome: datos de red/ransomware
                if 'netflow_bytes' in row.index and pd.notna(row['netflow_bytes']):
                    features['network_traffic'] = float(row['netflow_bytes'])
                if 'clusters' in row.index and pd.notna(row['clusters']):
                    features['cluster_activity'] = float(row['clusters'])
                if 'threats' in row.index and pd.notna(row['threats']):
                    threats_str = str(row['threats'])
                    features['has_threats'] = 1 if threats_str != '0' and threats_str != '' else 0

                if 'btc' in row.index and pd.notna(row['btc']):
                    features['crypto_activity'] = float(row['btc'])

            # FEATURES UNIVERSALES
            features['dataset_source'] = f"{dataset_name}/{file_name}"
            features['is_ransomware'] = 1

            batch_features.append(features)

        features_list.extend(batch_features)

        # Limpiar memoria cada lote
        if len(features_list) > 10000:
            gc.collect()

    return pd.DataFrame(features_list)

def create_network_ransomware_features_optimized():
    """Crea características de red sintéticas optimizadas"""
    print("  🔄 Generando características de red de ransomware...")

    n_samples = 3000  # Reducido de 5000 a 3000

    # Generar todas las muestras de una vez (más eficiente)
    features = {
        'dns_query_entropy': np.random.exponential(2, n_samples),
        'external_connections': np.random.poisson(50, n_samples),
        'port_scan_activity': np.random.beta(2, 8, n_samples),
        'encryption_traffic': np.random.exponential(100, n_samples),
        'data_exfiltration': np.random.exponential(200, n_samples),
        'connection_bursts': np.random.poisson(10, n_samples),
        'failed_connections': np.random.poisson(5, n_samples),
        'suspicious_ports': np.random.poisson(3, n_samples),
        'protocol_diversity': np.random.beta(3, 3, n_samples),
        'traffic_asymmetry': np.random.exponential(5, n_samples),
        'is_ransomware': np.ones(n_samples),
        'dataset_source': ['synthetic/network_ransomware'] * n_samples
    }

    return pd.DataFrame(features)

def prepare_ransomware_training_data_optimized():
    """Prepara datos de entrenamiento optimizados para memoria"""
    print_section("PREPARING OPTIMIZED RANSOMWARE TRAINING DATA")

    # 1. Cargar datos reales de ransomware (OPTIMIZADO)
    print("📥 Cargando datos reales de ransomware (optimizado)...")
    ransomware_data = load_ransomware_data_optimized()

    # 2. Extraer características optimizadas
    print("\n🔧 Extrayendo características específicas...")
    all_features = []

    # Procesar por tipo de dataset con límites de memoria
    for dataset_name in ransomware_data['dataset_source'].str.split('/').str[0].unique():
        dataset_data = ransomware_data[ransomware_data['dataset_source'].str.startswith(dataset_name)]

        print(f"  📊 Procesando {dataset_name} ({len(dataset_data):,} muestras)...")

        # Procesar por archivos individuales
        for source in dataset_data['dataset_source'].unique():
            source_data = dataset_data[dataset_data['dataset_source'] == source]
            file_name = source.split('/')[-1]

            features = extract_ransomware_features_optimized(source_data, dataset_name, file_name)
            if len(features) > 0:
                all_features.append(features)

            # Limpiar memoria
            gc.collect()

    # 3. Combinar características reales
    if all_features:
        real_features = pd.concat(all_features, ignore_index=True)
        print(f"  ✅ Características reales: {len(real_features):,} muestras")
    else:
        real_features = pd.DataFrame()
        print("  ⚠️  No se pudieron extraer características reales")

    # 4. Agregar características de red sintéticas (OPTIMIZADAS)
    print("\n🌐 Generando características de red de ransomware...")
    synthetic_features = create_network_ransomware_features_optimized()
    print(f"  ✅ Características sintéticas: {len(synthetic_features):,} muestras")

    # 5. Combinar todo optimizando memoria
    if len(real_features) > 0:
        common_cols = list(set(real_features.columns).intersection(set(synthetic_features.columns)))
        if 'is_ransomware' not in common_cols:
            common_cols.append('is_ransomware')
        if 'dataset_source' not in common_cols:
            common_cols.append('dataset_source')

        combined_data = pd.concat([
            real_features[common_cols],
            synthetic_features[common_cols]
        ], ignore_index=True)
    else:
        combined_data = synthetic_features

    print(f"\n📊 Dataset final OPTIMIZADO de ransomware:")
    print(f"   Muestras totales: {len(combined_data):,}")
    print(f"   Características: {len(combined_data.columns)}")
    print(f"   Memoria usada: {combined_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    return combined_data

# ... (las funciones restantes create_anomaly_detection_dataset, train_ransomware_only_model, y main
# se mantienen igual pero usando las funciones optimizadas)

def train_ransomware_only_model():
    """Entrena modelo específico para detección de ransomware"""
    print_section("TRAINING RANSOMWARE-ONLY DETECTOR (OPTIMIZED)")

    # 1. Preparar datos de ransomware OPTIMIZADOS
    ransomware_features = prepare_ransomware_training_data_optimized()

    # 2. Crear dataset de detección de anomalías
    training_data = create_anomaly_detection_dataset(ransomware_features)

    # 3. Preparar features y labels
    feature_cols = [col for col in training_data.columns
                    if col not in ['is_ransomware', 'dataset_source']]

    X = training_data[feature_cols].fillna(0)
    y = training_data['is_ransomware']

    print(f"\n🎯 Configuración del modelo OPTIMIZADO:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Muestras: {len(X):,}")
    print(f"   Ransomware: {sum(y==1):,} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"   No ransomware: {sum(y==0):,} ({sum(y==0)/len(y)*100:.1f}%)")

    # Limpiar memoria de datos intermedios
    del ransomware_features, training_data
    gc.collect()

    # ... (resto del código de entrenamiento igual)

def main():
    print("=" * 80)
    print("🚀 RANSOMWARE-ONLY DETECTOR - MEMORY OPTIMIZED (60% LARGE DATASETS)")
    print("=" * 80)
    print("Modelo específico para detección de ransomware")
    print("OPTIMIZADO para memoria - Carga 60% de datasets grandes")
    print("=" * 80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        model_dir, metadata = train_ransomware_only_model()

        print_section("TRAINING COMPLETED")
        print(f"✅ Modelo ransomware-only entrenado exitosamente")
        print(f"✅ Recall: {metadata['metrics']['recall']:.3f} (detección de ransomware)")
        print(f"✅ Precision: {metadata['metrics']['precision']:.3f}")
        print(f"✅ ROC-AUC: {metadata['metrics']['roc_auc']:.3f}")
        print(f"✅ Dataset: {metadata['dataset_info']['total_samples']:,} muestras")
        print(f"✅ Estrategia: 60% datasets grandes + 100% datasets pequeños")
        print(f"✅ Guardado en: {model_dir}")
        print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
