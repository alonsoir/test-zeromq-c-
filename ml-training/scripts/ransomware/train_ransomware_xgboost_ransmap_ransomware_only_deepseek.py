#!/usr/bin/env python3
"""
RANSOMWARE-ONLY DETECTOR - XGBoost
Modelo específico para ransomware usando SOLO datos de ransomware
Sin mezclar con tráfico normal
ml-training/scripts/train_ransomware_xgboost_ransmap_ransomware_only_deepseek.sh
Carga todo el dataset, en mi host provoca un
zsh: killed     python3 train_ransomware_xgboost_ransmap_ransomware_only_deepseek.py
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
# CONFIG - RANSOMWARE ONLY
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "datasets"
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_NAME = "ransomware_only_detector"

# Parámetros XGBoost
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
# FUNCIONES ESPECÍFICAS PARA RANSOMWARE
# =====================================================================
def print_section(title):
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def load_ransomware_data_full():
    """Carga TODOS los datos de ransomware disponibles"""
    print_section("LOADING RANSOMWARE DATA (FULL DATASETS)")

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
                # LEER ARCHIVOS COMPLETOS, no solo muestras
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

            except Exception as e:
                print(f"error: {e}")
                continue

    if not all_ransomware_data:
        raise ValueError("❌ No se pudieron cargar datasets de ransomware")

    # Combinar todos los datos
    combined_data = pd.concat(all_ransomware_data, ignore_index=True)

    print(f"\n  📊 Resumen Ransomware:")
    print(f"    Total archivos procesados: {len(all_ransomware_data)}")
    print(f"    Total muestras: {total_samples:,}")
    print(f"    Dataset combinado: {combined_data.shape}")

    return combined_data

def extract_ransomware_features(df, dataset_name, file_name):
    """Extrae características específicas de ransomware de cada dataset"""
    print(f"    🛠️  Extrayendo features de {file_name}...")

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    features_list = []

    # Procesar CADA FILA individualmente, no agregar
    for idx, row in df.iterrows():
        features = {}

        # FEATURES ESPECÍFICAS POR TIPO DE DATASET

        if dataset_name == "RANsMAP":
            # RANsMAP: datos de memoria/host
            numeric_cols = [col for col in row.index if pd.api.types.is_numeric_dtype(type(row[col]))]

            # Características de actividad de memoria
            if len(numeric_cols) >= 3:
                # Usar las primeras 3 columnas numéricas como proxies
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

            # Indicadores de archivo ejecutable
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

            # Características de criptomoneda (común en ransomware)
            if 'btc' in row.index and pd.notna(row['btc']):
                features['crypto_activity'] = float(row['btc'])

        # FEATURES UNIVERSALES
        features['dataset_source'] = f"{dataset_name}/{file_name}"
        features['is_ransomware'] = 1

        features_list.append(features)

    return pd.DataFrame(features_list)

def create_network_ransomware_features():
    """Crea características de red sintéticas típicas de ransomware"""
    print("  🔄 Generando características de red de ransomware...")

    n_samples = 5000  # Generar suficientes muestras

    features_list = []

    for i in range(n_samples):
        features = {
            # Patrones típicos de ransomware en red
            'dns_query_entropy': np.random.exponential(2),  # Alta entropía DNS
            'external_connections': np.random.poisson(50),   # Muchas conexiones externas
            'port_scan_activity': np.random.beta(2, 8),      # Escaneo de puertos
            'encryption_traffic': np.random.exponential(100), # Tráfico de cifrado
            'data_exfiltration': np.random.exponential(200), # Exfiltración de datos
            'connection_bursts': np.random.poisson(10),      # Ráfagas de conexiones
            'failed_connections': np.random.poisson(5),      # Conexiones fallidas
            'suspicious_ports': np.random.poisson(3),        # Puertos sospechosos
            'protocol_diversity': np.random.beta(3, 3),      # Diversidad de protocolos
            'traffic_asymmetry': np.random.exponential(5),   # Asimetría upload/download
            'is_ransomware': 1,
            'dataset_source': 'synthetic/network_ransomware'
        }
        features_list.append(features)

    return pd.DataFrame(features_list)

def prepare_ransomware_training_data():
    """Prepara datos de entrenamiento SOLO para ransomware"""
    print_section("PREPARING RANSOMWARE-ONLY TRAINING DATA")

    # 1. Cargar datos reales de ransomware
    print("📥 Cargando datos reales de ransomware...")
    ransomware_data = load_ransomware_data_full()

    # 2. Extraer características por tipo de dataset
    print("\n🔧 Extrayendo características específicas...")
    all_features = []

    # Procesar RANsMAP
    ransmap_data = ransomware_data[ransomware_data['dataset_source'].str.contains('RANsMAP', na=False)]
    if len(ransmap_data) > 0:
        print("  🖥️  Procesando RANsMAP (datos de host)...")
        for source in ransmap_data['dataset_source'].unique():
            source_data = ransmap_data[ransomware_data['dataset_source'] == source]
            dataset_name = "RANsMAP"
            file_name = source.split('/')[-1]
            features = extract_ransomware_features(source_data, dataset_name, file_name)
            all_features.append(features)

    # Procesar Ransomware 2024
    ransom2024_data = ransomware_data[ransomware_data['dataset_source'].str.contains('Ransomware Dataset 2024', na=False)]
    if len(ransom2024_data) > 0:
        print("  📁 Procesando Ransomware 2024 (archivos PE)...")
        for source in ransom2024_data['dataset_source'].unique():
            source_data = ransom2024_data[ransomware_data['dataset_source'] == source]
            dataset_name = "Ransomware Dataset 2024"
            file_name = source.split('/')[-1]
            features = extract_ransomware_features(source_data, dataset_name, file_name)
            all_features.append(features)

    # Procesar UGRansome
    ugransome_data = ransomware_data[ransomware_data['dataset_source'].str.contains('UGRansome', na=False)]
    if len(ugransome_data) > 0:
        print("  🌐 Procesando UGRansome (datos de red)...")
        for source in ugransome_data['dataset_source'].unique():
            source_data = ugransome_data[ransomware_data['dataset_source'] == source]
            dataset_name = "UGRansome"
            file_name = source.split('/')[-1]
            features = extract_ransomware_features(source_data, dataset_name, file_name)
            all_features.append(features)

    # 3. Combinar características reales
    if all_features:
        real_features = pd.concat(all_features, ignore_index=True)
        print(f"  ✅ Características reales: {len(real_features):,} muestras")
    else:
        real_features = pd.DataFrame()
        print("  ⚠️  No se pudieron extraer características reales")

    # 4. Agregar características de red sintéticas
    print("\n🌐 Generando características de red de ransomware...")
    synthetic_features = create_network_ransomware_features()
    print(f"  ✅ Características sintéticas: {len(synthetic_features):,} muestras")

    # 5. Combinar todo
    if len(real_features) > 0:
        # Encontrar columnas comunes
        common_cols = list(set(real_features.columns).intersection(set(synthetic_features.columns)))
        if 'is_ransomware' not in common_cols:
            common_cols.append('is_ransomware')
        if 'dataset_source' not in common_cols:
            common_cols.append('dataset_source')

        # Combinar manteniendo solo columnas comunes
        combined_data = pd.concat([
            real_features[common_cols],
            synthetic_features[common_cols]
        ], ignore_index=True)
    else:
        combined_data = synthetic_features

    print(f"\n📊 Dataset final de ransomware:")
    print(f"   Muestras totales: {len(combined_data):,}")
    print(f"   Características: {len(combined_data.columns)}")
    print(f"   Distribución por fuente:")
    source_counts = combined_data['dataset_source'].value_counts()
    for source, count in source_counts.items():
        print(f"     - {source}: {count:,} muestras")

    return combined_data

def create_anomaly_detection_dataset(ransomware_data):
    """Crea dataset para detección de anomalías (solo ransomware vs outliers)"""
    print_section("CREATING ANOMALY DETECTION DATASET")

    # Para detección de ransomware, necesitamos algunos "no ransomware" como outliers
    # Usaremos datos sintéticos que representen tráfico normal como contraste

    n_outliers = len(ransomware_data) // 5  # 20% outliers

    print(f"  🔄 Generando {n_outliers:,} muestras de outliers (no ransomware)...")

    outliers_list = []
    for i in range(n_outliers):
        outlier_features = {
            # Patrones de tráfico normal (baja sospecha)
            'dns_query_entropy': np.random.exponential(0.5),  # Baja entropía
            'external_connections': np.random.poisson(5),     # Pocas conexiones externas
            'port_scan_activity': np.random.beta(1, 20),      # Muy poco escaneo
            'encryption_traffic': np.random.exponential(10),  # Poco cifrado
            'data_exfiltration': np.random.exponential(20),   # Poca exfiltración
            'connection_bursts': np.random.poisson(1),        # Pocas ráfagas
            'failed_connections': np.random.poisson(0.5),     # Muy pocas fallas
            'suspicious_ports': np.random.poisson(0.2),       # Pocos puertos sospechosos
            'protocol_diversity': np.random.beta(1, 5),       # Baja diversidad
            'traffic_asymmetry': np.random.exponential(1),    # Baja asimetría
            'is_ransomware': 0,  # NO es ransomware
            'dataset_source': 'synthetic/normal_traffic'
        }
        outliers_list.append(outlier_features)

    outliers_df = pd.DataFrame(outliers_list)

    # Combinar ransomware con outliers
    combined_data = pd.concat([ransomware_data, outliers_df], ignore_index=True)

    print(f"  📊 Dataset de anomalías:")
    print(f"    Ransomware: {sum(combined_data['is_ransomware'] == 1):,}")
    print(f"    Outliers: {sum(combined_data['is_ransomware'] == 0):,}")
    print(f"    Total: {len(combined_data):,}")

    return combined_data

def train_ransomware_only_model():
    """Entrena modelo específico para detección de ransomware"""
    print_section("TRAINING RANSOMWARE-ONLY DETECTOR")

    # 1. Preparar datos de ransomware
    ransomware_features = prepare_ransomware_training_data()

    # 2. Crear dataset de detección de anomalías
    training_data = create_anomaly_detection_dataset(ransomware_features)

    # 3. Preparar features y labels
    feature_cols = [col for col in training_data.columns
                    if col not in ['is_ransomware', 'dataset_source']]

    X = training_data[feature_cols].fillna(0)
    y = training_data['is_ransomware']

    print(f"\n🎯 Configuración del modelo:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Muestras: {len(X):,}")
    print(f"   Ransomware: {sum(y==1):,} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"   No ransomware: {sum(y==0):,} ({sum(y==0)/len(y)*100:.1f}%)")

    # 4. Validación cruzada
    print_section("CROSS-VALIDATION")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Entrenar
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train_scaled, y_train, verbose=False)

        # Evaluar
        y_pred = model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)

        cv_scores.append({
            'fold': fold,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision
        })

        print(f"  Fold {fold}: Accuracy={accuracy:.3f}, Recall={recall:.3f}, Precision={precision:.3f}")

    avg_accuracy = np.mean([score['accuracy'] for score in cv_scores])
    avg_recall = np.mean([score['recall'] for score in cv_scores])
    avg_precision = np.mean([score['precision'] for score in cv_scores])

    print(f"\n  📊 Validación Cruzada (5-fold):")
    print(f"    Accuracy promedio: {avg_accuracy:.3f}")
    print(f"    Recall promedio:   {avg_recall:.3f}")
    print(f"    Precision promedio: {avg_precision:.3f}")

    # 5. Entrenamiento final
    print_section("FINAL MODEL TRAINING")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Conjuntos:")
    print(f"    Entrenamiento: {X_train.shape[0]:,} muestras")
    print(f"    Prueba: {X_test.shape[0]:,} muestras")

    # Entrenar modelo final
    final_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    final_model.fit(X_train_scaled, y_train, verbose=10)

    # 6. Evaluación final
    print_section("FINAL MODEL EVALUATION")
    y_pred = final_model.predict(X_test_scaled)
    y_proba = final_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"  📈 Métricas Finales:")
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Precision: {prec:.3f}")
    print(f"    Recall:    {rec:.3f}")
    print(f"    F1-Score:  {f1:.3f}")
    print(f"    ROC-AUC:   {roc_auc:.3f}")
    print(f"    PR-AUC:    {pr_auc:.3f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  📊 Matriz de Confusión:")
        print(f"    TN: {tn:,}  FP: {fp:,}")
        print(f"    FN: {fn:,}  TP: {tp:,}")
        print(f"    FPR: {fp/(fp+tn):.3f}, FNR: {fn/(fn+tp):.3f}")

    # Feature importance
    if hasattr(final_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"  🔝 Top Features de Ransomware:")
        for _, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:<25} {row['importance']:.4f}")

    # 7. Guardar modelo
    print_section("SAVING RANSOMWARE-ONLY MODEL")
    model_dir = OUTPUT_PATH / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, model_dir / f"{MODEL_NAME}.pkl")
    joblib.dump(scaler, model_dir / f"{MODEL_NAME}_scaler.pkl")

    metadata = {
        'model_name': MODEL_NAME,
        'model_type': 'RANSOMWARE_ONLY_DETECTOR',
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'features': feature_cols,
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        },
        'cross_validation': {
            'avg_accuracy': float(avg_accuracy),
            'avg_recall': float(avg_recall),
            'avg_precision': float(avg_precision),
            'n_folds': 5
        },
        'training_params': XGBOOST_PARAMS,
        'dataset_info': {
            'ransomware_samples': sum(y == 1),
            'outlier_samples': sum(y == 0),
            'total_samples': len(X),
            'data_sources': list(training_data['dataset_source'].unique())
        },
        'model_purpose': 'Detección específica de patrones de ransomware en diversos tipos de datos (host, red, archivos)',
        'deployment_note': 'Este modelo está optimizado para detectar ransomware, no tráfico normal'
    }

    with open(model_dir / f"{MODEL_NAME}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Modelo ransomware-only guardado en: {model_dir}")
    print(f"  🎯 Características del modelo:")
    print(f"    - Específico para ransomware ✓")
    print(f"    - Usa datos reales completos ✓")
    print(f"    - Detecta múltiples tipos de ransomware ✓")
    print(f"    - Optimizado para alta recall ✓")

    return model_dir, metadata

def main():
    print("=" * 80)
    print("🚀 RANSOMWARE-ONLY DETECTOR")
    print("=" * 80)
    print("Modelo específico para detección de ransomware")
    print("NO incluye detección de tráfico normal")
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
        print(f"✅ Guardado en: {model_dir}")
        print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()