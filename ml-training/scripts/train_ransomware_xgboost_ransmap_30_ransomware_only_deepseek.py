#!/usr/bin/env python3
"""
RANSOMWARE-ONLY DETECTOR - XGBoost - ULTRA MEMORY OPTIMIZED
Carga máximo 30% de datasets grandes y limita muestras totales
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
# CONFIG - ULTRA MEMORY OPTIMIZED
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "datasets"
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_NAME = "ransomware_only_detector_30pct"

# ULTRA MEMORY OPTIMIZATION SETTINGS
MEMORY_LIMIT_PCT = 0.30  # Solo 30% de datasets grandes
LARGE_DATASET_THRESHOLD = 50000  # Más agresivo: >50K muestras = grande
MAX_SAMPLES_PER_FILE = 20000  # Máximo 20K muestras por archivo
MAX_TOTAL_SAMPLES = 300000  # Máximo 300K muestras totales

# Parámetros XGBoost optimizados para menos memoria
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,  # Reducido
    'learning_rate': 0.15,
    'n_estimators': 150,  # Reducido
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 1,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# =====================================================================
# FUNCIONES ULTRA OPTIMIZADAS PARA MEMORIA
# =====================================================================
def print_section(title):
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def load_ransomware_data_ultra_optimized():
    """Carga datos con límites estrictos de memoria"""
    print_section("LOADING RANSOMWARE DATA (ULTRA OPTIMIZED - 30% LARGE DATASETS)")

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
            if total_samples >= MAX_TOTAL_SAMPLES:
                print(f"    ⏹️  Límite total alcanzado ({MAX_TOTAL_SAMPLES:,})")
                break

            print(f"    📖 {f.name}...", end=" ")
            try:
                # Verificar tamaño del archivo
                file_size = f.stat().st_size / (1024 * 1024)  # MB
                print(f"({file_size:.1f} MB)", end=" ")

                # ESTRATEGIA ULTRA OPTIMIZADA
                if file_size > 10:  # Archivo > 10MB
                    # Para archivos grandes, leer solo las columnas necesarias
                    print(f"GRANDE → muestreando...", end=" ")

                    # Leer solo una muestra del archivo
                    sample_frac = MEMORY_LIMIT_PCT  # 30%
                    df = pd.read_csv(f, low_memory=False)

                    # Muestrear agresivamente
                    if len(df) > 10000:
                        df = df.sample(n=int(len(df) * sample_frac), random_state=42)

                else:
                    # Archivo pequeño: cargar completo pero con límite
                    print(f"pequeño → cargando...", end=" ")
                    df = pd.read_csv(f, low_memory=False)
                    if len(df) > MAX_SAMPLES_PER_FILE:
                        df = df.sample(n=MAX_SAMPLES_PER_FILE, random_state=42)

                if df is None or len(df) == 0:
                    print("vacío")
                    continue

                # Verificar límite total
                if total_samples + len(df) > MAX_TOTAL_SAMPLES:
                    remaining = MAX_TOTAL_SAMPLES - total_samples
                    if remaining > 0:
                        df = df.sample(n=remaining, random_state=42)
                    else:
                        break

                # Marcar como ransomware
                df['is_ransomware'] = 1
                df['dataset_source'] = f"{dataset_name}/{f.name}"

                all_ransomware_data.append(df)
                total_samples += len(df)
                print(f"{len(df):,} muestras")

                # Limpiar memoria inmediatamente
                if len(all_ransomware_data) % 3 == 0:
                    gc.collect()

            except Exception as e:
                print(f"error: {e}")
                continue

        if total_samples >= MAX_TOTAL_SAMPLES:
            break

    if not all_ransomware_data:
        raise ValueError("❌ No se pudieron cargar datasets de ransomware")

    # Combinar todos los datos
    combined_data = pd.concat(all_ransomware_data, ignore_index=True)

    print(f"\n  📊 Resumen Ransomware (ULTRA OPTIMIZADO):")
    print(f"    Total archivos procesados: {len(all_ransomware_data)}")
    print(f"    Total muestras: {total_samples:,}")
    print(f"    Dataset combinado: {combined_data.shape}")
    print(f"    Memoria usada: {combined_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    return combined_data

def extract_ransomware_features_ultra_optimized(df, dataset_name, file_name):
    """Extrae características con límites estrictos de memoria"""
    print(f"    🛠️  Extrayendo features de {file_name}...")

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # LIMITAR muestras más agresivamente
    if len(df) > MAX_SAMPLES_PER_FILE:
        df = df.sample(n=MAX_SAMPLES_PER_FILE, random_state=42)
        print(f"      (muestreado a {MAX_SAMPLES_PER_FILE:,} muestras)")

    features_list = []

    # Procesar en lotes pequeños
    batch_size = 500  # Más pequeño
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        batch_features = []
        for idx, row in batch.iterrows():
            features = {}

            try:
                # FEATURES SIMPLIFICADAS - solo las más importantes
                if dataset_name == "RANsMAP":
                    numeric_cols = [col for col in row.index if pd.api.types.is_numeric_dtype(type(row[col]))]
                    if len(numeric_cols) >= 2:
                        val1 = float(row[numeric_cols[0]]) if pd.notna(row[numeric_cols[0]]) else 0
                        val2 = float(row[numeric_cols[1]]) if pd.notna(row[numeric_cols[1]]) else 0
                        features['memory_activity'] = val1 + val2
                        features['memory_intensity'] = abs(val1 * val2) / 1e6

                elif dataset_name == "Ransomware Dataset 2024":
                    if 'network_connections' in row.index and pd.notna(row['network_connections']):
                        features['network_activity'] = float(row['network_connections'])
                    if 'files_malicious' in row.index and pd.notna(row['files_malicious']):
                        features['malicious_files'] = float(row['files_malicious'])

                elif dataset_name == "UGRansome":
                    if 'netflow_bytes' in row.index and pd.notna(row['netflow_bytes']):
                        features['network_traffic'] = float(row['netflow_bytes'])
                    if 'threats' in row.index and pd.notna(row['threats']):
                        threats_str = str(row['threats'])
                        features['has_threats'] = 1 if threats_str != '0' and threats_str != '' else 0

                # FEATURES UNIVERSALES
                features['dataset_source'] = f"{dataset_name}/{file_name}"
                features['is_ransomware'] = 1

                batch_features.append(features)

            except Exception as e:
                continue

        features_list.extend(batch_features)

        # Limpiar memoria cada lote
        if len(features_list) > 5000:
            partial_df = pd.DataFrame(features_list)
            features_list = [partial_df]
            gc.collect()

    return pd.DataFrame(features_list) if features_list else pd.DataFrame()

def create_network_ransomware_features_ultra_optimized():
    """Crea características de red sintéticas ultra optimizadas"""
    print("  🔄 Generando características de red de ransomware...")

    n_samples = 1500  # Reducido a 1500

    # Generar eficientemente
    features = {
        'dns_query_entropy': np.random.exponential(2, n_samples),
        'external_connections': np.random.poisson(50, n_samples),
        'port_scan_activity': np.random.beta(2, 8, n_samples),
        'encryption_traffic': np.random.exponential(100, n_samples),
        'data_exfiltration': np.random.exponential(200, n_samples),
        'connection_bursts': np.random.poisson(10, n_samples),
        'is_ransomware': np.ones(n_samples),
        'dataset_source': ['synthetic/network_ransomware'] * n_samples
    }

    return pd.DataFrame(features)

def prepare_ransomware_training_data_ultra_optimized():
    """Prepara datos de entrenamiento ultra optimizados para memoria"""
    print_section("PREPARING ULTRA OPTIMIZED RANSOMWARE TRAINING DATA")

    # 1. Cargar datos reales de ransomware (ULTRA OPTIMIZADO)
    print("📥 Cargando datos reales de ransomware (ultra optimizado)...")
    ransomware_data = load_ransomware_data_ultra_optimized()

    # 2. Extraer características ultra optimizadas
    print("\n🔧 Extrayendo características específicas...")
    all_features = []
    total_processed = 0

    for dataset_name in ransomware_data['dataset_source'].str.split('/').str[0].unique():
        if total_processed >= MAX_TOTAL_SAMPLES:
            break

        dataset_data = ransomware_data[ransomware_data['dataset_source'].str.startswith(dataset_name)]

        print(f"  📊 Procesando {dataset_name} ({len(dataset_data):,} muestras)...")

        # Procesar por archivos individuales con límite
        for source in dataset_data['dataset_source'].unique()[:3]:  # Máximo 3 archivos por dataset
            if total_processed >= MAX_TOTAL_SAMPLES:
                break

            source_data = dataset_data[dataset_data['dataset_source'] == source]
            file_name = source.split('/')[-1]

            features = extract_ransomware_features_ultra_optimized(source_data, dataset_name, file_name)
            if len(features) > 0:
                all_features.append(features)
                total_processed += len(features)

            # Limpiar memoria agresivamente
            del source_data
            gc.collect()

    # 3. Combinar características reales
    if all_features:
        real_features = pd.concat(all_features, ignore_index=True)
        print(f"  ✅ Características reales: {len(real_features):,} muestras")
    else:
        real_features = pd.DataFrame()
        print("  ⚠️  No se pudieron extraer características reales")

    # 4. Agregar características de red sintéticas (ULTRA OPTIMIZADAS)
    print("\n🌐 Generando características de red de ransomware...")
    synthetic_features = create_network_ransomware_features_ultra_optimized()
    print(f"  ✅ Características sintéticas: {len(synthetic_features):,} muestras")

    # 5. Combinar todo optimizando memoria
    if len(real_features) > 0:
        common_cols = list(set(real_features.columns).intersection(set(synthetic_features.columns)))
        common_cols = [col for col in common_cols if col in real_features.columns and col in synthetic_features.columns]

        combined_data = pd.concat([
            real_features[common_cols],
            synthetic_features[common_cols]
        ], ignore_index=True)
    else:
        combined_data = synthetic_features

    # Aplicar límite final
    if len(combined_data) > MAX_TOTAL_SAMPLES:
        combined_data = combined_data.sample(n=MAX_TOTAL_SAMPLES, random_state=42)

    print(f"\n📊 Dataset final ULTRA OPTIMIZADO:")
    print(f"   Muestras totales: {len(combined_data):,}")
    print(f"   Características: {len(combined_data.columns)}")
    print(f"   Memoria usada: {combined_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    return combined_data

def create_anomaly_detection_dataset_optimized(ransomware_data):
    """Crea dataset para detección de anomalías optimizado"""
    print_section("CREATING OPTIMIZED ANOMALY DETECTION DATASET")

    # Reducir outliers también
    n_outliers = min(len(ransomware_data) // 10, 5000)  # Solo 10% outliers, máximo 5K

    print(f"  🔄 Generando {n_outliers:,} muestras de outliers...")

    # Generar eficientemente
    outliers_data = {
        'dns_query_entropy': np.random.exponential(0.5, n_outliers),
        'external_connections': np.random.poisson(5, n_outliers),
        'port_scan_activity': np.random.beta(1, 20, n_outliers),
        'encryption_traffic': np.random.exponential(10, n_outliers),
        'data_exfiltration': np.random.exponential(20, n_outliers),
        'connection_bursts': np.random.poisson(1, n_outliers),
        'is_ransomware': np.zeros(n_outliers),
        'dataset_source': ['synthetic/normal_traffic'] * n_outliers
    }

    outliers_df = pd.DataFrame(outliers_data)

    # Combinar manteniendo solo columnas comunes
    common_cols = list(set(ransomware_data.columns).intersection(set(outliers_df.columns)))
    combined_data = pd.concat([
        ransomware_data[common_cols],
        outliers_df[common_cols]
    ], ignore_index=True)

    print(f"  📊 Dataset de anomalías optimizado:")
    print(f"    Ransomware: {sum(combined_data['is_ransomware'] == 1):,}")
    print(f"    Outliers: {sum(combined_data['is_ransomware'] == 0):,}")
    print(f"    Total: {len(combined_data):,}")

    return combined_data

def train_ransomware_only_model_optimized():
    """Entrena modelo optimizado para memoria"""
    print_section("TRAINING RANSOMWARE-ONLY DETECTOR (ULTRA OPTIMIZED)")

    # 1. Preparar datos ultra optimizados
    ransomware_features = prepare_ransomware_training_data_ultra_optimized()

    # 2. Crear dataset optimizado
    training_data = create_anomaly_detection_dataset_optimized(ransomware_features)

    # 3. Preparar features y labels
    feature_cols = [col for col in training_data.columns
                    if col not in ['is_ransomware', 'dataset_source']]

    X = training_data[feature_cols].fillna(0)
    y = training_data['is_ransomware']

    print(f"\n🎯 Configuración del modelo ULTRA OPTIMIZADO:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Muestras: {len(X):,}")
    print(f"   Ransomware: {sum(y==1):,} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"   No ransomware: {sum(y==0):,} ({sum(y==0)/len(y)*100:.1f}%)")

    # Limpiar memoria inmediatamente
    del ransomware_features, training_data
    gc.collect()

    # 4. Validación cruzada optimizada
    print_section("OPTIMIZED CROSS-VALIDATION")

    # Usar menos folds para ahorrar memoria
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3 folds en lugar de 5
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Entrenar con parámetros optimizados
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

        # Limpiar memoria después de cada fold
        del X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, model
        gc.collect()

    avg_accuracy = np.mean([score['accuracy'] for score in cv_scores])
    avg_recall = np.mean([score['recall'] for score in cv_scores])
    avg_precision = np.mean([score['precision'] for score in cv_scores])

    print(f"\n  📊 Validación Cruzada (3-fold):")
    print(f"    Accuracy promedio: {avg_accuracy:.3f}")
    print(f"    Recall promedio:   {avg_recall:.3f}")
    print(f"    Precision promedio: {avg_precision:.3f}")

    # 5. Entrenamiento final optimizado
    print_section("FINAL MODEL TRAINING (OPTIMIZED)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Conjuntos optimizados:")
    print(f"    Entrenamiento: {X_train.shape[0]:,} muestras")
    print(f"    Prueba: {X_test.shape[0]:,} muestras")

    # Entrenar modelo final con early stopping
    final_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    final_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=20,
        verbose=10
    )

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

    print(f"  📈 Métricas Finales (Optimizadas):")
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

    # Feature importance (solo top 5 para ahorrar)
    if hasattr(final_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"  🔝 Top 5 Features de Ransomware:")
        for _, row in importance_df.head(5).iterrows():
            print(f"    {row['feature']:<25} {row['importance']:.4f}")

    # 7. Guardar modelo
    print_section("SAVING ULTRA OPTIMIZED MODEL")
    model_dir = OUTPUT_PATH / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, model_dir / f"{MODEL_NAME}.pkl")
    joblib.dump(scaler, model_dir / f"{MODEL_NAME}_scaler.pkl")

    metadata = {
        'model_name': MODEL_NAME,
        'model_type': 'RANSOMWARE_ONLY_DETECTOR_ULTRA_OPTIMIZED',
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'features': feature_cols,
        'memory_optimization': {
            'large_dataset_pct': MEMORY_LIMIT_PCT,
            'max_samples_per_file': MAX_SAMPLES_PER_FILE,
            'max_total_samples': MAX_TOTAL_SAMPLES,
            'final_memory_mb': X.memory_usage(deep=True).sum() / (1024**2)
        },
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
            'n_folds': 3
        },
        'training_params': XGBOOST_PARAMS,
        'dataset_info': {
            'ransomware_samples': sum(y == 1),
            'outlier_samples': sum(y == 0),
            'total_samples': len(X),
            'data_sources': list(pd.Series([src for src in X.index if 'dataset_source' in locals()]).unique()[:3]) if 'dataset_source' in locals() else []
        }
    }

    with open(model_dir / f"{MODEL_NAME}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Modelo ultra optimizado guardado en: {model_dir}")
    print(f"  🎯 Optimizaciones aplicadas:")
    print(f"    - Límite de 30% datasets grandes ✓")
    print(f"    - Máximo 20K muestras por archivo ✓")
    print(f"    - Máximo 300K muestras totales ✓")
    print(f"    - Procesamiento por lotes pequeños ✓")
    print(f"    - Limpieza agresiva de memoria ✓")

    return model_dir, metadata

def main():
    print("=" * 80)
    print("🚀 RANSOMWARE-ONLY DETECTOR - ULTRA MEMORY OPTIMIZED")
    print("=" * 80)
    print("Modelo específico para detección de ransomware")
    print("ULTRA OPTIMIZADO - 30% datasets grandes + límites estrictos")
    print("=" * 80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        model_dir, metadata = train_ransomware_only_model_optimized()

        print_section("TRAINING COMPLETED SUCCESSFULLY")
        print(f"✅ Modelo ransomware-only entrenado exitosamente")
        print(f"✅ Recall: {metadata['metrics']['recall']:.3f} (detección de ransomware)")
        print(f"✅ Precision: {metadata['metrics']['precision']:.3f}")
        print(f"✅ ROC-AUC: {metadata['metrics']['roc_auc']:.3f}")
        print(f"✅ Dataset: {metadata['dataset_info']['total_samples']:,} muestras")
        print(f"✅ Memoria optimizada: {metadata['memory_optimization']['final_memory_mb']:.1f} MB")
        print(f"✅ Estrategia: 30% datasets grandes + límites estrictos")
        print(f"✅ Guardado en: {model_dir}")
        print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()