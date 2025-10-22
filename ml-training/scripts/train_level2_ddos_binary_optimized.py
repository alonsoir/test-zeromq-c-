#!/usr/bin/env python3
"""
Train Level 2 DDoS Detector - Binary Classification (MEMORY OPTIMIZED)
Para VMs con RAM limitada (4-8GB)
ml-training/scripts/train_level2_ddos_binary_optimized.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURACIÓN - AJUSTAR SEGÚN TU RAM DISPONIBLE
# =====================================================================

DATASET_PATH = Path("datasets/CIC-DDoS-2019")
OUTPUT_PATH = Path("outputs")
MODEL_NAME = "level2_ddos_binary_detector"

# 🔧 CONFIGURACIÓN DE MEMORIA
# Para VM con 4GB RAM: usar SAMPLE_SIZE=200000
# Para VM con 6GB RAM: usar SAMPLE_SIZE=500000
# Para VM con 8GB+ RAM: usar SAMPLE_SIZE=1000000 o None (todo)
SAMPLE_SIZE = 200000  # 300k flows (~500MB en memoria)
MAX_ROWS_PER_FILE = 50000  # Leer máximo 50k filas por CSV

# Features a usar (reducido para ahorrar memoria)
FEATURES_TO_USE = [
    # Core features (las más importantes según paper CIC-DDoS-2019)
    'Source Port', 'Destination Port', 'Protocol',
    'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Max',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Max',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Fwd Header Length', 'Bwd Header Length',
    'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std',
    'Down/Up Ratio', 'Average Packet Size',
    'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
    'Fwd PSH Flags', 'Bwd PSH Flags',
    'Init Fwd Win Bytes', 'Init Bwd Win Bytes',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================

def print_memory_usage():
    """Mostrar uso de memoria actual"""
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"  💾 Memoria usada: {mem_mb:.1f} MB")


def load_ddos_dataset_optimized():
    """
    Cargar dataset con manejo inteligente de memoria
    - Lee archivos en chunks
    - Aplica sampling estratificado
    - Libera memoria agresivamente
    """
    print("=" * 80)
    print("📊 CARGANDO CIC-DDoS-2019 DATASET (OPTIMIZADO)")
    print("=" * 80)
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {DATASET_PATH}")
    
    csv_files = list(DATASET_PATH.rglob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSVs en: {DATASET_PATH}")
    
    print(f"\n✅ Archivos encontrados: {len(csv_files)}")
    print(f"🎯 Target sample size: {SAMPLE_SIZE:,} flows")
    print(f"📦 Max rows per file: {MAX_ROWS_PER_FILE:,}")
    
    # Calcular cuánto leer de cada archivo
    samples_per_file = SAMPLE_SIZE // len(csv_files)
    
    dfs = []
    total_loaded = 0
    
    for i, csv_file in enumerate(sorted(csv_files), 1):
        if total_loaded >= SAMPLE_SIZE:
            print(f"\n⏸️  Meta alcanzada: {total_loaded:,} samples")
            break
        
        print(f"[{i}/{len(csv_files)}] {csv_file.name:50s} ", end="")
        
        try:
            # Leer solo las columnas necesarias + Label
            cols_to_load = FEATURES_TO_USE + [' Label']
            
            # Leer en chunks para no agotar memoria
            chunk_size = min(MAX_ROWS_PER_FILE, samples_per_file)
            
            df_chunk = pd.read_csv(
                csv_file,
                encoding='utf-8',
                low_memory=False,
                nrows=chunk_size,
                usecols=lambda col: col in cols_to_load or 'label' in col.lower()
            )
            
            # Limpiar nombres de columnas
            df_chunk.columns = df_chunk.columns.str.strip()
            
            # Verificar que tiene label
            label_cols = [col for col in df_chunk.columns if 'label' in col.lower()]
            if not label_cols:
                print("❌ No label column")
                continue
            
            # Sampling estratificado si es muy grande
            if len(df_chunk) > samples_per_file:
                # Balancear BENIGN/ATTACK
                label_col = label_cols[0]
                benign = df_chunk[df_chunk[label_col].str.strip().str.upper() == 'BENIGN']
                attack = df_chunk[df_chunk[label_col].str.strip().str.upper() != 'BENIGN']
                
                n_benign = min(len(benign), samples_per_file // 2)
                n_attack = min(len(attack), samples_per_file // 2)
                
                df_chunk = pd.concat([
                    benign.sample(n=n_benign, random_state=42) if len(benign) > 0 else benign,
                    attack.sample(n=n_attack, random_state=42) if len(attack) > 0 else attack
                ], ignore_index=True)
            
            dfs.append(df_chunk)
            total_loaded += len(df_chunk)
            
            print(f"✅ {len(df_chunk):>6,} flows (total: {total_loaded:,})")
            
            # Liberar memoria
            del df_chunk
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not dfs:
        raise ValueError("No se pudieron cargar archivos")
    
    print(f"\n🔄 Concatenando {len(dfs)} chunks...")
    df_full = pd.concat(dfs, ignore_index=True)
    
    # Liberar memoria de chunks individuales
    del dfs
    gc.collect()
    
    print(f"✅ Dataset cargado: {df_full.shape[0]:,} flows, {df_full.shape[1]} columnas")
    print_memory_usage()
    
    return df_full


def preprocess_data(df):
    """Preprocesar datos con manejo eficiente de memoria"""
    print("\n" + "=" * 80)
    print("🧹 PREPROCESAMIENTO")
    print("=" * 80)
    
    # Buscar columna de labels
    label_cols = [col for col in df.columns if 'label' in col.lower()]
    
    if not label_cols:
        raise ValueError("No se encontró columna de labels")
    
    label_col = label_cols[0]
    print(f"\nColumna de labels: '{label_col}'")
    
    # Ver distribución original
    print("\n📊 Distribución original:")
    label_dist = df[label_col].value_counts()
    for label, count in list(label_dist.items())[:10]:  # Solo top 10
        pct = count / len(df) * 100
        print(f"  {label:30s}: {count:>10,} ({pct:>6.2f}%)")
    
    if len(label_dist) > 10:
        print(f"  ... y {len(label_dist)-10} clases más")
    
    # Convertir a binario
    y = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    
    print(f"\n📊 Distribución binaria:")
    print(f"  BENIGN (0): {(y == 0).sum():>10,} ({(y == 0).sum()/len(y)*100:>6.2f}%)")
    print(f"  DDOS   (1): {(y == 1).sum():>10,} ({(y == 1).sum()/len(y)*100:>6.2f}%)")
    
    # Seleccionar features disponibles
    print(f"\n🔍 Seleccionando features...")
    
    available_features = []
    missing_features = []
    
    for feat in FEATURES_TO_USE:
        if feat in df.columns:
            available_features.append(feat)
        else:
            missing_features.append(feat)
    
    print(f"  Features disponibles: {len(available_features)}/{len(FEATURES_TO_USE)}")
    
    if missing_features:
        print(f"  ⚠️  Features faltantes: {len(missing_features)}")
        if len(missing_features) <= 5:
            for feat in missing_features:
                print(f"    - {feat}")
    
    # Extraer features
    X = df[available_features].copy()
    
    # Liberar memoria del dataframe original
    del df
    gc.collect()
    
    # Limpiar datos
    print(f"\n🧹 Limpiando datos...")
    
    # Reemplazar infinitos
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(X[numeric_cols]).sum().sum()
    print(f"  Infinitos: {inf_count:,}")
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Rellenar nulos con mediana
    null_count = X.isnull().sum().sum()
    print(f"  Nulos: {null_count:,}")
    
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    X.fillna(0, inplace=True)
    
    print(f"\n✅ Datos limpios: {X.shape[0]:,} samples, {X.shape[1]} features")
    print_memory_usage()
    
    return X, y, available_features


def train_model(X, y, use_smote=True):
    """Entrenar modelo con configuración optimizada para RAM limitada"""
    print("\n" + "=" * 80)
    print("🌲 ENTRENAMIENTO RANDOM FOREST")
    print("=" * 80)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\n📊 Split:")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print_memory_usage()
    
    # SMOTE solo si el dataset no está muy desbalanceado
    if use_smote:
        benign_count = (y_train == 0).sum()
        ddos_count = (y_train == 1).sum()
        imbalance_ratio = max(benign_count, ddos_count) / min(benign_count, ddos_count)
        
        if imbalance_ratio > 3:  # Solo si está muy desbalanceado
            print(f"\n🔄 Aplicando SMOTE...")
            print(f"  Antes: BENIGN={benign_count:,}, DDOS={ddos_count:,}")
            
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            print(f"  Después: BENIGN={(y_train==0).sum():,}, DDOS={(y_train==1).sum():,}")
            print_memory_usage()
        else:
            print(f"\n⚖️  Dataset ya balanceado (ratio {imbalance_ratio:.2f}), saltando SMOTE")
    
    # Random Forest con parámetros optimizados para RAM
    print(f"\n🌲 Entrenando Random Forest...")
    print(f"  Parámetros (optimizados para RAM):")
    print(f"    n_estimators: 100")
    print(f"    max_depth: 20")
    print(f"    min_samples_split: 20")
    print(f"    min_samples_leaf: 10")
    print(f"    max_features: sqrt")
    print(f"    n_jobs: 2")
    
    rf = RandomForestClassifier(
        n_estimators=100,  # Menos árboles = menos RAM
        max_depth=20,      # Menor profundidad = menos RAM
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',  # sqrt en vez de None = menos RAM
        random_state=42,
        n_jobs=2,  # Solo 2 threads para no agotar RAM
        class_weight='balanced',
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    print("\n✅ Modelo entrenado")
    print_memory_usage()
    
    return rf, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluar modelo"""
    print("\n" + "=" * 80)
    print("📊 EVALUACIÓN")
    print("=" * 80)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas Test
    print("\n📈 Test Metrics:")
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['BENIGN', 'DDOS'],
                yticklabels=['BENIGN', 'DDOS'])
    plt.title('Confusion Matrix - Level 2 DDoS Detector')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = OUTPUT_PATH / "plots" / f"{MODEL_NAME}_confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Confusion matrix: {output_path}")
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist()
    }


def save_model(model, metrics, features_used):
    """Guardar modelo y metadata"""
    print("\n" + "=" * 80)
    print("💾 GUARDANDO MODELO")
    print("=" * 80)
    
    # Guardar modelo
    model_path = OUTPUT_PATH / "models" / f"{MODEL_NAME}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✅ Modelo: {model_path}")
    
    # Metadata
    metadata = {
        "model_name": MODEL_NAME,
        "model_type": "RandomForest",
        "level": 2,
        "purpose": "DDoS Binary Detection (BENIGN vs DDOS)",
        "version": "1.0.0",
        "training_date": datetime.now().isoformat(),
        "dataset": "CIC-DDoS-2019",
        "sample_size": SAMPLE_SIZE,
        "n_features": len(features_used),
        "feature_names": features_used,
        "classes": ["BENIGN", "DDOS"],
        "class_mapping": {"0": "BENIGN", "1": "DDOS"},
        "rf_params": {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 20,
            "min_samples_leaf": 10
        },
        "metrics": metrics,
        "onnx_compatible": True
    }
    
    metadata_path = OUTPUT_PATH / "metadata" / f"{MODEL_NAME}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata: {metadata_path}")


def main():
    """Función principal"""
    print("=" * 80)
    print("🎯 LEVEL 2 DDOS DETECTOR - MEMORY OPTIMIZED")
    print("=" * 80)
    print(f"\nConfiguración:")
    print(f"  Sample size: {SAMPLE_SIZE:,} flows")
    print(f"  Max rows/file: {MAX_ROWS_PER_FILE:,}")
    print(f"  Features: {len(FEATURES_TO_USE)}")
    
    try:
        # 1. Cargar dataset
        df = load_ddos_dataset_optimized()
        
        # 2. Preprocesar
        X, y, features_used = preprocess_data(df)
        
        # 3. Entrenar
        model, X_train, X_test, y_train, y_test = train_model(X, y, use_smote=True)
        
        # 4. Evaluar
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # 5. Guardar
        save_model(model, metrics, features_used)
        
        # Resumen
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETADO")
        print("=" * 80)
        print("\n📊 Métricas finales:")
        print(f"  Accuracy:  {metrics['accuracy']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1-Score:  {metrics['f1_score']:.2%}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        print("\n🎯 Siguiente paso:")
        print("  python scripts/convert_level2_ddos_to_onnx.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
