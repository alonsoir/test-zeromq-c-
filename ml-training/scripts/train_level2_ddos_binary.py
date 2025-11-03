#!/usr/bin/env python3
"""
Train Level 2 DDoS Detector - Binary Classification
BENIGN vs DDOS (all DDoS types aggregated)
Dataset: CIC-DDoS-2019
ml-training/scripts/train_level2_ddos_binary.py
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
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

DATASET_PATH = Path("datasets/CIC-DDoS-2019")
OUTPUT_PATH = Path("outputs")
MODEL_NAME = "level2_ddos_binary_detector"

# Features numéricas que mapean a NetworkFeatures del protobuf
# Basadas en las features individuales (lines 26-130 del .proto)
FEATURES_TO_USE = [
    # Puertos y protocolo
    'Source Port', 'Destination Port', 'Protocol',
    
    # Estadísticas de paquetes
    'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    
    # Longitudes de paquetes - Forward
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    
    # Longitudes de paquetes - Backward  
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    
    # Velocidades y ratios
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    
    # Inter-arrival times - Forward
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min',
    
    # Inter-arrival times - Backward
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min',
    
    # TCP Flags
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count',
    
    # Tamaños de segmentos
    'min_seg_size_forward', 'avg_seg_size_forward',
    'Fwd Header Length', 'Bwd Header Length',
    
    # Bulk transfer
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    
    # Estadísticas adicionales
    'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    
    # Ratios
    'Down/Up Ratio', 'Average Packet Size',
    'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
    
    # Sub-flow metrics
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    
    # Flags direccionales
    'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags',
    
    # Inicializaciones
    'Fwd Header Length.1', 'Init Fwd Win Bytes', 'Init Bwd Win Bytes',
    'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    
    # Active/Idle times
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================

def load_ddos_dataset(sample_size=None):
    """
    Cargar dataset CIC-DDoS-2019
    
    Args:
        sample_size: Si se especifica, limita el número de samples (útil para pruebas)
    
    Returns:
        DataFrame con todos los datos
    """
    print("=" * 80)
    print("📊 CARGANDO CIC-DDoS-2019 DATASET")
    print("=" * 80)
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset no encontrado en: {DATASET_PATH}")
    
    # Buscar todos los CSVs
    csv_files = list(DATASET_PATH.rglob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSVs en: {DATASET_PATH}")
    
    print(f"\n✅ Archivos encontrados: {len(csv_files)}")
    
    dfs = []
    total_rows = 0
    
    for csv_file in sorted(csv_files):
        print(f"  Cargando {csv_file.name}...", end=" ")
        
        try:
            # Cargar con encoding robusto
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
            
            # Si hay sample_size, limitar
            if sample_size and total_rows >= sample_size:
                print(f"[SKIP - ya tenemos {sample_size:,} samples]")
                continue
                
            if sample_size:
                remaining = sample_size - total_rows
                df = df.head(remaining)
            
            dfs.append(df)
            total_rows += len(df)
            print(f"✅ {len(df):,} flows")
            
            if sample_size and total_rows >= sample_size:
                print(f"\n⏸️  Límite alcanzado: {sample_size:,} samples")
                break
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not dfs:
        raise ValueError("No se pudieron cargar archivos")
    
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Dataset completo: {df_full.shape[0]:,} flows, {df_full.shape[1]} columnas")
    
    return df_full


def preprocess_data(df):
    """
    Preprocesar datos: limpiar, mapear labels, seleccionar features
    
    Args:
        df: DataFrame crudo
    
    Returns:
        X, y: Features y labels procesados
    """
    print("\n" + "=" * 80)
    print("🧹 PREPROCESAMIENTO")
    print("=" * 80)
    
    # Buscar columna de labels (puede variar)
    label_cols = [col for col in df.columns if 'label' in col.lower()]
    
    if not label_cols:
        raise ValueError("No se encontró columna de labels")
    
    label_col = label_cols[0]
    print(f"\nColumna de labels: '{label_col}'")
    
    # Ver distribución original
    print("\n📊 Distribución original:")
    label_dist = df[label_col].value_counts()
    for label, count in label_dist.items():
        pct = count / len(df) * 100
        print(f"  {label:30s}: {count:>10,} ({pct:>6.2f}%)")
    
    # Convertir a binario: BENIGN (0) vs DDOS (1)
    # Todos los ataques DDoS se agrupan como clase 1
    y = df[label_col].apply(lambda x: 0 if x.strip().upper() == 'BENIGN' else 1)
    
    print(f"\n📊 Distribución binaria:")
    print(f"  BENIGN (0): {(y == 0).sum():>10,} ({(y == 0).sum()/len(y)*100:>6.2f}%)")
    print(f"  DDOS   (1): {(y == 1).sum():>10,} ({(y == 1).sum()/len(y)*100:>6.2f}%)")
    
    # Seleccionar features
    print(f"\n🔍 Seleccionando features...")
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Verificar qué features existen
    available_features = []
    missing_features = []
    
    for feat in FEATURES_TO_USE:
        if feat in df.columns:
            available_features.append(feat)
        else:
            missing_features.append(feat)
    
    print(f"  Features disponibles: {len(available_features)}/{len(FEATURES_TO_USE)}")
    
    if missing_features:
        print(f"\n⚠️  Features faltantes ({len(missing_features)}):")
        for feat in missing_features[:10]:
            print(f"    - {feat}")
        if len(missing_features) > 10:
            print(f"    ... y {len(missing_features)-10} más")
    
    # Usar solo features disponibles
    X = df[available_features].copy()
    
    # Limpiar datos
    print(f"\n🧹 Limpiando datos...")
    
    # Reemplazar infinitos
    inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    print(f"  Valores infinitos: {inf_count:,}")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Rellenar nulos con mediana
    null_count = X.isnull().sum().sum()
    print(f"  Valores nulos: {null_count:,}")
    
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
    
    X.fillna(0, inplace=True)
    
    print(f"\n✅ Datos limpios: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    return X, y, available_features


def train_model(X, y, use_smote=True):
    """
    Entrenar modelo Random Forest con balanceo SMOTE
    
    Args:
        X: Features
        y: Labels
        use_smote: Aplicar SMOTE para balancear clases
    
    Returns:
        model, X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 80)
    print("🌲 ENTRENAMIENTO RANDOM FOREST")
    print("=" * 80)
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\n📊 Split:")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    
    # Aplicar SMOTE si está desbalanceado
    if use_smote and (y_train == 0).sum() != (y_train == 1).sum():
        print(f"\n🔄 Aplicando SMOTE para balancear clases...")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"  Antes: BENIGN={( y_train==0).sum():,}, DDOS={( y_train==1).sum():,}")
        print(f"  Después: BENIGN={(y_train_balanced==0).sum():,}, DDOS={(y_train_balanced==1).sum():,}")
        
        X_train = X_train_balanced
        y_train = y_train_balanced
    
    # Entrenar Random Forest
    print(f"\n🌲 Entrenando Random Forest...")
    print(f"  Parámetros:")
    print(f"    n_estimators: 150")
    print(f"    max_depth: 25")
    print(f"    min_samples_split: 10")
    print(f"    min_samples_leaf: 4")
    print(f"    class_weight: balanced")
    
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    print("\n✅ Modelo entrenado")
    
    # Cross-validation
    print("\n🔄 Validación cruzada (5-fold)...")
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    print(f"  F1 scores: {cv_scores}")
    print(f"  F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return rf, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluar modelo y generar métricas
    
    Args:
        model: Modelo entrenado
        X_train, X_test: Features train/test
        y_train, y_test: Labels train/test
    
    Returns:
        dict con métricas
    """
    print("\n" + "=" * 80)
    print("📊 EVALUACIÓN DEL MODELO")
    print("=" * 80)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas Train
    print("\n📈 Train Metrics:")
    print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f"  Recall:    {recall_score(y_train, y_train_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_train, y_train_pred):.4f}")
    
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
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['BENIGN', 'DDOS']))
    
    # Plot Confusion Matrix
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
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Level 2 DDoS Detector')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = OUTPUT_PATH / "plots" / f"{MODEL_NAME}_roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve: {roc_path}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    top_n = feature_importance.head(30)
    plt.barh(range(len(top_n)), top_n['importance'], color='red', alpha=0.7)
    plt.yticks(range(len(top_n)), top_n['feature'])
    plt.xlabel('Importance')
    plt.title('Top 30 Features - Level 2 DDoS Detector')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    feat_path = OUTPUT_PATH / "plots" / f"{MODEL_NAME}_feature_importance.png"
    plt.savefig(feat_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance: {feat_path}")
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')
    }


def save_model(model, metrics, features_used):
    """
    Guardar modelo y metadata
    
    Args:
        model: Modelo entrenado
        metrics: Métricas de evaluación
        features_used: Lista de features usadas
    """
    print("\n" + "=" * 80)
    print("💾 GUARDANDO MODELO")
    print("=" * 80)
    
    # Guardar modelo sklearn
    model_path = OUTPUT_PATH / "models" / f"{MODEL_NAME}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✅ Modelo: {model_path}")
    
    # Metadata completo
    metadata = {
        "model_name": MODEL_NAME,
        "model_type": "RandomForest",
        "level": 2,
        "purpose": "DDoS Binary Detection (BENIGN vs DDOS)",
        "version": "1.0.0",
        "training_date": datetime.now().isoformat(),
        "dataset": "CIC-DDoS-2019",
        "n_features": len(features_used),
        "feature_names": features_used,
        "classes": ["BENIGN", "DDOS"],
        "class_mapping": {
            "0": "BENIGN",
            "1": "DDOS"
        },
        "rf_params": {
            "n_estimators": 150,
            "max_depth": 25,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "class_weight": "balanced"
        },
        "preprocessing": {
            "smote_applied": True,
            "inf_replacement": "nan",
            "nan_replacement": "median"
        },
        "metrics": metrics,
        "onnx_compatible": True,
        "target_platform": "ml-detector-cpp",
        "protobuf_mapping": {
            "message": "NetworkFeatures",
            "fields": "Individual numeric fields (lines 26-130)",
            "notes": "Uses ~70 numeric features from NetworkFeatures message"
        }
    }
    
    metadata_path = OUTPUT_PATH / "metadata" / f"{MODEL_NAME}_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata: {metadata_path}")


def main():
    """Función principal"""
    print("=" * 80)
    print("🎯 LEVEL 2 DDOS DETECTOR - BINARY CLASSIFICATION")
    print("=" * 80)
    print("\nDataset: CIC-DDoS-2019")
    print("Classes: BENIGN (0) vs DDOS (1)")
    print("Model: Random Forest")
    print("Features: ~70 numeric from NetworkFeatures")
    
    try:
        # 1. Cargar dataset
        # Para pruebas, puedes usar sample_size=100000
        df = load_ddos_dataset(sample_size=None)  # None = cargar todo
        
        # 2. Preprocesar
        X, y, features_used = preprocess_data(df)
        
        # 3. Entrenar
        model, X_train, X_test, y_train, y_test = train_model(X, y, use_smote=True)
        
        # 4. Evaluar
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # 5. Guardar
        save_model(model, metrics, features_used)
        
        # Resumen final
        print("\n" + "=" * 80)
        print("✅ LEVEL 2 DDOS BINARY MODEL TRAINING COMPLETADO")
        print("=" * 80)
        print("\n📊 Métricas finales (Test):")
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
