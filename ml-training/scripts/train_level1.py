#!/usr/bin/env python3
"""
Train Level 1 Model - Binary Attack Detector
Fast training with 23 selected features
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_selected_features():
    """Cargar las 23 features seleccionadas"""
    with open("outputs/metadata/level1_features_selected.json") as f:
        metadata = json.load(f)
    return metadata['level1_features']

def load_and_prepare_data(selected_features):
    """Cargar y preparar datos"""
    print("=" * 80)
    print("📊 CARGANDO DATOS")
    print("=" * 80)
    
    ids_path = Path("datasets/CIC-IDS-2017/MachineLearningCVE")
    
    # Cargar todos los archivos para máxima diversidad
    files = list(ids_path.glob("*.csv"))
    print(f"\nCargando {len(files)} archivos...")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding='latin-1')
        dfs.append(df)
        print(f"  {f.name}: {df.shape[0]:,} flows")
    
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Total: {df_full.shape[0]:,} flows")
    
    # Separar features y labels
    label_col = ' Label'
    y = (df_full[label_col] != 'BENIGN').astype(int)
    
    # Seleccionar solo las 23 features
    X = df_full[selected_features]
    
    # Limpiar
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.fillna(0)
    
    print(f"\nClases:")
    print(f"  BENIGN: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.2f}%)")
    print(f"  ATTACK: {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.2f}%)")
    
    return X, y

def train_model(X, y):
    """Entrenar Random Forest con GridSearch"""
    print("\n" + "=" * 80)
    print("🌲 ENTRENANDO RANDOM FOREST")
    print("=" * 80)
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {X_train.shape[0]:,} samples")
    print(f"Test:  {X_test.shape[0]:,} samples")
    
    # Random Forest con buenos hiperparámetros
    print("\nEntrenando modelo (100 estimators)...")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',  # Importante para datasets desbalanceados
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    print("\n✅ Modelo entrenado")
    
    return rf, X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluar modelo"""
    print("\n" + "=" * 80)
    print("📊 EVALUACIÓN DEL MODELO")
    print("=" * 80)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilidades para ROC AUC
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
    print(classification_report(y_test, y_test_pred, target_names=['BENIGN', 'ATTACK']))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['BENIGN', 'ATTACK'],
                yticklabels=['BENIGN', 'ATTACK'])
    plt.title('Confusion Matrix - Level 1 Attack Detector')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = Path("outputs/plots/level1_confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Confusion matrix guardada: {output_path}")
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist()
    }

def save_model(model, metrics, selected_features):
    """Guardar modelo y metadata"""
    print("\n" + "=" * 80)
    print("💾 GUARDANDO MODELO")
    print("=" * 80)
    
    # Guardar modelo sklearn
    model_path = Path("outputs/models/level1_attack_detector.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✅ Modelo guardado: {model_path}")
    
    # Metadata completo
    metadata = {
        "model_name": "level1_attack_detector",
        "model_type": "RandomForest",
        "version": "1.0.0",
        "training_date": datetime.now().isoformat(),
        "dataset": "CIC-IDS-2017",
        "n_features": len(selected_features),
        "feature_names": selected_features,
        "classes": ["BENIGN", "ATTACK"],
        "rf_params": {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "class_weight": "balanced"
        },
        "metrics": metrics,
        "onnx_compatible": True,
        "target_platform": "ml-detector-cpp"
    }
    
    metadata_path = Path("outputs/metadata/level1_model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata guardada: {metadata_path}")

def main():
    print("=" * 80)
    print("🎯 TRAINING LEVEL 1 MODEL - ATTACK DETECTOR")
    print("=" * 80)
    
    # 1. Cargar features seleccionadas
    selected_features = load_selected_features()
    print(f"\n✅ Features seleccionadas: {len(selected_features)}")
    
    # 2. Cargar y preparar datos
    X, y = load_and_prepare_data(selected_features)
    
    # 3. Entrenar modelo
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # 4. Evaluar modelo
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # 5. Guardar modelo
    save_model(model, metrics, selected_features)
    
    print("\n" + "=" * 80)
    print("✅ LEVEL 1 MODEL TRAINING COMPLETADO")
    print("=" * 80)
    print("\n📊 Métricas finales:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1_score']:.2%}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print("\n🎯 Siguiente paso: Conversión a ONNX")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
