#!/usr/bin/env python3
"""
Feature Selection - Seleccionar top 23 features para Level 1
Automatiza el proceso de feature engineering
ml-training/scripts/select_features.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Cargar datos con ataques"""
    print("=" * 80)
    print("📊 CARGANDO DATASETS")
    print("=" * 80)
    
    ids_path = Path("datasets/CIC-IDS-2017/MachineLearningCVE")
    
    if not ids_path.exists():
        print(f"❌ Error: {ids_path} no existe")
        return None, None
    
    # Archivos con diversidad de ataques
    files_to_load = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
    ]
    
    dfs = []
    
    for filename in files_to_load:
        filepath = ids_path / filename
        if filepath.exists():
            print(f"  Cargando {filename}...")
            df = pd.read_csv(filepath, encoding='latin-1')
            dfs.append(df)
            print(f"    {df.shape[0]:,} rows")
    
    if not dfs:
        print("❌ No se pudieron cargar archivos")
        return None, None
    
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Dataset completo: {df_full.shape[0]:,} flows")
    
    # Separar features y labels
    label_col = ' Label'
    X = df_full.drop(columns=[label_col])
    y = df_full[label_col]
    
    return X, y


def clean_data(X):
    """Limpiar datos (infinitos y nulos)"""
    print("\n" + "=" * 80)
    print("🧹 LIMPIANDO DATOS")
    print("=" * 80)
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    # Reemplazar infinitos
    inf_count = np.isinf(X[numeric_cols]).sum().sum()
    print(f"\n  Valores infinitos: {inf_count:,}")
    
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Rellenar nulos con mediana
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # Rellenar nulos restantes con 0
    X.fillna(0, inplace=True)
    
    print(f"  Después: 0 infinitos, 0 nulos")
    print("\n✅ Datos limpios")
    
    return X


def calculate_feature_importance(X, y, n_features=23):
    """Calcular feature importance y seleccionar top N"""
    print("\n" + "=" * 80)
    print(f"🌲 FEATURE IMPORTANCE (top {n_features})")
    print("=" * 80)
    
    # Convertir labels a binario
    y_binary = (y != 'BENIGN').astype(int)
    
    print(f"\nClases:")
    print(f"  BENIGN: {(y_binary == 0).sum():,}")
    print(f"  ATTACK: {(y_binary == 1).sum():,}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary,
        test_size=0.2,
        random_state=42,
        stratify=y_binary
    )
    
    print(f"\nTrain: {X_train.shape[0]:,} samples")
    print(f"Test:  {X_test.shape[0]:,} samples")
    
    # Random Forest
    print("\nEntrenando Random Forest (10 estimators)...")
    rf = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(n_features)['feature'].tolist()
    
    print(f"\n📊 Top {n_features} features:")
    for i, (_, row) in enumerate(feature_importance.head(n_features).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:50s} ({row['importance']:.6f})")
    
    return feature_importance, top_features


def plot_feature_importance(feature_importance, n_top=30):
    """Generar gráfico de feature importance"""
    print("\n📊 Generando gráfico...")
    
    plt.figure(figsize=(12, 10))
    top_n = feature_importance.head(n_top)
    
    plt.barh(range(len(top_n)), top_n['importance'], color='steelblue')
    plt.yticks(range(len(top_n)), top_n['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {n_top} Features por Importancia (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_path = Path("outputs/plots/feature_importance_top30.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Guardado: {output_path}")


def save_results(feature_importance, top_features):
    """Guardar resultados en JSON y CSV"""
    print("\n💾 Guardando resultados...")
    
    # Metadata JSON
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "dataset": "CIC-IDS-2017",
        "total_features_available": len(feature_importance),
        "level1_features_selected": len(top_features),
        "selection_method": "RandomForest Feature Importance",
        "rf_params": {
            "n_estimators": 10,
            "max_depth": 10,
            "random_state": 42
        },
        "level1_features": top_features,
        "feature_importance": [
            {
                "feature": row['feature'],
                "importance": float(row['importance'])
            }
            for _, row in feature_importance.head(len(top_features)).iterrows()
        ]
    }
    
    # Guardar JSON
    json_path = Path("outputs/metadata/level1_features_selected.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✅ JSON: {json_path}")
    
    # Guardar CSV
    csv_path = Path("outputs/metadata/all_features_importance.csv")
    feature_importance.to_csv(csv_path, index=False)
    
    print(f"  ✅ CSV:  {csv_path}")


def main():
    """Función principal"""
    print("=" * 80)
    print("🎯 FEATURE SELECTION - ML DETECTOR LEVEL 1")
    print("=" * 80)
    
    # 1. Cargar datos
    X, y = load_data()
    if X is None:
        return 1
    
    # 2. Limpiar datos
    X_clean = clean_data(X)
    
    # 3. Calcular feature importance
    feature_importance, top_23 = calculate_feature_importance(X_clean, y, n_features=23)
    
    # 4. Generar gráfico
    plot_feature_importance(feature_importance)
    
    # 5. Guardar resultados
    save_results(feature_importance, top_23)
    
    print("\n" + "=" * 80)
    print("✅ FEATURE SELECTION COMPLETADA")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  - outputs/metadata/level1_features_selected.json")
    print("  - outputs/metadata/all_features_importance.csv")
    print("  - outputs/plots/feature_importance_top30.png")
    print("\n🎯 Siguiente paso: Entrenamiento de modelo Level 1")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
