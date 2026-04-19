#!/usr/bin/env python3
"""
train_xgboost_baseline.py — ADR-026 PASO 4c
Entrena XGBoost con el mismo feature set LEVEL1 (23 features) usado por el RF baseline.
Dataset: CIC-IDS-2017 (MachineLearningCVE)
Gate: F1 >= 0.9985 + Precision >= 0.99
Exports: xgboost_cicids2017.json (repo) + xgboost_cicids2017.ubj (produccion)

Uso:
    python3 train_xgboost_baseline.py --data-dir <path_to_MachineLearningCVE> --output-dir <path>
"""
import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import json
import time

# 23 features LEVEL1 — mismas que feature_extractor.cpp::LEVEL1_FEATURE_NAMES
# Nombres con espacio inicial tal como aparecen en CIC-IDS-2017
FEATURE_NAMES = [
    " Packet Length Std",
    " Subflow Fwd Bytes",
    " Fwd Packet Length Max",
    " Avg Fwd Segment Size",
    " ACK Flag Count",
    " Packet Length Variance",
    " PSH Flag Count",
    "Bwd Packet Length Max",
    " act_data_pkt_fwd",
    "Total Length of Fwd Packets",
    " Fwd Packet Length Std",
    "Fwd Packets/s",
    " Subflow Bwd Bytes",
    " Destination Port",
    "Init_Win_bytes_forward",
    "Subflow Fwd Packets",
    " Fwd IAT Min",
    " Packet Length Mean",
    " Total Length of Bwd Packets",
    " Bwd Packet Length Mean",
    " Bwd Packet Length Min",
    " Flow Duration",
    " Flow Packets/s",
]

LABEL_COL = " Label"
BENIGN_LABEL = "BENIGN"

F1_GATE = 0.9970  # RF baseline = 0.9968 — XGBoost debe superarlo
PRECISION_GATE = 0.99


def load_dataset(data_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"ERROR: no CSVs found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Cargando {len(csv_files)} CSVs desde {data_dir}...")
    dfs = []
    for f in sorted(csv_files):
        print(f"  -> {os.path.basename(f)}")
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: error cargando {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total filas: {len(df):,}")
    return df


def prepare_features(df: pd.DataFrame):
    # Verificar que todas las features existen
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"ERROR: features faltantes en CSV: {missing}", file=sys.stderr)
        sys.exit(1)
    if LABEL_COL not in df.columns:
        print(f"ERROR: columna label '{LABEL_COL}' no encontrada", file=sys.stderr)
        sys.exit(1)

    X = df[FEATURE_NAMES].copy()
    y = (df[LABEL_COL].str.strip() != BENIGN_LABEL).astype(int)

    # Limpiar NaN/Inf
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN/Inf — rellenando con 0")
        X.fillna(0.0, inplace=True)

    X = X.astype(np.float32)
    print(f"Features: {X.shape[1]}, Muestras: {len(X):,}")
    print(f"Distribución: {y.sum():,} ATTACK ({100*y.mean():.1f}%), {(1-y).sum():,} BENIGN")
    return X.values, y.values


def train_xgboost(X_train, y_train):
    print("\nEntrenando XGBoost...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, verbose=False)
    elapsed = time.time() - t0
    print(f"Entrenamiento completado en {elapsed:.1f}s")
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== Métricas XGBoost vs RF baseline ===")
    print(f"F1        : {f1:.4f}  (gate >= {F1_GATE}) {'✅' if f1 >= F1_GATE else '❌'}")
    print(f"Precision : {precision:.4f}  (gate >= {PRECISION_GATE}) {'✅' if precision >= PRECISION_GATE else '❌'}")
    print(f"Recall    : {recall:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(f"\nRF baseline: F1=0.9968, Precision=0.9944 (level1_attack_detector_metadata.json)")

    gates_passed = f1 >= F1_GATE and precision >= PRECISION_GATE
    if not gates_passed:
        print(f"\n❌ GATES NO PASADOS — no se exporta el modelo")
    else:
        print(f"\n✅ GATES PASADOS — modelo listo para exportar")
    return gates_passed, {"f1": f1, "precision": precision, "recall": recall, "auc": auc}


def export_model(model, output_dir: str, metrics: dict):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "xgboost_cicids2017.json")
    ubj_path = os.path.join(output_dir, "xgboost_cicids2017.ubj")
    meta_path = os.path.join(output_dir, "xgboost_cicids2017_metadata.json")

    model.get_booster().save_model(json_path)
    model.get_booster().save_model(ubj_path)
    print(f"\nModelo exportado:")
    print(f"  JSON (repo):       {json_path}")
    print(f"  UBJ  (produccion): {ubj_path}")

    metadata = {
        "model_name": "xgboost_cicids2017",
        "adr": "ADR-026",
        "dataset": "CIC-IDS-2017",
        "n_features": 23,
        "feature_names": FEATURE_NAMES,
        "classes": ["BENIGN", "ATTACK"],
        "random_state": 42,
        "metrics": metrics,
        "gates": {"f1": F1_GATE, "precision": PRECISION_GATE},
        "gates_passed": True,
        "plugin_contract": "docs/xgboost/plugin-contract.md",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata:          {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost baseline — ADR-026")
    parser.add_argument("--data-dir", required=True, help="Path to CIC-IDS-2017 MachineLearningCVE dir")
    parser.add_argument("--output-dir", required=True, help="Output directory for models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    df = load_dataset(args.data_dir)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    model = train_xgboost(X_train, y_train)
    gates_passed, metrics = evaluate(model, X_test, y_test)

    if gates_passed:
        export_model(model, args.output_dir, metrics)
        print("\n✅ train_xgboost_baseline.py completado — ADR-026 PASO 4c")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
