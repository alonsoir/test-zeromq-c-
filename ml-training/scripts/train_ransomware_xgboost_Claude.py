#!/usr/bin/env python3
"""
Train Level 2 Ransomware Detector - XGBoost Binary Classifier
Optimized for Raspberry Pi 5 (Edge Computing)

Target: >99% accuracy, <10ms inference, <50MB model size
Features: 28 (20 RansomwareFeatures + 8 NetworkFeatures base)

ml-training/scripts/train_ransomware_xgboost_Claude.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
import warnings
import gc
import sys
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "datasets"
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_NAME = "level2_ransomware_xgboost"

# Memoria optimizada para macOS training
SAMPLE_SIZE = 100000  # 100k samples para training rápido
USE_SMOTE = True
SMOTE_RATIO = 0.5  # Oversample attack class to 50% of benign (mejor balance)

# XGBoost hiperparámetros optimizados para RECALL (detectar ataques)
# Ajustado según recomendaciones para imbalanced datasets
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',           # Precision-Recall curve (mejor que logloss para imbalanced)
    'scale_pos_weight': 3.33,         # Peso para clase minoritaria (70K/21K ratio)
    'max_depth': 6,                   # Profundidad moderada
    'learning_rate': 0.05,            # Más lento pero más preciso
    'n_estimators': 300,              # Más árboles para mejor aprendizaje
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,            # Más flexible (antes 5)
    'gamma': 0.1,
    'reg_alpha': 0.1,                 # L1 regularization
    'reg_lambda': 1.0,                # L2 regularization
    'tree_method': 'hist',            # Rápido y eficiente en memoria
    'random_state': 42,
    'n_jobs': -1
}

# Features a usar (28 total - alineadas con protobuf)
FEATURES_TO_USE = [
    # Network Base Features (8) - directamente del CSV
    'Flow Byts/s',
    'Flow Pkts/s',
    'Tot Fwd Pkts',
    'Tot Bwd Pkts',
    'Pkt Len Mean',
    'Pkt Len Std',
    'SYN Flag Cnt',
    'RST Flag Cnt',
    'PSH Flag Cnt',
    'ACK Flag Cnt',
    'Dst Port',

    # Ransomware Behavior Proxies (17) - aproximaciones desde CSV
    'Flow IAT Std',              # dns_query_entropy proxy
    'Flow Duration',             # connection patterns
    'Down/Up Ratio',             # upload_download_ratio
    'TotLen Fwd Pkts',          # large_upload_sessions
    'TotLen Bwd Pkts',
    'Fwd IAT Mean',             # connection_rate_stddev proxy
    'Bwd IAT Mean',
    'Fwd IAT Std',
    'Bwd IAT Std',
    'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean',
    'Fwd Pkt Len Std',
    'Bwd Pkt Len Std',
    'Protocol',                  # protocol_diversity
    'Init Fwd Win Byts',
    'Init Bwd Win Byts',
    'Subflow Fwd Pkts'
]

# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)


def print_memory_usage():
    """Show current memory usage"""
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"  💾 Memory: {mem_mb:.1f} MB")


def load_cic_ids_2018_infiltration():
    """
    Load CIC-IDS-2018 Infiltration samples
    02-28-2018.csv: 68,871 Infilteration + 544,200 Benign
    """
    print_section("LOADING CIC-IDS-2018 (Infiltration)")

    file_path = DATASET_PATH / "CIC-IDS-2018" / "02-28-2018.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print(f"📂 Loading: {file_path.name}")

    # Load with memory optimization
    df = pd.read_csv(
        file_path,
        encoding='utf-8',
        low_memory=False
    )

    # Clean column names
    df.columns = df.columns.str.strip()

    # Filter Label column
    df = df[df['Label'].notna()]

    # Separate attack and benign
    attack = df[df['Label'] == 'Infilteration'].copy()
    benign = df[df['Label'] == 'Benign'].copy()

    print(f"  ✅ Infilteration: {len(attack):,}")
    print(f"  ✅ Benign: {len(benign):,}")
    print_memory_usage()

    return attack, benign


def load_cic_ids_2017_bot():
    """
    Load CIC-IDS-2017 Bot samples
    1,966 Bot samples across multiple files
    """
    print_section("LOADING CIC-IDS-2017 (Bot)")

    dataset_path = DATASET_PATH / "CIC-IDS-2017" / "MachineLearningCVE"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    csv_files = list(dataset_path.glob("*.csv"))
    print(f"📂 Found {len(csv_files)} files")

    attack_dfs = []
    benign_dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='latin-1', low_memory=False)
        df.columns = df.columns.str.strip()

        # Find label column
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if not label_cols:
            continue

        label_col = label_cols[0]

        # Extract Bot samples
        bot = df[df[label_col] == 'Bot'].copy()
        if len(bot) > 0:
            attack_dfs.append(bot)
            print(f"  ✅ {csv_file.name}: {len(bot)} Bot samples")

        # Sample some benign
        benign = df[df[label_col] == 'BENIGN'].copy()
        if len(benign) > 0:
            benign_sample = benign.sample(n=min(10000, len(benign)), random_state=42)
            benign_dfs.append(benign_sample)

    attack = pd.concat(attack_dfs, ignore_index=True) if attack_dfs else pd.DataFrame()
    benign = pd.concat(benign_dfs, ignore_index=True) if benign_dfs else pd.DataFrame()

    print(f"\n  📊 Total Bot: {len(attack):,}")
    print(f"  📊 Total Benign: {len(benign):,}")
    print_memory_usage()

    return attack, benign


def prepare_dataset(attack_2018, benign_2018, attack_2017, benign_2017):
    """
    Combine datasets and prepare features
    """
    print_section("PREPARING DATASET")

    # Combine attacks
    attack = pd.concat([attack_2018, attack_2017], ignore_index=True)
    benign = pd.concat([benign_2018, benign_2017], ignore_index=True)

    print(f"📊 Combined Attack samples: {len(attack):,}")
    print(f"📊 Combined Benign samples: {len(benign):,}")

    # Sample if too large
    if SAMPLE_SIZE and len(benign) > SAMPLE_SIZE:
        benign = benign.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"  ⚠️  Sampled Benign to {SAMPLE_SIZE:,}")

    if SAMPLE_SIZE and len(attack) > SAMPLE_SIZE // 10:
        attack = attack.sample(n=SAMPLE_SIZE // 10, random_state=42)
        print(f"  ⚠️  Sampled Attack to {SAMPLE_SIZE // 10:,}")

    # Combine and create labels
    attack['Label_Binary'] = 1
    benign['Label_Binary'] = 0

    df_full = pd.concat([attack, benign], ignore_index=True)

    print(f"\n📊 Final dataset: {len(df_full):,} samples")
    print(f"  🔴 Attack: {len(attack):,} ({len(attack)/len(df_full)*100:.2f}%)")
    print(f"  🟢 Benign: {len(benign):,} ({len(benign)/len(df_full)*100:.2f}%)")

    # Extract features
    available_features = [f for f in FEATURES_TO_USE if f in df_full.columns]
    missing_features = [f for f in FEATURES_TO_USE if f not in df_full.columns]

    if missing_features:
        print(f"\n  ⚠️  Missing features: {len(missing_features)}")
        for f in missing_features[:5]:
            print(f"    - {f}")
        if len(missing_features) > 5:
            print(f"    ... and {len(missing_features) - 5} more")

    print(f"\n  ✅ Using {len(available_features)} features")

    X = df_full[available_features].copy()
    y = df_full['Label_Binary'].copy()

    # Clean data
    print("\n🧹 Cleaning data...")

    # Convert all columns to numeric (some CSVs have numeric data as strings)
    print("  🔄 Converting to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median (or 0 if column is all NaN)
    print("  🔄 Filling missing values...")
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            fill_value = median_val if not np.isnan(median_val) else 0
            X[col] = X[col].fillna(fill_value)

    # Final verification - ensure all numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"  ⚠️  Non-numeric columns found: {list(non_numeric)}")
        for col in non_numeric:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    print(f"  ✅ Cleaned: {X.shape[0]:,} samples, {X.shape[1]} features")
    print_memory_usage()

    return X, y, available_features


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance dataset
    """
    print_section("APPLYING SMOTE")

    print(f"Before SMOTE:")
    print(f"  Attack: {(y_train == 1).sum():,}")
    print(f"  Benign: {(y_train == 0).sum():,}")

    smote = SMOTE(
        sampling_strategy=SMOTE_RATIO,
        random_state=42
    )

    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"\nAfter SMOTE:")
    print(f"  Attack: {(y_train_balanced == 1).sum():,}")
    print(f"  Benign: {(y_train_balanced == 0).sum():,}")
    print_memory_usage()

    return X_train_balanced, y_train_balanced


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model with early stopping
    """
    print_section("TRAINING XGBOOST MODEL")

    print(f"📊 Training samples: {X_train.shape[0]:,}")
    print(f"📊 Validation samples: {X_val.shape[0]:,}")
    print(f"📊 Features: {X_train.shape[1]}")

    print("\n🎯 Hyperparameters:")
    for key, value in XGBOOST_PARAMS.items():
        print(f"  {key}: {value}")

    # Create XGBoost model
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)

    # Train model (sin early stopping para compatibilidad XGBoost 3.1.1)
    print("\n🏋️ Training...")

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10
    )

    print(f"\n✅ Training complete!")

    # En XGBoost 3.x, best_iteration y best_score pueden no estar disponibles
    try:
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best score: {model.best_score:.6f}")
    except AttributeError:
        print(f"  Training completed with {model.n_estimators} estimators")

    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Comprehensive model evaluation
    """
    print_section("MODEL EVALUATION")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 Test Set Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n🎯 Confusion Matrix:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n⚠️  Error Rates:")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    print(f"  False Negative Rate: {fnr*100:.2f}%")

    # Feature Importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\n🔥 Top 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }

    return metrics, feature_importance


def plot_results(metrics, feature_importance, y_test, y_pred_proba):
    """
    Generate visualization plots
    """
    print_section("GENERATING PLOTS")

    output_dir = OUTPUT_PATH / "plots" / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150)
    plt.close()
    print(f"  ✅ Saved: feature_importance.png")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {metrics["roc_auc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150)
    plt.close()
    print(f"  ✅ Saved: roc_curve.png")

    print(f"\n📁 Plots saved to: {output_dir}")


def save_model(model, scaler, metrics, feature_names):
    """
    Save model, scaler, and metadata
    """
    print_section("SAVING MODEL")

    model_dir = OUTPUT_PATH / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save XGBoost model
    model_path = model_dir / f"{MODEL_NAME}.pkl"
    joblib.dump(model, model_path)
    print(f"  ✅ Model saved: {model_path}")

    # Save scaler
    scaler_path = model_dir / f"{MODEL_NAME}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ Scaler saved: {scaler_path}")

    # Save metadata
    metadata = {
        'model_name': MODEL_NAME,
        'model_type': 'XGBoost Binary Classifier',
        'version': '2.0.0',
        'created_date': datetime.now().isoformat(),
        'target_hardware': 'Raspberry Pi 5',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'hyperparameters': XGBOOST_PARAMS,
        'metrics': metrics,
        'dataset_info': {
            'cic_ids_2018_infiltration': 68871,
            'cic_ids_2017_bot': 1966,
            'sample_size': SAMPLE_SIZE,
            'smote_ratio': SMOTE_RATIO if USE_SMOTE else None
        }
    }

    metadata_path = model_dir / f"{MODEL_NAME}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Metadata saved: {metadata_path}")

    # Export to ONNX for Pi5
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

        onnx_path = model_dir / f"{MODEL_NAME}.onnx"
        onnxmltools.utils.save_model(onnx_model, onnx_path)
        print(f"  ✅ ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"  ⚠️  ONNX export failed: {e}")
        print(f"     Install with: pip install onnxmltools skl2onnx")

    print(f"\n📁 All files saved to: {model_dir}")

    return model_dir


# =====================================================================
# MAIN TRAINING PIPELINE
# =====================================================================

def main():
    print("=" * 80)
    print("🚀 TRAINING LEVEL 2 RANSOMWARE DETECTOR (XGBoost)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: >99% accuracy, <10ms inference, <50MB model")
    print_memory_usage()

    # 1. Load datasets
    attack_2018, benign_2018 = load_cic_ids_2018_infiltration()
    attack_2017, benign_2017 = load_cic_ids_2017_bot()

    # 2. Prepare dataset
    X, y, feature_names = prepare_dataset(
        attack_2018, benign_2018,
        attack_2017, benign_2017
    )

    # Free memory
    del attack_2018, benign_2018, attack_2017, benign_2017
    gc.collect()

    # 3. Split data
    print_section("SPLITTING DATA")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # 4. Scale features
    print_section("SCALING FEATURES")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("  ✅ Features scaled (StandardScaler)")

    # 5. Apply SMOTE
    if USE_SMOTE:
        X_train_scaled, y_train = apply_smote(X_train_scaled, y_train)

    # 6. Train model
    model = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)

    # 7. Evaluate
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics, feature_importance = evaluate_model(
        model, X_test_scaled, y_test, feature_names
    )

    # 8. Plot results
    plot_results(metrics, feature_importance, y_test, y_pred_proba)

    # 9. Save model
    model_dir = save_model(model, scaler, metrics, feature_names)

    # Final summary
    print_section("TRAINING COMPLETE")
    print(f"✅ Model: {MODEL_NAME}")
    print(f"✅ Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"✅ F1 Score: {metrics['f1_score']*100:.2f}%")
    print(f"✅ Saved to: {model_dir}")
    print(f"\n🎉 Model ready for deployment on Raspberry Pi 5!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)