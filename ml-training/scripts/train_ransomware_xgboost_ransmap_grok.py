#!/usr/bin/env python3
"""
Train Proto-Aligned Ransomware Detector - XGBoost con 20 RansomwareFeatures
Alineado con network_security.proto (RansomwareFeatures). Proxies de UNSW/internal.
Target: Recall >90%, <5ms inferencia en Pi5, serializable a ModelPrediction.
"""

import pandas as pd
import numpy as np
import tarfile
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
import warnings
import gc
import sys
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG (Alineado con Proto)
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "datasets"
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_NAME = "proto_ransomware_xgboost"

SAMPLE_SIZE_ATTACK = 20000  # De UNSW exploits (proxy ransomware)
SAMPLE_SIZE_BENIGN = 50000  # De internal-normal
USE_SMOTE = True
SMOTE_RATIO = 1.0

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'scale_pos_weight': 1,  # Dinámico para recall
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# 20 EXACTAS de RansomwareFeatures en proto
FEATURES_TO_USE = [
    'dns_query_entropy', 'new_external_ips_30s', 'dns_query_rate_per_min',
    'failed_dns_queries_ratio', 'tls_self_signed_cert_count', 'non_standard_port_http_count',
    'smb_connection_diversity', 'rdp_failed_auth_count', 'new_internal_connections_30s',
    'port_scan_pattern_score', 'upload_download_ratio_30s', 'burst_connections_count',
    'unique_destinations_30s', 'large_upload_sessions_count', 'nocturnal_activity_flag',
    'connection_rate_stddev', 'protocol_diversity_score', 'avg_flow_duration_seconds',
    'tcp_rst_ratio', 'syn_without_ack_ratio'
]

# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================
def print_section(title):
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def print_memory_usage():
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"  💾 Memory: {mem_mb:.1f} MB")

def decompress_if_needed(dataset_name):
    """Descomprime .tar.gz si no existe dir"""
    tar_path = DATASET_PATH / f"{dataset_name}.tar.gz"
    dir_path = DATASET_PATH / dataset_name
    if tar_path.exists() and not dir_path.exists():
        print(f"  📦 Descomprimiendo {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(DATASET_PATH)
        print("  ✅ Descomprimido")

def compute_ransomware_proxies(df):
    """Computa las 20 features de proto desde flows base. Robust to missing cols."""
    # Strip columns for consistency
    df.columns = df.columns.str.strip().str.lower()

    # Helper to safe get col
    def safe_col(col, default=0):
        return df.get(col, pd.Series([default] * len(df)))

    # 1-6: C&C proxies
    pkt_var = safe_col('pkt_len_var', (safe_col('pkt_len_std', 0)**2).fillna(0))
    df['dns_query_entropy'] = np.log(pkt_var + 1) / np.log(256)  # Shannon proxy
    df['new_external_ips_30s'] = df.groupby('srcip')['dstip'].transform('nunique').fillna(0)
    df['dns_query_rate_per_min'] = safe_col('flow_pkts_s', 0) * 60
    df['failed_dns_queries_ratio'] = safe_col('rst_cnt', 0) / (safe_col('tot_fwd_pkts', 1) + 1)
    df['tls_self_signed_cert_count'] = ((safe_col('dst_port', 0) == 443) & (safe_col('init_win_bytes_fwd', 65535) < 65535)).astype(int).sum()
    df['non_standard_port_http_count'] = ((safe_col('dst_port', 0) != 80) & (safe_col('dst_port', 0) != 443)).sum()

    # 7-10: Lateral
    df['smb_connection_diversity'] = ((safe_col('dst_port', 0) == 445) * safe_col('tot_fwd_pkts', 0)).fillna(0)
    df['rdp_failed_auth_count'] = ((safe_col('dst_port', 0) == 3389) * safe_col('rst_cnt', 0)).fillna(0)
    df['new_internal_connections_30s'] = safe_col('tot_fwd_pkts', 0)
    df['port_scan_pattern_score'] = safe_col('dst_port', 0).nunique() / len(df)

    # 11-14: Exfil
    df['upload_download_ratio_30s'] = safe_col('tot_fwd_byt', 0) / (safe_col('tot_bwd_byt', 1) + 1)
    burst_thresh = safe_col('flow_byts_s', 0).quantile(0.9)
    df['burst_connections_count'] = (safe_col('flow_byts_s', 0) > burst_thresh).sum()
    df['unique_destinations_30s'] = safe_col('dstip', '').nunique()
    df['large_upload_sessions_count'] = (safe_col('tot_fwd_byt', 0) > 1e6).sum()

    # 15-20: Behavioral
    if 'stime' in df.columns:
        df['flow_start_time'] = pd.to_datetime(df['stime'], errors='coerce', format='%d/%m/%Y %H:%M:%S')  # UNSW format
    else:
        df['flow_start_time'] = pd.Timestamp.now()  # Fallback
    df['nocturnal_activity_flag'] = df['flow_start_time'].dt.hour.isin([22,23,0,1,2,3,4,5,6]).astype(int)
    df['connection_rate_stddev'] = safe_col('fwd_iat_std', 0).fillna(0)
    df['protocol_diversity_score'] = safe_col('proto', 0).nunique() / len(safe_col('proto', 0).unique())
    df['avg_flow_duration_seconds'] = safe_col('dur', 0).mean()
    df['tcp_rst_ratio'] = safe_col('rst_cnt', 0) / (safe_col('tot_fwd_pkts', 0) + safe_col('tot_bwd_pkts', 0) + 1)
    df['syn_without_ack_ratio'] = safe_col('syn_cnt', 0) / (safe_col('ack_cnt', 1) + 1)

    # Fill NaN and select
    for feat in FEATURES_TO_USE:
        if feat not in df.columns:
            df[feat] = 0
        df[feat] = df[feat].fillna(0)

    return df[FEATURES_TO_USE].copy()

def load_unsw_nb15():
    """Carga UNSW-NB15: Attacks (Exploits/Backdoors como ransomware proxy)"""
    print_section("LOADING UNSW-NB15 (Attack Proxy)")
    decompress_if_needed("UNSW-NB15")
    unsw_path = DATASET_PATH / "UNSW-NB15"
    if not unsw_path.exists(): raise FileNotFoundError(f"No UNSW-NB15: {unsw_path}")

    csv_files = list(unsw_path.glob("*.csv"))
    attack_dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()  # Fix: Strip y lowercase
        # Filtrar por attack_cat para ransomware-like (Exploits, Backdoors, Fuzzers, Recon)
        if 'attack_cat' in df.columns:
            attack = df[df['attack_cat'].isin(['Exploits', 'Backdoors', 'Fuzzers', 'Reconnaissance'])]
        elif 'label' in df.columns:
            attack = df[df['label'] == 1]
        else:
            print(f"  ⚠️  No 'Label' or 'attack_cat' in {f.name}; skipping")
            continue
        if len(attack) > 0:
            attack_dfs.append(attack.sample(n=min(2000, len(attack)), random_state=42))

    if not attack_dfs:
        raise ValueError("No attack samples in UNSW-NB15! Check columns.")

    attack = pd.concat(attack_dfs, ignore_index=True).sample(n=min(SAMPLE_SIZE_ATTACK, len(pd.concat(attack_dfs))))
    print(f"  ✅ UNSW Attacks: {len(attack):,}")
    print_memory_usage()
    return attack, pd.DataFrame()  # Solo attacks

def load_internal_normal():
    """Benign de internal-normal"""
    print_section("LOADING Internal-Normal (Benign)")
    internal_path = DATASET_PATH / "internal-normal"
    if not internal_path.exists(): raise FileNotFoundError(f"No internal-normal: {internal_path}")

    csv_files = list(internal_path.rglob("*.csv"))
    benign_dfs = []
    for f in csv_files[:100]:
        df = pd.read_csv(f, nrows=1000, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()  # Fix columns
        df['label'] = 0  # Benign
        df['attack_cat'] = 'Normal'  # Para consistencia
        if 'stime' not in df.columns:
            df['stime'] = '2025-11-05 12:00:00'  # Fallback
        benign_dfs.append(df.sample(n=min(500, len(df)), random_state=42))

    if not benign_dfs:
        raise ValueError("No benign samples in internal-normal!")

    benign = pd.concat(benign_dfs, ignore_index=True).sample(n=min(SAMPLE_SIZE_BENIGN, len(pd.concat(benign_dfs))))
    print(f"  ✅ Internal Benign: {len(benign):,}")
    print_memory_usage()
    return pd.DataFrame(), benign

def prepare_dataset(attack_unsw, benign_int):
    print_section("PREPARING PROTO-ALIGNED DATASET")
    df_full = pd.concat([attack_unsw, benign_int], ignore_index=True)
    df_full['Label'] = df_full['label']  # Asegura binary label

    # Computa proxies
    X = compute_ransomware_proxies(df_full)
    y = df_full['Label'].copy()

    print(f"  ✅ {X.shape[0]:,} samples, {X.shape[1]} proto features")
    print_memory_usage()
    return X, y, FEATURES_TO_USE

# [Resto de funciones iguales a la versión anterior: apply_smote, train_xgboost, evaluate_model, plot_results, save_model]

def apply_smote(X_train, y_train):
    print_section("APPLYING SMOTE")
    print(f"Before: Attack {(y_train == 1).sum():,}, Benign {(y_train == 0).sum():,}")
    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"After: Attack {(y_bal == 1).sum():,}, Benign {(y_bal == 0).sum():,}")
    print_memory_usage()
    return X_bal, y_bal

def train_xgboost(X_train, y_train, X_val, y_val):
    print_section("TRAINING XGBOOST")
    print(f"Samples: Train {X_train.shape[0]:,}, Val {X_val.shape[0]:,}, Features {X_train.shape[1]}")
    print("\nHyperparams:")
    for k, v in XGBOOST_PARAMS.items(): print(f"  {k}: {v}")
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=10)
    print(f"✅ Complete! {model.n_estimators} estimators")
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    print_section("EVALUATION")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)
    print(f"Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}%, ROC: {roc*100:.2f}%, PR: {pr*100:.2f}%")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"CM: TN {tn:,} FP {fp:,} | FN {fn:,} TP {tp:,}")
    print(f"FPR: {fpr*100:.2f}%, FNR: {fnr*100:.2f}%")
    imp_df = pd.DataFrame({'feature': feature_names, 'imp': model.feature_importances_}).sort_values('imp', ascending=False)
    print("\nTop Features (Proto Order):")
    for _, row in imp_df.head().iterrows(): print(f"  {row['feature']:<30} {row['imp']:.4f}")
    metrics = {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1_score': float(f1), 'roc_auc': float(roc), 'pr_auc': float(pr), 'fpr': float(fpr), 'fnr': float(fnr), 'cm': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}}
    return metrics, imp_df

def plot_results(metrics, feature_importance, y_test, y_pred_proba):
    print_section("PLOTS")
    out_dir = OUTPUT_PATH / "plots" / MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    # Feature Imp
    plt.figure(figsize=(12, 8))
    top = feature_importance.head(20)
    plt.barh(range(len(top)), top['imp'])
    plt.yticks(range(len(top)), top['feature'])
    plt.title('RansomwareFeatures Importance (Proto-Aligned)')
    plt.tight_layout()
    plt.savefig(out_dir / 'features_proto.png', dpi=150)
    plt.close()
    # ROC & PR
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC {metrics["roc_auc"]:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(out_dir / 'roc.png', dpi=150)
    plt.close()
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, label=f'PR {metrics["pr_auc"]:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(out_dir / 'pr.png', dpi=150)
    plt.close()
    print(f"📁 Plots: {out_dir}")

def save_model(model, scaler, metrics, feature_names):
    print_section("SAVING (Proto-Compatible)")
    model_dir = OUTPUT_PATH / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{MODEL_NAME}.pkl")
    joblib.dump(scaler, model_dir / f"{MODEL_NAME}_scaler.pkl")
    metadata = {
        'model_name': MODEL_NAME,
        'proto_aligned': True,
        'ransomware_features_count': len(feature_names),
        'feature_names': feature_names,  # Para validar en C++ protobuf loader
        'model_type': 'RANDOM_FOREST_RANSOMWARE',  # Enum en proto
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'target_hardware': 'Raspberry Pi 5',
        'hyperparams': XGBOOST_PARAMS,
        'metrics': metrics,
        'dataset_info': {'unsw_exploits': SAMPLE_SIZE_ATTACK, 'internal_benign': SAMPLE_SIZE_BENIGN}
    }
    with open(model_dir / f"{MODEL_NAME}_proto_meta.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved {model_dir} (Carga en C++ via ModelPrediction)")
    # ONNX (fix strings: ya coerce en prepare)
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        from skl2onnx.helper import save_data
        save_data(onnx_model, model_dir / f"{MODEL_NAME}.onnx")
        print("✅ ONNX for Pi5")
    except Exception as e:
        print(f"⚠️ ONNX fail: {e} (pip install scikit-learn-onnx)")
    return model_dir

def main():
    print("=" * 80)
    print("🚀 PROTO-ALIGNED RANSOMWARE DETECTOR (20 Features)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_memory_usage()

    attack_unsw, _ = load_unsw_nb15()
    _, benign_int = load_internal_normal()

    X, y, features = prepare_dataset(attack_unsw, benign_int)
    del attack_unsw, benign_int
    gc.collect()

    print_section("SPLITTING")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    print_section("SCALING")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    if USE_SMOTE:
        X_train_s, y_train = apply_smote(X_train_s, y_train)

    pos_w = len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
    XGBOOST_PARAMS['scale_pos_weight'] = pos_w
    print(f"Pos weight: {pos_w:.2f}")

    model = train_xgboost(X_train_s, y_train, X_val_s, y_val)

    y_proba = model.predict_proba(X_test_s)[:, 1]
    metrics, imp = evaluate_model(model, X_test_s, y_test, features)

    plot_results(metrics, imp, y_test, y_proba)
    model_dir = save_model(model, scaler, metrics, features)

    print_section("COMPLETE")
    print(f"✅ Recall: {metrics['recall']*100:.2f}% | PR-AUC: {metrics['pr_auc']*100:.2f}%")
    print(f"✅ Proto-Aligned Model: {model_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)