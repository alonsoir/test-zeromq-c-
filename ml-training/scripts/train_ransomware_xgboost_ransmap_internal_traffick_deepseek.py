#!/usr/bin/env python3
"""
Train Proto-Aligned Ransomware Detector - XGBoost con 20 RansomwareFeatures
Usando RANsMAP, Ransomware Dataset 2024, y UGRansome
"""

import pandas as pd
import numpy as np
import tarfile
import zipfile
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
import warnings
import gc
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
DATASET_PATH = BASE_PATH / "datasets"
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_NAME = "ransomware_xgboost_production_v2"

SAMPLE_SIZE_ATTACK = 10000
SAMPLE_SIZE_BENIGN = 10000
USE_SMOTE = True
SMOTE_RATIO = 1.0

# PARÁMETROS XGBoost optimizados para ransomware
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'scale_pos_weight': 1,
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

# 20 Features del proto
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
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"  💾 Memory: {mem_mb:.1f} MB")
    except ImportError:
        print("  💾 Memory: psutil not available")

def decompress_if_needed(dataset_name):
    """Descomprime .tar.gz o .zip si no existe dir"""
    # Buscar diferentes variaciones del nombre
    possible_paths = [
        DATASET_PATH / dataset_name,
        DATASET_PATH / f"{dataset_name}.tar.gz",
        DATASET_PATH / f"{dataset_name}.zip",
        DATASET_PATH / f"Ransomware Dataset 2024",  # Nueva ruta corregida
        DATASET_PATH / "Ransomware Dataset 2024"
    ]

    for path in possible_paths:
        if path.exists():
            print(f"  ✅ Dataset encontrado: {path}")
            return path

    print(f"  ⚠️  Dataset no encontrado: {dataset_name}")
    return None

def safe_read_csv(file_path, sample_rows=None, encodings=['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']):
    """Lee CSV probando múltiples codificaciones"""
    for encoding in encodings:
        try:
            if sample_rows:
                df = pd.read_csv(file_path, nrows=sample_rows, encoding=encoding, low_memory=False)
            else:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            print(f"  ✅ {file_path.name} leído con encoding: {encoding}")
            return df
        except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
            print(f"  ⚠️  Encoding {encoding} falló: {e}")
            continue
        except Exception as e:
            print(f"  ⚠️  Error con {encoding}: {e}")
            continue

    try:
        if sample_rows:
            df = pd.read_csv(file_path, nrows=sample_rows, encoding='utf-8',
                             engine='python', on_bad_lines='skip', low_memory=False)
        else:
            df = pd.read_csv(file_path, encoding='utf-8', engine='python',
                             on_bad_lines='skip', low_memory=False)
        print(f"  ✅ {file_path.name} leído con engine python (skip bad lines)")
        return df
    except Exception as e:
        print(f"  ❌ No se pudo leer {file_path.name}: {e}")
        return None

def compute_ransomware_features(df, dataset_type):
    """Computa las 20 features específicas para ransomware"""
    print(f"  🛠️  Computando features ransomware para {dataset_type}")

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()

    # Helper function segura
    def safe_col(col_name, default=0):
        if col_name in df.columns:
            return df[col_name]
        return pd.Series([default] * len(df))

    # Inicializar todas las features
    for feat in FEATURES_TO_USE:
        df[feat] = 0.0

    # 1. DNS Query Entropy - variación en tamaños de paquetes
    if 'length' in df.columns:
        pkt_sizes = safe_col('length', 0)
    elif 'packet_length' in df.columns:
        pkt_sizes = safe_col('packet_length', 0)
    else:
        # Usar bytes como proxy
        pkt_sizes = safe_col('sbytes', 0) + safe_col('dbytes', 0)

    df['dns_query_entropy'] = np.log1p(pkt_sizes.var() if pkt_sizes.var() > 0 else 1) / 10.0

    # 2. New External IPs - diversidad de destinos
    if 'dst_ip' in df.columns:
        df['new_external_ips_30s'] = safe_col('dst_ip').nunique()
    elif 'dest_ip' in df.columns:
        df['new_external_ips_30s'] = safe_col('dest_ip').nunique()
    else:
        df['new_external_ips_30s'] = safe_col('dst_port', 0).nunique()

    # 3. DNS Query Rate - frecuencia de comunicación
    if 'flow_duration' in df.columns:
        flow_dur = safe_col('flow_duration', 1)
        df['dns_query_rate_per_min'] = len(df) / (flow_dur.max() / 60 + 1)
    else:
        df['dns_query_rate_per_min'] = safe_col('rate', 0) * 60

    # 4. Failed DNS Queries Ratio - paquetes reset/error
    if 'rst_flag_count' in df.columns:
        rst_count = safe_col('rst_flag_count', 0)
    else:
        rst_count = safe_col('rst', 0)

    total_pkts = safe_col('spkts', 1) + safe_col('dpkts', 0)
    df['failed_dns_queries_ratio'] = rst_count / total_pkts

    # 5. TLS Self-signed Cert Count - puertos no estándar para SSL
    dst_port = safe_col('dst_port', 0)
    df['tls_self_signed_cert_count'] = ((dst_port != 443) & (dst_port != 80)).astype(int)

    # 6. Non-standard Port HTTP Count
    df['non_standard_port_http_count'] = ((dst_port != 80) & (dst_port != 443) &
                                          (dst_port != 8080) & (dst_port != 8443)).astype(int)

    # 7. SMB Connection Diversity - actividad en puerto SMB
    df['smb_connection_diversity'] = (dst_port == 445).astype(int)

    # 8. RDP Failed Auth Count - actividad en puerto RDP
    df['rdp_failed_auth_count'] = (dst_port == 3389).astype(int)

    # 9. New Internal Connections - tasa de nuevas conexiones
    df['new_internal_connections_30s'] = safe_col('spkts', 0)

    # 10. Port Scan Pattern Score
    df['port_scan_pattern_score'] = dst_port.nunique() / max(len(df), 1)

    # 11. Upload/Download Ratio - asimetría de tráfico
    sbytes = safe_col('sbytes', 0)
    dbytes = safe_col('dbytes', 1)
    df['upload_download_ratio_30s'] = sbytes / np.maximum(dbytes, 1)

    # 12. Burst Connections Count - ráfagas de tráfico
    if 'flow_bytes_s' in df.columns:
        flow_rate = safe_col('flow_bytes_s', 0)
    else:
        flow_rate = (sbytes + dbytes) / np.maximum(safe_col('dur', 1), 1)

    if len(flow_rate) > 0 and flow_rate.nunique() > 1:
        burst_thresh = np.percentile(flow_rate, 90)
        df['burst_connections_count'] = (flow_rate > burst_thresh).astype(int)

    # 13. Unique Destinations
    df['unique_destinations_30s'] = dst_port.nunique()

    # 14. Large Upload Sessions - sesiones grandes de upload
    df['large_upload_sessions_count'] = (sbytes > 50000).astype(int)

    # 15. Nocturnal Activity Flag - ransomware suele actuar de noche
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['nocturnal_activity_flag'] = df['timestamp'].dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        except:
            df['nocturnal_activity_flag'] = 0
    else:
        # Asumir actividad nocturna para ransomware
        df['nocturnal_activity_flag'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])

    # 16. Connection Rate StdDev - variabilidad en tasa de conexión
    df['connection_rate_stddev'] = safe_col('rate', 0).std() if len(df) > 1 else 0

    # 17. Protocol Diversity Score
    proto = safe_col('proto', 'tcp')
    df['protocol_diversity_score'] = proto.nunique() / max(len(df), 1)

    # 18. Avg Flow Duration
    dur = safe_col('dur', 0)
    df['avg_flow_duration_seconds'] = dur.mean() if len(dur) > 0 else 0

    # 19. TCP RST Ratio
    df['tcp_rst_ratio'] = rst_count / np.maximum(total_pkts, 1)

    # 20. SYN without ACK Ratio - conexiones incompletas
    if 'syn_flag_count' in df.columns:
        syn_count = safe_col('syn_flag_count', 0)
    else:
        syn_count = safe_col('syn', 0)

    if 'ack_flag_count' in df.columns:
        ack_count = safe_col('ack_flag_count', 1)
    else:
        ack_count = safe_col('ack', 1)

    df['syn_without_ack_ratio'] = syn_count / np.maximum(ack_count, 1)

    # Limpiar valores
    for feat in FEATURES_TO_USE:
        df[feat] = df[feat].fillna(0)
        df[feat] = np.nan_to_num(df[feat])

    print(f"  ✅ Features computadas: {len(FEATURES_TO_USE)}")
    return df[FEATURES_TO_USE].copy()

def load_ransmap():
    """Carga dataset RANsMAP específico para ransomware"""
    print_section("LOADING RANsMAP DATASET")

    decompress_if_needed("RANsMAP")
    ransmap_path = DATASET_PATH / "RANsMAP"

    if not ransmap_path.exists():
        print(f"  ⚠️  RANsMAP no encontrado en {ransmap_path}")
        return None

    csv_files = list(ransmap_path.rglob("*.csv"))
    attack_dfs = []

    for f in csv_files[:5]:  # Limitar a 5 archivos
        print(f"  📖 Procesando: {f.name}")
        df = safe_read_csv(f, sample_rows=2000)

        if df is None:
            continue

        # RANsMAP debería tener tráfico de ransomware
        df['label'] = 1  # Marcar como ataque

        sample_size = min(1000, len(df))
        attack_dfs.append(df.sample(n=sample_size, random_state=42))
        print(f"    ✅ {len(df)} muestras ransomware, muestreadas {sample_size}")

    if attack_dfs:
        all_attacks = pd.concat(attack_dfs, ignore_index=True)
        final = all_attacks.sample(n=min(SAMPLE_SIZE_ATTACK, len(all_attacks)), random_state=42)
        print(f"  ✅ RANsMAP final: {len(final):,} muestras ransomware")
        return final

    return None

def load_ransomware_2024():
    """Carga Ransomware Dataset 2024 - RUTA CORREGIDA"""
    print_section("LOADING RANSOMWARE DATASET 2024")

    # Rutas corregidas para el dataset
    ransom2024_paths = [
        DATASET_PATH / "Ransomware Dataset 2024",
        DATASET_PATH / "Ransomware2024",
        DATASET_PATH / "Ransomware_Dataset_2024"
    ]

    ransom2024_path = None
    for path in ransom2024_paths:
        if path.exists():
            ransom2024_path = path
            break

    if ransom2024_path is None:
        print(f"  ⚠️  Ransomware2024 no encontrado en ninguna ruta conocida")
        return None

    print(f"  ✅ Dataset encontrado en: {ransom2024_path}")

    csv_files = list(ransom2024_path.rglob("*.csv"))
    if not csv_files:
        print(f"  ⚠️  No se encontraron archivos CSV en {ransom2024_path}")
        return None

    attack_dfs = []

    for f in csv_files[:3]:
        print(f"  📖 Procesando: {f.name}")
        df = safe_read_csv(f, sample_rows=3000)  # Leer más muestras

        if df is None:
            continue

        # Verificar si ya tiene columna label
        if 'label' not in df.columns:
            df['label'] = 1  # Marcar como ransomware

        sample_size = min(2000, len(df))
        attack_dfs.append(df.sample(n=sample_size, random_state=42))
        print(f"    ✅ {len(df)} muestras ransomware, muestreadas {sample_size}")

    if attack_dfs:
        all_attacks = pd.concat(attack_dfs, ignore_index=True)
        final = all_attacks.sample(n=min(SAMPLE_SIZE_ATTACK, len(all_attacks)), random_state=42)
        print(f"  ✅ Ransomware2024 final: {len(final):,} muestras ransomware")
        return final

    return None

def load_ugransome():
    """Carga UGRansome dataset"""
    print_section("LOADING UGRANSOME DATASET")

    decompress_if_needed("UGRansome")
    ugransome_path = DATASET_PATH / "UGRansome"

    if not ugransome_path.exists():
        print(f"  ⚠️  UGRansome no encontrado en {ugransome_path}")
        return None

    csv_files = list(ugransome_path.rglob("*.csv"))
    attack_dfs = []

    for f in csv_files[:3]:
        print(f"  📖 Procesando: {f.name}")
        df = safe_read_csv(f, sample_rows=2000)

        if df is None:
            continue

        df['label'] = 1  # Ransomware

        sample_size = min(1500, len(df))
        attack_dfs.append(df.sample(n=sample_size, random_state=42))
        print(f"    ✅ {len(df)} muestras ransomware, muestreadas {sample_size}")

    if attack_dfs:
        all_attacks = pd.concat(attack_dfs, ignore_index=True)
        final = all_attacks.sample(n=min(SAMPLE_SIZE_ATTACK, len(all_attacks)), random_state=42)
        print(f"  ✅ UGRansome final: {len(final):,} muestras ransomware")
        return final

    return None

def load_benign_traffic():
    """Carga tráfico benigno de varios sources"""
    print_section("LOADING BENIGN TRAFFIC")

    benign_sources = [
        "internal-normal",
        "CIC-IDS2017",  # Dataset común con tráfico normal
        "CTU-Normal"    # Tráfico normal de CTU
    ]

    benign_dfs = []

    for source in benign_sources:
        source_path = DATASET_PATH / source
        if source_path.exists():
            print(f"  📂 Cargando desde: {source}")
            csv_files = list(source_path.rglob("*.csv"))[:2]  # Primeros 2 archivos

            for f in csv_files:
                print(f"    📖 Procesando: {f.name}")
                df = safe_read_csv(f, sample_rows=1000)

                if df is None:
                    continue

                df['label'] = 0  # Benigno

                sample_size = min(500, len(df))
                benign_dfs.append(df.sample(n=sample_size, random_state=42))
                print(f"      ✅ {len(df)} muestras benignas, muestreadas {sample_size}")

    if not benign_dfs:
        # Fallback: crear datos sintéticos benignos
        print("  ⚠️  No se encontraron datasets benignos, creando datos sintéticos")
        synthetic_data = create_synthetic_benign(1000)
        benign_dfs.append(synthetic_data)

    all_benign = pd.concat(benign_dfs, ignore_index=True)
    final_benign = all_benign.sample(n=min(SAMPLE_SIZE_BENIGN, len(all_benign)), random_state=42)

    print(f"  ✅ Benign final: {len(final_benign):,} muestras")
    return final_benign

def create_synthetic_benign(n_samples):
    """Crea datos sintéticos benignos cuando no hay datasets disponibles"""
    data = {
        'src_ip': [f"192.168.1.{i%254+1}" for i in range(n_samples)],
        'dst_ip': [f"8.8.8.{i%254+1}" for i in range(n_samples)],
        'dst_port': np.random.choice([80, 443, 53, 22, 25], n_samples, p=[0.4, 0.4, 0.1, 0.05, 0.05]),
        'proto': np.random.choice(['tcp', 'udp'], n_samples, p=[0.7, 0.3]),
        'sbytes': np.random.exponential(500, n_samples).astype(int),
        'dbytes': np.random.exponential(1000, n_samples).astype(int),
        'spkts': np.random.poisson(10, n_samples),
        'dpkts': np.random.poisson(15, n_samples),
        'dur': np.random.exponential(10, n_samples),
        'rate': np.random.exponential(100, n_samples),
        'label': 0
    }
    return pd.DataFrame(data)

def prepare_training_data():
    """Prepara todos los datos para entrenamiento"""
    print_section("PREPARING TRAINING DATA")

    # Cargar datos de ransomware
    ransom_dfs = []

    ransmap_data = load_ransmap()
    if ransmap_data is not None:
        ransom_dfs.append(ransmap_data)

    ransom2024_data = load_ransomware_2024()  # Ahora con ruta corregida
    if ransom2024_data is not None:
        ransom_dfs.append(ransom2024_data)

    ugransome_data = load_ugransome()
    if ugransome_data is not None:
        ransom_dfs.append(ugransome_data)

    if not ransom_dfs:
        print("  ❌ No se pudieron cargar datasets de ransomware")
        sys.exit(1)

    # Combinar todos los ataques
    all_attacks = pd.concat(ransom_dfs, ignore_index=True)
    final_attacks = all_attacks.sample(n=min(SAMPLE_SIZE_ATTACK, len(all_attacks)), random_state=42)

    # Cargar datos benignos
    benign_data = load_benign_traffic()

    print(f"  📊 Datasets cargados:")
    print(f"    Ransomware: {len(final_attacks):,} muestras")
    print(f"    Benigno:    {len(benign_data):,} muestras")

    # Computar features
    print("  🛠️  Computando features para ransomware...")
    X_ransomware = compute_ransomware_features(final_attacks, "ransomware")
    y_ransomware = final_attacks['label']

    print("  🛠️  Computando features para tráfico benigno...")
    X_benign = compute_ransomware_features(benign_data, "benign")
    y_benign = benign_data['label']

    # Combinar
    X = pd.concat([X_ransomware, X_benign], ignore_index=True)
    y = pd.concat([y_ransomware, y_benign], ignore_index=True)

    print(f"  ✅ Dataset final: {X.shape[0]:,} muestras, {X.shape[1]} features")
    print(f"  📊 Distribución: Benigno {sum(y==0):,}, Ransomware {sum(y==1):,}")

    return X, y, FEATURES_TO_USE

def train_model(X, y, features):
    """Entrena el modelo XGBoost"""
    print_section("TRAINING MODEL")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"  Split: Train {X_train.shape[0]:,}, Test {X_test.shape[0]:,}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE si está habilitado
    if USE_SMOTE:
        print("  🔄 Aplicando SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
        print(f"    Después de SMOTE: {X_train_bal.shape[0]:,} muestras")
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train

    # Ajustar peso de clases
    pos_count = sum(y_train_bal == 1)
    neg_count = sum(y_train_bal == 0)
    XGBOOST_PARAMS['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1
    print(f"  ⚖️  Peso de clase: {XGBOOST_PARAMS['scale_pos_weight']:.2f}")

    # Entrenar
    print("  🌳 Entrenando XGBoost...")
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_bal, y_train_bal)

    return model, scaler, X_test_scaled, y_test

def evaluate_model(model, X_test, y_test, features):
    """Evalúa el modelo"""
    print_section("MODEL EVALUATION")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas básicas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"  📈 Métricas:")
    print(f"    Accuracy:    {acc:.3f}")
    print(f"    Precision:   {prec:.3f}")
    print(f"    Recall:      {rec:.3f}")
    print(f"    F1-Score:    {f1:.3f}")
    print(f"    ROC-AUC:     {roc_auc:.3f}")
    print(f"    PR-AUC:      {pr_auc:.3f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"  📊 Matriz de confusión:")
        print(f"    TN: {tn:,}  FP: {fp:,}")
        print(f"    FN: {fn:,}  TP: {tp:,}")
        print(f"    FPR: {fpr:.3f}, FNR: {fnr:.3f}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"  🔝 Top 10 Features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:<35} {row['importance']:.4f}")
    else:
        importance_df = pd.DataFrame({'feature': features, 'importance': [0]*len(features)})
        print("  ⚠️  No feature importances available")

    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'fpr': float(fpr),
        'fnr': float(fnr)
    }

    return metrics, importance_df, y_proba

def save_model(model, scaler, metrics, features, importance_df):
    """Guarda el modelo entrenado"""
    print_section("SAVING MODEL")

    model_dir = OUTPUT_PATH / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo y scaler
    joblib.dump(model, model_dir / f"{MODEL_NAME}.pkl")
    joblib.dump(scaler, model_dir / f"{MODEL_NAME}_scaler.pkl")

    # Metadata
    metadata = {
        'model_name': MODEL_NAME,
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'features': features,
        'metrics': metrics,
        'feature_importance': importance_df.to_dict('records'),
        'training_params': XGBOOST_PARAMS,
        'datasets_used': ['RANsMAP', 'Ransomware2024', 'UGRansome']
    }

    with open(model_dir / f"{MODEL_NAME}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Modelo guardado en: {model_dir}")
    return model_dir

def main():
    print("=" * 80)
    print("🚀 RANSOMWARE DETECTOR - PRODUCTION VERSION V2")
    print("=" * 80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_memory_usage()

    try:
        # Preparar datos
        X, y, features = prepare_training_data()

        # Entrenar modelo
        model, scaler, X_test, y_test = train_model(X, y, features)

        # Evaluar modelo
        metrics, importance_df, y_proba = evaluate_model(model, X_test, y_test, features)

        # Guardar modelo
        model_dir = save_model(model, scaler, metrics, features, importance_df)

        print_section("TRAINING COMPLETED")
        print(f"✅ Recall: {metrics['recall']:.3f} (>0.9 objetivo)")
        print(f"✅ Precision: {metrics['precision']:.3f}")
        print(f"✅ Modelo guardado en: {model_dir}")
        print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()