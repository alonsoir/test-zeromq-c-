#!/usr/bin/env python3
"""
MODELO RANSOMWARE - ALINEADO CON .PROTO
Usa SOLO las 20 features de ransomware del message RansomwareFeatures
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

# =====================================================================
# CONFIG - ALINEADO CON .PROTO
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_NAME = "ransomware_network_detector_proto_aligned"

# PARÁMETROS XGBOOST OPTIMIZADOS
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.15,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# =====================================================================
# FEATURES EXACTAS DEL .PROTO - RansomwareFeatures
# =====================================================================
PROTO_FEATURES = [
    # C&C Communication
    'dns_query_entropy',
    'new_external_ips_30s',
    'dns_query_rate_per_min',
    'failed_dns_queries_ratio',
    'tls_self_signed_cert_count',
    'non_standard_port_http_count',

    # Lateral Movement
    'smb_connection_diversity',
    'rdp_failed_auth_count',
    'new_internal_connections_30s',
    'port_scan_pattern_score',

    # Exfiltration
    'upload_download_ratio_30s',
    'burst_connections_count',
    'unique_destinations_30s',
    'large_upload_sessions_count',

    # Behavioral
    'nocturnal_activity_flag',
    'connection_rate_stddev',
    'protocol_diversity_score',
    'avg_flow_duration_seconds',
    'tcp_rst_ratio',
    'syn_without_ack_ratio'
]

def generate_ransomware_training_data():
    """Genera datos de entrenamiento basados en patrones de ransomware de red"""
    print("🎯 GENERANDO DATOS DE RANSOMWARE (PATRONES DE RED)")

    n_samples = 10000  # Dataset manejable

    data = []

    for i in range(n_samples):
        # 50% ransomware, 50% normal
        is_ransomware = 1 if i < n_samples // 2 else 0

        if is_ransomware:
            # PATRONES DE RANSOMWARE (alto entropía DNS, muchas conexiones, etc.)
            features = {
                'dns_query_entropy': np.random.exponential(3.0),
                'new_external_ips_30s': np.random.poisson(15),
                'dns_query_rate_per_min': np.random.exponential(50),
                'failed_dns_queries_ratio': np.random.beta(2, 8),
                'tls_self_signed_cert_count': np.random.poisson(3),
                'non_standard_port_http_count': np.random.poisson(5),
                'smb_connection_diversity': np.random.poisson(8),
                'rdp_failed_auth_count': np.random.poisson(6),
                'new_internal_connections_30s': np.random.poisson(12),
                'port_scan_pattern_score': np.random.exponential(0.7),
                'upload_download_ratio_30s': np.random.exponential(5.0),
                'burst_connections_count': np.random.poisson(10),
                'unique_destinations_30s': np.random.poisson(20),
                'large_upload_sessions_count': np.random.poisson(4),
                'nocturnal_activity_flag': np.random.choice([0, 1], p=[0.3, 0.7]),
                'connection_rate_stddev': np.random.exponential(25.0),
                'protocol_diversity_score': np.random.beta(3, 2),
                'avg_flow_duration_seconds': np.random.exponential(120.0),
                'tcp_rst_ratio': np.random.beta(2, 8),
                'syn_without_ack_ratio': np.random.beta(3, 7),
                'is_ransomware': 1
            }
        else:
            # PATRONES NORMALES (baja entropía, pocas conexiones)
            features = {
                'dns_query_entropy': np.random.exponential(0.5),
                'new_external_ips_30s': np.random.poisson(2),
                'dns_query_rate_per_min': np.random.exponential(5),
                'failed_dns_queries_ratio': np.random.beta(1, 20),
                'tls_self_signed_cert_count': np.random.poisson(0.1),
                'non_standard_port_http_count': np.random.poisson(0.2),
                'smb_connection_diversity': np.random.poisson(1),
                'rdp_failed_auth_count': np.random.poisson(0.1),
                'new_internal_connections_30s': np.random.poisson(1),
                'port_scan_pattern_score': np.random.beta(1, 20),
                'upload_download_ratio_30s': np.random.exponential(0.3),
                'burst_connections_count': np.random.poisson(0.5),
                'unique_destinations_30s': np.random.poisson(3),
                'large_upload_sessions_count': np.random.poisson(0.1),
                'nocturnal_activity_flag': np.random.choice([0, 1], p=[0.8, 0.2]),
                'connection_rate_stddev': np.random.exponential(5.0),
                'protocol_diversity_score': np.random.beta(1, 5),
                'avg_flow_duration_seconds': np.random.exponential(30.0),
                'tcp_rst_ratio': np.random.beta(1, 15),
                'syn_without_ack_ratio': np.random.beta(1, 10),
                'is_ransomware': 0
            }

        data.append(features)

    df = pd.DataFrame(data)
    print(f"✅ Dataset generado: {len(df)} muestras")
    print(f"   Ransomware: {sum(df['is_ransomware'] == 1)}")
    print(f"   Normal: {sum(df['is_ransomware'] == 0)}")

    return df

def train_proto_aligned_model():
    """Entrena modelo alineado con las features del .proto"""
    print("🚀 ENTRENANDO MODELO ALINEADO CON .PROTO")

    # 1. Generar datos de entrenamiento
    training_data = generate_ransomware_training_data()

    # 2. Preparar features (exactamente las del .proto)
    X = training_data[PROTO_FEATURES]
    y = training_data['is_ransomware']

    print(f"📊 Dataset final:")
    print(f"   Muestras: {len(X):,}")
    print(f"   Features: {len(PROTO_FEATURES)} (alineadas con .proto)")
    print(f"   Ransomware: {sum(y==1):,} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"   Normal: {sum(y==0):,} ({sum(y==0)/len(y)*100:.1f}%)")

    # 3. Split y escalado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Entrenar modelo
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_scaled, y_train, verbose=10)

    # 5. Evaluar
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"📈 MÉTRICAS FINALES:")
    print(f"   Accuracy:  {acc:.3f}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall:    {rec:.3f}")
    print(f"   F1-Score:  {f1:.3f}")
    print(f"   ROC-AUC:   {roc_auc:.3f}")

    # 6. Guardar modelo
    model_dir = OUTPUT_PATH / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / f"{MODEL_NAME}.pkl")
    joblib.dump(scaler, model_dir / f"{MODEL_NAME}_scaler.pkl")

    # 7. Metadata
    metadata = {
        'model_name': MODEL_NAME,
        'model_type': 'RANSOMWARE_NETWORK_DETECTOR',
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'features': PROTO_FEATURES,
        'metrics': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'training_params': XGBOOST_PARAMS,
        'dataset_info': {
            'total_samples': len(X),
            'ransomware_samples': sum(y == 1),
            'normal_samples': sum(y == 0),
            'data_source': 'synthetic_based_on_ransomware_patterns'
        },
        'proto_alignment': {
            'message_name': 'RansomwareFeatures',
            'feature_count': 20,
            'fully_aligned': True
        }
    }

    with open(model_dir / f"{MODEL_NAME}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Modelo guardado en: {model_dir}")
    print(f"✅ ALINEADO CON .PROTO: {len(PROTO_FEATURES)} features exactas")

    return model_dir, metadata

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 RANSOMWARE DETECTOR - ALINEADO CON .PROTO")
    print("=" * 60)
    print(f"Features: {len(PROTO_FEATURES)} del message RansomwareFeatures")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        model_dir, metadata = train_proto_aligned_model()

        print("\n✅ ENTRENAMIENTO COMPLETADO")
        print(f"🎯 Recall: {metadata['metrics']['recall']:.3f} (detección ransomware)")
        print(f"📊 Dataset: {metadata['dataset_info']['total_samples']:,} muestras")
        print(f"🔧 Features: {len(metadata['features'])} alineadas con .proto")
        print(f"💾 Guardado en: {model_dir}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()