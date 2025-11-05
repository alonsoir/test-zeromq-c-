#!/usr/bin/env python3
"""
PRODUCTION - SISTEMA DE DETECCIÓN RANSOMWARE DOBLE CAPA
Integración lista para el sistema enterprise
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

class ProductionRansomwareDetector:
    def __init__(self, models_base_path="ml-training/outputs/models"):
        self.models_base_path = Path(models_base_path)
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Carga ambos modelos en memoria"""
        try:
            # Modelo conservador
            conservative_path = self.models_base_path / "ransomware_anomaly_detector"
            self.conservative_model = joblib.load(conservative_path / "ransomware_anomaly_detector.pkl")
            self.scaler_conservative = joblib.load(conservative_path / "ransomware_anomaly_detector_scaler.pkl")
            
            # Modelo agresivo  
            aggressive_path = self.models_base_path / "ransomware_detector_optimized"
            self.aggressive_model = joblib.load(aggressive_path / "ransomware_detector_optimized.pkl")
            self.scaler_aggressive = joblib.load(aggressive_path / "ransomware_detector_optimized_scaler.pkl")
            
            # Features
            with open(conservative_path / "ransomware_anomaly_detector_features.json", 'r') as f:
                self.features_cons = json.load(f)
            with open(aggressive_path / "ransomware_detector_optimized_features.json", 'r') as f:
                self.features_agg = json.load(f)
                
            self.models_loaded = True
            print("✅ Sistema de detección ransomware cargado")
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {e}")
            raise
    
    def analyze_network_flow(self, flow_features):
        """
        Analiza un flujo de red usando el sistema de doble capa
        Retorna: dict con resultado y metadatos
        """
        if not self.models_loaded:
            raise RuntimeError("Modelos no cargados")
        
        # Preparar features para ambos modelos
        def prepare_features(features, required_features):
            prepared = {}
            for feat in required_features:
                prepared[feat] = features.get(feat, 0.0)
            return pd.DataFrame([prepared])[required_features]
        
        # Capa 1: Modelo conservador
        features_cons = prepare_features(flow_features, self.features_cons)
        features_cons_scaled = self.scaler_conservative.transform(features_cons)
        pred_cons = self.conservative_model.predict(features_cons_scaled)[0]
        score_cons = self.conservative_model.decision_function(features_cons_scaled)[0]
        
        if pred_cons == -1:  # Anomalía detectada
            return {
                'ransomware_detected': True,
                'confidence_level': 'HIGH',
                'detection_method': 'conservative_model',
                'anomaly_score': float(score_cons),
                'risk_category': 'CRITICAL',
                'recommendation': 'Bloquear tráfico inmediatamente',
                'timestamp': datetime.now().isoformat()
            }
        
        # Capa 2: Modelo agresivo
        features_agg = prepare_features(flow_features, self.features_agg)
        features_agg_scaled = self.scaler_aggressive.transform(features_agg)
        pred_agg = self.aggressive_model.predict(features_agg_scaled)[0]
        score_agg = self.aggressive_model.decision_function(features_agg_scaled)[0]
        
        if pred_agg == -1:  # Anomalía detectada
            return {
                'ransomware_detected': True,
                'confidence_level': 'MEDIUM', 
                'detection_method': 'aggressive_model',
                'anomaly_score': float(score_agg),
                'risk_category': 'HIGH',
                'recommendation': 'Investigar y monitorear tráfico',
                'timestamp': datetime.now().isoformat()
            }
        
        # Tráfico normal
        return {
            'ransomware_detected': False,
            'confidence_level': 'HIGH',
            'detection_method': 'both_models_agree',
            'anomaly_scores': {
                'conservative': float(score_cons),
                'aggressive': float(score_agg)
            },
            'risk_category': 'LOW',
            'recommendation': 'Tráfico normal - Continuar monitoreo',
            'timestamp': datetime.now().isoformat()
        }

# Instancia global para uso en producción
ransomware_detector = ProductionRansomwareDetector()

if __name__ == "__main__":
    # Ejemplo de uso en producción
    sample_flow = {
        ' Flow Duration': 120000,
        ' Total Fwd Packets': 45,
        'Total Length of Fwd Packets': 56000,
        ' Flow Bytes/s': 15000,
        # ... agregar más features según .proto
    }
    
    result = ransomware_detector.analyze_network_flow(sample_flow)
    print("🔍 Resultado del análisis:", result)
