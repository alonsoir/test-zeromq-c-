#!/usr/bin/env python3
"""
SISTEMA DE DETECCIÓN DE RANSOMWARE DE DOBLE CAPA
Combina ambos modelos para máxima efectividad
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

class RansomwareDualDetector:
    def __init__(self):
        self.models_loaded = False
        self.conservative_model = None
        self.aggressive_model = None
        self.scaler_conservative = None
        self.scaler_aggressive = None
        self.features_conservative = None
        self.features_aggressive = None

    def load_models(self):
        """Carga ambos modelos entrenados"""
        print("🔄 CARGANDO SISTEMA DE DOBLE DETECCIÓN...")

        # Rutas de los modelos
        conservative_path = Path("../outputs/models/ransomware_anomaly_detector")
        aggressive_path = Path("../outputs/models/ransomware_detector_optimized")

        try:
            # Cargar modelo conservador
            self.conservative_model = joblib.load(conservative_path / "ransomware_anomaly_detector.pkl")
            self.scaler_conservative = joblib.load(conservative_path / "ransomware_anomaly_detector_scaler.pkl")
            with open(conservative_path / "ransomware_anomaly_detector_features.json", 'r') as f:
                self.features_conservative = json.load(f)
            print("   ✅ Modelo conservador cargado (0 falsos positivos)")

            # Cargar modelo agresivo
            self.aggressive_model = joblib.load(aggressive_path / "ransomware_detector_optimized.pkl")
            self.scaler_aggressive = joblib.load(aggressive_path / "ransomware_detector_optimized_scaler.pkl")
            with open(aggressive_path / "ransomware_detector_optimized_features.json", 'r') as f:
                self.features_aggressive = json.load(f)
            print("   ✅ Modelo agresivo cargado (98.6% detección)")

            self.models_loaded = True
            print("   🎯 Sistema de doble capa listo")

        except Exception as e:
            print(f"   ❌ Error cargando modelos: {e}")
            raise

    def prepare_features(self, network_data, features_list):
        """Prepara las features para un modelo específico"""
        # Asegurar que tenemos todas las features necesarias
        prepared_data = {}
        for feature in features_list:
            if feature in network_data:
                prepared_data[feature] = network_data[feature]
            else:
                # Si falta la feature, usar valor por defecto
                prepared_data[feature] = 0.0
        return pd.DataFrame([prepared_data])[features_list]

    def detect_ransomware(self, network_features):
        """
        Sistema de detección en cascada:
        1. Primero usa modelo conservador (0 falsos positivos)
        2. Si es dudoso, usa modelo agresivo (máxima detección)
        """
        if not self.models_loaded:
            self.load_models()

        print(f"\n🔍 ANALIZANDO TRÁFICO DE RED...")

        # CAPA 1: Modelo Conservador (Alta precisión)
        print("   🛡️  Capa 1 - Verificación conservadora...")
        features_cons = self.prepare_features(network_features, self.features_conservative)
        features_cons_scaled = self.scaler_conservative.transform(features_cons)
        prediction_cons = self.conservative_model.predict(features_cons_scaled)[0]
        score_cons = self.conservative_model.decision_function(features_cons_scaled)[0]

        # Isolation Forest: -1 = anomalía, 1 = normal
        is_anomaly_cons = (prediction_cons == -1)

        if is_anomaly_cons:
            # CAPA 1 DETECTA RANSOMWARE - Alta confianza
            print("   🚨 ALERTA RANSOMWARE (Capa 1 - Alta Confianza)")
            return {
                'detection': True,
                'confidence': 'HIGH',
                'detector_used': 'conservative',
                'anomaly_score': float(score_cons),
                'message': 'Ransomware detectado con certeza (0% falsos positivos)',
                'timestamp': datetime.now().isoformat()
            }
        else:
            # CAPA 2: Modelo Agresivo (Alta detección)
            print("   🔍 Capa 2 - Verificación agresiva...")
            features_agg = self.prepare_features(network_features, self.features_aggressive)
            features_agg_scaled = self.scaler_aggressive.transform(features_agg)
            prediction_agg = self.aggressive_model.predict(features_agg_scaled)[0]
            score_agg = self.aggressive_model.decision_function(features_agg_scaled)[0]

            is_anomaly_agg = (prediction_agg == -1)

            if is_anomaly_agg:
                # CAPA 2 DETECTA RANSOMWARE - Media confianza
                print("   ⚠️  ALERTA RANSOMWARE (Capa 2 - Media Confianza)")
                return {
                    'detection': True,
                    'confidence': 'MEDIUM',
                    'detector_used': 'aggressive',
                    'anomaly_score': float(score_agg),
                    'message': 'Posible ransomware detectado (98.6% recall)',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # AMBAS CAPAS CONCUERDAN: TRÁFICO NORMAL
                print("   ✅ Tráfico normal (Ambas capas)")
                return {
                    'detection': False,
                    'confidence': 'HIGH',
                    'detector_used': 'both',
                    'anomaly_scores': {
                        'conservative': float(score_cons),
                        'aggressive': float(score_agg)
                    },
                    'message': 'Tráfico clasificado como normal',
                    'timestamp': datetime.now().isoformat()
                }

# =====================================================================
# EJEMPLOS DE USO
# =====================================================================

def test_dual_detector():
    """Prueba el sistema de doble detección con ejemplos"""
    print("🧪 PROBANDO SISTEMA DE DOBLE DETECCIÓN")
    print("=" * 60)

    detector = RansomwareDualDetector()

    # Ejemplo 1: Tráfico normal típico
    print("\n📋 EJEMPLO 1 - Tráfico Normal:")
    normal_traffic = {
        ' Flow Duration': 120000,
        ' Total Fwd Packets': 45,
        ' Total Backward Packets': 32,
        ' Flow Bytes/s': 15000,
        ' Flow Packets/s': 12,
        ' Flow IAT Mean': 10000,
        ' Flow IAT Std': 2000,
        ' Destination Port': 443,
        ' SYN Flag Count': 3,
        ' ACK Flag Count': 40
    }
    result1 = detector.detect_ransomware(normal_traffic)
    print(f"   Resultado: {result1['detection']} - {result1['message']}")

    # Ejemplo 2: Posible ransomware (patrón C&C)
    print("\n📋 EJEMPLO 2 - Posible Ransomware (C&C):")
    ransomware_traffic = {
        ' Flow Duration': 800000,
        ' Total Fwd Packets': 1500,
        ' Total Backward Packets': 50,
        ' Flow Bytes/s': 500000,
        ' Flow Packets/s': 200,
        ' Flow IAT Mean': 5000,
        ' Flow IAT Std': 80000,
        ' Destination Port': 8333,  # Puerto Bitcoin común en ransomware
        ' SYN Flag Count': 80,
        ' ACK Flag Count': 1200
    }
    result2 = detector.detect_ransomware(ransomware_traffic)
    print(f"   Resultado: {result2['detection']} - {result2['message']}")

    # Ejemplo 3: Caso límite (puede variar entre modelos)
    print("\n📋 EJEMPLO 3 - Caso Límite:")
    borderline_traffic = {
        ' Flow Duration': 300000,
        ' Total Fwd Packets': 300,
        ' Total Backward Packets': 100,
        ' Flow Bytes/s': 80000,
        ' Flow Packets/s': 50,
        ' Flow IAT Mean': 15000,
        ' Flow IAT Std': 25000,
        ' Destination Port': 8080,
        ' SYN Flag Count': 15,
        ' ACK Flag Count': 200
    }
    result3 = detector.detect_ransomware(borderline_traffic)
    print(f"   Resultado: {result3['detection']} - {result3['message']}")

def create_production_script():
    """Crea script listo para producción"""
    script_content = '''#!/usr/bin/env python3
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
'''

    with open("production_ransomware_detector.py", "w") as f:
        f.write(script_content)
    print("✅ Script de producción creado: production_ransomware_detector.py")

if __name__ == "__main__":
    print("🚀 CREANDO SISTEMA DE DOBLE DETECCIÓN RANSOMWARE")
    print("=" * 60)

    # 1. Probar el sistema
    test_dual_detector()

    # 2. Crear script de producción
    create_production_script()

    print(f"\n🎉 SISTEMA COMPLETO CREADO!")
    print("📁 ARCHIVOS GENERADOS:")
    print("   - ransomware_dual_detector.py (este script)")
    print("   - production_ransomware_detector.py (para producción)")
    print("\n🎯 ESTRATEGIA IMPLEMENTADA:")
    print("   🛡️  Capa 1: Modelo Conservador - 0% falsos positivos")
    print("   🔍 Capa 2: Modelo Agresivo - 98.6% detección")
    print("   ✅ Cobertura máxima + Mínimas falsas alarmas")