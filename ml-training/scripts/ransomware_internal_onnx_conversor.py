#!/usr/bin/env python3
"""
CONVERSOR DE MODELOS A ONNX
Convierte todos los modelos XGBoost a formato ONNX para producción
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import onnx
from onnxruntime import InferenceSession
import xgboost as xgb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# =====================================================================
# CONFIG - CONVERSIÓN ONNX
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
MODELS_PATH = BASE_PATH / "outputs" / "models"
ONNX_OUTPUT_PATH = BASE_PATH / "outputs" / "onnx_models"

def convert_xgboost_to_onnx(model_path, scaler_path, features_path, output_path):
    """Convierte un modelo XGBoost a ONNX"""
    print(f"🔄 Convirtiendo {model_path.name} a ONNX...")

    try:
        # Cargar modelo y scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Cargar features
        with open(features_path, 'r') as f:
            features = json.load(f)

        # Crear datos de ejemplo para inferir tipos
        n_features = len(features)
        example_data = np.random.random((1, n_features)).astype(np.float32)

        # Convertir modelo
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Guardar modelo ONNX
        onnx.save(onnx_model, output_path)

        # Verificar que el modelo funciona
        ort_session = InferenceSession(output_path)
        inputs = {ort_session.get_inputs()[0].name: example_data}
        outputs = ort_session.run(None, inputs)

        print(f"   ✅ Conversión exitosa: {output_path.name}")
        print(f"   📊 Input shape: {example_data.shape}")
        print(f"   🎯 Output shape: {outputs[0].shape}")

        return True

    except Exception as e:
        print(f"   ❌ Error en conversión: {e}")
        return False

def convert_all_models_to_onnx():
    """Convierte todos los modelos a ONNX"""
    print("🚀 INICIANDO CONVERSIÓN MASIVA A ONNX")
    print("=" * 60)

    # Crear directorio de salida
    ONNX_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    models_to_convert = [
        {
            'name': 'internal_traffic_detector_onnx_ready',
            'description': 'Detector de tráfico interno'
        },
        {
            'name': 'ransomware_anomaly_detector',
            'description': 'Detector conservador de ransomware'
        },
        {
            'name': 'ransomware_detector_optimized',
            'description': 'Detector agresivo de ransomware'
        }
    ]

    conversion_results = {}

    for model_info in models_to_convert:
        model_name = model_info['name']
        model_dir = MODELS_PATH / model_name

        if not model_dir.exists():
            print(f"❌ Modelo {model_name} no encontrado")
            continue

        print(f"\n🎯 PROCESANDO: {model_info['description']}")

        # Archivos necesarios
        model_file = model_dir / f"{model_name}.pkl"
        scaler_file = model_dir / f"{model_name}_scaler.pkl"
        features_file = model_dir / f"{model_name}_features.json"
        output_file = ONNX_OUTPUT_PATH / f"{model_name}.onnx"

        # Verificar que existen todos los archivos
        missing_files = []
        for file_path in [model_file, scaler_file, features_file]:
            if not file_path.exists():
                missing_files.append(file_path.name)

        if missing_files:
            print(f"   ⚠️  Archivos faltantes: {missing_files}")
            continue

        # Convertir a ONNX
        success = convert_xgboost_to_onnx(
            model_file, scaler_file, features_file, output_file
        )

        conversion_results[model_name] = {
            'success': success,
            'onnx_path': str(output_file),
            'description': model_info['description'],
            'timestamp': datetime.now().isoformat()
        }

    # Generar reporte de conversión
    print(f"\n📊 REPORTE DE CONVERSIÓN ONNX:")
    print("=" * 60)

    successful = 0
    for model_name, result in conversion_results.items():
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {model_name}: {result['description']}")
        if result['success']:
            successful += 1

    print(f"\n🎯 RESUMEN: {successful}/{len(models_to_convert)} modelos convertidos")

    # Guardar metadata de conversión
    conversion_metadata = {
        'conversion_date': datetime.now().isoformat(),
        'total_models': len(models_to_convert),
        'successful_conversions': successful,
        'results': conversion_results
    }

    with open(ONNX_OUTPUT_PATH / "conversion_metadata.json", 'w') as f:
        json.dump(conversion_metadata, f, indent=2)

    print(f"💾 Metadata guardada en: {ONNX_OUTPUT_PATH / 'conversion_metadata.json'}")

def test_onnx_models():
    """Prueba los modelos ONNX convertidos"""
    print(f"\n🧪 PROBANDO MODELOS ONNX...")
    print("=" * 60)

    onnx_files = list(ONNX_OUTPUT_PATH.glob("*.onnx"))

    for onnx_file in onnx_files:
        print(f"🔍 Probando {onnx_file.name}...")

        try:
            # Cargar modelo ONNX
            ort_session = InferenceSession(str(onnx_file))

            # Obtener información del modelo
            inputs = ort_session.get_inputs()
            outputs = ort_session.get_outputs()

            print(f"   ✅ Modelo cargado correctamente")
            print(f"   📥 Input: {inputs[0].name} - Shape: {inputs[0].shape}")
            print(f"   📤 Output: {outputs[0].name} - Shape: {outputs[0].shape}")

            # Crear datos de prueba
            input_shape = inputs[0].shape
            if input_shape[0] is None:  # Dimension dinámica
                batch_size = 1
                feature_size = input_shape[1]
            else:
                batch_size, feature_size = input_shape

            test_data = np.random.random((batch_size, feature_size)).astype(np.float32)

            # Realizar inferencia
            input_name = inputs[0].name
            outputs = ort_session.run(None, {input_name: test_data})

            print(f"   🎯 Inferencia exitosa - Output: {outputs[0].shape}")

        except Exception as e:
            print(f"   ❌ Error probando modelo: {e}")

if __name__ == "__main__":
    print("🎯 CONVERSOR DE MODELOS ML A ONNX")
    print("=" * 60)
    print("Objetivo: Preparar modelos para producción enterprise")
    print("=" * 60)

    # 1. Convertir todos los modelos a ONNX
    convert_all_models_to_onnx()

    # 2. Probar modelos convertidos
    test_onnx_models()

    print(f"\n🎉 PROCESO DE CONVERSIÓN COMPLETADO!")
    print(f"📁 Modelos ONNX guardados en: {ONNX_OUTPUT_PATH}")
    print(f"🚀 Listos para integración en sistemas enterprise")