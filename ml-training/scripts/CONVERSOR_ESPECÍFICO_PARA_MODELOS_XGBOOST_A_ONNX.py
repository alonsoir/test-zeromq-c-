#!/usr/bin/env python3
"""
CONVERSOR XGBOOST A ONNX USANDO HUMMINGBIRD
Solución definitiva para convertir XGBoost a ONNX
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import onnx
from onnxruntime import InferenceSession

# =====================================================================
# CONFIG - HUMMINGBIRD CONVERSION
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
MODELS_PATH = BASE_PATH / "outputs" / "models"
ONNX_OUTPUT_PATH = BASE_PATH / "outputs" / "onnx_models"

def install_hummingbird():
    """Instala hummingbird si no está disponible"""
    try:
        import hummingbird.ml
        return True
    except ImportError:
        print("🔧 Instalando Hummingbird...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hummingbird-ml"])
            import hummingbird.ml
            print("✅ Hummingbird instalado correctamente")
            return True
        except:
            print("❌ No se pudo instalar Hummingbird")
            return False

def convert_xgboost_with_hummingbird(model_path, scaler_path, features_path, output_path):
    """Convierte modelo XGBoost a ONNX usando Hummingbird"""
    print(f"🔄 Convirtiendo {model_path.name} a ONNX con Hummingbird...")

    try:
        # Instalar/verificar Hummingbird
        if not install_hummingbird():
            return False

        import hummingbird.ml

        # Cargar modelo y scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Cargar features
        with open(features_path, 'r') as f:
            features = json.load(f)

        # Crear datos de ejemplo
        n_features = len(features)
        example_data = np.random.random((1, n_features)).astype(np.float32)

        # Aplicar scaler a los datos de ejemplo
        example_data_scaled = scaler.transform(example_data)

        print(f"   📊 Modelo: {type(model).__name__}")
        print(f"   🔧 Features: {n_features}")

        # Convertir usando Hummingbird
        hb_model = hummingbird.ml.convert(model, "onnx", example_data_scaled)

        # Guardar modelo ONNX
        hb_model.save(str(output_path))

        # Verificar que funciona
        ort_session = InferenceSession(str(output_path))

        # Preparar entrada para ONNX Runtime
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: example_data_scaled.astype(np.float32)})

        print(f"   ✅ Conversión exitosa: {output_path.name}")
        print(f"   📥 Input: {example_data_scaled.shape}")
        print(f"   📤 Output: {outputs[0].shape}")

        return True

    except Exception as e:
        print(f"   ❌ Error en conversión Hummingbird: {e}")
        return False

def convert_all_models_hummingbird():
    """Convierte todos los modelos usando Hummingbird"""
    print("🚀 CONVERSIÓN CON HUMMINGBIRD")
    print("=" * 60)

    ONNX_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    models_to_convert = [
        {
            'name': 'internal_traffic_detector_onnx_ready',
            'description': 'Detector de tráfico interno'
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

        # Verificar archivos
        missing_files = []
        for file_path in [model_file, scaler_file, features_file]:
            if not file_path.exists():
                missing_files.append(file_path.name)

        if missing_files:
            print(f"   ⚠️  Archivos faltantes: {missing_files}")
            continue

        # Convertir a ONNX con Hummingbird
        success = convert_xgboost_with_hummingbird(
            model_file, scaler_file, features_file, output_file
        )

        conversion_results[model_name] = {
            'success': success,
            'onnx_path': str(output_file),
            'description': model_info['description'],
            'timestamp': datetime.now().isoformat()
        }

    # Reporte
    print(f"\n📊 REPORTE DE CONVERSIÓN HUMMINGBIRD:")
    print("=" * 60)

    successful = 0
    for model_name, result in conversion_results.items():
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {model_name}: {result['description']}")
        if result['success']:
            successful += 1

    print(f"\n🎯 RESUMEN: {successful}/{len(models_to_convert)} modelos convertidos")

    # Guardar metadata
    conversion_metadata = {
        'conversion_date': datetime.now().isoformat(),
        'total_models': len(models_to_convert),
        'successful_conversions': successful,
        'results': conversion_results,
        'conversion_tool': 'Hummingbird'
    }

    with open(ONNX_OUTPUT_PATH / "conversion_metadata.json", 'w') as f:
        json.dump(conversion_metadata, f, indent=2)

if __name__ == "__main__":
    print("🎯 CONVERSOR XGBOOST A ONNX - HUMMINGBIRD")
    print("=" * 60)
    print("Solución definitiva para conversión XGBoost")
    print("=" * 60)

    convert_all_models_hummingbird()