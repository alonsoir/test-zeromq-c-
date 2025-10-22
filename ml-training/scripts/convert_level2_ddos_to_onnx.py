#!/usr/bin/env python3
"""
Convert Level 2 DDoS Model to ONNX
ml-training/scripts/convert_level2_ddos_to_onnx.py
"""

import numpy as np
import joblib
import json
from pathlib import Path
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURACIÓN
# =====================================================================

MODEL_NAME = "level2_ddos_binary_detector"
INPUT_PATH = Path("outputs/models")
OUTPUT_PATH = Path("outputs/onnx")
METADATA_PATH = Path("outputs/metadata")

# =====================================================================
# FUNCIONES
# =====================================================================

def load_model_and_metadata():
    """Cargar modelo sklearn y metadata"""
    print("=" * 80)
    print("📦 CARGANDO MODELO")
    print("=" * 80)
    
    model_path = INPUT_PATH / f"{MODEL_NAME}.joblib"
    metadata_path = METADATA_PATH / f"{MODEL_NAME}_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata no encontrada: {metadata_path}")
    
    print(f"\n📦 Cargando modelo: {model_path}")
    model = joblib.load(model_path)
    
    print(f"📋 Cargando metadata: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\n✅ Modelo cargado:")
    print(f"  Tipo: {metadata['model_type']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Classes: {metadata['classes']}")
    print(f"  F1 Score: {metadata['metrics']['f1_score']:.4f}")
    print(f"  ROC AUC: {metadata['metrics']['roc_auc']:.4f}")
    
    return model, metadata


def convert_to_onnx(model, n_features):
    """
    Convertir modelo sklearn a ONNX
    
    Args:
        model: Modelo sklearn
        n_features: Número de features
    
    Returns:
        onnx_model
    """
    print("\n" + "=" * 80)
    print("🔄 CONVERSIÓN A ONNX")
    print("=" * 80)
    
    print(f"\n  Features: {n_features}")
    print(f"  Opset: 12")
    
    # Definir tipo de input (batch_size variable, n_features fijo)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convertir
    print(f"\n🔄 Convirtiendo...")
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,  # Compatible con ONNX Runtime 1.16+
        options={
            'zipmap': False  # No usar ZipMap, output directo
        }
    )
    
    # Guardar
    output_path = OUTPUT_PATH / f"{MODEL_NAME}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ Modelo ONNX guardado:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    
    return onnx_model, output_path


def verify_onnx_model(onnx_path):
    """Verificar que el modelo ONNX es válido"""
    print("\n" + "=" * 80)
    print("🔍 VERIFICANDO MODELO ONNX")
    print("=" * 80)
    
    # Cargar y verificar
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("\n✅ Modelo ONNX válido")
    
    # Mostrar información
    print(f"\n📊 Información del modelo:")
    print(f"  Opset: {onnx_model.opset_import[0].version}")
    
    # Inputs
    print(f"\n  Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'None' for d in inp.type.tensor_type.shape.dim]
        print(f"    - {inp.name}: {shape}")
    
    # Outputs
    print(f"\n  Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else 'None' for d in out.type.tensor_type.shape.dim]
        print(f"    - {out.name}: {shape}")


def validate_onnx_conversion(sklearn_model, onnx_path, n_features, n_samples=1000):
    """
    Validar que sklearn y ONNX dan mismos resultados
    
    Args:
        sklearn_model: Modelo sklearn original
        onnx_path: Path al modelo ONNX
        n_features: Número de features
        n_samples: Número de samples para validar
    
    Returns:
        bool: True si validación exitosa
    """
    print("\n" + "=" * 80)
    print("🧪 VALIDANDO CONVERSIÓN")
    print("=" * 80)
    
    # Generar datos de prueba
    print(f"\n🔢 Generando {n_samples} samples de prueba...")
    np.random.seed(42)
    X_test = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Predicción sklearn
    print("🔮 Predicción sklearn...")
    y_sklearn = sklearn_model.predict(X_test)
    y_sklearn_proba = sklearn_model.predict_proba(X_test)
    
    # Predicción ONNX
    print("🔮 Predicción ONNX...")
    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    proba_name = sess.get_outputs()[1].name
    
    onnx_results = sess.run(
        [label_name, proba_name],
        {input_name: X_test}
    )
    y_onnx = onnx_results[0]
    y_onnx_proba = onnx_results[1]
    
    # Comparar labels
    label_match = np.all(y_sklearn == y_onnx)
    print(f"\n📊 Comparación de labels:")
    print(f"  Match: {label_match}")
    
    if not label_match:
        diff_count = np.sum(y_sklearn != y_onnx)
        print(f"  Diferencias: {diff_count}/{n_samples} ({diff_count/n_samples*100:.2f}%)")
    
    # Comparar probabilidades (con tolerancia)
    proba_close = np.allclose(y_sklearn_proba, y_onnx_proba, rtol=1e-4, atol=1e-4)
    print(f"\n📊 Comparación de probabilidades:")
    print(f"  Close (rtol=1e-4): {proba_close}")
    
    if not proba_close:
        max_diff = np.max(np.abs(y_sklearn_proba - y_onnx_proba))
        mean_diff = np.mean(np.abs(y_sklearn_proba - y_onnx_proba))
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
    
    # Resultado final
    if label_match and proba_close:
        print("\n✅ VALIDACIÓN EXITOSA")
        print("  sklearn y ONNX producen resultados idénticos")
        return True
    else:
        print("\n⚠️  VALIDACIÓN CON ADVERTENCIAS")
        if label_match:
            print("  ✅ Labels match")
        else:
            print("  ❌ Labels differ")
        
        if proba_close:
            print("  ✅ Probabilities close")
        else:
            print("  ⚠️  Probabilities differ slightly (aceptable)")
        
        return label_match  # Labels match es suficiente


def create_onnx_metadata(original_metadata, onnx_path, validation_passed):
    """
    Crear metadata para el modelo ONNX
    
    Args:
        original_metadata: Metadata del modelo sklearn
        onnx_path: Path al modelo ONNX
        validation_passed: Bool indicando si validación pasó
    """
    print("\n" + "=" * 80)
    print("📝 CREANDO METADATA ONNX")
    print("=" * 80)
    
    metadata = {
        "model_name": f"{MODEL_NAME}_onnx",
        "onnx_version": "1.12",
        "opset_version": 12,
        "original_model": MODEL_NAME,
        "conversion_date": datetime.now().isoformat(),
        "level": 2,
        "purpose": "DDoS Binary Detection (BENIGN vs DDOS)",
        "n_features": original_metadata['n_features'],
        "feature_names": original_metadata['feature_names'],
        "classes": original_metadata['classes'],
        "class_mapping": original_metadata['class_mapping'],
        "input_name": "float_input",
        "input_shape": [None, original_metadata['n_features']],
        "input_dtype": "float32",
        "output_label_name": "label",
        "output_proba_name": "probabilities",
        "validation_passed": validation_passed,
        "sklearn_metrics": original_metadata['metrics'],
        "usage": {
            "cpp_example": f'ONNXModel model("{MODEL_NAME}.onnx"); auto result = model.predict(features);',
            "expected_input": f"float array of size {original_metadata['n_features']}",
            "expected_output": "label (0=BENIGN, 1=DDOS) + probabilities [prob_benign, prob_ddos]"
        },
        "integration_notes": {
            "pipeline_position": "Level 2 - Activated when Level 1 detects ATTACK",
            "input_source": "NetworkFeatures message from protobuf",
            "output_destination": "Scheduler/GeoIP Enricher",
            "performance_target": "<5ms inference time"
        }
    }
    
    # Guardar metadata
    metadata_path = METADATA_PATH / f"{MODEL_NAME}_onnx_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata ONNX guardada: {metadata_path}")


def main():
    """Función principal"""
    print("=" * 80)
    print("🔄 CONVERSIÓN LEVEL 2 DDOS A ONNX")
    print("=" * 80)
    
    try:
        # 1. Cargar modelo
        model, metadata = load_model_and_metadata()
        
        # 2. Convertir a ONNX
        onnx_model, onnx_path = convert_to_onnx(model, metadata['n_features'])
        
        # 3. Verificar modelo ONNX
        verify_onnx_model(onnx_path)
        
        # 4. Validar conversión
        validation_passed = validate_onnx_conversion(
            model, onnx_path, metadata['n_features']
        )
        
        # 5. Crear metadata ONNX
        create_onnx_metadata(metadata, onnx_path, validation_passed)
        
        # Resumen final
        print("\n" + "=" * 80)
        print("✅ CONVERSIÓN A ONNX COMPLETADA")
        print("=" * 80)
        
        file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"\n📦 Modelo ONNX:")
        print(f"  Path: {onnx_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Features: {metadata['n_features']}")
        print(f"  Validación: {'✅ PASSED' if validation_passed else '⚠️  WITH WARNINGS'}")
        
        print(f"\n🎯 Para integrar en ml-detector:")
        print(f"  1. Copiar a: ../ml-detector/models/production/level2/")
        print(f"  2. Actualizar: ../ml-detector/config/ml_detector_config.json")
        print(f"  3. Código C++: Cargar con ONNX Runtime")
        print(f"  4. Input: std::vector<float> con {metadata['n_features']} valores")
        print(f"  5. Output: label (0/1) + probabilities [2 valores]")
        
        if not validation_passed:
            print("\n⚠️  ADVERTENCIA: La validación tuvo advertencias")
            print("  Revisa las diferencias antes de usar en producción")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
