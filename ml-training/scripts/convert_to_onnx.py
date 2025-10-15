#!/usr/bin/env python3
"""
Convert Models to ONNX - Conversión de modelos sklearn a ONNX
Para uso en ml-detector C++
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

def load_model_and_metadata(model_path, metadata_path):
    """Cargar modelo sklearn y metadata"""
    print(f"📦 Cargando modelo: {model_path}")
    model = joblib.load(model_path)
    
    print(f"📋 Cargando metadata: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    return model, metadata

def convert_to_onnx(model, n_features, output_path):
    """Convertir modelo sklearn a ONNX"""
    print(f"\n🔄 Convirtiendo a ONNX...")
    print(f"   Features: {n_features}")
    print(f"   Destino: {output_path}")
    
    # Definir tipo de input (batch_size variable, n_features fijo)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convertir
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,  # Compatible con ONNX Runtime 1.16
        options={
            'zipmap': False  # No usar ZipMap, output directo
        }
    )
    
    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"✅ Modelo ONNX guardado")
    
    return onnx_model

def validate_onnx_conversion(sklearn_model, onnx_path, n_features, n_samples=100):
    """Validar que sklearn y ONNX dan mismos resultados"""
    print(f"\n🧪 Validando conversión ONNX...")
    
    # Generar datos de prueba
    np.random.seed(42)
    X_test = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Predicción sklearn
    y_sklearn = sklearn_model.predict(X_test)
    y_sklearn_proba = sklearn_model.predict_proba(X_test)
    
    # Predicción ONNX
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
    print(f"   Labels match: {label_match}")
    
    # Comparar probabilidades (con tolerancia)
    proba_close = np.allclose(y_sklearn_proba, y_onnx_proba, rtol=1e-4, atol=1e-4)
    print(f"   Probabilities close: {proba_close}")
    
    if label_match and proba_close:
        print("✅ Validación exitosa - sklearn y ONNX coinciden")
        return True
    else:
        print("❌ Validación fallida - diferencias entre sklearn y ONNX")
        
        # Mostrar diferencias
        if not label_match:
            diff_count = np.sum(y_sklearn != y_onnx)
            print(f"   Diferencias en labels: {diff_count}/{n_samples}")
        
        if not proba_close:
            max_diff = np.max(np.abs(y_sklearn_proba - y_onnx_proba))
            print(f"   Max diff en probabilidades: {max_diff}")
        
        return False

def verify_onnx_model(onnx_path):
    """Verificar que el modelo ONNX es válido"""
    print(f"\n🔍 Verificando modelo ONNX...")
    
    # Cargar y verificar
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✅ Modelo ONNX válido")
    
    # Mostrar información
    print(f"\n📊 Información del modelo ONNX:")
    print(f"   Opset: {onnx_model.opset_import[0].version}")
    
    # Inputs
    print(f"   Inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'None' for d in inp.type.tensor_type.shape.dim]
        print(f"      - {inp.name}: {inp.type.tensor_type.elem_type} {shape}")
    
    # Outputs
    print(f"   Outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else 'None' for d in out.type.tensor_type.shape.dim]
        print(f"      - {out.name}: {out.type.tensor_type.elem_type} {shape}")

def create_onnx_metadata(original_metadata, onnx_path, validation_passed):
    """Crear metadata para el modelo ONNX"""
    metadata = {
        "model_name": original_metadata['model_name'] + "_onnx",
        "onnx_version": "1.12",
        "opset_version": 12,
        "original_model": original_metadata['model_name'],
        "conversion_date": datetime.now().isoformat(),
        "n_features": original_metadata['n_features'],
        "feature_names": original_metadata['feature_names'],
        "classes": original_metadata['classes'],
        "input_name": "float_input",
        "input_shape": [None, original_metadata['n_features']],
        "input_dtype": "float32",
        "output_label_name": "label",
        "output_proba_name": "probabilities",
        "validation_passed": validation_passed,
        "sklearn_metrics": original_metadata['metrics'],
        "usage": {
            "cpp_example": "ONNXModel model(\"level1_attack_detector.onnx\"); auto result = model.predict(features);",
            "expected_input": f"float array of size {original_metadata['n_features']}",
            "expected_output": "label (0=BENIGN, 1=ATTACK) + probabilities [prob_benign, prob_attack]"
        }
    }
    
    # Guardar metadata
    metadata_path = onnx_path.parent / f"{onnx_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata ONNX guardada: {metadata_path}")
    
    return metadata

def main():
    print("=" * 80)
    print("🔄 CONVERSIÓN A ONNX - ML DETECTOR")
    print("=" * 80)
    
    # Rutas
    model_path = Path("outputs/models/level1_attack_detector.joblib")
    metadata_path = Path("outputs/metadata/level1_model_metadata.json")
    onnx_path = Path("outputs/onnx/level1_attack_detector.onnx")
    
    # Verificar que existen
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        print("   Ejecuta primero: python scripts/train_level1.py")
        return 1
    
    # 1. Cargar modelo
    model, metadata = load_model_and_metadata(model_path, metadata_path)
    print(f"✅ Modelo cargado: {metadata['model_type']}")
    print(f"   Features: {metadata['n_features']}")
    print(f"   Classes: {metadata['classes']}")
    
    # 2. Convertir a ONNX
    onnx_model = convert_to_onnx(model, metadata['n_features'], onnx_path)
    
    # 3. Verificar modelo ONNX
    verify_onnx_model(onnx_path)
    
    # 4. Validar conversión
    validation_passed = validate_onnx_conversion(
        model, onnx_path, metadata['n_features']
    )
    
    # 5. Crear metadata ONNX
    onnx_metadata = create_onnx_metadata(metadata, onnx_path, validation_passed)
    
    # 6. Resumen
    print("\n" + "=" * 80)
    print("✅ CONVERSIÓN A ONNX COMPLETADA")
    print("=" * 80)
    
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n📦 Modelo ONNX generado:")
    print(f"   Archivo: {onnx_path}")
    print(f"   Tamaño: {file_size_mb:.2f} MB")
    print(f"   Features: {metadata['n_features']}")
    print(f"   Validación: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    
    print(f"\n🎯 Para usar en C++:")
    print(f"   1. Copiar a: ../ml-detector/models/production/level1/")
    print(f"   2. Cargar con: ONNXModel(\"level1_attack_detector.onnx\")")
    print(f"   3. Input: std::vector<float> con {metadata['n_features']} valores")
    print(f"   4. Output: label (0/1) + probabilities [2 valores]")
    
    if not validation_passed:
        print("\n⚠️  ADVERTENCIA: La validación falló")
        print("   Revisa el modelo antes de usarlo en producción")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
