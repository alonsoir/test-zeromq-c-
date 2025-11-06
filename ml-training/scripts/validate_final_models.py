#!/usr/bin/env python3
"""
VALIDADOR FINAL PARA MODELOS ONNX
Enfocado en compatibilidad con C++
"""

import onnx
import onnxruntime as ort
import json
import numpy as np
from pathlib import Path

def validate_for_cpp(onnx_path):
    """Valida específicamente para uso en C++"""
    try:
        print(f"🔍 {onnx_path.name}")

        # 1. Verificar que el archivo se puede cargar
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("   ✅ Sintaxis ONNX válida")

        # 2. Verificar opset (debe ser compatible con ONNX Runtime C++)
        opset = model.opset_import[0].version
        print(f"   🔧 Opset: {opset}")

        # 3. Configuración optimizada para C++
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.enable_profiling = False

        # 4. Crear sesión
        providers = ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(onnx_path, so, providers=providers)
        print("   ✅ Sesión ONNX Runtime creada")

        # 5. Información de E/S
        inputs = ort_session.get_inputs()
        outputs = ort_session.get_outputs()

        print(f"   📥 Entrada: {inputs[0].name}")
        print(f"      Tipo: {inputs[0].type}")
        print(f"      Shape: {inputs[0].shape}")

        print(f"   📤 Salidas: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"      {i+1}. {out.name} - {out.type} - {out.shape}")

        # 6. Probar inferencia con shape real
        input_shape = list(inputs[0].shape)
        # Reemplazar dimensiones dinámicas
        for i, dim in enumerate(input_shape):
            if dim is None or dim == 'None':
                input_shape[i] = 1

        # Crear datos de prueba
        test_data = np.random.random(input_shape).astype(np.float32)

        # 7. Ejecutar inferencia
        results = ort_session.run(None, {inputs[0].name: test_data})

        print(f"   ✅ Inferencia exitosa")
        for i, result in enumerate(results):
            result_type = type(result).__name__
            try:
                result_shape = result.shape
                print(f"      Output {i+1}: {result_type} - Shape: {result_shape}")
            except AttributeError:
                print(f"      Output {i+1}: {result_type} - Sin shape")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_metadata_quality(metadata_path):
    """Verifica la calidad de los metadatos"""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        checks = {
            'Tipo de modelo': 'model_type' in metadata,
            'Forma de entrada': 'input_shape' in metadata,
            'Nombres de features': 'feature_names' in metadata,
            'Clases de salida': 'output_classes' in metadata
        }

        print("   📊 Metadatos:")
        passed = 0
        for check_name, check_passed in checks.items():
            status = "✅" if check_passed else "❌"
            print(f"      {status} {check_name}")
            if check_passed:
                passed += 1

        # Información adicional
        if 'feature_names' in metadata:
            print(f"      📐 Número de features: {len(metadata['feature_names'])}")

        if 'scaler' in metadata:
            print(f"      ⚖️  Scaler: {metadata['scaler'].get('scaler_type', 'N/A')}")

        return passed >= 3  # Al menos 3 de 4 checks pasan

    except Exception as e:
        print(f"   ❌ Error en metadatos: {e}")
        return False

def main():
    models_dir = Path("/Users/aironman/CLionProjects/test-zeromq-docker/ml-detector/models/production/level3")

    print("🔍 VALIDACIÓN FINAL PARA C++")
    print("=" * 50)

    if not models_dir.exists():
        print(f"❌ Directorio no encontrado: {models_dir}")
        return

    # Encontrar modelos
    onnx_files = list(models_dir.rglob("*.onnx"))

    if not onnx_files:
        print("❌ No se encontraron modelos ONNX")
        return

    print(f"📁 Encontrados {len(onnx_files)} modelos")

    validation_results = {}

    for onnx_path in onnx_files:
        category = onnx_path.parent.name
        model_name = onnx_path.stem

        print(f"\n📂 {category.upper()}/{model_name}")
        print("-" * 40)

        # Validar modelo
        model_valid = validate_for_cpp(onnx_path)

        # Validar metadatos
        metadata_path = onnx_path.with_suffix('.json')
        metadata_valid = check_metadata_quality(metadata_path) if metadata_path.exists() else False

        validation_results[model_name] = {
            'category': category,
            'model_valid': model_valid,
            'metadata_valid': metadata_valid,
            'ready_for_cpp': model_valid and metadata_valid
        }

    # Resumen final
    print(f"\n{'='*60}")
    print("🎯 RESUMEN FINAL - PREPARACIÓN PARA C++")
    print(f"{'='*60}")

    total = len(validation_results)
    ready_for_cpp = sum(1 for r in validation_results.values() if r['ready_for_cpp'])
    model_valid = sum(1 for r in validation_results.values() if r['model_valid'])
    metadata_valid = sum(1 for r in validation_results.values() if r['metadata_valid'])

    print(f"📊 ESTADÍSTICAS:")
    print(f"   • Total de modelos: {total}")
    print(f"   • Modelos ONNX válidos: {model_valid}/{total}")
    print(f"   • Metadatos válidos: {metadata_valid}/{total}")
    print(f"   • LISTOS PARA C++: {ready_for_cpp}/{total}")

    print(f"\n📋 POR CATEGORÍA:")
    categories = {}
    for result in validation_results.values():
        cat = result['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'ready': 0}
        categories[cat]['total'] += 1
        if result['ready_for_cpp']:
            categories[cat]['ready'] += 1

    for cat, stats in categories.items():
        readiness = stats['ready'] / stats['total'] * 100
        print(f"   📁 {cat.upper()}: {stats['ready']}/{stats['total']} listos ({readiness:.1f}%)")

    print(f"\n💡 RECOMENDACIONES FINALES:")
    if ready_for_cpp == total:
        print("   ✅ TODOS LOS MODELOS ESTÁN LISTOS PARA INTEGRACIÓN EN C++")
        print("   🚀 Puedes proceder con la implementación en el detector")
    elif ready_for_cpp >= total * 0.7:
        print("   ⚠️  La mayoría de modelos están listos")
        print("   🔧 Revisa los que fallaron antes de integrar")
    else:
        print("   ❌ Problemas significativos de compatibilidad")
        print("   🛠️  Revisa la conversión de los modelos fallidos")

if __name__ == "__main__":
    main()