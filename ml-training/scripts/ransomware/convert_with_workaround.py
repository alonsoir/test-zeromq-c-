#!/usr/bin/env python3
"""
Convert candidate to ONNX - XGBoost 3.x compatible
Workaround for base_score array format issue
"""
import joblib
import json
from pathlib import Path
import numpy as np
import sys

# Rutas
candidate_dir = Path("model_candidates/ransomware_xgboost_candidate_v2_20251106_095308")
output_dir = Path("../../../ml-detector/models/production/level3/ransomware")

print("=" * 70)
print("🔄 CONVERTING CANDIDATE MODEL (XGBoost 3.x Compatible)")
print("=" * 70)
print(f"\n📂 Input:  {candidate_dir}")
print(f"📂 Output: {output_dir}")

# Load model
model_path = candidate_dir / f"{candidate_dir.name}.pkl"
print(f"\n🔄 Loading model...")
model = joblib.load(model_path)
print(f"   ✅ Model loaded: {type(model).__name__}")

# Load metadata
metadata_path = candidate_dir / f"{candidate_dir.name}_metadata.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
print(f"   ✅ Metadata loaded")

n_features = len(model.feature_importances_)
print(f"   📊 Features: {n_features}")

# Step 1: Always save XGBoost JSON (this always works)
print(f"\n📋 Step 1: Exporting to XGBoost native JSON...")
json_output = output_dir / f"{candidate_dir.name}.json"
output_dir.mkdir(parents=True, exist_ok=True)
model.save_model(str(json_output))
print(f"   ✅ XGBoost JSON saved: {json_output.name}")

# Step 2: Try ONNX conversion with skl2onnx
print(f"\n🔄 Step 2: Attempting ONNX conversion via skl2onnx...")
onnx_success = False

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12
    )

    # Save ONNX
    onnx_path = output_dir / f"{candidate_dir.name}.onnx"
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print(f"   ✅ ONNX conversion successful!")
    print(f"   💾 ONNX saved: {onnx_path.name}")
    onnx_success = True

except Exception as e:
    print(f"   ⚠️  skl2onnx failed: {e}")
    print(f"   🔄 Trying XGBoost native ONNX export...")

    # Step 3: Try native ONNX export (XGBoost 2.0+)
    try:
        onnx_path = output_dir / f"{candidate_dir.name}.onnx"

        # XGBoost native ONNX export (may not be available)
        model.save_model(str(onnx_path), format='onnx')

        print(f"   ✅ Native ONNX export successful!")
        print(f"   💾 ONNX saved: {onnx_path.name}")
        onnx_success = True

    except Exception as e2:
        print(f"   ⚠️  Native ONNX export also failed: {e2}")
        print(f"   ℹ️  This is expected with XGBoost 3.1.1")

# Save comprehensive metadata
print(f"\n💾 Saving metadata...")
output_metadata = {
    'model_name': candidate_dir.name,
    'model_type': 'XGBoost Binary Classifier',
    'n_features': n_features,
    'input_name': 'float_input',
    'parent_model': 'ransomware_xgboost_production_v2',
    'xgboost_version': '3.1.1',
    'formats_available': [],
    'metrics': metadata.get('metrics', {}),
    'retraining_info': {
        'synthetic_data': True,
        'improvement_threshold': metadata.get('retraining_config', {}).get('improvement_threshold', 0.001),
        'dataset_info': metadata.get('dataset_info', {})
    }
}

# Track which formats are available
output_metadata['formats_available'].append({
    'format': 'XGBoost JSON',
    'file': json_output.name,
    'status': 'success',
    'note': 'Use with XGBoost C API - native format'
})

if onnx_success:
    output_metadata['formats_available'].append({
        'format': 'ONNX',
        'file': f"{candidate_dir.name}.onnx",
        'status': 'success',
        'opset_version': 12
    })
else:
    output_metadata['formats_available'].append({
        'format': 'ONNX',
        'status': 'failed',
        'note': 'XGBoost 3.x compatibility issue - use JSON format instead'
    })

metadata_output_path = output_dir / f"{candidate_dir.name}_metadata.json"
with open(metadata_output_path, 'w') as f:
    json.dump(output_metadata, f, indent=2)
print(f"   ✅ Metadata saved: {metadata_output_path.name}")

# Final summary
print("\n" + "=" * 70)
print("✅ CONVERSION COMPLETE")
print("=" * 70)
print(f"\n📁 Files created:")
print(f"   ✅ {json_output.name} (XGBoost JSON - READY)")
if onnx_success:
    print(f"   ✅ {candidate_dir.name}.onnx (ONNX - READY)")
else:
    print(f"   ⚠️  ONNX conversion failed (expected with XGBoost 3.x)")
print(f"   ✅ {metadata_output_path.name} (Metadata)")

print(f"\n🎯 Integration Options:")
print(f"   OPTION A (Recommended): Use XGBoost JSON with native C API")
print(f"   - File: {json_output.name}")
print(f"   - C++ API: #include <xgboost/c_api.h>")
print(f"   - Function: XGBoosterLoadModel()")

if onnx_success:
    print(f"\n   OPTION B: Use ONNX format")
    print(f"   - File: {candidate_dir.name}.onnx")
    print(f"   - C++ API: ONNXRuntime")
else:
    print(f"\n   ℹ️  ONNX not available due to XGBoost 3.x format changes")

print(f"\n📊 Model Info:")
print(f"   - Features: {n_features}")
print(f"   - F1 Score: {metadata.get('metrics', {}).get('f1', 'N/A')}")
print(f"   - Improvement: {metadata.get('metrics', {}).get('improvement', 'N/A')}")

print("=" * 70)