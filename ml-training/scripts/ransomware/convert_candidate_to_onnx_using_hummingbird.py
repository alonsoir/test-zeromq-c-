#!/usr/bin/env python3
"""
ONNX Conversion Workaround for XGBoost 3.x using Hummingbird
"""
import joblib
from pathlib import Path
import numpy as np

# Rutas
candidate_dir = Path("model_candidates/ransomware_xgboost_candidate_v2_20251106_095308")
output_dir = Path("../../../ml-detector/models/production/level3/ransomware")

print("🔄 Loading model for Hummingbird conversion...")
model = joblib.load(candidate_dir / f"{candidate_dir.name}.pkl")

try:
    from hummingbird.ml import convert

    # Convert using Hummingbird
    hb_model = convert(model, 'onnx',
                       extra_config={"n_features": 45,
                                     "test_input": np.random.rand(1, 45).astype(np.float32)})

    # Save ONNX
    onnx_path = output_dir / f"{candidate_dir.name}.onnx"
    hb_model.save_model(str(onnx_path))

    print(f"✅ ONNX via Hummingbird: {onnx_path}")

except ImportError:
    print("❌ Hummingbird not available - install with: pip install hummingbird-ml")
except Exception as e:
    print(f"❌ Hummingbird conversion failed: {e}")