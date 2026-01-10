#!/usr/bin/env python3
"""
ML Defender - PCA to ONNX Converter
Convierte modelo PCA sklearn a formato ONNX para inferencia C++
Via Appia Quality - Production-ready model export
"""

import numpy as np
import argparse
import pickle
from pathlib import Path
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import time

def load_sklearn_models(model_dir):
    """Carga scaler y PCA desde pickle"""
    model_dir = Path(model_dir)

    scaler_path = model_dir / 'scaler.pkl'
    pca_path = model_dir / 'pca_model.pkl'

    print(f"[INFO] Loading models from: {model_dir}")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[INFO] ✅ Scaler loaded: {scaler_path}")

    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    print(f"[INFO] ✅ PCA model loaded: {pca_path}")

    return scaler, pca

def create_pipeline(scaler, pca):
    """Crea pipeline sklearn: scaler + PCA"""
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('scaler', scaler),
        ('pca', pca)
    ])

    print(f"[INFO] ✅ Pipeline created: StandardScaler → PCA")
    return pipeline

def convert_to_onnx(pipeline, n_features=102):
    """Convierte pipeline a ONNX"""
    print("\n" + "=" * 70)
    print("ONNX Conversion")
    print("=" * 70)

    # Define input type: (batch_size, 83 features)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    # Convert
    print(f"[INFO] Converting to ONNX...")
    print(f"       Input: float32[batch_size, {n_features}]")
    print(f"       Output: float32[batch_size, {pipeline.named_steps['pca'].n_components}]")

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=12
    )

    print(f"[INFO] ✅ ONNX conversion complete")

    return onnx_model

def validate_onnx(onnx_model, pipeline, n_samples=100):
    """Valida que ONNX produce mismos resultados que sklearn"""
    print("\n" + "=" * 70)
    print("ONNX Validation")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    X_test = np.random.randn(n_samples, 102).astype(np.float32)

    # sklearn prediction
    start = time.time()
    y_sklearn = pipeline.transform(X_test)
    sklearn_time = (time.time() - start) / n_samples * 1_000_000  # μs

    # ONNX prediction
    session = ort.InferenceSession(onnx_model.SerializeToString())
    input_name = session.get_inputs()[0].name

    start = time.time()
    y_onnx = session.run(None, {input_name: X_test})[0]
    onnx_time = (time.time() - start) / n_samples * 1_000_000  # μs

    # Compare
    max_diff = np.abs(y_sklearn - y_onnx).max()
    mean_diff = np.abs(y_sklearn - y_onnx).mean()

    print(f"[INFO] Validation results ({n_samples} samples):")
    print(f"       sklearn output shape: {y_sklearn.shape}")
    print(f"       ONNX output shape:    {y_onnx.shape}")
    print(f"       Max difference:  {max_diff:.2e}")
    print(f"       Mean difference: {mean_diff:.2e}")
    print(f"       sklearn time: {sklearn_time:.2f} μs/sample")
    print(f"       ONNX time:    {onnx_time:.2f} μs/sample")

    if max_diff < 1e-5:
        print(f"[INFO] ✅ Validation PASSED (max_diff < 1e-5)")
        return True, onnx_time
    else:
        print(f"[WARNING] ⚠️  Large difference detected!")
        return False, onnx_time

def save_onnx(onnx_model, output_path, metadata=None):
    """Guarda modelo ONNX con metadata"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    if metadata:
        onnx_model.metadata_props.extend([
            onnx.StringStringEntryProto(key=k, value=str(v))
            for k, v in metadata.items()
        ])

    # Save
    onnx.save(onnx_model, str(output_path))

    print(f"\n[INFO] ✅ ONNX model saved: {output_path}")
    print(f"       Size: {output_path.stat().st_size / 1024:.2f} KB")

    # Model info
    print(f"\n[INFO] ONNX Model Info:")
    print(f"       IR Version: {onnx_model.ir_version}")
    print(f"       Producer: {onnx_model.producer_name}")
    print(f"       Opset: {onnx_model.opset_import[0].version}")
    print(f"       Inputs:  {len(onnx_model.graph.input)}")
    print(f"       Outputs: {len(onnx_model.graph.output)}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert PCA sklearn model to ONNX format"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/vagrant/contrib/claude/pca_pipeline/models',
        help='Directory with sklearn models (scaler.pkl, pca_model.pkl)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/vagrant/contrib/claude/pca_pipeline/models/pca_embedder.onnx',
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation tests'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ML Defender - PCA to ONNX Converter")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # Load sklearn models
    scaler, pca = load_sklearn_models(args.input)

    # Create pipeline
    pipeline = create_pipeline(scaler, pca)

    # Convert to ONNX
    onnx_model = convert_to_onnx(pipeline)

    # Validate
    if args.validate:
        valid, onnx_time = validate_onnx(onnx_model, pipeline)
        if not valid:
            print("[ERROR] Validation failed!")
            return 1
    else:
        onnx_time = None

    # Metadata
    metadata = {
        'model_type': 'pca_embedder',
        'n_features_in': 102,
        'n_components': pca.n_components,
        'variance_explained': f"{pca.explained_variance_ratio_.sum():.4f}",
        'created_by': 'claude_pca_pipeline',
    }

    if onnx_time:
        metadata['inference_time_us'] = f"{onnx_time:.2f}"

    # Save
    save_onnx(onnx_model, args.output, metadata)

    print("\n" + "=" * 70)
    print("✅ ONNX CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Model ready for C++ inference: {args.output}")
    print(f"\nUsage in C++:")
    print(f"  Ort::Session session(env, \"{args.output}\", session_options);")

if __name__ == '__main__':
    main()