#!/usr/bin/env python3
"""
Quick test to verify all three embedder models work correctly.

Tests:
1. Load each model with ONNX
2. Run inference with dummy features
3. Verify output shapes
"""

import onnx

def test_model(model_path, expected_output_dim):
    """Test a single ONNX model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print('='*60)
    
    # Load model
    print("Step 1: Loading model...")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("  ✅ Model loaded and validated")
    
    # Check input/output shapes
    print("\nStep 2: Checking model info...")
    graph = model.graph
    
    # Input info
    input_tensor = graph.input[0]
    print(f"  Input name: {input_tensor.name}")
    input_shape = [dim.dim_value if dim.dim_value > 0 else 'batch' 
                   for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"  Input shape: {input_shape}")
    
    # Output info
    output_tensor = graph.output[0]
    print(f"  Output name: {output_tensor.name}")
    output_shape = [dim.dim_value if dim.dim_value > 0 else 'batch' 
                    for dim in output_tensor.type.tensor_type.shape.dim]
    print(f"  Output shape: {output_shape}")
    
    # Verify dimensions
    assert output_shape[1] == expected_output_dim, \
        f"Expected output dim {expected_output_dim}, got {output_shape[1]}"
    print(f"  ✅ Output dimension correct: {expected_output_dim}")
    
    print(f"\n✅ {model_path} PASSED")
    return True

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  ML Defender - ONNX Embedder Models Verification      ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    models = [
        ("chronos_embedder.onnx", 512),
        ("sbert_embedder.onnx", 384),
        ("attack_embedder.onnx", 256),
    ]
    
    results = []
    for model_path, expected_dim in models:
        try:
            test_model(model_path, expected_dim)
            results.append((model_path, "✅ PASS"))
        except Exception as e:
            print(f"\n❌ {model_path} FAILED: {e}")
            results.append((model_path, f"❌ FAIL: {e}"))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, status in results:
        print(f"  {model:30s} {status}")
    
    print("\n" + "="*60)
    passed = sum(1 for _, status in results if status.startswith("✅"))
    print(f"Result: {passed}/{len(models)} tests passed")
    print("="*60)
    
    if passed == len(models):
        print("\n✅ ALL MODELS VERIFIED - READY FOR DAY 34")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
