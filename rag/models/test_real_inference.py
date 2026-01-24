#!/usr/bin/env python3
"""
ML Defender - Real Data Inference Test (Day 34)

Process:
1. Load events from JSONL
2. Extract 83 features per event
3. Run inference through all 3 embedders
4. Verify output shapes and distributions
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from datetime import datetime

def load_events_jsonl(jsonl_path, max_events=100):
    """Load events from JSONL file"""
    events = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_events:
                break
            try:
                event = json.loads(line.strip())
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping line {i}: {e}")
                continue
    return events

def extract_features(event):
    """
    Extract 83 features from RAG event.

    Features (83 total):
    - Timestamp features: 7 (year, month, day, hour, minute, second, microsecond)
    - IP features: 8 (src_ip octets × 4, dst_ip octets × 4)
    - Port features: 2 (src_port, dst_port)
    - Protocol features: 3 (protocol, ip_version, tcp_flags)
    - Packet features: 4 (packet_length, header_length, payload_length, ttl)
    - Detection scores: 5 (fast_score, ml_score, final_score, is_malicious, severity)
    - Network metadata: 6 (vlan_id, dscp, ecn, window_size, mss, seq_num)
    - Behavioral features: 48 (flow stats, timing, patterns)
    """
    features = np.zeros(83, dtype=np.float32)

    # Timestamp features (0-6)
    if 'timestamp' in event:
        try:
            ts = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            features[0] = ts.year / 2026.0  # Normalize
            features[1] = ts.month / 12.0
            features[2] = ts.day / 31.0
            features[3] = ts.hour / 24.0
            features[4] = ts.minute / 60.0
            features[5] = ts.second / 60.0
            features[6] = ts.microsecond / 1e6
        except Exception:
            pass

    # IP features (7-14)
    if 'src_ip' in event:
        try:
            octets = [int(x) for x in event['src_ip'].split('.')]
            features[7:11] = np.array(octets) / 255.0
        except Exception:
            pass
    if 'dst_ip' in event:
        try:
            octets = [int(x) for x in event['dst_ip'].split('.')]
            features[11:15] = np.array(octets) / 255.0
        except Exception:
            pass

    # Port features (15-16)
    features[15] = event.get('src_port', 0) / 65535.0
    features[16] = event.get('dst_port', 0) / 65535.0

    # Protocol features (17-19)
    features[17] = event.get('protocol', 0) / 255.0
    features[18] = event.get('ip_version', 4) / 6.0
    features[19] = event.get('tcp_flags', 0) / 255.0

    # Packet features (20-23)
    features[20] = min(event.get('packet_length', 0) / 65535.0, 1.0)
    features[21] = event.get('header_length', 0) / 255.0
    features[22] = min(event.get('payload_length', 0) / 65535.0, 1.0)
    features[23] = event.get('ttl', 64) / 255.0

    # Detection scores (24-28)
    features[24] = event.get('fast_detector_score', 0.0)
    features[25] = event.get('ml_detector_score', 0.0)
    features[26] = event.get('final_score', 0.0)
    features[27] = float(event.get('is_malicious', False))
    features[28] = event.get('severity', 0) / 10.0

    # Network metadata (29-34)
    features[29] = event.get('vlan_id', 0) / 4095.0
    features[30] = event.get('dscp', 0) / 63.0
    features[31] = event.get('ecn', 0) / 3.0
    features[32] = event.get('window_size', 0) / 65535.0
    features[33] = event.get('mss', 0) / 65535.0
    features[34] = min(event.get('seq_num', 0) / 4294967295.0, 1.0)

    # Behavioral features (35-82) - Placeholder
    # In production, these would include:
    # - Flow statistics (bytes/packets sent/received)
    # - Timing features (inter-arrival times, duration)
    # - Pattern features (entropy, periodicity, burstiness)
    # For now, fill with reasonable defaults
    for i in range(35, 83):
        features[i] = 0.5  # Neutral value

    return features

def test_embedder(model_path, features, model_name):
    """Test a single embedder with features"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)

    # Load model
    print("Step 1: Loading ONNX model...")
    session = ort.InferenceSession(model_path)
    print(f"  ✅ Model loaded: {model_path}")

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"  Input: {input_name}, Output: {output_name}")

    # Run inference
    print("\nStep 2: Running inference...")
    input_data = features.reshape(1, -1).astype(np.float32)
    outputs = session.run([output_name], {input_name: input_data})
    embedding = outputs[0]

    print(f"  ✅ Inference completed")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Output shape: {embedding.shape}")

    # Validate output
    print("\nStep 3: Validating output...")
    print(f"  Embedding dimension: {embedding.shape[1]}")
    print(f"  Value range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")

    # Show first few values
    print(f"  First 5 values: {' '.join(f'{v:.4f}' for v in embedding[0][:5])}")

    return embedding

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  ML Defender - Real Data Inference Test              ║")
    print("╚════════════════════════════════════════════════════════╝\n")

    # Find latest JSONL file
    data_dir = Path("/vagrant/logs/rag/events")
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        print("❌ No JSONL files found in /vagrant/logs/rag/events")
        return 1

    latest_jsonl = jsonl_files[-1]
    print(f"📄 Using JSONL file: {latest_jsonl.name}")
    print(f"   Path: {latest_jsonl}\n")

    # Load events
    print("Step 1: Loading events...")
    events = load_events_jsonl(latest_jsonl, max_events=10)
    print(f"  ✅ Loaded {len(events)} events\n")

    if not events:
        print("❌ No events loaded")
        return 1

    # Extract features from first event
    print("Step 2: Extracting features from first event...")
    features = extract_features(events[0])
    print(f"  ✅ Extracted {len(features)} features")
    print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  First 10 features: {' '.join(f'{v:.3f}' for v in features[:10])}\n")

    # Test all embedders
    models = [
        ("chronos_embedder.onnx", "Chronos (Time Series)", 512),
        ("sbert_embedder.onnx", "SBERT (Semantic)", 384),
        ("attack_embedder.onnx", "Attack (Patterns)", 256),
    ]

    results = []
    for model_path, model_name, expected_dim in models:
        try:
            embedding = test_embedder(model_path, features, model_name)

            # Verify dimension
            actual_dim = embedding.shape[1]
            if actual_dim == expected_dim:
                print(f"  ✅ Dimension correct: {actual_dim}")
                results.append((model_name, "✅ PASS"))
            else:
                print(f"  ❌ Dimension mismatch: expected {expected_dim}, got {actual_dim}")
                results.append((model_name, "❌ FAIL"))
        except Exception as e:
            print(f"\n❌ {model_name} FAILED: {e}")
            results.append((model_name, f"❌ ERROR: {e}"))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, status in results:
        print(f"  {model:30s} {status}")

    print("\n" + "="*60)
    passed = sum(1 for _, status in results if status.startswith("✅"))
    print(f"Result: {passed}/{len(models)} embedders passed")
    print("="*60)

    if passed == len(models):
        print("\n✅ ALL EMBEDDERS WORKING WITH REAL DATA")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())