#!/usr/bin/env python3
"""
ML Defender - Batch Processing Test (Day 34)

Tests:
- Load 100 events
- Generate embeddings for all
- Measure throughput
- Check consistency
"""

import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Import from test_real_inference.py
import json
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
    """Extract 83 features from RAG event (same as test_real_inference.py)"""
    features = np.zeros(83, dtype=np.float32)

    # Timestamp features (0-6)
    if 'timestamp' in event:
        try:
            ts = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            features[0] = ts.year / 2026.0
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

    # Behavioral features (35-82)
    for i in range(35, 83):
        features[i] = 0.5

    return features

def batch_process(model_path, features_batch, batch_size=10):
    """Process features in batches"""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    embeddings = []
    num_batches = (len(features_batch) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(features_batch))
        batch = features_batch[start_idx:end_idx]

        # Pad batch if needed
        if len(batch) < batch_size:
            padding = np.zeros((batch_size - len(batch), 83), dtype=np.float32)
            batch = np.vstack([batch, padding])

        # Run inference
        outputs = session.run([output_name], {input_name: batch.astype(np.float32)})
        embeddings.append(outputs[0][:end_idx - start_idx])

    return np.vstack(embeddings) if embeddings else np.array([])

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  ML Defender - Batch Processing Test                 ║")
    print("╚════════════════════════════════════════════════════════╝\n")

    # Load events
    data_dir = Path("/vagrant/logs/rag/events")
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        print("❌ No JSONL files found in /vagrant/logs/rag/events")
        return 1

    latest_jsonl = jsonl_files[-1]

    print(f"📄 Loading events from: {latest_jsonl.name}\n")
    events = load_events_jsonl(latest_jsonl, max_events=100)
    print(f"  ✅ Loaded {len(events)} events\n")

    if not events:
        print("❌ No events loaded")
        return 1

    # Extract all features
    print("Extracting features...")
    features_list = []
    for i, event in enumerate(events):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(events)}")
        features = extract_features(event)
        features_list.append(features)

    features_batch = np.array(features_list)
    print(f"  ✅ Extracted features: {features_batch.shape}\n")

    # Test each embedder
    models = [
        ("chronos_embedder.onnx", "Chronos", 512),
        ("sbert_embedder.onnx", "SBERT", 384),
        ("attack_embedder.onnx", "Attack", 256),
    ]

    results = []

    for model_path, name, expected_dim in models:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)

        try:
            # Warm-up
            _ = batch_process(model_path, features_batch[:10], batch_size=10)

            # Benchmark
            start = time.time()
            embeddings = batch_process(model_path, features_batch, batch_size=10)
            elapsed = time.time() - start

            throughput = len(features_batch) / elapsed

            print(f"  ✅ Processed {len(features_batch)} events")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {throughput:.1f} events/sec")
            print(f"  Embedding shape: {embeddings.shape}")
            print(f"  Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")

            # Verify dimension
            if embeddings.shape[1] == expected_dim:
                print(f"  ✅ Dimension correct: {expected_dim}")
                results.append((name, "PASS", throughput))
            else:
                print(f"  ❌ Dimension mismatch: expected {expected_dim}, got {embeddings.shape[1]}")
                results.append((name, "FAIL", 0))

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((name, "ERROR", 0))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status, throughput in results:
        if status == "PASS":
            print(f"  {name:15s} ✅ {status:6s} {throughput:8.1f} events/sec")
        else:
            print(f"  {name:15s} ❌ {status}")

    print("\n╔════════════════════════════════════════════════════════╗")
    passed = sum(1 for _, status, _ in results if status == "PASS")
    if passed == len(models):
        print("║  BATCH PROCESSING COMPLETE ✅                          ║")
    else:
        print("║  SOME TESTS FAILED ❌                                  ║")
    print("╚════════════════════════════════════════════════════════╝")

    return 0 if passed == len(models) else 1

if __name__ == "__main__":
    exit(main())