#!/usr/bin/env python3
"""
ML Defender - PCA Training Data Generator
Genera datos sintéticos con 83 features basadas en network_security.proto
Via Appia Quality - Data generation para PCA embedder training
"""

import numpy as np
import argparse
from pathlib import Path

# ============================================================================
# FEATURE SCHEMA - Basado en network_security.proto
# ============================================================================

FEATURE_SCHEMA = {
    # DDoSFeatures (10 features) - field 112
    'ddos_syn_ack_ratio': (0.0, 10.0),
    'ddos_packet_symmetry': (0.0, 1.0),
    'ddos_source_ip_dispersion': (0.0, 1.0),
    'ddos_protocol_anomaly_score': (0.0, 1.0),
    'ddos_packet_size_entropy': (0.0, 5.0),
    'ddos_traffic_amplification_factor': (0.0, 100.0),
    'ddos_flow_completion_rate': (0.0, 1.0),
    'ddos_geographical_concentration': (0.0, 1.0),
    'ddos_traffic_escalation_rate': (0.0, 1.0),
    'ddos_resource_saturation_score': (0.0, 1.0),

    # RansomwareEmbeddedFeatures (10 features) - field 113
    'ransomware_io_intensity': (0.0, 1.0),
    'ransomware_entropy': (0.0, 1.0),
    'ransomware_resource_usage': (0.0, 1.0),
    'ransomware_network_activity': (0.0, 1.0),
    'ransomware_file_operations': (0.0, 1.0),
    'ransomware_process_anomaly': (0.0, 1.0),
    'ransomware_temporal_pattern': (0.0, 1.0),
    'ransomware_access_frequency': (0.0, 1.0),
    'ransomware_data_volume': (0.0, 1.0),
    'ransomware_behavior_consistency': (0.0, 1.0),

    # TrafficFeatures (10 features) - field 114
    'traffic_packet_rate': (0.0, 1.0),
    'traffic_connection_rate': (0.0, 1.0),
    'traffic_tcp_udp_ratio': (0.0, 1.0),
    'traffic_avg_packet_size': (0.0, 1.0),
    'traffic_port_entropy': (0.0, 1.0),
    'traffic_flow_duration_std': (0.0, 1.0),
    'traffic_src_ip_entropy': (0.0, 1.0),
    'traffic_dst_ip_concentration': (0.0, 1.0),
    'traffic_protocol_variety': (0.0, 1.0),
    'traffic_temporal_consistency': (0.0, 1.0),

    # InternalFeatures (10 features) - field 115
    'internal_connection_rate': (0.0, 1.0),
    'internal_service_port_consistency': (0.0, 1.0),
    'internal_protocol_regularity': (0.0, 1.0),
    'internal_packet_size_consistency': (0.0, 1.0),
    'internal_connection_duration_std': (0.0, 1.0),
    'internal_lateral_movement_score': (0.0, 1.0),
    'internal_service_discovery_patterns': (0.0, 1.0),
    'internal_data_exfiltration_indicators': (0.0, 1.0),
    'internal_temporal_anomaly_score': (0.0, 1.0),
    'internal_access_pattern_entropy': (0.0, 1.0),

    # NetworkFeatures - Estadísticas adicionales (43 features)
    'flow_duration_microseconds': (0, 10_000_000),  # 0-10s
    'total_forward_packets': (1, 10000),
    'total_backward_packets': (0, 10000),
    'total_forward_bytes': (60, 1_500_000),
    'total_backward_bytes': (0, 1_500_000),

    # Length stats forward (4)
    'forward_packet_length_max': (60, 1500),
    'forward_packet_length_min': (60, 1500),
    'forward_packet_length_mean': (60.0, 1500.0),
    'forward_packet_length_std': (0.0, 500.0),

    # Length stats backward (4)
    'backward_packet_length_max': (0, 1500),
    'backward_packet_length_min': (0, 1500),
    'backward_packet_length_mean': (0.0, 1500.0),
    'backward_packet_length_std': (0.0, 500.0),

    # Velocidades y ratios (8)
    'flow_bytes_per_second': (0.0, 1_000_000.0),
    'flow_packets_per_second': (0.0, 10_000.0),
    'forward_packets_per_second': (0.0, 10_000.0),
    'backward_packets_per_second': (0.0, 10_000.0),
    'download_upload_ratio': (0.0, 100.0),
    'average_packet_size': (60.0, 1500.0),
    'average_forward_segment_size': (0.0, 1500.0),
    'average_backward_segment_size': (0.0, 1500.0),

    # Inter-arrival times flow (4)
    'flow_iat_mean': (0.0, 1_000_000.0),
    'flow_iat_std': (0.0, 500_000.0),
    'flow_iat_max': (0, 10_000_000),
    'flow_iat_min': (0, 10_000_000),

    # Inter-arrival times forward (5)
    'forward_iat_total': (0.0, 100_000_000.0),
    'forward_iat_mean': (0.0, 1_000_000.0),
    'forward_iat_std': (0.0, 500_000.0),
    'forward_iat_max': (0, 10_000_000),
    'forward_iat_min': (0, 10_000_000),

    # Inter-arrival times backward (5)
    'backward_iat_total': (0.0, 100_000_000.0),
    'backward_iat_mean': (0.0, 1_000_000.0),
    'backward_iat_std': (0.0, 500_000.0),
    'backward_iat_max': (0, 10_000_000),
    'backward_iat_min': (0, 10_000_000),

    # TCP flags (12)
    'fin_flag_count': (0, 100),
    'syn_flag_count': (0, 100),
    'rst_flag_count': (0, 100),
    'psh_flag_count': (0, 100),
    'ack_flag_count': (0, 10000),
    'urg_flag_count': (0, 10),
    'cwe_flag_count': (0, 10),
    'ece_flag_count': (0, 10),
    'forward_psh_flags': (0, 100),
    'backward_psh_flags': (0, 100),
    'forward_urg_flags': (0, 10),
    'backward_urg_flags': (0, 10),

    # Headers y bulk (8)
    'forward_header_length': (20.0, 60.0),
    'backward_header_length': (0.0, 60.0),
    'forward_average_bytes_bulk': (0.0, 100_000.0),
    'forward_average_packets_bulk': (0.0, 1000.0),
    'forward_average_bulk_rate': (0.0, 1_000_000.0),
    'backward_average_bytes_bulk': (0.0, 100_000.0),
    'backward_average_packets_bulk': (0.0, 1000.0),
    'backward_average_bulk_rate': (0.0, 1_000_000.0),

    # Packet length stats (5)
    'minimum_packet_length': (60, 1500),
    'maximum_packet_length': (60, 1500),
    'packet_length_mean': (60.0, 1500.0),
    'packet_length_std': (0.0, 500.0),
    'packet_length_variance': (0.0, 250_000.0),

    # Active/Idle (2)
    'active_mean': (0.0, 10_000_000.0),
    'idle_mean': (0.0, 10_000_000.0),
}

assert len(FEATURE_SCHEMA) == 102, f"Expected 83 features, got {len(FEATURE_SCHEMA)}"

def generate_synthetic_data(n_samples=100_000, seed=42):
    """
    Genera datos sintéticos con 83 features

    Args:
        n_samples: Número de eventos a generar
        seed: Semilla para reproducibilidad

    Returns:
        np.array de shape (n_samples, 83)
    """
    print(f"[INFO] Generando {n_samples:,} eventos sintéticos con {len(FEATURE_SCHEMA)} features...")

    np.random.seed(seed)
    data = np.zeros((n_samples, len(FEATURE_SCHEMA)))

    for idx, (feature_name, (min_val, max_val)) in enumerate(FEATURE_SCHEMA.items()):
        # Generar valores uniformes en el rango especificado
        data[:, idx] = np.random.uniform(min_val, max_val, n_samples)

        if idx % 10 == 0:
            print(f"  [{idx+1}/{len(FEATURE_SCHEMA)}] Generated {feature_name}")

    print(f"[INFO] ✅ Datos generados: shape={data.shape}")
    return data

def save_training_data(data, output_path):
    """Guarda datos en formato .npz comprimido"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = list(FEATURE_SCHEMA.keys())

    np.savez_compressed(
        output_path,
        data=data,
        feature_names=feature_names,
        n_features=len(feature_names)
    )

    print(f"[INFO] ✅ Datos guardados en: {output_path}")
    print(f"       Shape: {data.shape}")
    print(f"       Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for PCA embedder"
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=100_000,
        help='Number of samples to generate (default: 100,000)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/vagrant/ml_defender/pca_training/training_data.npz',
        help='Output path for training data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ML Defender - PCA Training Data Generator")
    print("=" * 70)
    print(f"Samples: {args.samples:,}")
    print(f"Features: {len(FEATURE_SCHEMA)}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # Generate data
    data = generate_synthetic_data(args.samples, args.seed)

    # Save
    save_training_data(data, args.output)

    print("\n✅ Training data generation complete!")
    print(f"Next step: python3 train_pca_embedder.py --input {args.output}")

if __name__ == '__main__':
    main()