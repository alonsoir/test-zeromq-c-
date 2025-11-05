#!/usr/bin/env python3
"""
ANALIZADOR RÁPIDO - FEATURES CLAVE RANSOMWARE
"""

import pandas as pd
from pathlib import Path
import json

def quick_ransomware_features_analysis():
    """Análisis rápido enfocado en features de ransomware"""
    print("🔍 ANÁLISIS RÁPIDO - FEATURES RANSOMWARE")
    print("=" * 50)

    datasets = {
        "RanSMAP": Path("../datasets/RanSMAP"),
        "ransomware": Path("../datasets/ransomware"),
        "ugransome": Path("../datasets/ugransome")
    }

    # Features clave que buscamos en ransomware
    ransomware_keywords = [
        'memory', 'mem', 'encrypt', 'crypto', 'ransom', 'bitcoin', 'btc',
        'network', 'traffic', 'flow', 'packet', 'dns', 'registry', 'file',
        'process', 'thread', 'api', 'call', 'write', 'read', 'exec'
    ]

    summary = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\n📂 {dataset_name}:")

        if not dataset_path.exists():
            print("   ❌ No encontrado")
            continue

        csv_files = list(dataset_path.rglob("*.csv"))[:2]  # Solo primeros 2 archivos

        for csv_file in csv_files:
            print(f"   📄 {csv_file.name}...", end=" ")
            try:
                df = pd.read_csv(csv_file, nrows=3)  # Solo 3 filas

                # Buscar features relevantes
                relevant_cols = []
                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ransomware_keywords):
                        relevant_cols.append(col)

                print(f"✅ {len(relevant_cols)} features ransomware de {len(df.columns)} total")

                if relevant_cols:
                    print(f"      🎯 RELEVANTES: {relevant_cols}")

                summary[f"{dataset_name}/{csv_file.name}"] = {
                    'total_columns': len(df.columns),
                    'ransomware_features': relevant_cols,
                    'sample_values': {col: str(df[col].iloc[0])[:50] for col in relevant_cols[:3]} if relevant_cols else {}
                }

            except Exception as e:
                print(f"❌ Error: {e}")

    print("\n🎯 RESUMEN EJECUTIVO:")
    print("=" * 50)

    total_features = 0
    for file, info in summary.items():
        features_count = len(info['ransomware_features'])
        total_features += features_count
        print(f"📁 {file.split('/')[-1]}:")
        print(f"   {features_count} features ransomware")
        if info['ransomware_features']:
            print(f"   → {info['ransomware_features'][:3]}...")  # Primeras 3

    print(f"\n📊 TOTAL: {total_features} features ransomware identificadas")

    return summary

if __name__ == "__main__":
    summary = quick_ransomware_features_analysis()

    print("\n" + "=" * 50)
    print("🚀 PRÓXIMO PASO CRÍTICO:")
    print("POR FAVOR COMPARTE tu archivo .proto")
    print("=" * 50)
    print("Sin el .proto no podemos crear un modelo compatible")
    print("Pega aquí el contenido o comparte el archivo")