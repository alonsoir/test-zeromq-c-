#!/usr/bin/env python3
"""
BUSCADOR DE DATASETS RANSOMWARE DE RED
- Encuentra datasets que coincidan con las features del .proto
"""

import pandas as pd
from pathlib import Path
import json

def find_network_ransomware_datasets():
    """Busca datasets de ransomware que tengan features de red"""
    print("🔍 BUSCANDO DATASETS RANSOMWARE DE RED")
    print("=" * 60)

    # Features de red que necesitamos del .proto
    network_features_needed = [
        'dns', 'query', 'entropy', 'ip', 'connection', 'port', 'tls',
        'cert', 'http', 'smb', 'rdp', 'upload', 'download', 'flow',
        'packet', 'byte', 'protocol', 'tcp', 'syn', 'ack', 'rst'
    ]

    datasets_path = Path("../datasets")
    compatible_datasets = {}

    # Revisar TODOS los datasets disponibles
    for dataset_dir in datasets_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        print(f"\n📂 Revisando: {dataset_dir.name}")

        csv_files = list(dataset_dir.rglob("*.csv"))
        if not csv_files:
            continue

        for csv_file in csv_files[:3]:  # Revisar primeros 3 archivos
            try:
                # Leer solo encabezados para eficiencia
                df_sample = pd.read_csv(csv_file, nrows=2)
                columns_lower = [str(col).lower() for col in df_sample.columns]

                # Buscar coincidencias con features de red
                matching_features = []
                for feature in network_features_needed:
                    matches = [col for col in columns_lower if feature in col]
                    matching_features.extend(matches)

                if matching_features:
                    print(f"   ✅ {csv_file.name}: {len(matching_features)} features red")
                    print(f"      → {matching_features[:5]}...")

                    compatible_datasets[f"{dataset_dir.name}/{csv_file.name}"] = {
                        'matching_features': matching_features,
                        'total_columns': len(df_sample.columns),
                        'sample_size': len(pd.read_csv(csv_file, nrows=1))
                    }

            except Exception as e:
                continue

    # Análisis de compatibilidad
    print(f"\n🎯 RESUMEN DE COMPATIBILIDAD:")
    print("=" * 60)

    if compatible_datasets:
        for dataset, info in compatible_datasets.items():
            print(f"📁 {dataset}:")
            print(f"   {len(info['matching_features'])} features de red")
            print(f"   {info['total_columns']} columnas totales")

        # Guardar análisis
        with open("network_datasets_analysis.json", "w") as f:
            json.dump(compatible_datasets, f, indent=2)

        print(f"\n💾 Análisis guardado en: network_datasets_analysis.json")

    else:
        print("❌ No se encontraron datasets con features de red")
        print("\n🔍 BUSQUEDA ALTERNATIVA: Revisando nombres de archivos...")

        # Buscar por nombres que suenen a red/ransomware
        potential_files = []
        for pattern in ["*flow*", "*packet*", "*dns*", "*network*", "*ransom*", "*malware*"]:
            potential_files.extend(datasets_path.rglob(f"*{pattern}*.csv"))

        for file_path in potential_files[:10]:
            print(f"   📄 {file_path.relative_to(datasets_path)}")

    return compatible_datasets

def check_ugransome_for_network_features():
    """Revisa específicamente UGRansome que menciona Netflow_Bytes"""
    print(f"\n🔍 ANALIZANDO UGRANSOME EN PROFUNDIDAD")

    ugransome_path = Path("../datasets/ugransome")
    if not ugransome_path.exists():
        print("❌ UGRansome no encontrado")
        return

    for csv_file in ugransome_path.rglob("*.csv"):
        try:
            print(f"\n📊 Analizando: {csv_file.name}")
            df = pd.read_csv(csv_file, nrows=10)

            print(f"   Columnas: {list(df.columns)}")
            print(f"   Forma: {df.shape}")

            # Verificar si tiene columnas de red
            network_cols = [col for col in df.columns if any(x in str(col).lower() for x in
                                                             ['flow', 'packet', 'byte', 'ip', 'port', 'protocol', 'dns'])]

            if network_cols:
                print(f"   ✅ Features de red: {network_cols}")

                # Mostrar estadísticas básicas
                for col in network_cols[:3]:
                    if col in df.columns:
                        print(f"      {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 BUSCADOR DE DATASETS RANSOMWARE DE RED")
    print("OBJETIVO: Encontrar datos REALES que coincidan con el .proto")
    print("=" * 60)

    # 1. Buscar datasets compatibles
    compatible = find_network_ransomware_datasets()

    # 2. Analizar UGRansome en profundidad
    check_ugransome_for_network_features()

    print(f"\n🎯 PRÓXIMOS PASOS:")
    if compatible:
        print("1. ✅ Tenemos datasets con features de red")
        print("2. 🔄 Podemos crear modelo con datos REALES")
        print("3. 🎯 Alineado con .proto")
    else:
        print("1. ❌ No hay datasets de red disponibles")
        print("2. 💡 Alternativas:")
        print("   - Usar UGRansome (Netflow_Bytes) como base")
        print("   - Enriquecer con datos de otros datasets")
        print("   - Crear transformación de datos de host a red")