#!/usr/bin/env python3
"""
Exploración de Datos - CIC-IDS-2017 y CIC-DDoS-2019
Script para análisis inicial de los datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

def explore_cic_ids_2017():
    """Explorar dataset CIC-IDS-2017"""
    print("=" * 80)
    print("📊 CIC-IDS-2017 DATASET")
    print("=" * 80)

    ids_path = Path("datasets/CIC-IDS-2017/MachineLearningCVE")
    
    if not ids_path.exists():
        print(f"❌ Error: No se encuentra {ids_path}")
        print(f"   Ruta absoluta buscada: {ids_path.absolute()}")
        return None
    
    ids_files = sorted(ids_path.glob("*.csv"))

    if not ids_files:
        print(f"❌ Error: No se encontraron archivos CSV en {ids_path}")
        return None

    print(f"\n✅ Archivos encontrados: {len(ids_files)}")
    for i, f in enumerate(ids_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {f.name}: {size_mb:.1f} MB")

    # Buscar Monday o usar el primer archivo disponible
    monday_files = [f for f in ids_files if 'Monday' in f.name]
    
    if monday_files:
        sample_file = monday_files[0]
        print(f"\n📋 Cargando {sample_file.name}...")
    else:
        sample_file = ids_files[0]
        print(f"\n📋 Monday no encontrado, cargando {sample_file.name}...")

    try:
        df_monday = pd.read_csv(sample_file, encoding='latin-1')
    except Exception as e:
        print(f"❌ Error al cargar archivo: {e}")
        return None

    print(f"\n✅ Dimensiones: {df_monday.shape}")
    print(f"   - Filas (flows): {df_monday.shape[0]:,}")
    print(f"   - Columnas (features): {df_monday.shape[1]}")

    # Ver primeras columnas
    print("\n📋 Primeras 10 columnas:")
    for i, col in enumerate(df_monday.columns[:10], 1):
        print(f"{i:3d}. {col}")

    # Ver últimas columnas (incluyendo Label)
    print("\n📋 Últimas 10 columnas:")
    start_idx = max(1, df_monday.shape[1] - 9)
    for i, col in enumerate(df_monday.columns[-10:], start_idx):
        print(f"{i:3d}. {col}")

    # Información de tipos
    print("\n📊 Tipos de datos:")
    type_counts = df_monday.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"  {dtype}: {count}")

    # Valores nulos
    print("\n🔍 Valores nulos:")
    null_counts = df_monday.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Total de valores nulos: {null_counts.sum():,}")
        print("\nColumnas con nulos (top 10):")
        top_nulls = null_counts[null_counts > 0].sort_values(ascending=False).head(10)
        for col, count in top_nulls.items():
            print(f"  {col}: {count:,} ({count/len(df_monday)*100:.2f}%)")
    else:
        print("✅ No hay valores nulos")

    # Valores infinitos
    print("\n🔍 Valores infinitos:")
    numeric_cols = df_monday.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df_monday[numeric_cols]).sum()
    total_inf = inf_counts.sum()
    if total_inf > 0:
        print(f"Total de valores infinitos: {total_inf:,}")
        print("\nColumnas con infinitos (top 10):")
        top_infs = inf_counts[inf_counts > 0].sort_values(ascending=False).head(10)
        for col, count in top_infs.items():
            print(f"  {col}: {count:,} ({count/len(df_monday)*100:.2f}%)")
    else:
        print("✅ No hay valores infinitos")

    # Distribución de Labels
    print("\n🏷️ Distribución de Labels:")
    
    # Buscar columna de label (puede tener espacio o no)
    label_cols = [col for col in df_monday.columns if 'label' in col.lower()]
    
    if not label_cols:
        print("❌ No se encontró columna de labels")
        return df_monday
    
    label_col = label_cols[0]
    print(f"Columna de labels: '{label_col}'")
    
    label_dist = df_monday[label_col].value_counts()
    print(f"\nTotal de clases: {len(label_dist)}")
    print("\nDistribución:")
    for label, count in label_dist.items():
        pct = count / len(df_monday) * 100
        print(f"  {label:30s}: {count:>10,} ({pct:>6.2f}%)")

    # Total de flows por archivo
    print("\n📊 Total de flows por archivo:")
    total_flows = 0
    for csv_file in ids_files:
        try:
            # Leer solo primera línea para contar
            df_temp = pd.read_csv(csv_file, encoding='latin-1', nrows=1)
            # Ahora contar líneas de forma eficiente
            with open(csv_file) as f:
                flows = sum(1 for _ in f) - 1  # -1 para el header
            total_flows += flows
            print(f"  {csv_file.name:60s}: {flows:>10,} flows")
        except Exception as e:
            print(f"  {csv_file.name:60s}: Error al contar ({e})")

    print(f"\n{'TOTAL':60s}: {total_flows:>10,} flows")
    
    return df_monday


def explore_cic_ddos_2019():
    """Explorar dataset CIC-DDoS-2019"""
    print("\n" + "=" * 80)
    print("📊 CIC-DDoS-2019 DATASET")
    print("=" * 80)

    ddos_path = Path("datasets/CIC-DDoS-2019")
    
    if not ddos_path.exists():
        print(f"❌ Error: No se encuentra {ddos_path}")
        print(f"   Ruta absoluta buscada: {ddos_path.absolute()}")
        return None
    
    ddos_files = sorted(ddos_path.rglob("*.csv"))

    if not ddos_files:
        print(f"❌ Error: No se encontraron archivos CSV en {ddos_path}")
        return None

    print(f"\n✅ Archivos encontrados: {len(ddos_files)}")

    # Agrupar por carpeta
    from collections import defaultdict
    by_folder = defaultdict(list)
    for f in ddos_files:
        folder = f.parent.name
        size_mb = f.stat().st_size / (1024 * 1024)
        by_folder[folder].append((f.name, size_mb))

    for folder, files in sorted(by_folder.items()):
        print(f"\n📁 {folder}/:")
        for name, size in files:
            print(f"  - {name}: {size:.1f} MB")

    # Cargar un archivo pequeño de ejemplo
    print("\n📋 Cargando archivo de ejemplo (primeras 10,000 filas)...")
    
    # Buscar archivo Syn o usar el primero
    syn_files = [f for f in ddos_files if 'Syn.csv' in f.name]
    
    if syn_files:
        sample_file = syn_files[0]
    else:
        sample_file = ddos_files[0]
    
    print(f"Archivo seleccionado: {sample_file.name}")
    
    try:
        df_ddos = pd.read_csv(sample_file, encoding='latin-1', nrows=10000)
    except Exception as e:
        print(f"❌ Error al cargar archivo: {e}")
        return None

    print(f"\n✅ Dimensiones (sample): {df_ddos.shape}")
    print(f"✅ Columnas: {df_ddos.shape[1]}")

    # Ver columnas
    print("\n📋 Primeras 10 columnas:")
    for i, col in enumerate(df_ddos.columns[:10], 1):
        print(f"{i:3d}. {col}")

    print("\n📋 Últimas 10 columnas:")
    start_idx = max(1, df_ddos.shape[1] - 9)
    for i, col in enumerate(df_ddos.columns[-10:], start_idx):
        print(f"{i:3d}. {col}")

    # Labels
    label_cols = [col for col in df_ddos.columns if 'label' in col.lower()]
    
    if label_cols:
        label_col_ddos = label_cols[0]
        print(f"\n🏷️ Labels en DDoS dataset (columna: '{label_col_ddos}'):")
        label_dist = df_ddos[label_col_ddos].value_counts()
        for label, count in label_dist.items():
            pct = count / len(df_ddos) * 100
            print(f"  {label:30s}: {count:>6,} ({pct:>6.2f}%)")
    
    return df_ddos


def compare_features(df_ids, df_ddos):
    """Comparar features entre ambos datasets"""
    if df_ids is None or df_ddos is None:
        print("\n⚠️ No se puede comparar: uno o ambos datasets no se cargaron")
        return
    
    print("\n" + "=" * 80)
    print("🔗 COMPARACIÓN DE FEATURES")
    print("=" * 80)

    ids_features = set(df_ids.columns)
    ddos_features = set(df_ddos.columns)

    print(f"\nCIC-IDS-2017 features: {len(ids_features)}")
    print(f"CIC-DDoS-2019 features: {len(ddos_features)}")

    common = ids_features & ddos_features
    only_ids = ids_features - ddos_features
    only_ddos = ddos_features - ids_features

    print(f"\nFeatures comunes: {len(common)}")
    print(f"Solo en IDS-2017: {len(only_ids)}")
    print(f"Solo en DDoS-2019: {len(only_ddos)}")

    if only_ids:
        print("\n📋 Solo en IDS-2017:")
        for feat in sorted(only_ids)[:10]:
            print(f"  - {feat}")
        if len(only_ids) > 10:
            print(f"  ... y {len(only_ids)-10} más")

    if only_ddos:
        print("\n📋 Solo en DDoS-2019:")
        for feat in sorted(only_ddos)[:10]:
            print(f"  - {feat}")
        if len(only_ddos) > 10:
            print(f"  ... y {len(only_ddos)-10} más")


def main():
    """Función principal"""
    print("=" * 80)
    print("🔍 EXPLORACIÓN DE DATASETS CIC")
    print("=" * 80)
    
    # Mostrar ruta actual
    print(f"\nDirectorio de trabajo: {Path.cwd()}")
    
    # Explorar ambos datasets
    df_ids = explore_cic_ids_2017()
    df_ddos = explore_cic_ddos_2019()
    
    # Comparar si ambos se cargaron correctamente
    compare_features(df_ids, df_ddos)
    
    print("\n" + "=" * 80)
    print("✅ EXPLORACIÓN COMPLETADA")
    print("=" * 80)
    
    # Retornar código de salida
    if df_ids is None or df_ddos is None:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
