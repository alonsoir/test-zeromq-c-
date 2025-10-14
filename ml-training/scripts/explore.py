# ============================================================================
# Exploración de Datos - CIC-IDS-2017 y CIC-DDoS-2019
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. CIC-IDS-2017 - Exploración
# ============================================================================

print("=" * 80)
print("📊 CIC-IDS-2017 DATASET")
print("=" * 80)

ids_path = Path("../datasets/CIC-IDS-2017/MachineLearningCVE")
ids_files = sorted(ids_path.glob("*.csv"))

print(f"\n✅ Archivos encontrados: {len(ids_files)}")
for f in ids_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  - {f.name}: {size_mb:.1f} MB")

# Cargar Monday (más pequeño para explorar)
print("\n📋 Cargando Monday-WorkingHours...")
df_monday = pd.read_csv(ids_files[3], encoding='latin-1')

print(f"\n✅ Dimensiones: {df_monday.shape}")
print(f"   - Filas (flows): {df_monday.shape[0]:,}")
print(f"   - Columnas (features): {df_monday.shape[1]}")

# Ver primeras columnas
print("\n📋 Primeras 10 columnas:")
for i, col in enumerate(df_monday.columns[:10], 1):
    print(f"{i:3d}. {col}")

# Ver últimas columnas (incluyendo Label)
print("\n📋 Últimas 10 columnas:")
for i, col in enumerate(df_monday.columns[-10:], df_monday.shape[1]-9):
    print(f"{i:3d}. {col}")

# Información de tipos
print("\n📊 Tipos de datos:")
print(df_monday.dtypes.value_counts())

# Valores nulos
print("\n🔍 Valores nulos:")
null_counts = df_monday.isnull().sum()
if null_counts.sum() > 0:
    print(f"Total de valores nulos: {null_counts.sum():,}")
    print("\nColumnas con nulos:")
    print(null_counts[null_counts > 0].head(10))
else:
    print("✅ No hay valores nulos")

# Valores infinitos
print("\n🔍 Valores infinitos:")
numeric_cols = df_monday.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df_monday[numeric_cols]).sum()
total_inf = inf_counts.sum()
if total_inf > 0:
    print(f"Total de valores infinitos: {total_inf:,}")
    print("\nColumnas con infinitos:")
    print(inf_counts[inf_counts > 0].head(10))
else:
    print("✅ No hay valores infinitos")

# Distribución de Labels
print("\n🏷️ Distribución de Labels:")
label_col = ' Label'  # CIC-IDS-2017 tiene espacio antes
if label_col not in df_monday.columns:
    label_col = 'Label'

print(f"Columna de labels: '{label_col}'")
label_dist = df_monday[label_col].value_counts()
print(label_dist)
print(f"\nTotal de clases: {len(label_dist)}")
print(f"Porcentaje BENIGN: {label_dist.get('BENIGN', 0) / len(df_monday) * 100:.2f}%")

# ============================================================================
# 2. CIC-DDoS-2019 - Exploración
# ============================================================================

print("\n" + "=" * 80)
print("📊 CIC-DDoS-2019 DATASET")
print("=" * 80)

ddos_path = Path("../datasets/CIC-DDoS-2019")
ddos_files = sorted(ddos_path.rglob("*.csv"))

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
sample_file = [f for f in ddos_files if 'Syn.csv' in f.name and '01-12' in str(f.parent)][0]
df_ddos = pd.read_csv(sample_file, encoding='latin-1', nrows=10000)

print(f"\n✅ Archivo: {sample_file.name}")
print(f"✅ Dimensiones (sample): {df_ddos.shape}")
print(f"✅ Columnas: {df_ddos.shape[1]}")

# Ver columnas
print("\n📋 Primeras 10 columnas:")
for i, col in enumerate(df_ddos.columns[:10], 1):
    print(f"{i:3d}. {col}")

print("\n📋 Últimas 10 columnas:")
for i, col in enumerate(df_ddos.columns[-10:], df_ddos.shape[1]-9):
    print(f"{i:3d}. {col}")

# Labels
label_col_ddos = ' Label' if ' Label' in df_ddos.columns else 'Label'
print(f"\n🏷️ Labels en DDoS dataset:")
print(df_ddos[label_col_ddos].value_counts())

# ============================================================================
# 3. COMPARACIÓN DE FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("🔗 COMPARACIÓN DE FEATURES")
print("=" * 80)

ids_features = set(df_monday.columns)
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
    for feat in sorted(only_ids)[:5]:
        print(f"  - {feat}")
    if len(only_ids) > 5:
        print(f"  ... y {len(only_ids)-5} más")

if only_ddos:
    print("\n📋 Solo en DDoS-2019:")
    for feat in sorted(only_ddos)[:5]:
        print(f"  - {feat}")
    if len(only_ddos) > 5:
        print(f"  ... y {len(only_ddos)-5} más")

# ============================================================================
# 4. ESTADÍSTICAS BÁSICAS
# ============================================================================

print("\n" + "=" * 80)
print("📊 ESTADÍSTICAS BÁSICAS - CIC-IDS-2017")
print("=" * 80)

# Total de flows por archivo
print("\nTotal de flows por archivo:")
total_flows = 0
for csv_file in ids_files:
    df_temp = pd.read_csv(csv_file, encoding='latin-1')
    flows = len(df_temp)
    total_flows += flows
    print(f"  {csv_file.name:60s}: {flows:>10,} flows")

print(f"\n{'TOTAL':60s}: {total_flows:>10,} flows")

print("\n" + "=" * 80)
print("✅ EXPLORACIÓN COMPLETADA")
print("=" * 80)

