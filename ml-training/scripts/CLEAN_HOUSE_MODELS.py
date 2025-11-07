#!/usr/bin/env python3
"""
LIMPIADOR DE MODELOS DE BAJA CALIDAD
Elimina automáticamente los modelos no aprovechables identificados
"""

import pandas as pd
from pathlib import Path
import shutil
import json
from datetime import datetime

# =====================================================================
# CONFIG - LIMPIADOR DE MODELOS
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
OUTPUT_PATH = BASE_PATH / "outputs"
MODELS_PATH = OUTPUT_PATH / "models"
BACKUP_PATH = OUTPUT_PATH / "model_backup_before_cleanup"

class ModelCleaner:
    def __init__(self):
        self.models_to_keep = []
        self.models_to_delete = []
        self.deletion_summary = []

    def load_analysis_results(self):
        """Carga los resultados del análisis anterior"""
        analysis_path = OUTPUT_PATH / "model_analysis_report_final" / "model_analysis.csv"

        if not analysis_path.exists():
            print("❌ No se encontró el reporte de análisis")
            return None

        df = pd.read_csv(analysis_path)
        return df

    def classify_models(self, df):
        """Clasifica modelos en mantener vs eliminar"""
        print("📊 CLASIFICANDO MODELOS...")

        # Modelos para mantener (calidad >= 80)
        self.models_to_keep = df[
            (df['quality_score'] >= 80) &
            (df['completeness_score'] >= 50)
            ][['model_name', 'directory', 'quality_score']].to_dict('records')

        # Modelos para eliminar (calidad < 80 o incompletos)
        self.models_to_delete = df[
            (df['quality_score'] < 80) |
            (df['completeness_score'] < 50)
            ][['model_name', 'directory', 'quality_score', 'completeness_score']].to_dict('records')

        print(f"✅ Modelos para mantener: {len(self.models_to_keep)}")
        print(f"🗑️  Modelos para eliminar: {len(self.models_to_delete)}")

        return self.models_to_keep, self.models_to_delete

    def create_backup(self):
        """Crea backup de todos los modelos antes de eliminar"""
        print("\n💾 CREANDO BACKUP...")

        if BACKUP_PATH.exists():
            shutil.rmtree(BACKUP_PATH)

        BACKUP_PATH.mkdir(parents=True, exist_ok=True)

        # Copiar todo el directorio de modelos
        if MODELS_PATH.exists():
            shutil.copytree(MODELS_PATH, BACKUP_PATH / "models")
            print(f"✅ Backup creado en: {BACKUP_PATH}")
        else:
            print("❌ No se pudo crear backup - directorio models no existe")

    def find_model_files(self, model_info):
        """Encuentra todos los archivos relacionados con un modelo"""
        model_dir = MODELS_PATH / model_info['directory']
        base_name = model_info['model_name']

        if not model_dir.exists():
            return []

        # Buscar todos los archivos relacionados
        related_files = []

        # Archivos principales
        patterns = [
            f"{base_name}.pkl",
            f"{base_name}.joblib",
            f"{base_name}_scaler.pkl",
            f"{base_name}_metadata.json",
            f"{base_name}_meta.json",
            f"{base_name}_features.json",
            f"{base_name}_feature.json"
        ]

        for pattern in patterns:
            files = list(model_dir.glob(pattern))
            related_files.extend(files)

        # También buscar en el directorio completo si es necesario
        all_files = list(model_dir.glob(f"*{base_name}*"))
        related_files.extend([f for f in all_files if f not in related_files])

        return list(set(related_files))  # Remover duplicados

    def delete_model_files(self, model_info):
        """Elimina todos los archivos de un modelo"""
        files_to_delete = self.find_model_files(model_info)
        deleted_files = []

        for file_path in files_to_delete:
            try:
                if file_path.exists():
                    file_path.unlink()
                    deleted_files.append(file_path.name)
            except Exception as e:
                print(f"   ❌ Error eliminando {file_path.name}: {e}")

        return deleted_files

    def delete_empty_directories(self):
        """Elimina directorios vacíos después de la limpieza"""
        print("\n🧹 LIMPIANDO DIRECTORIOS VACÍOS...")

        empty_dirs = []
        for model_dir in MODELS_PATH.iterdir():
            if model_dir.is_dir():
                # Verificar si el directorio está vacío
                if not any(model_dir.iterdir()):
                    try:
                        model_dir.rmdir()
                        empty_dirs.append(model_dir.name)
                    except Exception as e:
                        print(f"   ❌ Error eliminando directorio {model_dir.name}: {e}")

        if empty_dirs:
            print(f"✅ Directorios vacíos eliminados: {len(empty_dirs)}")
            for dir_name in empty_dirs:
                print(f"   - {dir_name}")

    def clean_low_quality_models(self):
        """Ejecuta la limpieza completa"""
        print("🚀 INICIANDO LIMPIEZA DE MODELOS...")
        print("=" * 60)

        # 1. Cargar análisis
        df = self.load_analysis_results()
        if df is None:
            return

        # 2. Clasificar modelos
        self.classify_models(df)

        # 3. Crear backup
        self.create_backup()

        # 4. Mostrar modelos a eliminar
        print(f"\n🗑️  ELIMINANDO {len(self.models_to_delete)} MODELOS...")
        print("=" * 60)

        total_deleted = 0
        total_files_deleted = 0

        for i, model in enumerate(self.models_to_delete, 1):
            print(f"\n{i}/{len(self.models_to_delete)} Eliminando: {model['directory']}/{model['model_name']}")
            print(f"   📊 Calidad: {model['quality_score']:.1f}/100, Completitud: {model['completeness_score']:.1f}%")

            deleted_files = self.delete_model_files(model)

            if deleted_files:
                print(f"   ✅ Eliminados: {len(deleted_files)} archivos")
                for file_name in deleted_files:
                    print(f"      - {file_name}")

                total_deleted += 1
                total_files_deleted += len(deleted_files)

                self.deletion_summary.append({
                    'model': f"{model['directory']}/{model['model_name']}",
                    'quality_score': model['quality_score'],
                    'files_deleted': len(deleted_files),
                    'file_names': deleted_files
                })
            else:
                print(f"   ⚠️  No se encontraron archivos para eliminar")

        # 5. Limpiar directorios vacíos
        self.delete_empty_directories()

        # 6. Generar reporte de limpieza
        self.generate_cleanup_report(total_deleted, total_files_deleted)

        return total_deleted, total_files_deleted

    def generate_cleanup_report(self, total_models, total_files):
        """Genera reporte de la limpieza realizada"""
        print(f"\n📋 GENERANDO REPORTE DE LIMPIEZA...")

        report = {
            'cleanup_date': datetime.now().isoformat(),
            'summary': {
                'models_deleted': total_models,
                'files_deleted': total_files,
                'models_kept': len(self.models_to_keep),
                'backup_location': str(BACKUP_PATH)
            },
            'models_kept': self.models_to_keep,
            'models_deleted': self.deletion_summary
        }

        # Guardar reporte JSON
        cleanup_report_path = OUTPUT_PATH / "cleanup_report.json"
        with open(cleanup_report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generar reporte legible
        self.generate_human_readable_report(report)

        print(f"✅ Reporte de limpieza guardado en: {cleanup_report_path}")

    def generate_human_readable_report(self, report):
        """Genera reporte legible para humanos"""
        md = f"""# 🧹 REPORTE DE LIMPIEZA DE MODELOS ML

**Fecha de limpieza:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 RESUMEN

| Métrica | Valor |
|---------|-------|
| 🗑️  Modelos eliminados | {report['summary']['models_deleted']} |
| 📄 Archivos eliminados | {report['summary']['files_deleted']} |
| 💾 Modelos conservados | {report['summary']['models_kept']} |
| 💽 Backup | {report['summary']['backup_location']} |

## 🚀 MODELOS CONSERVADOS (Producción)

"""

        if report['models_kept']:
            md += "| Modelo | Calidad |\n"
            md += "|--------|---------|\n"
            for model in report['models_kept']:
                md += f"| {model['model_name']} | {model['quality_score']:.1f}/100 |\n"
        else:
            md += "❌ No hay modelos conservados\n"

        md += "\n## 🗑️ MODELOS ELIMINADOS\n\n"

        if report['models_deleted']:
            for model in report['models_deleted']:
                md += f"### ❌ {model['model']}\n"
                md += f"- **Calidad:** {model['quality_score']:.1f}/100\n"
                md += f"- **Archivos eliminados:** {model['files_deleted']}\n"
                if model['file_names']:
                    md += "- **Archivos:**\n"
                    for file_name in model['file_names']:
                        md += f"  - `{file_name}`\n"
                md += "\n"
        else:
            md += "✅ No se eliminaron modelos\n"

        md += "---\n"
        md += "*Limpieza automática ejecutada por Model Cleaner*"

        readable_report_path = OUTPUT_PATH / "cleanup_report.md"
        with open(readable_report_path, 'w') as f:
            f.write(md)

    def show_final_state(self):
        """Muestra el estado final después de la limpieza"""
        print(f"\n🎯 ESTADO FINAL DESPUÉS DE LA LIMPIEZA")
        print("=" * 60)

        # Contar modelos restantes
        remaining_models = []
        for model_dir in MODELS_PATH.iterdir():
            if model_dir.is_dir():
                pkl_files = list(model_dir.glob("*.pkl"))
                joblib_files = list(model_dir.glob("*.joblib"))
                model_files = [f for f in pkl_files + joblib_files if '_scaler' not in f.stem]

                for model_file in model_files:
                    remaining_models.append({
                        'name': model_file.stem,
                        'directory': model_dir.name,
                        'path': model_file
                    })

        print(f"📁 Modelos restantes: {len(remaining_models)}")

        if remaining_models:
            print("\n🏆 MODELOS DE ALTA CALIDAD CONSERVADOS:")
            for i, model in enumerate(remaining_models, 1):
                print(f"{i:2d}. {model['directory']}/{model['name']}")

        print(f"\n💾 Backup disponible en: {BACKUP_PATH}")
        print("🔧 Puedes restaurar desde el backup si es necesario")

def main():
    print("🧹 LIMPIADOR DE MODELOS DE BAJA CALIDAD")
    print("=" * 60)
    print("ADVERTENCIA: Esta acción ELIMINARÁ permanentemente modelos")
    print("Se creará un backup automáticamente")
    print("=" * 60)

    # Confirmación de seguridad
    response = input("¿Continuar con la limpieza? (sí/no): ").strip().lower()
    if response not in ['sí', 'si', 's', 'yes', 'y']:
        print("❌ Limpieza cancelada")
        return

    cleaner = ModelCleaner()

    try:
        # Ejecutar limpieza
        total_models, total_files = cleaner.clean_low_quality_models()

        # Mostrar estado final
        cleaner.show_final_state()

        print(f"\n🎉 LIMPIEZA COMPLETADA!")
        print(f"🗑️  {total_models} modelos eliminados")
        print(f"📄 {total_files} archivos liberados")
        print(f"💾 Backup guardado en: {BACKUP_PATH}")

    except Exception as e:
        print(f"❌ Error durante la limpieza: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()