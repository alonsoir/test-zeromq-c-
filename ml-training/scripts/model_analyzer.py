#!/usr/bin/env python3
"""
ANALIZADOR CORREGIDO DE MODELOS ML - VERSIÓN 2.1
Corrige el error de model_name y mejora la robustez
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIG - ANALIZADOR CORREGIDO
# =====================================================================
BASE_PATH = Path(__file__).parent.parent
OUTPUT_PATH = BASE_PATH / "outputs"
MODELS_PATH = OUTPUT_PATH / "models"
REPORT_PATH = OUTPUT_PATH / "model_analysis_report_final"

class FixedModelAnalyzer:
    def __init__(self):
        self.model_groups = []
        self.analysis_results = []

    def discover_model_groups(self):
        """Descubre y agrupa modelos de forma robusta"""
        print("🔍 BUSCANDO MODELOS...")

        # Buscar todos los archivos
        all_files = []
        all_files.extend(MODELS_PATH.rglob("*.pkl"))
        all_files.extend(MODELS_PATH.rglob("*.joblib"))

        # Organizar por grupos
        model_groups = {}

        for file_path in all_files:
            dir_name = file_path.parent.name
            file_stem = file_path.stem

            # Determinar si es modelo principal o archivo auxiliar
            if any(x in file_stem for x in ['_scaler', '_metadata', '_features']):
                base_name = file_stem.split('_scaler')[0].split('_metadata')[0].split('_features')[0]
            else:
                base_name = file_stem

            group_key = f"{dir_name}/{base_name}"

            if group_key not in model_groups:
                model_groups[group_key] = {
                    'directory': dir_name,
                    'base_name': base_name,
                    'model_file': None,
                    'scaler_file': None,
                    'metadata_file': None,
                    'features_file': None
                }

            # Clasificar archivo
            if '_scaler' in file_stem:
                model_groups[group_key]['scaler_file'] = file_path
            elif any(x in file_stem for x in ['_metadata', '_meta']):
                model_groups[group_key]['metadata_file'] = file_path
            elif any(x in file_stem for x in ['_features', '_feature']):
                model_groups[group_key]['features_file'] = file_path
            else:
                model_groups[group_key]['model_file'] = file_path

        # Solo grupos que tienen modelo principal
        self.model_groups = [group for group in model_groups.values() if group['model_file']]

        print(f"📁 Encontrados {len(self.model_groups)} modelos principales")
        return self.model_groups

    def load_model(self, model_path):
        """Carga modelo de forma robusta"""
        try:
            if model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            else:
                return joblib.load(model_path)
        except Exception as e:
            print(f"   ❌ Error cargando {model_path.name}: {e}")
            return None

    def analyze_model(self, group):
        """Analiza un modelo individual"""
        print(f"🔍 Analizando: {group['directory']}/{group['base_name']}")

        # Cargar modelo
        model = self.load_model(group['model_file'])
        if model is None:
            return None

        # Identificar algoritmo
        algorithm = self.get_algorithm(model)

        # Analizar metadata
        metadata = self.load_metadata(group)

        # Analizar completitud
        completeness = self.analyze_completeness(group)

        # Analizar rendimiento
        performance = self.analyze_performance(metadata)

        # Resultado
        result = {
            'model_name': group['base_name'],
            'directory': group['directory'],
            'algorithm': algorithm,
            'model_type': type(model).__name__,
            **completeness,
            **performance
        }

        # Mostrar resumen
        print(f"   🎯 {algorithm} - Calidad: {result['quality_score']:.1f}/100")
        print(f"   📦 {result['completeness_status']} ({result['completeness_score']:.1f}%)")
        print(f"   💡 {result['recommendation']}")

        return result

    def get_algorithm(self, model):
        """Identifica el algoritmo del modelo"""
        model_type = type(model).__name__.lower()

        if 'xgboost' in model_type or 'xgb' in str(model).lower():
            return 'XGBoost'
        elif 'randomforest' in model_type:
            return 'RandomForest'
        elif 'isolationforest' in model_type:
            return 'IsolationForest'
        elif 'logistic' in model_type:
            return 'LogisticRegression'
        else:
            return 'Unknown'

    def load_metadata(self, group):
        """Carga metadatos del modelo"""
        metadata = {}

        # Intentar cargar desde archivo metadata
        if group['metadata_file']:
            try:
                with open(group['metadata_file'], 'r') as f:
                    metadata = json.load(f)
            except:
                pass

        # Si no hay metadata file, buscar en el directorio
        if not metadata:
            metadata_files = list(Path(group['model_file']).parent.glob("*metadata*"))
            for meta_file in metadata_files:
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    break
                except:
                    continue

        return metadata

    def analyze_completeness(self, group):
        """Analiza completitud del modelo"""
        components = [
            group['model_file'] is not None,
            group['scaler_file'] is not None,
            group['metadata_file'] is not None,
            group['features_file'] is not None
        ]

        completeness_score = (sum(components) / len(components)) * 100

        if completeness_score >= 90:
            status = "📦 COMPLETO"
        elif completeness_score >= 70:
            status = "📁 CASI COMPLETO"
        elif completeness_score >= 50:
            status = "📋 PARCIAL"
        else:
            status = "❌ INCOMPLETO"

        return {
            'completeness_score': completeness_score,
            'completeness_status': status
        }

    def analyze_performance(self, metadata):
        """Analiza rendimiento del modelo"""
        metrics = metadata.get('metrics', {})

        performance = {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'roc_auc': metrics.get('roc_auc', 0)
        }

        # Calcular score de calidad
        quality_score = self.calculate_quality_score(performance)
        recommendation = self.get_recommendation(quality_score)

        return {
            **performance,
            'quality_score': quality_score,
            'recommendation': recommendation
        }

    def calculate_quality_score(self, performance):
        """Calcula score de calidad 0-100"""
        weights = {'accuracy': 0.2, 'precision': 0.25, 'recall': 0.25, 'f1_score': 0.2, 'roc_auc': 0.1}
        score = 0

        for metric, weight in weights.items():
            score += performance[metric] * weight * 100

        return min(100, score)

    def get_recommendation(self, quality_score):
        """Genera recomendación basada en calidad"""
        if quality_score >= 90:
            return "🎯 EXCELENTE - Listo para producción"
        elif quality_score >= 80:
            return "✅ BUENO - Aprovechable"
        elif quality_score >= 70:
            return "⚠️  REGULAR - Validar"
        elif quality_score >= 50:
            return "🔧 MEJORABLE - Reentrenar"
        else:
            return "❌ DESCARTAR - Baja calidad"

    def analyze_all(self):
        """Analiza todos los modelos"""
        print("\n🎯 ANALIZANDO MODELOS...")
        print("=" * 60)

        for group in self.model_groups:
            result = self.analyze_model(group)
            if result:
                self.analysis_results.append(result)

        return self.analysis_results

    def generate_simple_report(self):
        """Genera reporte simple y robusto"""
        print("\n📊 GENERANDO REPORTE...")

        if not self.analysis_results:
            print("❌ No hay resultados")
            return

        # Crear DataFrame
        df = pd.DataFrame(self.analysis_results)

        # Modelos recomendados
        production_ready = df[
            (df['quality_score'] >= 80) &
            (df['completeness_score'] >= 50)
            ].sort_values('quality_score', ascending=False)

        # Guardar reporte CSV
        REPORT_PATH.mkdir(parents=True, exist_ok=True)
        df.to_csv(REPORT_PATH / "model_analysis.csv", index=False)

        # Generar reporte markdown simple
        self.generate_simple_markdown(df, production_ready)

        return df, production_ready

    def generate_simple_markdown(self, df, production_ready):
        """Genera reporte markdown simple"""
        md = f"""# 🚀 REPORTE DE MODELOS ML - ANÁLISIS FINAL

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total de modelos analizados:** {len(df)}

## 🏆 MODELOS RECOMENDADOS PARA PRODUCCIÓN

"""

        if len(production_ready) > 0:
            md += "| Modelo | Algoritmo | Calidad | Completitud | Recomendación |\n"
            md += "|--------|-----------|---------|-------------|---------------|\n"

            for _, model in production_ready.iterrows():
                md += f"| {model['model_name']} | {model['algorithm']} | {model['quality_score']:.1f}/100 | {model['completeness_score']:.1f}% | {model['recommendation']} |\n"
        else:
            md += "❌ No hay modelos recomendados para producción\n"

        md += f"\n## 📊 ESTADÍSTICAS\n\n"
        md += f"- **Total modelos:** {len(df)}\n"
        md += f"- **Recomendados para producción:** {len(production_ready)}\n"
        md += f"- **XGBoost models:** {len(df[df['algorithm'] == 'XGBoost'])}\n"
        md += f"- **RandomForest models:** {len(df[df['algorithm'] == 'RandomForest'])}\n"

        with open(REPORT_PATH / "production_recommendations.md", 'w') as f:
            f.write(md)

    def print_final_summary(self, production_ready):
        """Imprime resumen final"""
        print("\n" + "=" * 80)
        print("🎯 RESUMEN FINAL - MODELOS APROVECHABLES")
        print("=" * 80)

        if len(production_ready) > 0:
            print(f"\n🚀 **TOP {len(production_ready)} MODELOS PARA PRODUCCIÓN:**")
            for i, (_, model) in enumerate(production_ready.iterrows(), 1):
                print(f"{i:2d}. {model['model_name']}")
                print(f"    🔧 {model['algorithm']} | 📊 {model['quality_score']:.1f}/100 | 📦 {model['completeness_score']:.1f}%")
                print(f"    📍 {model['directory']}/")
                print()
        else:
            print("❌ No se encontraron modelos recomendados para producción")

        print(f"📁 Reporte completo en: {REPORT_PATH}")

def main():
    print("🚀 ANALIZADOR DE MODELOS ML - VERSIÓN FINAL")
    print("=" * 60)

    analyzer = FixedModelAnalyzer()

    # 1. Descubrir modelos
    model_groups = analyzer.discover_model_groups()

    if not model_groups:
        print("❌ No se encontraron modelos")
        return

    # 2. Analizar
    analyzer.analyze_all()

    # 3. Generar reporte
    df, production_ready = analyzer.generate_simple_report()

    # 4. Mostrar resumen
    analyzer.print_final_summary(production_ready)

    print(f"\n🎉 ANÁLISIS COMPLETADO!")

if __name__ == "__main__":
    main()