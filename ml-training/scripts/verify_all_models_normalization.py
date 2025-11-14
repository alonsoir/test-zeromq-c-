# /vagrant/ml-training/scripts/verify_all_models_normalization.py
import re
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class ModelNormalizationVerifier:
    def __init__(self):
        self.results = {}
        self.report_data = []

    def analyze_header(self, header_path):
        """Análisis completo de un header"""
        try:
            with open(header_path, 'r') as f:
                content = f.read()

            # Estadísticas básicas
            file_size = os.path.getsize(header_path)
            lines = content.count('\n')

            # Análisis de thresholds
            thresholds = re.findall(r'[-+]?[0-9]*\.[0-9]+f', content)
            numeric_thresholds = [float(t.replace('f', '')) for t in thresholds
                                  if t.replace('f', '') not in ['-2.0000000000']]

            if numeric_thresholds:
                stats = {
                    'file': str(header_path),
                    'file_size_kb': file_size / 1024,
                    'lines': lines,
                    'total_thresholds': len(numeric_thresholds),
                    'min_threshold': min(numeric_thresholds),
                    'max_threshold': max(numeric_thresholds),
                    'avg_threshold': sum(numeric_thresholds) / len(numeric_thresholds),
                    'large_thresholds': len([t for t in numeric_thresholds if t > 1.0]),
                    'status': 'NORMALIZED' if all(t <= 1.0 for t in numeric_thresholds) else 'NOT_NORMALIZED'
                }

                return stats
        except Exception as e:
            print(f"❌ Error analizando {header_path}: {e}")
            return None

    def generate_report(self):
        """Genera reporte detallado"""
        if not self.report_data:
            print("⚠️ No hay datos para generar reporte")
            return

        df = pd.DataFrame(self.report_data)

        print("\n📊 REPORTE DETALLADO DE NORMALIZACIÓN")
        print("=" * 60)

        for _, row in df.iterrows():
            status_icon = "✅" if row['status'] == 'NORMALIZED' else "❌"
            print(f"{status_icon} {Path(row['file']).parent.name:20} | "
                  f"Thresholds: {row['total_thresholds']:4d} | "
                  f"Rango: [{row['min_threshold']:.4f}, {row['max_threshold']:.4f}] | "
                  f">1.0: {row['large_thresholds']:2d}")

        # Estadísticas globales
        total_models = len(df)
        normalized_models = len(df[df['status'] == 'NORMALIZED'])
        total_thresholds = df['total_thresholds'].sum()
        problem_thresholds = df['large_thresholds'].sum()

        print(f"\n📈 ESTADÍSTICAS GLOBALES:")
        print(f"   Modelos verificados: {total_models}")
        print(f"   Modelos normalizados: {normalized_models}/{total_models} ({normalized_models/total_models*100:.1f}%)")
        print(f"   Thresholds totales: {total_thresholds}")
        print(f"   Thresholds problemáticos: {problem_thresholds} ({problem_thresholds/total_thresholds*100:.2f}%)")

        # Guardar reportes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'model_verification_report_{timestamp}.csv', index=False)

        report = {
            'timestamp': timestamp,
            'summary': {
                'total_models': int(total_models),
                'normalized_models': int(normalized_models),
                'total_thresholds': int(total_thresholds),
                'problem_thresholds': int(problem_thresholds),
                'all_normalized': bool(problem_thresholds == 0)
            },
            'details': [{
                'file': str(item['file']),
                'file_size_kb': float(item['file_size_kb']),
                'lines': int(item['lines']),
                'total_thresholds': int(item['total_thresholds']),
                'min_threshold': float(item['min_threshold']),
                'max_threshold': float(item['max_threshold']),
                'avg_threshold': float(item['avg_threshold']),
                'large_thresholds': int(item['large_thresholds']),
                'status': str(item['status'])
            } for item in self.report_data]
        }

        with open(f'model_verification_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Reportes guardados:")
        print(f"   - model_verification_report_{timestamp}.csv")
        print(f"   - model_verification_report_{timestamp}.json")

        return problem_thresholds == 0

    def run_verification(self):
        """Ejecuta verificación completa"""
        print("🔍 INICIANDO VERIFICACIÓN AUTOMÁTICA DE MODELOS")
        print("=" * 50)

        model_dirs = ["ddos_detection", "external_traffic", "internal_traffic", "ransomware"]

        for model_dir in model_dirs:
            if not Path(model_dir).exists():
                print(f"⚠️  Directorio no encontrado: {model_dir}")
                continue

            headers = list(Path(model_dir).glob("*.hpp"))
            if headers:
                print(f"\n📁 {model_dir.upper()}:")
                for header in headers:
                    stats = self.analyze_header(header)
                    if stats:
                        self.report_data.append(stats)
                        status_icon = "✅" if stats['status'] == 'NORMALIZED' else "❌"
                        print(f"   {status_icon} {header.name}: {stats['total_thresholds']} thresholds")
            else:
                print(f"📁 {model_dir.upper()}: ⚠️  No hay headers .hpp")

        return self.generate_report()

if __name__ == "__main__":
    verifier = ModelNormalizationVerifier()
    success = verifier.run_verification()

    if success:
        print("\n🎉 ¡VERIFICACIÓN EXITOSA! Todos los modelos están normalizados.")
        exit(0)
    else:
        print("\n❌ VERIFICACIÓN FALLIDA: Algunos modelos necesitan corrección.")
        exit(1)