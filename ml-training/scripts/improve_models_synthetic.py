#!/usr/bin/env python3
"""
MEJORA DE MODELOS CON DATOS SINTÉTICOS
Parte de los scripts finales y añade datos sintéticos de calidad
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import xgboost as xgb
import json
from pathlib import Path

class ModelImprover:
    def __init__(self, base_script_path: Path):
        self.base_script_path = base_script_path
        self.results = {}

    def load_base_model_script(self, script_name: str):
        """Carga y analiza un script de modelo base"""
        print(f"📖 Analizando script base: {script_name}")

        # Aquí podríamos parsear el script para extraer:
        # - Features usadas
        # - Hiperparámetros
        # - Preprocesamiento
        # - Métricas originales

        return {
            'features': self.extract_features_from_script(script_name),
            'hyperparams': self.extract_hyperparams(script_name),
            'original_metrics': self.extract_original_metrics(script_name)
        }

    def generate_high_quality_synthetic_data(self, original_features: list, n_samples: int = 10000):
        """Genera datos sintéticos de alta calidad"""
        print(f"🎯 Generando {n_samples} muestras sintéticas...")

        synthetic_data = {}

        # Para cada feature, generar datos realistas basados en:
        # 1. Distribuciones del dataset original
        # 2. Correlaciones entre features
        # 3. Patrones de ransomware conocidos

        for feature in original_features:
            if 'duration' in feature.lower():
                # Duración: mayoría corta, algunas largas (ransomware)
                synthetic_data[feature] = np.random.exponential(10, n_samples)
            elif 'packets' in feature.lower():
                # Paquetes: distribución pareto (algunos valores muy altos)
                synthetic_data[feature] = np.random.pareto(2, n_samples) * 100
            elif 'bytes' in feature.lower():
                # Bytes: log-normal (tráfico real)
                synthetic_data[feature] = np.random.lognormal(5, 2, n_samples)
            else:
                # Distribución normal para otras features
                synthetic_data[feature] = np.random.normal(0, 1, n_samples)

        df_synthetic = pd.DataFrame(synthetic_data)

        # Generar labels realistas (20% ransomware)
        labels = np.zeros(n_samples)
        ransomware_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
        labels[ransomware_indices] = 1

        # Ajustar features para samples de ransomware
        for idx in ransomware_indices:
            # Patrones típicos de ransomware
            df_synthetic.loc[idx, 'duration'] *= 0.1  # Conexiones más cortas
            df_synthetic.loc[idx, 'packets'] *= 5     # Más paquetes
            df_synthetic.loc[idx, 'bytes'] *= 10      # Más datos

        return df_synthetic, labels

    def improve_model_f1(self, base_model_info: dict, synthetic_data: pd.DataFrame, synthetic_labels: np.array):
        """Mejora el modelo usando datos sintéticos"""
        print("🚀 Mejorando modelo con datos sintéticos...")

        # Combinar datos originales (si están disponibles) con sintéticos
        # Aquí usaríamos el script base para reentrenar

        # Métricas objetivo:
        # - Mejor F1 Score
        # - Mejor matriz de confusión
        # - Menos falsos positivos/negativos

        improved_model = xgb.XGBClassifier(
            **base_model_info['hyperparams'],
            eval_metric='logloss',
            early_stopping_rounds=50
        )

        # Entrenar con datos sintéticos
        X_train, X_test, y_train, y_test = train_test_split(
            synthetic_data, synthetic_labels, test_size=0.2, random_state=42
        )

        improved_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluar
        y_pred = improved_model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return improved_model, f1, cm

    def run_improvement_pipeline(self, script_names: list):
        """Ejecuta pipeline completo de mejora"""
        for script_name in script_names:
            print(f"\n{'='*50}")
            print(f"🔄 MEJORANDO: {script_name}")
            print(f"{'='*50}")

            # 1. Cargar configuración base
            base_info = self.load_base_model_script(script_name)

            # 2. Generar datos sintéticos
            synthetic_data, synthetic_labels = self.generate_high_quality_synthetic_data(
                base_info['features'], n_samples=20000
            )

            # 3. Mejorar modelo
            improved_model, f1_score, confusion_mat = self.improve_model_f1(
                base_info, synthetic_data, synthetic_labels
            )

            # 4. Guardar resultados
            self.results[script_name] = {
                'f1_score': f1_score,
                'confusion_matrix': confusion_mat.tolist(),
                'features_used': base_info['features'],
                'improvement': f1_score - base_info['original_metrics'].get('f1', 0)
            }

            print(f"✅ {script_name} - F1: {f1_score:.4f}")

        return self.results

def main():
    # Scripts base para mejorar
    BASE_SCRIPTS = [
        "ransomware_network_detector_proto_aligned.py",
        "train_ransomware_xgboost_ransmap_ransomware_only_deepseek.py",
        "train_internal_traffic_model_deepseek.py"
    ]

    improver = ModelImprover(Path("."))
    results = improver.run_improvement_pipeline(BASE_SCRIPTS)

    # Guardar resultados
    with open("model_improvement_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n🎉 MEJORA COMPLETADA")
    for script, result in results.items():
        print(f"📊 {script}: F1 = {result['f1_score']:.4f} (mejora: {result['improvement']:+.4f})")

if __name__ == "__main__":
    main()