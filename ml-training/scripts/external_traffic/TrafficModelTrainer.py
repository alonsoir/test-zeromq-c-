# ml-training-scripts/external_traffic/TrafficModelTrainer.py
import json
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from SyntheticDataGenerator import TRAFFIC_FEATURES
from TrafficDataValidator import TrafficDataValidator

class TrafficModelTrainer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.model = None

    def load_and_prepare_data(self):
        """Carga y prepara datos para entrenamiento"""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data['dataset'])

        # Convertir labels a numéricos
        df['label_num'] = df['label'].map({'internet': 0, 'internal': 1})

        # Separar features y target
        X = df[TRAFFIC_FEATURES].values
        y = df['label_num'].values

        return X, y, data['model_info']

    def train_model(self, n_estimators=100, test_size=0.2):
        """Entrena el modelo RandomForest"""
        print("🎯 Entrenando modelo de clasificación de tráfico...")

        X, y, model_info = self.load_and_prepare_data()

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Entrenar RandomForest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluar
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ Modelo entrenado - Accuracy: {accuracy:.4f}")
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['internet', 'internal']))

        return self.model, accuracy, model_info

    def save_model(self, filename: str):
        """Guarda modelo en formato PKL"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train_model() primero.")

        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"💾 Modelo guardado: {filename}")

if __name__ == "__main__":
    # 1. Generar dataset
    from SyntheticDataGenerator import TrafficSyntheticGenerator
    generator = TrafficSyntheticGenerator(n_samples=50000)
    generator.save_dataset("traffic_classification_dataset.json")

    # 2. Entrenar modelo
    trainer = TrafficModelTrainer("traffic_classification_dataset.json")
    model, accuracy, model_info = trainer.train_model(n_estimators=100)
    trainer.save_model("traffic_classification_model.pkl")

    # 3. Validar
    validator = TrafficDataValidator("traffic_classification_dataset.json")
    validator.validate_separability()