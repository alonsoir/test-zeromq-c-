# train_simple_effective.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os

def train_simple_effective_model():
    """Entrenar modelo simple pero efectivo"""
    print("🎯 ENTRENANDO MODELO SIMPLE Y EFECTIVO...")

    # Cargar datasets garantizados
    domains = {
        'network': 'data/network_guaranteed.csv',
        'files': 'data/files_guaranteed.csv',
        'processes': 'data/processes_guaranteed.csv'
    }

    all_data = []

    for domain_name, path in domains.items():
        df = pd.read_csv(path)
        all_data.append(df)
        print(f"✅ {domain_name}: {len(df)} samples, {df['is_ransomware'].sum()} ransomware")

    # Combinar todos los datos
    combined_df = pd.concat(all_data, ignore_index=True)
    X = combined_df.drop('is_ransomware', axis=1)
    y = combined_df['is_ransomware']

    print(f"\n📊 DATASET COMBINADO: {X.shape}")
    print(f"🎯 BALANCE: {y.sum()}/{len(y)} ransomware ({y.mean():.1%})")

    # Validación cruzada simple
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"🎯 F1 Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Entrenar modelo final
    model.fit(X, y)

    # Guardar
    os.makedirs('models', exist_ok=True)
    model_path = 'models/simple_effective_model.pkl'
    joblib.dump(model, model_path)

    # Evaluación
    y_pred = model.predict(X)
    print(f"\n📈 EVALUACIÓN EN TRAIN:")
    print(classification_report(y, y_pred))

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔝 FEATURES MÁS IMPORTANTES:")
    print(importance_df.head(10))

    return model, X, y

if __name__ == "__main__":
    train_simple_effective_model()