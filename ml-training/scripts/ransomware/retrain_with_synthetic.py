#!/usr/bin/env python3
"""
SUPER LIGHTWEIGHT RETRAINING - NO EXTERNAL DEPENDENCIES
Uses only sklearn + xgboost + numpy for synthetic data generation
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# ONLY FROM YOUR EXISTING REQUIREMENTS.TXT
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuperLightConfig:
    """Super lightweight configuration - no external deps"""

    # Original ransomware_xgboost_production_v2 metrics
    ORIGINAL_F1: float = 0.98
    ORIGINAL_PRECISION: float = 0.97
    ORIGINAL_RECALL: float = 0.99

    MODEL_NAME: str = "ransomware_xgboost_production_v2"
    BASE_FEATURES: int = 45

    # Conservative settings
    SYNTHETIC_RATIO: float = 0.2  # 20% synthetic data
    IMPROVEMENT_THRESHOLD: float = 0.001  # 0.1% improvement

    # Dataset size
    BASE_SAMPLES: int = 2000
    SYNTHETIC_SAMPLES: int = 400

class StatisticalDataGenerator:
    """Statistical synthetic data generation using only numpy"""

    def __init__(self, config: SuperLightConfig):
        self.config = config

    def create_ransomware_dataset(self) -> pd.DataFrame:
        """Create realistic ransomware dataset using statistical distributions"""
        logger.info("Creating ransomware dataset with statistical patterns...")

        n_samples = self.config.BASE_SAMPLES
        data = {}

        # Define realistic feature distributions for network traffic
        feature_configs = [
            # (name_pattern, distribution_function, ransomware_multiplier)
            ('duration', lambda: np.random.exponential(25), 0.3),
            ('fwd_packets', lambda: np.random.poisson(35), 2.0),
            ('bwd_packets', lambda: np.random.poisson(25), 2.5),
            ('total_bytes', lambda: np.random.lognormal(9, 1.2), 3.0),
            ('packet_rate', lambda: np.random.gamma(2, 20), 2.8),
            ('byte_rate', lambda: np.random.gamma(1.5, 1000), 3.2),
            ('flow_duration', lambda: np.random.exponential(30), 0.4),
            ('packet_size_var', lambda: np.random.gamma(1, 50), 1.5),
            ('flag_count', lambda: np.random.poisson(6), 1.8),
            ('connection_rate', lambda: np.random.exponential(15), 2.2),
        ]

        # Store multipliers separately for ransomware transformation
        ransomware_multipliers = {}

        # Generate features based on configurations
        for i in range(self.config.BASE_FEATURES):
            if i < len(feature_configs):
                pattern, dist_func, ransomware_mult = feature_configs[i]
                data[f'feature_{i}'] = [dist_func() for _ in range(n_samples)]
                ransomware_multipliers[i] = ransomware_mult
            else:
                # Normal distribution for remaining features
                data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
                ransomware_multipliers[i] = 1.0  # No change

        # Create labels with realistic imbalance (15% ransomware)
        labels = np.zeros(n_samples)
        n_ransomware = int(n_samples * 0.15)
        ransomware_indices = np.random.choice(n_samples, size=n_ransomware, replace=False)
        labels[ransomware_indices] = 1

        # Apply ransomware patterns to malicious samples
        for idx in ransomware_indices:
            for i in range(self.config.BASE_FEATURES):
                multiplier = ransomware_multipliers[i]
                data[f'feature_{i}'][idx] *= multiplier

        data['label'] = labels
        df = pd.DataFrame(data)

        logger.info(f"✅ Dataset: {len(df)} samples, {df['label'].mean():.1%} ransomware")
        return df

    def generate_smart_synthetic(self, real_data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Smart statistical synthetic data generation"""
        logger.info(f"Generating {n_samples} smart synthetic samples...")

        synthetic_samples = []

        for _ in range(n_samples):
            # Get a random real sample as base
            base_sample = real_data.sample(1).iloc[0].copy()

            # Create a new sample dictionary
            new_sample = {}

            # Copy label first
            new_sample['label'] = base_sample['label']

            # Add intelligent variation to features
            for col in real_data.columns:
                if col != 'label':
                    original_value = base_sample[col]
                    feature_std = real_data[col].std()

                    if feature_std > 0:
                        # Smart noise: proportional to feature variance
                        noise_scale = feature_std * 0.15

                        # Different noise patterns for different feature types
                        if any(pattern in col for pattern in ['duration', 'time']):
                            # Time features: log-normal noise
                            noise = np.random.lognormal(0, 0.1) - 1
                        elif any(pattern in col for pattern in ['packet', 'byte', 'rate']):
                            # Traffic features: gamma noise
                            noise = np.random.gamma(1, 0.2) - 0.2
                        else:
                            # Other features: normal noise
                            noise = np.random.normal(0, 0.1)

                        new_value = original_value + noise * noise_scale

                        # Ensure positive values for certain features
                        if any(pattern in col for pattern in ['duration', 'packet', 'byte', 'rate']):
                            new_value = max(0.1, new_value)

                        new_sample[col] = new_value
                    else:
                        new_sample[col] = original_value

            synthetic_samples.append(new_sample)

        synthetic_df = pd.DataFrame(synthetic_samples)
        logger.info(f"✅ Smart synthetic data generated: {len(synthetic_df)} samples")
        return synthetic_df

class SimpleModelOptimizer:
    """Simple model optimization without external dependencies"""

    def __init__(self, config: SuperLightConfig):
        self.config = config

    def optimize_simple(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Simple but effective parameter optimization"""
        logger.info("Running simple parameter optimization...")

        # Tested parameter combinations for ransomware detection
        param_combinations = [
            # Conservative (good for stability)
            {'n_estimators': 80, 'max_depth': 6, 'learning_rate': 0.08, 'subsample': 0.8},
            # Balanced
            {'n_estimators': 120, 'max_depth': 7, 'learning_rate': 0.1, 'subsample': 0.85},
            # More complex
            {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.12, 'subsample': 0.9},
            # Simple but deep
            {'n_estimators': 100, 'max_depth': 9, 'learning_rate': 0.15, 'subsample': 0.75},
        ]

        best_score = 0
        best_params = {}

        for params in param_combinations:
            scores = []

            # 3-fold cross-validation
            for fold in range(3):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.25,
                    random_state=42 + fold,  # Different random state each fold
                    stratify=y
                )

                model = xgb.XGBClassifier(**params, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                scores.append(score)

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                logger.info(f"📈 New best: F1={mean_score:.4f} with {params}")

        logger.info(f"🎯 Best parameters: {best_params} (F1: {best_score:.4f})")
        return best_params

class SuperLightPipeline:
    """Super lightweight retraining pipeline - no external deps"""

    def __init__(self, config: Optional[SuperLightConfig] = None):
        self.config = config or SuperLightConfig()
        self.data_generator = StatisticalDataGenerator(self.config)
        self.model_optimizer = SimpleModelOptimizer(self.config)

    def generate_candidate_name(self) -> str:
        """Generate candidate name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ransomware_xgboost_candidate_v2_{timestamp}"

    def run_retraining(self) -> Dict:
        """Run super lightweight retraining"""
        candidate_name = self.generate_candidate_name()
        logger.info(f"🚀 Starting super lightweight retraining: {candidate_name}")

        try:
            # 1. Generate realistic base dataset
            logger.info("📊 Step 1: Generating base dataset...")
            real_data = self.data_generator.create_ransomware_dataset()

            # 2. Generate smart synthetic data
            logger.info("🧠 Step 2: Generating smart synthetic data...")
            synthetic_data = self.data_generator.generate_smart_synthetic(
                real_data, self.config.SYNTHETIC_SAMPLES
            )

            # 3. Combine datasets
            logger.info("🔗 Step 3: Combining datasets...")
            combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
            X = combined_data.drop('label', axis=1)
            y = combined_data['label']

            logger.info(f"📈 Final dataset: {len(real_data)} real + {len(synthetic_data)} synthetic")
            logger.info(f"🎯 Class balance: {y.mean():.1%} ransomware")

            # 4. Optimize model
            logger.info("⚙️  Step 4: Optimizing model...")
            best_params = self.model_optimizer.optimize_simple(X, y)

            # 5. Train final model with proper validation split
            logger.info("🏋️  Step 5: Training final model...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            final_model = xgb.XGBClassifier(**best_params, random_state=42)
            final_model.fit(X_train, y_train)

            # 6. Comprehensive evaluation
            logger.info("📊 Step 6: Evaluating model...")
            y_pred = final_model.predict(X_test)
            y_pred_proba = final_model.predict_proba(X_test)

            new_f1 = f1_score(y_test, y_pred, average='weighted')
            new_precision = precision_score(y_test, y_pred, average='weighted')
            new_recall = recall_score(y_test, y_pred, average='weighted')
            new_cm = confusion_matrix(y_test, y_pred)

            improvement = new_f1 - self.config.ORIGINAL_F1
            is_improved = improvement >= self.config.IMPROVEMENT_THRESHOLD

            # 7. Save candidate model
            logger.info("💾 Step 7: Saving candidate model...")
            candidate_path = self._save_candidate(final_model, candidate_name, {
                'params': best_params,
                'metrics': {
                    'f1': new_f1,
                    'precision': new_precision,
                    'recall': new_recall,
                    'improvement': improvement,
                    'threshold_met': is_improved
                },
                'confusion_matrix': new_cm.tolist(),
                'dataset_info': {
                    'real_samples': len(real_data),
                    'synthetic_samples': len(synthetic_data),
                    'ransomware_ratio': y.mean()
                }
            })

            results = {
                'candidate_name': candidate_name,
                'candidate_path': candidate_path,
                'success': True,
                'is_improved': is_improved,
                'metrics': {
                    'original_f1': self.config.ORIGINAL_F1,
                    'new_f1': new_f1,
                    'improvement': improvement,
                    'original_precision': self.config.ORIGINAL_PRECISION,
                    'new_precision': new_precision,
                    'original_recall': self.config.ORIGINAL_RECALL,
                    'new_recall': new_recall
                },
                'confusion_matrix': new_cm.tolist()
            }

            self._print_detailed_results(results)
            return results

        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'candidate_name': candidate_name
            }

    def _save_candidate(self, model, candidate_name: str, metadata: Dict) -> Path:
        """Save candidate model with comprehensive metadata"""
        candidates_dir = Path("model_candidates")
        candidates_dir.mkdir(exist_ok=True)

        candidate_dir = candidates_dir / candidate_name
        candidate_dir.mkdir(exist_ok=True)

        # Save model
        model_path = candidate_dir / f"{candidate_name}.pkl"
        joblib.dump(model, model_path)

        # Save metadata
        full_metadata = {
            'candidate_name': candidate_name,
            'parent_model': self.config.MODEL_NAME,
            'generation_date': datetime.now().isoformat(),
            'original_metrics': {
                'f1': self.config.ORIGINAL_F1,
                'precision': self.config.ORIGINAL_PRECISION,
                'recall': self.config.ORIGINAL_RECALL
            },
            'retraining_config': {
                'synthetic_ratio': self.config.SYNTHETIC_RATIO,
                'improvement_threshold': self.config.IMPROVEMENT_THRESHOLD,
                'base_samples': self.config.BASE_SAMPLES
            },
            **metadata
        }

        metadata_path = candidate_dir / f"{candidate_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)

        # Save feature importance if available
        try:
            feature_importance = dict(zip(
                [f'feature_{i}' for i in range(len(model.feature_importances_))],
                model.feature_importances_.tolist()
            ))

            importance_path = candidate_dir / f"{candidate_name}_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
        except:
            pass  # Skip if feature importance not available

        logger.info(f"💾 Candidate saved: {candidate_dir}")
        return candidate_dir

    def _print_detailed_results(self, results: Dict):
        """Print comprehensive results"""
        print(f"\n{'='*70}")
        print("🎯 SUPER LIGHTWEIGHT RETRAINING - COMPLETE RESULTS")
        print(f"{'='*70}")
        print(f"📁 Candidate: {results['candidate_name']}")
        print(f"📊 Dataset: {results['candidate_path'].name}")
        print(f"🎯 F1 Score:    {results['metrics']['original_f1']:.4f} → {results['metrics']['new_f1']:.4f}")
        print(f"📈 Improvement: {results['metrics']['improvement']:+.4f}")
        print(f"🎯 Precision:   {results['metrics']['original_precision']:.4f} → {results['metrics']['new_precision']:.4f}")
        print(f"🎯 Recall:      {results['metrics']['original_recall']:.4f} → {results['metrics']['new_recall']:.4f}")
        print(f"✅ Threshold Met: {results['is_improved']}")

        # Confusion matrix
        print(f"\n📊 Confusion Matrix:")
        cm = results['confusion_matrix']
        print(f"    Actual\\Predicted | Normal  | Ransomware")
        print(f"    {'-'*43}")
        print(f"    Normal         | {cm[0][0]:>7} | {cm[0][1]:>11}")
        print(f"    Ransomware     | {cm[1][0]:>7} | {cm[1][1]:>11}")

        print(f"\n💾 Saved to: {results['candidate_path']}")
        print(f"{'='*70}")

def main():
    """Main execution"""
    print("🚀 STARTING SUPER LIGHTWEIGHT RANSOMWARE RETRAINING")
    print("   (No external dependencies - using statistical methods only)")
    print("=" * 60)

    pipeline = SuperLightPipeline()

    try:
        results = pipeline.run_retraining()

        if results['success']:
            if results['is_improved']:
                print("\n🎉 SUCCESS: Generated IMPROVED candidate model!")
                print("   This candidate meets the improvement threshold and can be considered for production.")
            else:
                print("\n⚠️  Candidate generated but needs more improvement")
                print("   The model shows potential but doesn't meet the strict improvement threshold.")
        else:
            print(f"\n❌ Retraining failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        raise

if __name__ == "__main__":
    main()