#!/usr/bin/env python3
"""
Verify System Setup for Model #2 Training
Checks: Python version, dependencies, datasets, disk space

Usage: python verify_setup.py
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version (>= 3.11)"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("   ❌ Python 3.11+ required")
        return False
    print("   ✅ Python version OK")
    return True


def check_dependencies():
    """Check required Python packages"""
    print("\n📦 Checking dependencies...")
    
    required = {
        'xgboost': 'XGBoost',
        'sklearn': 'scikit-learn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'imblearn': 'imbalanced-learn',
        'joblib': 'Joblib'
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing).lower()}")
        return False
    
    return True


def check_datasets():
    """Check if required datasets exist"""
    print("\n💾 Checking datasets...")
    
    base_path = Path.cwd().parent / "datasets"
    
    datasets = {
        'CIC-IDS-2018': base_path / "CIC-IDS-2018" / "02-28-2018.csv",
        'CIC-IDS-2017': base_path / "CIC-IDS-2017" / "MachineLearningCVE"
    }
    
    all_ok = True
    
    for name, path in datasets.items():
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"   ✅ {name}: {size_mb:.1f} MB")
            else:
                files = list(path.glob("*.csv"))
                print(f"   ✅ {name}: {len(files)} files")
        else:
            print(f"   ❌ {name}: NOT FOUND")
            print(f"      Expected: {path}")
            all_ok = False
    
    return all_ok


def check_disk_space():
    """Check available disk space"""
    print("\n💿 Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (2**30)
        
        print(f"   Free space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print(f"   ⚠️  Low disk space (< 5 GB)")
            return False
        else:
            print(f"   ✅ Disk space OK")
            return True
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return True


def check_output_dirs():
    """Check/create output directories"""
    print("\n📁 Checking output directories...")
    
    base_path = Path.cwd().parent / "outputs"
    
    dirs = [
        base_path / "models" / "level2_ransomware_xgboost",
        base_path / "plots" / "level2_ransomware_xgboost",
        base_path / "metadata"
    ]
    
    for dir_path in dirs:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_path.name}")
        else:
            print(f"   ✅ Exists: {dir_path.name}")
    
    return True


def check_training_script():
    """Check if training script exists"""
    print("\n📄 Checking training script...")
    
    script_path = Path.cwd() / "train_ransomware_xgboost_Claude.py"
    
    if script_path.exists():
        size_kb = script_path.stat().st_size / 1024
        print(f"   ✅ train_ransomware_xgboost_Claude.py ({size_kb:.1f} KB)")
        return True
    else:
        print(f"   ❌ train_ransomware_xgboost_Claude.py NOT FOUND")
        print(f"      Copy from: outputs/train_ransomware_xgboost_Claude.py")
        return False


def main():
    print("=" * 70)
    print("🔍 VERIFYING SETUP FOR MODEL #2 TRAINING")
    print("=" * 70)
    
    checks = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Datasets': check_datasets(),
        'Disk Space': check_disk_space(),
        'Output Dirs': check_output_dirs(),
        'Training Script': check_training_script()
    }
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    for name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_ok = all(checks.values())
    
    if all_ok:
        print("\n🎉 All checks passed! Ready to train.")
        print("\nRun training with:")
        print("  cd scripts")
        print("  python train_ransomware_xgboost_Claude.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Fix issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
