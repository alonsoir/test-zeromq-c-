# 🔬 Synthetic Data Stability Curve Analysis

**Purpose:** Determine the optimal synthetic/real data mixing ratio for ransomware detection model retraining through rigorous scientific experimentation.

**Status:** Ready for execution - Phase 0 of ML Autonomous Evolution System

---

## 📋 Overview

This script performs comprehensive experiments to answer the scientific question:

> **"What is the optimal proportion of synthetic data to mix with real data to maximize ransomware detection performance?"**

### Scientific Method

1. **Hypothesis:** Adding synthetic data improves model performance up to a threshold
2. **Experiment:** Train models with ratios from 0% to 100% synthetic data
3. **Analysis:** Statistical validation with multiple runs per ratio
4. **Conclusion:** Identify optimal ratio with confidence intervals

---

## 🎯 Experiments

### Experiment A: Fixed Hyperparameters
- **Goal:** Isolate pure effect of synthetic ratio
- **Ratios tested:** 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
- **Runs per ratio:** 5 (statistical robustness)
- **Hyperparameters:** Fixed across all ratios
- **Total models:** 55 (11 ratios × 5 runs)
- **Time estimate:** 2-3 hours

### Experiment B: Optimized Hyperparameters
- **Goal:** Maximize performance at each ratio
- **Ratios tested:** Same as Experiment A
- **Runs per ratio:** 3
- **Hyperparameters:** Optimized independently for each ratio
- **Total models:** 33 (11 ratios × 3 runs)
- **Time estimate:** 3-4 hours

---

## 🚀 Usage

### Basic Execution

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker/ml-training/scripts/ransomware
python synthetic_stability_curve.py
```

### Interactive Menu

The script will prompt you to choose:
1. Experiment A (fixed hyperparameters)
2. Experiment B (optimized hyperparameters)
3. Both experiments

**Recommended for first run:** Start with Experiment A to get baseline results quickly.

---

## 📊 Outputs

All outputs are organized in structured directories:

```
ml-training/scripts/ransomware/
├── model_candidates/
│   └── experiment_A_20251112_070000/
│       ├── ratio_0.0/
│       │   ├── run_1/
│       │   │   ├── ransomware_xgboost_candidate_v2_..._ratio_0.0_run_1.pkl
│       │   │   ├── ..._metadata.json
│       │   │   └── ..._importance.json
│       │   ├── run_2/
│       │   └── ...
│       ├── ratio_0.1/
│       └── ...
│
└── results/
    └── experiment_A_20251112_070000/
        ├── plots/
        │   ├── stability_curve_main.pdf          ← Main figure for paper
        │   ├── stability_curve_main.png
        │   ├── confusion_matrices_grid.pdf
        │   ├── confusion_matrices_grid.png
        │   └── statistical_significance.pdf
        │
        ├── tables/
        │   ├── results_summary.csv
        │   ├── results_summary.tex               ← LaTeX table for paper
        │   └── pvalue_matrix.csv
        │
        ├── report/
        │   ├── executive_summary.txt             ← Read this first!
        │   └── executive_summary.md
        │
        └── raw_data/
            ├── all_runs_data.csv
            └── all_runs_data.json
```

---

## 📈 Key Visualizations

### 1. Main Stability Curve
4-subplot figure showing:
- (a) F1 Score vs Synthetic Ratio (with error bars)
- (b) Precision & Recall vs Ratio
- (c) Overfitting Gap Detection
- (d) Per-Class Performance (Benign vs Ransomware)

### 2. Confusion Matrices Grid
Shows how prediction patterns change across key ratios (0%, 20%, 40%, 60%, 100%)

### 3. Statistical Significance Heatmap
P-values matrix showing which ratio differences are statistically significant

---

## 🎯 Interpreting Results

### Executive Summary
Check `report/executive_summary.txt` for:
- Optimal synthetic ratio recommendation
- Statistical significance vs baseline
- Degradation zone identification
- Production recommendations

### Key Metrics to Monitor
1. **F1 Score:** Primary performance metric
2. **Overfitting Gap:** Train F1 - Test F1 (should be close to 0)
3. **Per-Class F1:** Ensure both Benign and Ransomware perform well
4. **Statistical Significance:** p-value < 0.05 vs baseline

### Decision Criteria
The script automatically identifies optimal ratio based on:
- Maximum F1 score (40% weight)
- Minimum overfitting (30% weight)
- Stability across runs (20% weight)
- Improvement over baseline (10% weight)

---

## 🔬 Scientific Rigor

### Statistical Validation
- Multiple runs per ratio (5 for Exp A, 3 for Exp B)
- Different random seeds per run
- 95% confidence intervals
- T-tests for significance
- Cohen's d for effect size

### Avoiding Bias
- Fixed test set across all ratios
- Stratified train/test split
- Same base dataset generation
- Documented random seeds for reproducibility

---

## 📚 For the arXiv Paper

The outputs are designed to be publication-ready:

### Figures to Include
1. `stability_curve_main.pdf` - Main result figure
2. `confusion_matrices_grid.pdf` - Qualitative analysis
3. `statistical_significance.pdf` - Validation evidence

### Tables to Include
1. `results_summary.tex` - Comprehensive metrics table
2. `pvalue_matrix.csv` - Statistical validation

### Supplementary Material
- `all_runs_data.json` - Full experimental data
- All trained models in `model_candidates/`

---

## ⚠️ Important Notes

### Memory Requirements
- Each experiment generates 33-55 models
- Each model ~5-10 MB
- Total disk space: ~500 MB - 1 GB per experiment

### Time Requirements
- Experiment A: 2-3 hours
- Experiment B: 3-4 hours
- Both: 5-7 hours total

### Dependencies
All dependencies are already in your environment:
- pandas, numpy, scipy
- matplotlib, seaborn
- sklearn, xgboost, joblib
- Imports from `retrain_with_synthetic.py`

---

## 🐛 Troubleshooting

### Issue: Import error from retrain_with_synthetic.py
**Solution:** Ensure you're running from the same directory as `retrain_with_synthetic.py`

### Issue: Out of memory
**Solution:** Reduce `runs_per_ratio` in ExperimentConfig class

### Issue: Plotting error
**Solution:** matplotlib backend is set to 'Agg' (non-interactive), this is correct for server environments

---

## 🎉 Expected Outcomes

### If Hypothesis Confirmed
- Clear peak in F1 score at specific ratio (e.g., 20-40%)
- Statistical significance vs baseline (p < 0.05)
- Degradation visible at high ratios (>50%)

### If Hypothesis Rejected
- No clear optimal ratio, or
- Performance decreases with any synthetic data, or
- Multiple local optima

**Either outcome is scientifically valuable and publishable!**

---

## 📞 Next Steps After Execution

1. **Review executive_summary.txt**
2. **Examine stability_curve_main.pdf**
3. **Check statistical significance**
4. **Compare Experiment A vs B** (if both ran)
5. **Validate optimal ratio on holdout dataset** (CIC-IDS-2017, Bot-IoT)
6. **Incorporate findings into paper**

---

## 🙏 Acknowledgments

This script embodies the principle:

> **"Conservative AI + Visionary Human = Breakthrough Innovation"**

- **Alonso:** Vision, domain expertise, scientific rigor
- **Claude:** Implementation, statistical methods, documentation

---

**Ready to discover the truth? Run the script and let the data speak! 🚀🔬**