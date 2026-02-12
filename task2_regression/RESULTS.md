# Regression Model - Results Summary

**Author:** Oleksandr  
**Date:** February 2026  
**Model:** XGBoost Regressor

## Model Performance

### Training Metrics
- **Validation RMSE:** 0.2753
- **Validation R² Score:** 0.9999
- **Cross-Validation RMSE:** 0.2718 ± 0.0037 (5-fold CV)

### Model Configuration
```
XGBoost Parameters:
- n_estimators: 300
- learning_rate: 0.05
- max_depth: 6
- min_child_weight: 3
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 0.1
```

## Predictions Summary

**Test Dataset:** hidden_test.csv  
**Number of Predictions:** 10,000

### Prediction Statistics
| Metric | Value |
|--------|-------|
| Minimum | -0.0584 |
| Maximum | 100.0826 |
| Mean | 49.9601 |
| Std Dev | 28.7576 |

## Output Files

### predictions.csv
Contains predictions for all 10,000 test samples.

**Format:**
```csv
prediction
11.109371
79.97326
7.4005485
...
```

### Model Files (in models/)
- `xgboost_model.joblib` - Trained XGBoost model (not included in repo - too large)
- `scaler.joblib` - StandardScaler fitted on training data (not included in repo)
- `metadata.json` - Model metadata and configuration

## Reproducing Results

To reproduce the results:

```bash
# 1. Train model
python train.py --model xgboost --cv 5

# 2. Generate predictions
python predict.py --model xgboost --output predictions.csv
```

## Key Findings from EDA

1. **Data Quality:** No missing values, clean dataset
2. **Features:** 53 numerical features with varying correlations to target
3. **Target Distribution:** Wide range with some outliers
4. **Best Correlations:** Features in columns 7, 26, 39 show strongest positive correlation

## Model Selection Rationale

**XGBoost was chosen because:**
- Excellent performance on tabular data
- Handles non-linear relationships well
- Built-in regularization prevents overfitting
- Fast training and prediction
- Robust to outliers

The exceptional R² score (0.9999) and low RMSE (0.2718) indicate the model learned the patterns in the data very effectively.

---

**Note:** For detailed exploratory data analysis, see [eda_notebook.ipynb](eda_notebook.ipynb)
