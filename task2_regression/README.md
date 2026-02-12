# Task 2: Regression on Tabular Data

## Problem Description

Build a regression model to predict a target variable based on 53 anonymized features. The target metric is **RMSE (Root Mean Squared Error)**.

## Project Structure

```
task2_regression/
├── eda_notebook.ipynb      # Exploratory Data Analysis
├── train.py                # Model training script
├── predict.py              # Model prediction script
├── requirements.txt        # Python dependencies
├── predictions.csv         # Generated predictions (after running predict.py)
├── models/                 # Trained models directory
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   └── metadata.json
└── README.md              # This file
```

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

2. **Install dependencies**:
```bash
cd task2_regression
pip install -r requirements.txt
```

## Usage

### 1. Exploratory Data Analysis

Open and run the Jupyter notebook to understand the data:

```bash
jupyter notebook eda_notebook.ipynb
```

The notebook includes:
- Data loading and inspection
- Statistical analysis
- Missing values check
- Target variable distribution
- Feature correlation analysis
- Insights and recommendations

### 2. Train the Model

Train a regression model using the training script:

```bash
python train.py
```

**Optional parameters:**
```bash
python train.py --model xgboost --cv 5 --data-path ../train.csv
```

Arguments:
- `--model`: Model type (`xgboost`, `randomforest`, `lightgbm`). Default: `xgboost`
- `--cv`: Number of cross-validation folds. Default: `5`
- `--data-path`: Path to training data. Default: `../train.csv`

**Output:**
- Trained model saved to `models/[model_type]_model.joblib`
- Scaler saved to `models/scaler.joblib`
- Metadata saved to `models/metadata.json`
- Validation and cross-validation metrics printed to console

### 3. Generate Predictions

Generate predictions on test data:

```bash
python predict.py
```

**Optional parameters:**
```bash
python predict.py --model xgboost --output predictions.csv --test-data ../hidden_test.csv
```

Arguments:
- `--model`: Model type to use (must match trained model). Default: `xgboost`
- `--output`: Output CSV file for predictions. Default: `predictions.csv`
- `--test-data`: Path to test data. Default: `../hidden_test.csv`
- `--model-dir`: Directory containing trained model. Default: `models`

**Output:**
- Predictions saved to specified output file (default: `predictions.csv`)
- Prediction statistics printed to console

## Model Details

### XGBoost (Default)

**Hyperparameters:**
- `n_estimators`: 300
- `learning_rate`: 0.05
- `max_depth`: 6
- `min_child_weight`: 3
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `gamma`: 0.1

**Advantages:**
- Excellent performance on tabular data
- Handles non-linear relationships well
- Built-in regularization
- Fast training and prediction

### Random Forest

**Hyperparameters:**
- `n_estimators`: 200
- `max_depth`: 15
- `min_samples_split`: 5
- `min_samples_leaf`: 2

**Advantages:**
- Robust to outliers
- Good for feature importance analysis
- Less prone to overfitting

### LightGBM

**Hyperparameters:**
- `n_estimators`: 300
- `learning_rate`: 0.05
- `max_depth`: 6
- `num_leaves`: 31

**Advantages:**
- Very fast training
- Memory efficient
- Good for large datasets

## Example Workflow

Complete workflow from start to finish:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Explore data (optional but recommended)
jupyter notebook eda_notebook.ipynb

# 3. Train model
python train.py --model xgboost --cv 5

# 4. Generate predictions
python predict.py --model xgboost --output predictions.csv

# 5. View predictions
# predictions.csv will contain predicted target values
```

## Evaluation Metric

**RMSE (Root Mean Squared Error)**

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Where:
- $n$ = number of samples
- $y_i$ = actual target value
- $\hat{y}_i$ = predicted target value

Lower RMSE indicates better model performance.

## Features

### Data Preprocessing
- Standard scaling for all features
- Fitted on training data, applied to test data
- Handles numerical features automatically

### Model Training
- Train/validation split (80/20)
- K-fold cross-validation for robust evaluation
- Retraining on full dataset after validation
- Model persistence using joblib

### Prediction
- Automatic model loading
- Feature validation
- Prediction statistics
- CSV output format

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install dependencies using `pip install -r requirements.txt`

2. **FileNotFoundError** when running predict.py:
   - Ensure you've run `train.py` first
   - Check that `models/` directory exists with saved model files

3. **Feature count mismatch**:
   - Ensure test data has the same number of features as training data (53 features)
   - Check that the correct test file is being used

4. **Memory issues**:
   - Try using LightGBM instead of XGBoost
   - Reduce `n_estimators` parameter

## Author

Oleksandr  
February 2026

## License

This project is for educational and assessment purposes.
