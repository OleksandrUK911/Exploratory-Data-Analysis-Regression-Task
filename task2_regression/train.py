"""
Model Training Script
Author: Oleksandr
Date: February 2026

This script trains a regression model on the provided dataset and saves the model.
The model predicts a target variable based on 53 anonymized features.

Usage:
    python train.py [--model MODEL_TYPE] [--cv CV_FOLDS]
    
Arguments:
    --model: Model type (xgboost, randomforest, lightgbm). Default: xgboost
    --cv: Number of cross-validation folds. Default: 5
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import json
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class RegressionModelTrainer:
    """
    A class to train and evaluate regression models.
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize the trainer with a specific model type.
        
        Args:
            model_type: Type of model to train ('xgboost', 'randomforest', 'lightgbm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def create_model(self):
        """
        Create a model based on the specified type.
        
        Returns:
            Initialized model
        """
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'randomforest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def load_data(self, filepath):
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data shape: {df.shape}")
        return df
    
    def preprocess_data(self, df, is_train=True):
        """
        Preprocess the data.
        
        Args:
            df: Input DataFrame
            is_train: Whether this is training data
            
        Returns:
            Preprocessed features and target (if train)
        """
        if is_train:
            # Separate features and target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Store feature names
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # Fit and transform features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
        else:
            # Transform features using fitted scaler
            X = df.values
            X_scaled = self.scaler.transform(X)
            return X_scaled
    
    def train(self, X, y, cv_folds=5):
        """
        Train the model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
        """
        print(f"\nTraining {self.model_type} model...")
        print("=" * 60)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        print(f"\nValidation Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Cross-validation
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=cv_folds, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        cv_rmse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"Cross-Validation RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})")
        
        # Retrain on full dataset
        print("\nRetraining on full dataset...")
        self.model.fit(X, y)
        
        return {
            'val_rmse': rmse,
            'val_r2': r2,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std
        }
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model and scaler.
        
        Args:
            model_dir: Directory to save the model
        """
        Path(model_dir).mkdir(exist_ok=True)
        
        model_path = Path(model_dir) / f'{self.model_type}_model.joblib'
        scaler_path = Path(model_dir) / 'scaler.joblib'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_count': len(self.feature_names)
        }
        
        metadata_path = Path(model_dir) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Main function to run the training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train regression model')
    parser.add_argument(
        '--model', 
        type=str, 
        default='xgboost',
        choices=['xgboost', 'randomforest', 'lightgbm'],
        help='Model type to train'
    )
    parser.add_argument(
        '--cv', 
        type=int, 
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../train.csv',
        help='Path to training data'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("REGRESSION MODEL TRAINING")
    print("=" * 60)
    print(f"Model Type: {args.model}")
    print(f"Cross-Validation Folds: {args.cv}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = RegressionModelTrainer(model_type=args.model)
    
    # Load data
    df = trainer.load_data(args.data_path)
    
    # Preprocess data
    X, y = trainer.preprocess_data(df, is_train=True)
    
    # Train model
    metrics = trainer.train(X, y, cv_folds=args.cv)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
