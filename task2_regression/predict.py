"""
Model Prediction Script
Author: Oleksandr
Date: February 2026

This script loads a trained regression model and generates predictions on test data.

Usage:
    python predict.py [--model MODEL_TYPE] [--output OUTPUT_FILE]
    
Arguments:
    --model: Model type to use (xgboost, randomforest, lightgbm). Default: xgboost
    --output: Output file for predictions. Default: predictions.csv
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class RegressionPredictor:
    """
    A class to load a trained model and make predictions.
    """
    
    def __init__(self, model_type='xgboost', model_dir='models'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to load
            model_dir: Directory containing the saved model
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        
    def load_model(self):
        """
        Load the trained model and scaler from disk.
        """
        model_path = self.model_dir / f'{self.model_type}_model.joblib'
        scaler_path = self.model_dir / 'scaler.joblib'
        metadata_path = self.model_dir / 'metadata.json'
        
        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train the model first using train.py"
            )
        
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found: {scaler_path}\n"
                f"Please train the model first using train.py"
            )
        
        # Load model and scaler
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        print(f"Loading scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata if exists
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Model metadata: {self.metadata}")
        
        print("Model and scaler loaded successfully!")
        
    def load_data(self, filepath):
        """
        Load test data from CSV file.
        
        Args:
            filepath: Path to the test CSV file
            
        Returns:
            DataFrame with loaded data
        """
        print(f"\nLoading test data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Test data shape: {df.shape}")
        
        # Validate feature count
        if self.metadata and df.shape[1] != self.metadata['feature_count']:
            raise ValueError(
                f"Feature count mismatch! Expected {self.metadata['feature_count']}, "
                f"got {df.shape[1]}"
            )
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the test data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed features
        """
        X = df.values
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def predict(self, X):
        """
        Make predictions on the preprocessed data.
        
        Args:
            X: Preprocessed feature matrix
            
        Returns:
            Array of predictions
        """
        print("\nGenerating predictions...")
        predictions = self.model.predict(X)
        print(f"Predictions generated: {len(predictions)} samples")
        
        # Print prediction statistics
        print(f"\nPrediction Statistics:")
        print(f"  Min: {predictions.min():.4f}")
        print(f"  Max: {predictions.max():.4f}")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std: {predictions.std():.4f}")
        
        return predictions
    
    def save_predictions(self, predictions, output_path):
        """
        Save predictions to a CSV file.
        
        Args:
            predictions: Array of predictions
            output_path: Path to save the predictions
        """
        # Create DataFrame with predictions
        pred_df = pd.DataFrame(predictions, columns=['prediction'])
        
        # Save to CSV
        pred_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")


def main():
    """Main function to run the prediction pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate predictions using trained model')
    parser.add_argument(
        '--model', 
        type=str, 
        default='xgboost',
        choices=['xgboost', 'randomforest', 'lightgbm'],
        help='Model type to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output file for predictions'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default='../hidden_test.csv',
        help='Path to test data'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing the trained model'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("REGRESSION MODEL PREDICTION")
    print("=" * 60)
    print(f"Model Type: {args.model}")
    print(f"Output File: {args.output}")
    print("=" * 60)
    
    # Initialize predictor
    predictor = RegressionPredictor(
        model_type=args.model,
        model_dir=args.model_dir
    )
    
    # Load model
    predictor.load_model()
    
    # Load test data
    test_df = predictor.load_data(args.test_data)
    
    # Preprocess data
    X_test = predictor.preprocess_data(test_df)
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Save predictions
    predictor.save_predictions(predictions, args.output)
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
