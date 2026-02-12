"""
Random Forest Model for MNIST Digit Classification
Author: Oleksandr
Date: February 2026

This module implements a Random Forest classifier for digit classification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .interface import DigitClassificationInterface


class RandomForestModel(DigitClassificationInterface):
    """
    Random Forest classifier for digit classification.
    
    Input: 28x28x1 image (flattened to 784-dimensional vector)
    Output: Single integer from 0 to 9
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 20, random_state: int = 42):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for Random Forest.
        
        Flattens the 28x28x1 image to a 784-dimensional vector.
        
        Args:
            image: Input image of shape (28, 28, 1)
            
        Returns:
            Flattened image of shape (1, 784)
        """
        # Ensure correct shape
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected image shape (28, 28, 1), got {image.shape}")
        
        # Flatten the image
        flattened = image.reshape(1, -1)
        
        # Normalize pixel values to [0, 1]
        processed = flattened.astype('float32') / 255.0
        
        return processed
    
    def predict(self, image: np.ndarray) -> int:
        """
        Predict the digit class for the given image.
        
        Args:
            image: Input image of shape (28, 28, 1)
            
        Returns:
            Predicted digit (0-9)
        """
        if not self.is_trained:
            # For demonstration purposes, return a mock prediction
            # In production, you would load a pretrained model
            print("Warning: Model is not trained. Returning random prediction.")
            return int(np.random.randint(0, 10))
        
        # Preprocess the image
        processed_image = self.preprocess(image)
        
        # Get prediction
        prediction = self.model.predict(processed_image)
        
        return int(prediction[0])
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Random Forest model.
        
        Raises:
            NotImplementedError: Training is not implemented for this assessment
        """
        raise NotImplementedError("Training is not implemented for this assessment")
    
    def load_pretrained_model(self, model_path: str):
        """
        Load a pretrained Random Forest model.
        
        Args:
            model_path: Path to the saved model file
        """
        import joblib
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(f"Loaded pretrained model from {model_path}")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"RandomForestModel(n_estimators={self.model.n_estimators}, status={status})"
