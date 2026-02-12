"""
Random Model for MNIST Digit Classification
Author: Oleksandr
Date: February 2026

This module implements a random baseline model for digit classification.
"""

import numpy as np
from .interface import DigitClassificationInterface


class RandomModel(DigitClassificationInterface):
    """
    Random baseline model for digit classification.
    
    This model serves as a baseline and returns random predictions.
    It takes the center 10x10 crop of the image as input but doesn't actually use it
    for prediction (returns random values as specified in the requirements).
    
    Input: 28x28x1 image (center 10x10 crop is extracted)
    Output: Random integer from 0 to 9
    """
    
    def __init__(self, random_state: int = None):
        """
        Initialize the Random model.
        
        Args:
            random_state: Random seed for reproducibility (optional)
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image by extracting the center 10x10 crop.
        
        Args:
            image: Input image of shape (28, 28, 1)
            
        Returns:
            Center crop of shape (10, 10)
        """
        # Ensure correct shape
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected image shape (28, 28, 1), got {image.shape}")
        
        # Calculate center crop coordinates
        # For a 28x28 image, the center 10x10 starts at index 9 and ends at 19
        start_idx = (28 - 10) // 2  # = 9
        end_idx = start_idx + 10      # = 19
        
        # Extract center 10x10 crop
        center_crop = image[start_idx:end_idx, start_idx:end_idx, 0]
        
        return center_crop
    
    def predict(self, image: np.ndarray) -> int:
        """
        Predict the digit class for the given image.
        
        For simplicity, this model returns a random value regardless of the input,
        as specified in the requirements.
        
        Args:
            image: Input image of shape (28, 28, 1)
            
        Returns:
            Random digit (0-9)
        """
        # Preprocess to validate input format
        _ = self.preprocess(image)
        
        # Return random prediction
        prediction = np.random.randint(0, 10)
        
        return int(prediction)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Random model.
        
        Raises:
            NotImplementedError: Training is not implemented for this assessment
        """
        raise NotImplementedError("Training is not implemented for this assessment")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"RandomModel(random_state={self.random_state})"
