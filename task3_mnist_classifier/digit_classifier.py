"""
Digit Classifier - Unified Interface for MNIST Classification
Author: Oleksandr
Date: February 2026

This module provides a unified interface for digit classification using different algorithms.
"""

import numpy as np
from typing import Literal
from models import DigitClassificationInterface, CNNModel, RandomForestModel, RandomModel


class DigitClassifier:
    """
    A unified classifier that can use different algorithms for digit classification.
    
    This class provides a consistent interface regardless of the underlying algorithm,
    allowing easy switching between CNN, Random Forest, and Random models.
    
    Usage:
        classifier = DigitClassifier(algorithm='cnn')
        prediction = classifier.predict(image)
    """
    
    SUPPORTED_ALGORITHMS = ['cnn', 'rf', 'rand']
    
    def __init__(self, algorithm: Literal['cnn', 'rf', 'rand']):
        """
        Initialize the DigitClassifier with the specified algorithm.
        
        Args:
            algorithm: The algorithm to use for classification.
                      Possible values: 'cnn', 'rf', 'rand'
                      - 'cnn': Convolutional Neural Network
                      - 'rf': Random Forest
                      - 'rand': Random baseline model
                      
        Raises:
            ValueError: If the algorithm is not supported
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )
        
        self.algorithm = algorithm
        self.model = self._create_model()
        
    def _create_model(self) -> DigitClassificationInterface:
        """
        Create and return the appropriate model based on the algorithm.
        
        Returns:
            Instance of a model implementing DigitClassificationInterface
        """
        if self.algorithm == 'cnn':
            return CNNModel()
        elif self.algorithm == 'rf':
            return RandomForestModel()
        elif self.algorithm == 'rand':
            return RandomModel(random_state=42)
        
    def predict(self, image: np.ndarray) -> int:
        """
        Predict the digit class for the given image.
        
        This method provides a unified interface regardless of the underlying algorithm.
        All models take the same input format (28x28x1) and return the same output format (int).
        
        Args:
            image: Input image as numpy array of shape (28, 28, 1)
                  Values should be in the range [0, 255]
            
        Returns:
            Predicted digit as an integer from 0 to 9
            
        Raises:
            ValueError: If the image shape is incorrect
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be a numpy array, got {type(image)}")
        
        if image.shape != (28, 28, 1):
            raise ValueError(
                f"Image must have shape (28, 28, 1), got {image.shape}"
            )
        
        # Use the underlying model to make prediction
        prediction = self.model.predict(image)
        
        return prediction
    
    def get_algorithm(self) -> str:
        """
        Get the current algorithm being used.
        
        Returns:
            Name of the algorithm
        """
        return self.algorithm
    
    def get_model(self) -> DigitClassificationInterface:
        """
        Get the underlying model instance.
        
        Returns:
            The model instance
        """
        return self.model
    
    def __repr__(self) -> str:
        """
        String representation of the classifier.
        
        Returns:
            String describing the classifier
        """
        return f"DigitClassifier(algorithm='{self.algorithm}', model={self.model})"


# Convenience function for quick usage
def create_classifier(algorithm: Literal['cnn', 'rf', 'rand']) -> DigitClassifier:
    """
    Factory function to create a DigitClassifier instance.
    
    Args:
        algorithm: The algorithm to use ('cnn', 'rf', or 'rand')
        
    Returns:
        DigitClassifier instance
    """
    return DigitClassifier(algorithm=algorithm)
