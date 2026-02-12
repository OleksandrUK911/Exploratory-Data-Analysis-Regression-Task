"""
Digit Classification Interface
Author: Oleksandr
Date: February 2026

This module defines the interface that all digit classification models must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class DigitClassificationInterface(ABC):
    """
    Abstract base class for digit classification models.
    
    All digit classification models must implement this interface to ensure
    consistent behavior and allow for easy addition of new models.
    """
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Union[np.ndarray, any]:
        """
        Preprocess the input image to the format required by the specific model.
        
        Args:
            image: Input image as numpy array of shape (28, 28, 1)
            
        Returns:
            Preprocessed data in the format required by the model
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Predict the digit class for the given image.
        
        Args:
            image: Input image as numpy array of shape (28, 28, 1)
            
        Returns:
            Predicted digit (integer from 0 to 9)
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on the given data.
        
        Note: For this assessment, this method should raise NotImplementedError.
        
        Args:
            X: Training images
            y: Training labels
            
        Raises:
            NotImplementedError: Training is not implemented for this assessment
        """
        raise NotImplementedError("Training is not implemented for this assessment")
    
    def __repr__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            Model class name
        """
        return f"{self.__class__.__name__}()"
