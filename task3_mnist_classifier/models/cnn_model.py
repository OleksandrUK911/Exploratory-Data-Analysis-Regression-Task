"""
CNN Model for MNIST Digit Classification
Author: Oleksandr
Date: February 2026

This module implements a Convolutional Neural Network for digit classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .interface import DigitClassificationInterface


class CNNModel(DigitClassificationInterface):
    """
    Convolutional Neural Network model for digit classification.
    
    Input: 28x28x1 tensor (grayscale image)
    Output: Single integer from 0 to 9
    """
    
    def __init__(self):
        """
        Initialize the CNN model.
        """
        self.model = self._build_model()
        self.is_trained = False
        
    def _build_model(self) -> keras.Model:
        """
        Build the CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for CNN.
        
        Args:
            image: Input image of shape (28, 28, 1)
            
        Returns:
            Preprocessed image of shape (1, 28, 28, 1) with values in [0, 1]
        """
        # Ensure correct shape
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected image shape (28, 28, 1), got {image.shape}")
        
        # Normalize pixel values to [0, 1]
        processed = image.astype('float32') / 255.0
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def predict(self, image: np.ndarray) -> int:
        """
        Predict the digit class for the given image.
        
        Args:
            image: Input image of shape (28, 28, 1)
            
        Returns:
            Predicted digit (0-9)
        """
        # Preprocess the image
        processed_image = self.preprocess(image)
        
        # Get predictions
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Return the class with highest probability
        predicted_class = np.argmax(predictions[0])
        
        return int(predicted_class)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the CNN model.
        
        Raises:
            NotImplementedError: Training is not implemented for this assessment
        """
        raise NotImplementedError("Training is not implemented for this assessment")
    
    def load_pretrained_weights(self, weights_path: str):
        """
        Load pretrained weights for the model.
        
        Args:
            weights_path: Path to the weights file
        """
        self.model.load_weights(weights_path)
        self.is_trained = True
        print(f"Loaded pretrained weights from {weights_path}")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"CNNModel(status={status})"
