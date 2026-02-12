"""
Example Usage of DigitClassifier
Author: Oleksandr
Date: February 2026

This script demonstrates how to use the DigitClassifier with different algorithms.
"""

import numpy as np
from digit_classifier import DigitClassifier


def create_sample_image() -> np.ndarray:
    """
    Create a sample 28x28x1 image for demonstration.
    
    Returns:
        Random image of shape (28, 28, 1)
    """
    # Create a random image with values in [0, 255]
    image = np.random.randint(0, 256, size=(28, 28, 1), dtype=np.uint8)
    return image


def demonstrate_algorithm(algorithm: str):
    """
    Demonstrate a specific algorithm.
    
    Args:
        algorithm: Algorithm name ('cnn', 'rf', or 'rand')
    """
    print(f"\n{'='*60}")
    print(f"Testing {algorithm.upper()} Algorithm")
    print(f"{'='*60}")
    
    # Create classifier
    classifier = DigitClassifier(algorithm=algorithm)
    print(f"Created: {classifier}")
    
    # Create sample image
    sample_image = create_sample_image()
    print(f"\nInput image shape: {sample_image.shape}")
    print(f"Input image dtype: {sample_image.dtype}")
    print(f"Input image range: [{sample_image.min()}, {sample_image.max()}]")
    
    # Make prediction
    prediction = classifier.predict(sample_image)
    print(f"\nPredicted digit: {prediction}")
    print(f"Prediction type: {type(prediction)}")
    
    # Verify prediction is valid
    assert 0 <= prediction <= 9, "Prediction must be between 0 and 9"
    assert isinstance(prediction, (int, np.integer)), "Prediction must be an integer"
    
    print("✓ Prediction is valid!")


def test_multiple_predictions(algorithm: str, num_predictions: int = 5):
    """
    Test multiple predictions with the same classifier.
    
    Args:
        algorithm: Algorithm name
        num_predictions: Number of predictions to make
    """
    print(f"\n{'='*60}")
    print(f"Testing {num_predictions} predictions with {algorithm.upper()}")
    print(f"{'='*60}")
    
    classifier = DigitClassifier(algorithm=algorithm)
    predictions = []
    
    for i in range(num_predictions):
        image = create_sample_image()
        prediction = classifier.predict(image)
        predictions.append(prediction)
        print(f"Prediction {i+1}: {prediction}")
    
    print(f"\nAll predictions: {predictions}")
    print(f"Unique values: {set(predictions)}")


def test_error_handling():
    """
    Test error handling with invalid inputs.
    """
    print(f"\n{'='*60}")
    print("Testing Error Handling")
    print(f"{'='*60}")
    
    # Test 1: Invalid algorithm
    print("\nTest 1: Invalid algorithm")
    try:
        classifier = DigitClassifier(algorithm='invalid')
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test 2: Invalid image shape
    print("\nTest 2: Invalid image shape")
    try:
        classifier = DigitClassifier(algorithm='rand')
        invalid_image = np.random.randint(0, 256, size=(32, 32, 1), dtype=np.uint8)
        classifier.predict(invalid_image)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test 3: Invalid image type
    print("\nTest 3: Invalid image type")
    try:
        classifier = DigitClassifier(algorithm='rand')
        classifier.predict([1, 2, 3])
        print("✗ Should have raised TypeError")
    except TypeError as e:
        print(f"✓ Correctly raised TypeError: {e}")


def main():
    """
    Main function to run all demonstrations.
    """
    print("="*60)
    print("DIGIT CLASSIFIER DEMONSTRATION")
    print("="*60)
    
    # Demonstrate each algorithm
    for algorithm in ['cnn', 'rf', 'rand']:
        demonstrate_algorithm(algorithm)
    
    # Test multiple predictions
    test_multiple_predictions('rand', num_predictions=10)
    
    # Test error handling
    test_error_handling()
    
    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
