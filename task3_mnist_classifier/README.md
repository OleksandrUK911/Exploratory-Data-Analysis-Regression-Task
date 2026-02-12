# Task 3: MNIST Digit Classifier with OOP

## Problem Description

Build an object-oriented system for MNIST digit classification that supports multiple algorithms:
- **CNN** (Convolutional Neural Network): Uses 28×28×1 tensor input
- **RF** (Random Forest): Uses flattened 784-dimensional array input  
- **RAND** (Random Model): Uses 10×10 center crop, returns random values

The system should provide a unified interface (`DigitClassifier`) that handles different algorithms transparently.

## Project Structure

```
task3_mnist_classifier/
├── models/
│   ├── __init__.py              # Package initialization
│   ├── interface.py             # Abstract base class (DigitClassificationInterface)
│   ├── cnn_model.py             # CNN implementation
│   ├── rf_model.py              # Random Forest implementation
│   └── random_model.py          # Random baseline implementation
├── digit_classifier.py          # Unified DigitClassifier wrapper
├── example_usage.py             # Usage examples and tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Architecture

### Class Hierarchy

```
DigitClassificationInterface (ABC)
├── CNNModel
├── RandomForestModel
└── RandomModel

DigitClassifier (Wrapper)
└── Uses any DigitClassificationInterface implementation
```

### Design Principles

1. **Interface Segregation**: All models implement `DigitClassificationInterface`
2. **Open/Closed Principle**: Easy to add new models without modifying existing code
3. **Dependency Inversion**: `DigitClassifier` depends on the interface, not concrete implementations
4. **Single Responsibility**: Each class has one clear purpose

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
cd task3_mnist_classifier
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from digit_classifier import DigitClassifier
import numpy as np

# Create a sample 28x28x1 image
image = np.random.randint(0, 256, size=(28, 28, 1), dtype=np.uint8)

# Use CNN algorithm
classifier = DigitClassifier(algorithm='cnn')
prediction = classifier.predict(image)
print(f"Predicted digit: {prediction}")  # Output: 0-9

# Use Random Forest algorithm
classifier = DigitClassifier(algorithm='rf')
prediction = classifier.predict(image)
print(f"Predicted digit: {prediction}")

# Use Random baseline
classifier = DigitClassifier(algorithm='rand')
prediction = classifier.predict(image)
print(f"Predicted digit: {prediction}")
```

### Running Examples

```bash
python example_usage.py
```

This will demonstrate:
- All three algorithms (CNN, RF, Random)
- Multiple predictions
- Error handling
- Input validation

### Command-Line Usage

You can also import and use the classes directly:

```python
from models import CNNModel, RandomForestModel, RandomModel
import numpy as np

# Direct model usage
cnn = CNNModel()
image = np.random.randint(0, 256, size=(28, 28, 1), dtype=np.uint8)
prediction = cnn.predict(image)
```

## API Reference

### DigitClassifier

Main wrapper class providing a unified interface.

```python
DigitClassifier(algorithm: Literal['cnn', 'rf', 'rand'])
```

**Methods:**
- `predict(image: np.ndarray) -> int`: Predict digit from image
- `get_algorithm() -> str`: Get current algorithm name
- `get_model() -> DigitClassificationInterface`: Get underlying model

**Parameters:**
- `algorithm`: Algorithm to use ('cnn', 'rf', or 'rand')

**Returns:**
- Integer from 0 to 9 representing the predicted digit

### DigitClassificationInterface

Abstract base class that all models must implement.

**Abstract Methods:**
- `preprocess(image: np.ndarray) -> Union[np.ndarray, any]`: Preprocess image for model
- `predict(image: np.ndarray) -> int`: Make prediction
- `train(X: np.ndarray, y: np.ndarray)`: Train model (raises NotImplementedError)

### CNNModel

Convolutional Neural Network implementation.

**Architecture:**
- Conv2D (32 filters, 3×3) → MaxPooling
- Conv2D (64 filters, 3×3) → MaxPooling
- Conv2D (64 filters, 3×3)
- Flatten → Dense (64) → Dropout(0.5) → Dense (10, softmax)

**Input:** 28×28×1 tensor (normalized to [0, 1])

**Key Methods:**
- `load_pretrained_weights(weights_path: str)`: Load saved weights

### RandomForestModel

Random Forest classifier implementation.

**Parameters:**
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: 20)
- `random_state`: Random seed (default: 42)

**Input:** Flattened 784-dimensional array (normalized to [0, 1])

**Key Methods:**
- `load_pretrained_model(model_path: str)`: Load saved model

### RandomModel

Random baseline implementation for comparison.

**Input:** 10×10 center crop of the image

**Output:** Random integer from 0 to 9

**Parameters:**
- `random_state`: Random seed for reproducibility (optional)

## Input/Output Specifications

### Input Format

All models accept the **same input format**:
- **Type:** `numpy.ndarray`
- **Shape:** `(28, 28, 1)`
- **Data Type:** `uint8` or `float32`
- **Value Range:** `[0, 255]`

### Output Format

All models return the **same output format**:
- **Type:** `int`
- **Value Range:** `[0, 9]`
- **Meaning:** Predicted digit class

## Internal Processing

Each model handles preprocessing differently:

| Model | Preprocessing |
|-------|--------------|
| CNN | Normalize to [0,1], add batch dimension → (1, 28, 28, 1) |
| Random Forest | Flatten to 784-dim vector, normalize to [0,1] |
| Random | Extract center 10×10 crop |

The `DigitClassifier` wrapper ensures consistent input/output regardless of internal preprocessing.

## Adding New Models

To add a new classification model:

1. **Create a new class** in `models/` directory:

```python
from models.interface import DigitClassificationInterface
import numpy as np

class MyNewModel(DigitClassificationInterface):
    def preprocess(self, image: np.ndarray):
        # Your preprocessing logic
        return processed_image
    
    def predict(self, image: np.ndarray) -> int:
        processed = self.preprocess(image)
        # Your prediction logic
        return prediction
    
    def train(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Training not implemented")
```

2. **Update `models/__init__.py`**:

```python
from .my_new_model import MyNewModel

__all__ = [..., 'MyNewModel']
```

3. **Update `digit_classifier.py`**:

```python
SUPPORTED_ALGORITHMS = ['cnn', 'rf', 'rand', 'mynew']

def _create_model(self):
    # ... existing code ...
    elif self.algorithm == 'mynew':
        return MyNewModel()
```

## Testing

The `example_usage.py` script includes comprehensive tests:

```bash
python example_usage.py
```

**Tests include:**
- ✓ All three algorithms work correctly
- ✓ Multiple predictions  
- ✓ Invalid algorithm handling
- ✓ Invalid image shape handling
- ✓ Invalid input type handling
- ✓ Output validation (must be 0-9)

## Notes

### Training Not Implemented

As per requirements, the `train()` method raises `NotImplementedError`. This is intentional as the focus is on the OOP structure and prediction interface.

### Model Weights

- **CNN**: Requires pretrained weights for production use. Call `load_pretrained_weights(path)` to load them.
- **Random Forest**: Requires pretrained model. Call `load_pretrained_model(path)` to load it.
- **Random**: No training needed (baseline model).

For demonstration purposes without trained models, the system will:
- **CNN**: Work with random initialization (poor accuracy)
- **Random Forest**: Return random predictions with a warning if not trained
- **Random**: Always work (returns random values by design)

## Example Output

```
==============================================================
Testing CNN Algorithm
==============================================================
Created: DigitClassifier(algorithm='cnn', model=CNNModel(status=untrained))

Input image shape: (28, 28, 1)
Input image dtype: uint8
Input image range: [0, 255]

Predicted digit: 7
Prediction type: <class 'int'>
✓ Prediction is valid!
```

## Design Benefits

1. **Extensibility**: Easy to add new models
2. **Consistency**: All models have the same interface
3. **Type Safety**: Abstract base class enforces method signatures
4. **Testability**: Each component can be tested independently
5. **Maintainability**: Clear separation of concerns

## Author

Oleksandr  
February 2026

## License

This project is for educational and assessment purposes.
