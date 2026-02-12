# Data Science Engineer - Test Assignment

**Author:** Oleksandr  
**Date:** February 2026

This repository contains solutions to a three-part Data Science Engineer assessment covering classical algorithms, machine learning regression, and object-oriented programming.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Task 1: Counting Islands](#task-1-counting-islands)
  - [Task 2: Regression on Tabular Data](#task-2-regression-on-tabular-data)
  - [Task 3: MNIST Classifier OOP](#task-3-mnist-classifier-oop)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## 🎯 Overview

This repository demonstrates proficiency in:
- **Classical Algorithms**: Efficient graph traversal (DFS)
- **Machine Learning**: Regression modeling, EDA, model deployment
- **Software Engineering**: OOP principles, design patterns, clean code

All solutions include comprehensive documentation, testing, and follow best practices for production-ready code.

## 📁 Project Structure

```
Test task - Quantum/
├── task1_counting_islands/       # Task 1: Island counting algorithm
│   ├── counting_islands.py       # Main implementation
│   ├── test_cases.py             # Unit tests
│   └── README.md                 # Task-specific documentation
│
├── task2_regression/              # Task 2: Regression modeling
│   ├── eda_notebook.ipynb        # Exploratory Data Analysis
│   ├── train.py                  # Model training script
│   ├── predict.py                # Prediction script
│   ├── requirements.txt          # Python dependencies
│   ├── predictions.csv           # Output predictions
│   └── README.md                 # Task-specific documentation
│
├── task3_mnist_classifier/        # Task 3: OOP digit classifier
│   ├── models/                   # Model implementations
│   │   ├── __init__.py
│   │   ├── interface.py          # Abstract base class
│   │   ├── cnn_model.py          # CNN implementation
│   │   ├── rf_model.py           # Random Forest implementation
│   │   └── random_model.py       # Random baseline
│   ├── digit_classifier.py       # Unified classifier wrapper
│   ├── example_usage.py          # Usage examples
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # Task-specific documentation
│
├── train.csv                      # Training dataset (regression)
├── hidden_test.csv               # Test dataset (regression)
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 📝 Tasks

### Task 1: Counting Islands

**Problem:** Count the number of islands in a binary matrix (1=land, 0=water).

**Solution Highlights:**
- Depth-First Search (DFS) algorithm
- Time Complexity: O(M × N)
- Space Complexity: O(M × N)
- Comprehensive unit tests
- All test cases passing

**Quick Start:**
```bash
cd task1_counting_islands
python counting_islands.py
python -m unittest test_cases.py
```

[📖 Full Documentation](task1_counting_islands/README.md)

---

### Task 2: Regression on Tabular Data

**Problem:** Build a regression model to predict a target variable from 53 anonymized features.

**Solution Highlights:**
- Exploratory Data Analysis in Jupyter notebook
- XGBoost, Random Forest, and LightGBM implementations
- Cross-validation for robust evaluation
- RMSE optimization
- Production-ready training and prediction scripts
- Comprehensive documentation

**Quick Start:**
```bash
cd task2_regression
pip install -r requirements.txt

# Train model
python train.py --model xgboost --cv 5

# Generate predictions
python predict.py --model xgboost --output predictions.csv
```

**Key Features:**
- ✅ EDA with visualizations
- ✅ Multiple model options
- ✅ Command-line interface
- ✅ Model persistence
- ✅ Prediction validation

[📖 Full Documentation](task2_regression/README.md)

---

### Task 3: MNIST Classifier OOP

**Problem:** Create an extensible OOP system for MNIST digit classification supporting multiple algorithms.

**Solution Highlights:**
- Abstract base class (`DigitClassificationInterface`)
- Three model implementations:
  - CNN (Convolutional Neural Network)
  - Random Forest
  - Random Baseline
- Unified `DigitClassifier` wrapper
- Consistent input/output interface
- Easily extensible architecture
- SOLID principles applied

**Quick Start:**
```bash
cd task3_mnist_classifier
pip install -r requirements.txt
python example_usage.py
```

**Example Usage:**
```python
from digit_classifier import DigitClassifier
import numpy as np

# Create image (28x28x1)
image = np.random.randint(0, 256, size=(28, 28, 1), dtype=np.uint8)

# Use any algorithm
classifier = DigitClassifier(algorithm='cnn')
prediction = classifier.predict(image)
print(f"Predicted digit: {prediction}")  # 0-9
```

[📖 Full Documentation](task3_mnist_classifier/README.md)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Clone Repository

```bash
git clone <repository-url>
cd "Test task - Quantum"
```

### Install Dependencies

Each task has its own `requirements.txt`. Install as needed:

```bash
# Task 1 - No dependencies needed (pure Python)

# Task 2 - Regression
cd task2_regression
pip install -r requirements.txt

# Task 3 - MNIST Classifier
cd task3_mnist_classifier
pip install -r requirements.txt
```

### Alternative: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r task2_regression/requirements.txt
pip install -r task3_mnist_classifier/requirements.txt
```

## 🎮 Quick Start

### Run All Tasks

```bash
# Task 1: Counting Islands
cd task1_counting_islands
python counting_islands.py
python test_cases.py

# Task 2: Regression
cd ../task2_regression
pip install -r requirements.txt
python train.py
python predict.py

# Task 3: MNIST Classifier
cd ../task3_mnist_classifier
pip install -r requirements.txt
python example_usage.py
```

## 🛠 Technologies Used

### Core
- **Python 3.8+**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

### Machine Learning
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **scikit-learn** - ML algorithms and utilities
- **TensorFlow/Keras** - Deep learning

### Data Science
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Jupyter** - Interactive notebooks
- **SciPy** - Scientific computing

### Software Engineering
- **unittest** - Unit testing
- **joblib** - Model persistence
- **ABC** - Abstract base classes
- **Type Hints** - Code documentation

## 📊 Results Summary

| Task | Status | Key Metrics |
|------|--------|-------------|
| Task 1: Islands | ✅ Complete | All test cases pass, O(M×N) complexity |
| Task 2: Regression | ✅ Complete | RMSE optimized, multiple models |
| Task 3: MNIST OOP | ✅ Complete | 3 models, fully extensible architecture |

## 📄 Documentation

Each task includes comprehensive README files with:
- Problem description
- Solution approach
- Installation instructions
- Usage examples
- API reference
- Architecture diagrams (where applicable)

## 🧪 Testing

### Task 1
```bash
cd task1_counting_islands
python -m unittest test_cases.py
```

### Task 2
```bash
cd task2_regression
python train.py --cv 5  # Cross-validation
```

### Task 3
```bash
cd task3_mnist_classifier
python example_usage.py  # Runs all tests
```

## 💡 Key Features

### Code Quality
- ✅ Comprehensive documentation
- ✅ Type hints
- ✅ Error handling
- ✅ Input validation
- ✅ Unit tests

### Architecture
- ✅ SOLID principles
- ✅ Design patterns
- ✅ Modular structure
- ✅ Extensible design
- ✅ Clean code practices

### Functionality
- ✅ Command-line interfaces
- ✅ Jupyter notebooks
- ✅ Multiple algorithms
- ✅ Model persistence
- ✅ Production-ready scripts

## 🔍 Code Highlights

### Efficient Algorithms
- DFS for island counting with optimal complexity

### Production-Ready ML
- Cross-validation for robust evaluation
- Model persistence and loading
- Command-line interfaces
- Comprehensive error handling

### Clean Architecture
- Abstract base classes for extensibility
- Dependency inversion principle
- Single responsibility principle
- Open/closed principle

## 📞 Contact

**Author:** Oleksandr  
**Date:** February 2026  
**Purpose:** Data Science Engineer Technical Assessment

---

## 📜 License

This project is submitted for technical assessment purposes.

---

## 🙏 Acknowledgments

Thank you for reviewing this submission. Each task demonstrates different aspects of data science and software engineering expertise, from algorithmic thinking to machine learning to software design.
