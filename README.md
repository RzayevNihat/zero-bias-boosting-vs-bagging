# Zero-Bias Boosting vs. Bagging 🤖

A comprehensive implementation of ensemble machine learning algorithms from scratch, comparing boosting and bagging approaches through empirical analysis on real-world datasets.

## Table of Contents

- [Overview]
- [Features]
- [Tech Stack]
- [Project Structure]
- [Prerequisites]
- [Installation]
- [Configuration]
- [Running the Project]
- [Usage]
- [Experiments]
- [API Documentation]
- [Testing]
- [Error Handling]
- [Security]
- [Performance Notes]
- [Known Limitations]
- [Future Improvements]
- [Deployment]
- [Acknowledgements]

---

## Overview

This project investigates the central question: **"Under what conditions does boosting outperform bagging, and vice versa, and why?"** 

By implementing three core machine learning algorithms from scratch — Decision Trees (CART), AdaBoost, and Random Forest — this project enables a rigorous empirical comparison of two fundamental ensemble learning philosophies. The analysis is supported by unsupervised learning techniques (PCA, K-Means, DBSCAN) for data exploration and visualization.

**Key Purpose**: Educational implementation of ensemble methods without relying on scikit-learn's built-in classifiers, allowing for deep understanding of how these algorithms work at a mathematical level.

---

## Features

- **Decision Tree Classifier**
  - Support for Gini impurity and entropy criteria
  - Configurable max depth, min samples split, and feature subsampling
  - Weighted sample handling for boosting algorithms
  - Feature importance calculation

- **AdaBoost Classifier**
  - SAMME algorithm for multi-class classification
  - SAMME.R variant with real-valued class probabilities
  - Decision stumps as weak learners
  - Sample weight reweighting during training

- **Gradient Boosting Classifier**
  - Binary classification with log-loss objective
  - Regression tree weak learners
  - Numerically stable sigmoid computations

- **Random Forest Classifier**
  - Bootstrap aggregation with configurable tree count
  - Out-of-bag (OOB) scoring for model evaluation
  - Parallel tree fitting using multiprocessing
  - Averaged class probabilities
  - Mean impurity-based feature importances
  - Random feature subsampling at each split

- **Comprehensive Metrics**
  - Accuracy, macro precision, recall, F1
  - ROC-AUC score
  - Calibration metrics (log loss, Brier score, ECE)
  - Paired statistical tests

- **Extensive Experiments**
  - Head-to-head comparison under fixed resources (5-fold CV)
  - Bias-variance decomposition analysis
  - Noise robustness evaluation
  - AdaBoost and Random Forest scaling studies
  - Gradient Boosting vs. AdaBoost comparison
  - Coverage audit for implementation completeness

- **Data Preprocessing**
  - Dataset loading and caching
  - Feature standardization
  - Stratified K-fold cross-validation support

---

## Tech Stack

| Component                         | Technology   | Version    |
|-----------------------------------|--------------|------------|
| **Language**                      | Python       | 3.9+       |
| **Numerical Computing**           | NumPy        | >=1.26, <3 |
| **Scientific Computing**          | SciPy        | >=1.11, <2 |
| **Machine Learning (comparison)** | scikit-learn | >=1.4, <2  |
| **Parallelization**               | joblib       | >=1.3, <2  |
| **Data Manipulation**             | pandas       | >=2.2, <3  |
| **Visualization**                 | matplotlib   | >=3.8, <4  |
| **Testing**                       | pytest       | >=8, <9    |

---

## Project Structure

```
zero-bias-boosting-vs-bagging/
├── src/                          # Core implementation modules
│   ├── trees/                    # Decision Tree implementation
│   │   ├── decision_tree.py      # Main DecisionTree classifier
│   │   ├── node.py               # Tree node structure
│   │   ├── impurity.py           # Gini/entropy calculations
│   │   ├── splitter.py           # Best split finding logic
│   │   └── stopping.py           # Tree stopping criteria
│   │
│   ├── boosting/                 # Boosting algorithms
│   │   ├── adaboost.py           # AdaBoost and SAMME variants
│   │   ├── gradient_boosting.py  # Gradient Boosting classifier
│   │   └── _validation.py        # Input validation utilities
│   │
│   ├── bagging/                  # Bagging/ensemble algorithms
│   │   ├── random_forest.py      # Random Forest classifier
│   │   └── __init__.py
│   │
│   ├── metrics/                  # Evaluation metrics
│   │   ├── evaluation.py         # Classification metrics (accuracy, F1, AUC, etc.)
│   │   └── __init__.py
│   │
│   ├── experiments/              # Experimental scripts
│   │   ├── head_to_head.py       # Experiment 4: head-to-head comparison
│   │   ├── bias_variance.py      # Experiment 6: bias-variance decomposition
│   │   ├── adaboost_scaling.py   # Experiment 2: AdaBoost scaling
│   │   ├── rf_scaling.py         # Experiment 3: Random Forest scaling
│   │   ├── noise_robustness.py   # Experiment 5: noise robustness
│   │   ├── gbm_comparison.py     # Gradient Boosting vs AdaBoost
│   │   ├── coverage_audit.py     # Implementation completeness check
│   │   ├── run_all.py            # Master experiment runner
│   │   ├── utils.py              # Experiment utilities
│   │   └── rf_utils.py           # Random Forest specific utilities
│   │
│   ├── unsupervised/             # Unsupervised learning (placeholder)
│   │   ├── pca.py                # PCA (not yet implemented)
│   │   ├── kmeans.py             # K-Means clustering (not yet implemented)
│   │   ├── dbscan.py             # DBSCAN clustering (not yet implemented)
│   │   └── __init__.py
│   │
│   └── utils/                    # Utility functions
│       ├── preprocessing.py      # Data loading and preprocessing
│       └── __init__.py
│
├── tests/                        # Comprehensive test suite
│   ├── conftest.py               # pytest fixtures (test data generation)
│   ├── test_decision_tree.py     # Decision Tree tests
│   ├── test_adaboost.py          # AdaBoost tests
│   ├── test_gradient_boosting.py # Gradient Boosting tests
│   ├── test_random_forest.py     # Random Forest tests
│   ├── test_impurity.py          # Impurity function tests
│   ├── test_splitter.py          # Split finding tests
│   ├── test_evaluation.py        # Metrics tests
│   ├── test_metrics.py           # Metric calculation tests
│   ├── test_preprocessing.py     # Data preprocessing tests
│   └── test_unsupervised.py      # Unsupervised learning tests
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── DecisionTreeNotebook.ipynb  # Decision Tree walkthrough
│   └── exploration.ipynb           # Data exploration
│
├── data/                         # Dataset storage
│   └── .gitkeep
│
├── figures/                      # Generated visualizations
│   └── .gitkeep
│
├── results/                      # Experiment results (CSV/JSON)
│   └── .gitkeep
│
├── contribution/                 # Project documentation
│   ├── contribution_report.pdf   # Team contribution breakdown
│   └── contribution_report.tex
│
├── presentation/                 # Project presentation
│   ├── presentation.pdf          # Slide deck for defense
│   └── presentation.tex
│
├── report/                       # Technical report
│   ├── report.pdf                # Compiled technical paper (IEEE format)
│   └── report.tex
│
├── requirements.txt              # Python dependencies
├── download_data.sh              # Data download script (optional)
├── README.md                     # This file
├── .gitignore                    # Git ignore patterns
└── .git/                         # Git repository (if cloned from remote)
```

### Key Directories Explained

| Directory          | Purpose                                                           |
|--------------------|-------------------------------------------------------------------|
| `src/trees/`       | Core Decision Tree implementation with all supporting components  |
| `src/boosting/`    | Boosting ensemble implementations (AdaBoost, Gradient Boosting)   |
| `src/bagging/`     | Bagging ensemble implementation (Random Forest)                   |
| `src/experiments/` | Controlled empirical studies comparing algorithms across datasets |
| `src/metrics/`     | Custom implementations of classification evaluation metrics       |
| `tests/`           | Comprehensive unit tests ensuring correctness against edge cases  |
| `notebooks/`       | Interactive Jupyter notebooks for exploration and demonstration   |
| `data/`            | Real-world datasets (WDBC, Adult, CoverType)                      |
| `results/`         | Experiment outputs (CSV/JSON format for analysis)                 |

---

## Prerequisites

- **Python**: 3.9 or higher
- **Package Manager**: pip, conda, or poetry
- **OS**: Linux, macOS, or Windows
- **Memory**: Minimum 2GB (4GB+ recommended for full experiments)
- **Disk Space**: ~500MB for data and results

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/RzayevNihat/zero-bias-boosting-vs-bagging.git
cd zero-bias-boosting-vs-bagging
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import numpy, scipy, sklearn, pytest; print('All dependencies installed!')"
```

---

## Configuration

### Environment Variables

The project supports the following environment variables:

| Variable      | Description                               | Required | Default      |
|---------------|-------------------------------------------|----------|--------------|
| `DATA_DIR`    | Path to dataset storage directory         | No       | `./data/`    |
| `RESULTS_DIR` | Path for experiment results output        | No       | `./results/` |
| `FIGURES_DIR` | Path for visualization output             | No       | `./figures/` |
| `RANDOM_SEED` | Global random seed for reproducibility    | No       | `42`         |
| `N_JOBS`      | Number of parallel jobs for Random Forest | No       | `-1`         |

### Example Configuration

```bash
# Set environment variables
export DATA_DIR="/path/to/data"
export RESULTS_DIR="/path/to/results"
export RANDOM_SEED=42
export N_JOBS=4

# Run experiments
python src/experiments/head_to_head.py
```

---

## Running the Project

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running in Development Mode

```bash
# Import and use modules directly in Python
python -c "from src.trees.decision_tree import DecisionTree; print('Tree imported successfully')"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_decision_tree.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Running Experiments

```bash
# Run individual experiments
python src/experiments/head_to_head.py
python src/experiments/bias_variance.py
python src/experiments/adaboost_scaling.py
python src/experiments/rf_scaling.py
python src/experiments/noise_robustness.py
python src/experiments/gbm_comparison.py

# Run all experiments (when implemented)
python src/experiments/run_all.py
```

### Linting and Formatting (if available)

```bash
# Check code style (if tools configured)
pytest tests/

# Format code (if formatter configured)
# No standard formatter configured in this project
```

---

## Usage

### Basic Example: Decision Tree

```python
import numpy as np
from src.trees.decision_tree import DecisionTree

# Generate sample data
X = np.array([[2.5], [5.0], [7.5], [3.0], [8.0]])
y = np.array([0, 1, 1, 0, 1])

# Create and train decision tree
tree = DecisionTree(max_depth=3, criterion="gini")
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
probabilities = tree.predict_proba(X)

print(f"Predictions: {predictions}")
print(f"Probabilities:\n{probabilities}")
```

### Example: AdaBoost Classifier

```python
from src.boosting.adaboost import AdaBoostClassifier
from sklearn.datasets import load_iris

# Load sample dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create and train AdaBoost
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada.fit(X, y)

# Predictions and probabilities
predictions = ada.predict(X)
proba = ada.predict_proba(X)

print(f"Accuracy: {(predictions == y).mean():.4f}")
```

### Example: Random Forest Classifier

```python
from src.bagging.random_forest import RandomForestClassifier
from sklearn.datasets import load_wine

# Load sample dataset
wine = load_wine()
X, y = wine.data, wine.target

# Create and train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    max_features="sqrt",
    n_jobs=-1,  # Use all available cores
    random_state=42
)
rf.fit(X, y)

# Predictions with out-of-bag (OOB) score
predictions = rf.predict(X)
oob_score = rf.oob_score_
feature_importances = rf.feature_importances_

print(f"OOB Score: {oob_score:.4f}")
print(f"Feature Importances: {feature_importances}")
```

### Example: Using Custom Metrics

```python
from src.metrics.evaluation import classification_metrics
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1, 1])
y_proba = np.array([
    [0.9, 0.1],
    [0.2, 0.8],
    [0.6, 0.4],
    [0.8, 0.2],
    [0.3, 0.7],
    [0.1, 0.9]
])

metrics = classification_metrics(y_true, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

---

## API Documentation

### Decision Tree

**Module**: `src.trees.decision_tree`

**Class**: `DecisionTree`

```python
class DecisionTree:
    """
    Decision Tree Classifier using CART algorithm.
    
    Parameters:
        max_depth (int | None): Maximum tree depth. Default: None (unlimited)
        min_samples_split (int): Minimum samples required to split. Default: 2
        criterion (str): Impurity measure - "gini" or "entropy". Default: "gini"
        max_features (str | int | None): Feature subsampling strategy. Default: None
        random_state (int | None): Random seed for reproducibility. Default: None
    """
    
    def fit(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> DecisionTree
    def predict(X: np.ndarray) -> np.ndarray
    def predict_proba(X: np.ndarray) -> np.ndarray
    def feature_importances_(self) -> np.ndarray
```

| Method            | Description             | HTTP Method |
|-------------------|-------------------------|-------------|
| `fit()`           | Train tree on data      |      -      |
| `predict()`       | Get class predictions   |      -      |
| `predict_proba()` | Get class probabilities |      -      |

### AdaBoost

**Module**: `src.boosting.adaboost`

**Class**: `AdaBoostClassifier` / `AdaBoostSAMMERClassifier`

```python
class AdaBoostClassifier:
    """
    AdaBoost Classifier using SAMME algorithm.
    
    Parameters:
        n_estimators (int): Number of weak learners. Default: 50
        learning_rate (float): Shrinkage factor. Default: 1.0
        criterion (str): Impurity measure for stumps. Default: "gini"
        random_state (int | None): Random seed. Default: None
    """
    
    def fit(X: np.ndarray, y: np.ndarray) -> AdaBoostClassifier
    def predict(X: np.ndarray) -> np.ndarray
    def predict_proba(X: np.ndarray) -> np.ndarray
```

### Random Forest

**Module**: `src.bagging.random_forest`

**Class**: `RandomForestClassifier`

```python
class RandomForestClassifier:
    """
    Random Forest Classifier with bootstrap aggregation.
    
    Parameters:
        n_estimators (int): Number of trees. Default: 100
        max_depth (int | None): Maximum tree depth. Default: None
        min_samples_split (int): Min samples to split. Default: 2
        max_features (str | int | None): Feature subsampling. Default: "sqrt"
        n_jobs (int): Number of parallel jobs. Default: 1
        random_state (int | None): Random seed. Default: None
    """
    
    def fit(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier
    def predict(X: np.ndarray) -> np.ndarray
    def predict_proba(X: np.ndarray) -> np.ndarray
    def oob_score_(self) -> float
    def feature_importances_(self) -> np.ndarray
```

### Metrics

**Module**: `src.metrics.evaluation`

**Functions**:

```python
def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: Sequence[Any] | None = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Returns dictionary with keys:
    - accuracy
    - macro_precision
    - macro_recall
    - macro_f1
    - roc_auc
    - log_loss
    - brier_score
    - expected_calibration_error
    """
```

---

## Screenshots

Screenshots and visualizations from experiments would appear in the `figures/` directory after running experiments. These include:

- Bias-variance decomposition plots
- Cross-validation performance comparisons
- Feature importance rankings
- Scaling analysis (dataset size vs. performance)
- Noise robustness curves
- Calibration plots

*(Placeholder: Run experiments to generate actual visualizations)*

---

## Deployment

### Running on Different Environments

**Local Development**:
```bash
python src/experiments/head_to_head.py
```

**On Server/Cluster** (with multiple datasets):
```bash
# Results saved to results/ directory
python src/experiments/run_all.py --output ./results/
```

---

## Error Handling

The project implements robust error handling:

- **Input Validation**: All classifiers validate input shapes and types
- **Edge Cases**: Handles single-feature data, pure label sets, empty splits
- **Numerical Stability**: Log-loss uses numerically stable sigmoid, prevents log(0)
- **Missing Implementations**: Graceful fallback when unsupervised modules not yet implemented
- **Cross-validation Errors**: Stratified K-fold ensures balanced folds

Example error messages:
```
ValueError: criterion must be either 'gini' or 'entropy'.
RuntimeError: Must call fit() before predict()
ValueError: X and y have inconsistent lengths
```

---

## Performance Notes

### Optimizations

- **Decision Tree**: O(n log n) per split using sorted midpoint search
- **AdaBoost**: Vectorized NumPy operations for weight updates
- **Random Forest**: Multiprocessing parallelization across tree fitting
- **Memory Efficiency**: Bootstrap samples avoid data duplication

### Benchmarks

Performance scales approximately as:

| Algorithm     | Time Complexity       | Space Complexity |
|---------------|-----------------------|------------------|
| Decision Tree | O(n × m × log n)      | O(n + m)         | 
| AdaBoost      | O(T × n × m × log n)  | O(n + m)         |
| Random Forest | O(B × n × m × log n)* | O(B × (n + m))   |

*T = num estimators, B = num trees, n = num samples, m = num features, with parallelization benefits on multi-core systems

### Dataset Sizes Tested

- WDBC: 569 samples, 30 features
- Adult: ~30,000 samples, 14 features
- CoverType: ~580,000 samples, 54 features

---

## Known Limitations

1. **Unsupervised Modules**: PCA, K-Means, and DBSCAN are placeholder stubs not yet implemented
2. **Multi-class Gradient Boosting**: GBM only supports binary classification (suitable for comparison experiments)
3. **Missing Data**: No built-in handling for NaN/missing values
4. **Categorical Features**: Input features must be numeric; categorical encoding required beforehand
5. **Imbalanced Data**: No built-in class weighting or SMOTE support
6. **Hyperparameter Tuning**: No grid/random search; manual parameter specification required
7. **Feature Scaling**: Tree-based models are scale-invariant, but some metrics benefit from standardization
8. **Large Sparse Data**: Optimized for dense matrices; sparse matrix support not implemented

---

## Future Improvements

1. **Unsupervised Learning**: Implement full PCA, K-Means, DBSCAN modules with visualizations
2. **Multi-class Gradient Boosting**: Extend GBM to multinomial objectives
3. **Missing Value Handling**: Add imputation and missing indicator support
4. **Categorical Features**: Native categorical feature support with one-hot encoding
5. **Feature Scaling Options**: Built-in StandardScaler, MinMaxScaler integration
6. **Hyperparameter Search**: Grid search, random search, Bayesian optimization
7. **Model Persistence**: Save/load models to disk (pickle/joblib format)
8. **Cross-validation Strategies**: Time-series split, group k-fold
9. **Early Stopping**: Validation set monitoring for boosting algorithms
10. **GPU Acceleration**: CUDA/CuPy support for large-scale experiments
11. **Web Interface**: REST API for model serving
12. **Additional Ensemble Methods**: XGBoost variants, Stacking, Blending

---

## Testing

### Test Coverage

The project includes comprehensive unit tests for:

- **Decision Tree**: Correctness on XOR/Gaussian data, edge cases, weighted samples, sklearn parity
- **AdaBoost**: Multi-class classification, weight updates, convergence
- **Random Forest**: Bootstrap correctness, OOB scoring, feature importance
- **Metrics**: All classification metrics against reference implementations
- **Preprocessing**: Data loading, standardization, splits

### Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_decision_tree.py::TestDecisionTreeCorrectness -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Test Statistics

- **Total Test Files**: 11
- **Estimated Test Count**: 100+ individual tests
- **Coverage Target**: >85% of critical paths

---

## Acknowledgements

This project implements algorithms and approaches from the following sources:

- **The Elements of Statistical Learning** (Hastie, Tibshirani & Friedman) — primary reference for tree-based methods, bias-variance decomposition, and ensemble learning theory
- **Decision Tree CART Algorithm** — Quinlan (1986) and follow-up classification and regression tree literature
- **AdaBoost SAMME Algorithm** — Zhu et al. multi-class extension to classic Freund & Schapire algorithm
- **Random Forest** — Breiman (2001) bootstrap aggregation with random features
- **Gradient Boosting** — Friedman (2001) functional gradient descent approach

Special acknowledgment to:
- **scikit-learn** developers for reference implementations used in validation
- Course instructors and teaching staff for guidance throughout the semester