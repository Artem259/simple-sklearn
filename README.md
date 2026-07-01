# simple-sklearn

[![CI/CD](https://github.com/Artem259/simple-sklearn/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Artem259/simple-sklearn/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/Artem259/simple-sklearn/graph/badge.svg)](https://codecov.io/gh/Artem259/simple-sklearn)
[![PyPI version](https://badge.fury.io/py/simple-sklearn.svg)](https://pypi.org/p/simple-sklearn)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-GitHub_Pages-blue.svg)](https://Artem259.github.io/simple-sklearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**simple-sklearn** is a Python package designed to provide clear, readable, and highly pythonic implementations of
fundamental machine learning algorithms. Abstracting away from complex low-level optimizations, this library
focuses on clarity and educational value using high-level libraries like `numpy`, `pandas`, and `scipy`.

Every model is designed to integrate seamlessly with scikit-learn's estimator API, inheriting from
`sklearn.base.BaseEstimator` and the appropriate mixins (`ClassifierMixin` or `ClusterMixin`). This allows you
to plug these simple implementations directly into `scikit-learn` pipelines and cross-validation workflows.

**WARNING: Not for Production Use.** This library is built for educational purposes and algorithmic transparency.
The implementations prioritize readability and simplicity over execution speed and memory optimization.

---

## Available Models

**Classification:**

* OneRClassifier: 1R (One Rule) classification.
* NaiveBayesClassifier: Categorical Naive Bayes classification.
* KNeighborsClassifier: K-Nearest Neighbors classification.
* DecisionTreeClassifier: Decision Tree classification using the ID3 algorithm.

**Clustering:**

* KMeans: K-Means clustering.
* KMedoids: K-Medoids clustering.
* DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
* AgglomerativeClustering: Hierarchical agglomerative clustering.

---

## Installation

**Requirements:**

* Python 3.10+

**Install directly from PyPI** using `pip`:

```bash
pip install simple-sklearn
```

Core Dependencies:

* scikit-learn (>= 1.6.1)
* numpy (>= 1.26)
* pandas (>= 2.2.3)
* scipy (>= 1.13.1)
* typing-extensions (>=4.1.0, <5.0)

---

## Quick Start

Because simple-sklearn strictly implements the scikit-learn API, you can fit and predict models exactly as you would
with scikit-learn.

### Classification Example

```python
import numpy as np
from simple_sklearn.classification import NaiveBayesClassifier

# Categorical data
X = np.array([[0, 0], [0, 1], [1, 0], [2, 2], [2, 3]])
y = np.array([0, 0, 0, 1, 1])

# Initialize and fit the model
clf = NaiveBayesClassifier()
clf.fit(X, y)

# Predict on new data
X_new = np.array([[2, 2], [1, 4]])
predictions = clf.predict(X_new)  # Handles unseen category '4' gracefully
print(f"Predictions: {predictions}")  # Output: [1 0]
```

### Clustering Example

```python
import numpy as np
from simple_sklearn.clustering import KMeans

# Continuous data
X = np.array([[0.1, 0.1], [0.2, 0.1], [10.1, 10.1], [10.2, 10.1]])

# Initialize and fit the model
clusterer = KMeans(n_clusters=2, max_iter=10, random_state=42)
clusterer.fit(X)

print(f"Cluster labels: {clusterer.labels_}")  # Output: [0 0 1 1]
print(f"Cluster centers: \n{clusterer.cluster_centers_}")  # Output: [[0.15, 0.1], [10.15, 10.1]]
```

---

## Documentation

Detailed API references, algorithm details, and Jupyter Notebook usage examples are available on the library website:

**[Artem259.github.io/simple-sklearn](https://Artem259.github.io/simple-sklearn)**

---

## Development & Contributing

Pull requests are welcome! If you'd like to contribute, please read the [Contributing Guide](CONTRIBUTING.md).

---

## Roadmap

Future development focuses on:

1. Estimator Enhancements and Performance Optimization:
    * Implementing a custom KDTree for faster nearest-neighbor searches.
    * Refactoring AgglomerativeClustering with Priority Queue (Min-Heap) to reduce time complexity
      during cluster merging.
    * Adding other estimator features (e.g., standard hyperparameters like `max_depth` for `DecisionTreeClassifier`).
2. Documentation Upgrades:
    * Configuring documentation versioning via the `mike` tool.

---

## License

This project is licensed under the [MIT License](LICENSE).
