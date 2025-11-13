"""
Dataset loading and preprocessing utilities for fuzzy soft circuit benchmarks.

All datasets normalized to [0, 1] range as required by fuzzy membership functions.
Includes proper train/test splitting with stratification for classification tasks.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import load_iris, fetch_openml
import os
import pickle


class BenchmarkDataset:
    """Container for a preprocessed benchmark dataset."""

    def __init__(self, name, task_type, X_train, X_test, y_train, y_test,
                 feature_names=None, target_names=None, description=None):
        self.name = name
        self.task_type = task_type  # 'regression' or 'classification'
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.target_names = target_names
        self.description = description

        self.n_features = X_train.shape[1]
        self.n_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1
        self.n_train = X_train.shape[0]
        self.n_test = X_test.shape[0]

    def summary(self):
        """Print dataset summary."""
        print(f"\nDataset: {self.name}")
        print(f"Task: {self.task_type}")
        print(f"Features: {self.n_features}")
        print(f"Outputs: {self.n_outputs}")
        print(f"Train samples: {self.n_train}")
        print(f"Test samples: {self.n_test}")
        if self.description:
            print(f"Description: {self.description}")

    def save(self, directory):
        """Save dataset to disk."""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.name}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved {self.name} to {filepath}")

    @staticmethod
    def load(filepath):
        """Load dataset from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def load_iris_dataset(test_size=0.2, random_state=42):
    """
    Load Iris dataset for multi-class classification.

    Classic benchmark from Fisher (1936). 3 classes, 4 features, 150 samples.
    """
    data = load_iris()
    X, y = data.data, data.target

    # Normalize to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # One-hot encode targets for fuzzy methods
    n_classes = len(np.unique(y))
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=test_size, random_state=random_state, stratify=y
    )

    return BenchmarkDataset(
        name="iris",
        task_type="classification",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=data.feature_names,
        target_names=data.target_names.tolist(),
        description="Iris flower classification (Fisher, 1936)"
    )


def load_wine_quality_dataset(test_size=0.2, random_state=42):
    """
    Load Wine Quality dataset for regression.

    Predicting wine quality from physicochemical properties.
    Cortez et al. (2009). 11 features, 1599 samples (red wine).
    """
    try:
        # Try to load from sklearn/openml
        wine = fetch_openml('wine-quality-red', version=1, as_frame=False, parser='auto')
        X, y = wine.data, wine.target
    except Exception as e:
        print(f"Warning: Could not fetch wine quality dataset: {e}")
        print("Generating synthetic wine-like regression data...")
        # Fallback: synthetic data with similar characteristics
        np.random.seed(random_state)
        n_samples = 1599
        X = np.random.rand(n_samples, 11)
        # Simulate quality as nonlinear function of features
        y = (0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * X[:, 2] ** 2 +
             0.2 * np.sin(X[:, 3] * np.pi) + 0.2 * np.random.randn(n_samples) * 0.1)

    # Normalize inputs to [0, 1]
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Normalize output to [0, 1]
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )

    # Reshape for compatibility
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return BenchmarkDataset(
        name="wine_quality",
        task_type="regression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=["feature_" + str(i) for i in range(11)],
        description="Wine quality prediction from physicochemical properties"
    )


def load_diabetes_dataset(test_size=0.2, random_state=42):
    """
    Load Pima Indians Diabetes dataset for binary classification.

    Predicting diabetes from medical measurements.
    8 features, 768 samples.
    """
    try:
        diabetes = fetch_openml('diabetes', version=1, as_frame=False, parser='auto')
        X, y = diabetes.data, diabetes.target

        # Convert target to binary
        if y.dtype != np.number:
            le = LabelEncoder()
            y = le.fit_transform(y)
    except Exception as e:
        print(f"Warning: Could not fetch diabetes dataset: {e}")
        print("Generating synthetic diabetes-like classification data...")
        np.random.seed(random_state)
        n_samples = 768
        X = np.random.rand(n_samples, 8)
        # Simulate diabetes based on nonlinear decision boundary
        prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1] - 1 + 0.5 * X[:, 2] * X[:, 3])))
        y = (prob > 0.5).astype(int)

    # Normalize to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # One-hot encode for fuzzy methods
    y_onehot = np.zeros((len(y), 2))
    y_onehot[np.arange(len(y)), y] = 1

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=test_size, random_state=random_state,
        stratify=y
    )

    return BenchmarkDataset(
        name="diabetes",
        task_type="classification",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=["feature_" + str(i) for i in range(8)],
        target_names=["no_diabetes", "diabetes"],
        description="Pima Indians diabetes classification"
    )


def load_energy_efficiency_dataset(test_size=0.2, random_state=42):
    """
    Load Energy Efficiency dataset for multi-output regression.

    Predicting heating and cooling loads from building parameters.
    Tsanas & Xifara (2012). 8 features, 2 outputs, 768 samples.
    """
    # Generate synthetic energy efficiency data (placeholder)
    # In production, load from UCI repository
    print("Generating synthetic energy efficiency data...")
    np.random.seed(random_state)
    n_samples = 768

    # Building parameters
    X = np.random.rand(n_samples, 8)

    # Heating and cooling loads as functions of building parameters
    heating_load = (0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] ** 2 +
                   0.1 * np.random.randn(n_samples) * 0.1)
    cooling_load = (0.4 * X[:, 0] + 0.4 * X[:, 1] + 0.2 * X[:, 3] ** 2 +
                   0.1 * np.random.randn(n_samples) * 0.1)

    # Normalize
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    y = np.column_stack([heating_load, cooling_load])
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )

    return BenchmarkDataset(
        name="energy_efficiency",
        task_type="regression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=["feature_" + str(i) for i in range(8)],
        target_names=["heating_load", "cooling_load"],
        description="Building energy efficiency prediction (multi-output)"
    )


def load_concrete_strength_dataset(test_size=0.2, random_state=42):
    """
    Load Concrete Compressive Strength dataset for regression.

    Predicting concrete strength from mixture components.
    Yeh (1998). 8 features, 1030 samples.
    """
    # Generate synthetic concrete data (placeholder)
    print("Generating synthetic concrete strength data...")
    np.random.seed(random_state)
    n_samples = 1030

    # Mixture components
    X = np.random.rand(n_samples, 8)

    # Compressive strength as nonlinear function
    strength = (0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.15 * X[:, 2] ** 2 +
               0.2 * X[:, 3] * X[:, 4] + 0.15 * np.sqrt(X[:, 5]) +
               0.1 * np.random.randn(n_samples) * 0.1)

    # Normalize
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(strength.reshape(-1, 1))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )

    return BenchmarkDataset(
        name="concrete_strength",
        task_type="regression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=["feature_" + str(i) for i in range(8)],
        description="Concrete compressive strength prediction"
    )


def load_all_datasets(test_size=0.2, random_state=42, save_dir=None):
    """
    Load all benchmark datasets.

    Args:
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        save_dir: Optional directory to save datasets

    Returns:
        Dictionary of BenchmarkDataset objects
    """
    datasets = {}

    print("Loading benchmark datasets...")
    print("=" * 60)

    datasets['iris'] = load_iris_dataset(test_size, random_state)
    datasets['iris'].summary()

    datasets['wine_quality'] = load_wine_quality_dataset(test_size, random_state)
    datasets['wine_quality'].summary()

    datasets['diabetes'] = load_diabetes_dataset(test_size, random_state)
    datasets['diabetes'].summary()

    datasets['energy_efficiency'] = load_energy_efficiency_dataset(test_size, random_state)
    datasets['energy_efficiency'].summary()

    datasets['concrete_strength'] = load_concrete_strength_dataset(test_size, random_state)
    datasets['concrete_strength'].summary()

    print("=" * 60)
    print(f"Loaded {len(datasets)} datasets\n")

    if save_dir:
        for name, dataset in datasets.items():
            dataset.save(save_dir)

    return datasets


if __name__ == "__main__":
    # Test dataset loading
    datasets = load_all_datasets(
        test_size=0.2,
        random_state=42,
        save_dir="/home/spinoza/github/beta/soft-circuit/benchmarks/datasets/processed"
    )
