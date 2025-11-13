"""
Multi-Layer Perceptron baseline for comparison.

Standard neural network without interpretability - serves as accuracy benchmark.
Uses scikit-learn for simplicity and fair comparison.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import time


class MLPBaseline:
    """
    MLP baseline using scikit-learn.

    Configuration chosen for fair comparison:
    - Similar capacity to fuzzy methods
    - Same training epochs
    - Simple architecture
    """

    def __init__(self, n_inputs, n_outputs, task_type='regression',
                 hidden_layers=(32, 16), random_state=42):
        """
        Initialize MLP.

        Args:
            n_inputs: Number of input features
            n_outputs: Number of outputs
            task_type: 'regression' or 'classification'
            hidden_layers: Tuple of hidden layer sizes
            random_state: Random seed
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.task_type = task_type
        self.hidden_layers = hidden_layers
        self.random_state = random_state

        if task_type == 'regression':
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=random_state,
                early_stopping=False,
                verbose=False
            )
        else:  # classification
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=random_state,
                early_stopping=False,
                verbose=False
            )

    def train(self, X_train, y_train, verbose=True):
        """
        Train MLP.

        Args:
            X_train: Training inputs
            y_train: Training outputs
            verbose: Print progress

        Returns:
            Training time in seconds
        """
        start_time = time.time()

        if self.task_type == 'classification':
            # Convert one-hot to class labels
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                y_train_labels = np.argmax(y_train, axis=1)
            else:
                y_train_labels = y_train.flatten()

            self.model.fit(X_train, y_train_labels)
        else:
            # For regression, handle single or multi-output
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)

            self.model.fit(X_train, y_train)

        train_time = time.time() - start_time

        if verbose:
            print(f"MLP trained in {train_time:.2f} seconds")
            print(f"Final loss: {self.model.loss_:.6f}")

        return train_time

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Input data

        Returns:
            Predictions (one-hot for classification, values for regression)
        """
        if self.task_type == 'classification':
            # Get class predictions
            class_pred = self.model.predict(X)

            # Convert to one-hot
            n_classes = self.model.n_outputs_
            predictions = np.zeros((len(X), n_classes))
            predictions[np.arange(len(X)), class_pred] = 1

            return predictions
        else:
            predictions = self.model.predict(X)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            return predictions

    def predict_proba(self, X):
        """Get prediction probabilities (classification only)."""
        if self.task_type == 'classification':
            return self.model.predict_proba(X)
        else:
            raise ValueError("predict_proba only available for classification")

    def count_parameters(self):
        """Count total number of parameters in the network."""
        total = 0
        for coef in self.model.coefs_:
            total += coef.size
        for intercept in self.model.intercepts_:
            total += intercept.size
        return total

    def get_loss_curve(self):
        """Get training loss curve."""
        return self.model.loss_curve_


if __name__ == "__main__":
    # Test MLP on simple regression problem
    print("Testing MLP on XOR-like problem")
    print("=" * 60)

    # Create training data
    X_train = np.array([
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.5, 0.5],
        [0.3, 0.7],
        [0.7, 0.3],
    ])

    y_train = np.array([
        [0.2],
        [0.8],
        [0.8],
        [0.2],
        [0.5],
        [0.6],
        [0.6],
    ])

    # Initialize MLP
    mlp = MLPBaseline(n_inputs=2, n_outputs=1, task_type='regression',
                     hidden_layers=(16, 8), random_state=42)

    # Train
    train_time = mlp.train(X_train, y_train)
    print(f"Parameters: {mlp.count_parameters()}")

    # Test
    print("\nTest predictions:")
    X_test = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ])

    y_test = np.array([
        [0.2],
        [0.8],
        [0.8],
        [0.2],
        [0.5],
    ])

    predictions = mlp.predict(X_test)

    for i in range(len(X_test)):
        print(f"Input: {X_test[i]} -> Predicted: {predictions[i][0]:.3f}, Target: {y_test[i][0]:.3f}")

    # Compute test MSE
    test_mse = mean_squared_error(y_test, predictions)
    print(f"\nTest MSE: {test_mse:.6f}")
