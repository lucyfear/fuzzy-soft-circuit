"""
Simplified ANFIS (Adaptive Neuro-Fuzzy Inference System) implementation.

Following Jang (1993) but simplified for fair comparison with FuzzySoftCircuit.
Uses gradient descent instead of hybrid learning for consistency.

Reference:
Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system.
IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.
"""

import autograd.numpy as np
from autograd import grad
import itertools


class SimplifiedANFIS:
    """
    Simplified ANFIS for benchmark comparison.

    Key simplifications for fair comparison:
    - Grid partitioning rule structure (standard ANFIS)
    - Gaussian membership functions (like FuzzySoftCircuit)
    - Pure gradient descent (instead of hybrid learning)
    - Same optimization framework as FuzzySoftCircuit
    """

    def __init__(self, n_inputs, n_outputs, n_memberships=3, max_rules=None):
        """
        Initialize ANFIS.

        Args:
            n_inputs: Number of input variables
            n_outputs: Number of output variables
            n_memberships: Number of membership functions per input
            max_rules: Maximum number of rules (None = grid partitioning)
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_memberships = n_memberships

        # Generate rule structure via grid partitioning
        # Each rule corresponds to one combination of membership functions
        self.rules = list(itertools.product(range(n_memberships), repeat=n_inputs))

        # Limit rules if max_rules specified (for high-dimensional inputs)
        if max_rules and len(self.rules) > max_rules:
            # Randomly sample rules
            np.random.seed(42)
            indices = np.random.choice(len(self.rules), max_rules, replace=False)
            self.rules = [self.rules[i] for i in indices]

        self.n_rules = len(self.rules)

        print(f"ANFIS initialized with {self.n_rules} rules "
              f"(grid partitioning: {n_memberships}^{n_inputs})")

        # Initialize parameters
        self.init_params()

    def init_params(self):
        """Initialize ANFIS parameters."""
        params = {}

        # Premise parameters (membership functions)
        # Shape: (n_inputs, n_memberships, 2) - center and width
        params['mf_centers'] = np.zeros((self.n_inputs, self.n_memberships))
        params['mf_widths'] = np.ones((self.n_inputs, self.n_memberships)) * 0.3

        # Initialize centers evenly spaced
        for i in range(self.n_inputs):
            for j in range(self.n_memberships):
                params['mf_centers'][i, j] = j / (self.n_memberships - 1) if self.n_memberships > 1 else 0.5

        # Consequent parameters (Takagi-Sugeno linear consequents)
        # For simplicity, use constant consequents (Mamdani-style)
        # Shape: (n_rules, n_outputs)
        params['consequents'] = np.random.randn(self.n_rules, self.n_outputs) * 0.1

        self.params = params

    def membership_value(self, x, center, width):
        """Compute Gaussian membership function value."""
        return np.exp(-((x - center) / width) ** 2)

    def forward(self, inputs, params=None):
        """
        ANFIS forward pass.

        Layer 1: Fuzzification
        Layer 2: Rule firing strengths (product T-norm)
        Layer 3: Normalized firing strengths
        Layer 4: Consequent calculation
        Layer 5: Defuzzification (weighted average)
        """
        if params is None:
            params = self.params

        # Layer 1: Compute membership values for each input
        memberships = []
        for i in range(self.n_inputs):
            input_memberships = []
            for j in range(self.n_memberships):
                center = params['mf_centers'][i, j]
                width = params['mf_widths'][i, j]
                mu = self.membership_value(inputs[i], center, width)
                input_memberships.append(mu)
            memberships.append(input_memberships)

        # Layer 2 & 3: Compute rule firing strengths
        firing_strengths = []
        for rule_idx, rule in enumerate(self.rules):
            # Rule is tuple like (0, 2, 1) meaning "IF x0 is MF0 AND x1 is MF2 AND x2 is MF1"
            strength = 1.0
            for input_idx, mf_idx in enumerate(rule):
                strength *= memberships[input_idx][mf_idx]
            firing_strengths.append(strength)

        firing_strengths = np.array(firing_strengths)

        # Normalize firing strengths
        total_strength = np.sum(firing_strengths) + 1e-10
        normalized_strengths = firing_strengths / total_strength

        # Layer 4 & 5: Compute output (weighted average of consequents)
        outputs = []
        for o in range(self.n_outputs):
            output_val = 0.0
            for r in range(self.n_rules):
                # Constant consequent (Mamdani-style for simplicity)
                consequent = 1 / (1 + np.exp(-params['consequents'][r, o]))
                output_val = output_val + normalized_strengths[r] * consequent
            outputs.append(output_val)

        return np.array(outputs)

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.1, verbose=True):
        """
        Train ANFIS using gradient descent.

        Args:
            X_train: Training inputs, shape (n_samples, n_inputs)
            y_train: Training outputs, shape (n_samples, n_outputs)
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Print progress

        Returns:
            Training history (losses)
        """
        # Flatten parameters for optimization
        def flatten(p):
            flat = []
            for key in ['mf_centers', 'mf_widths', 'consequents']:
                flat.append(p[key].flatten())
            return np.concatenate(flat)

        def unflatten(flat):
            p = {}
            idx = 0

            # mf_centers
            size = self.n_inputs * self.n_memberships
            p['mf_centers'] = flat[idx:idx+size].reshape(self.n_inputs, self.n_memberships)
            idx += size

            # mf_widths
            p['mf_widths'] = flat[idx:idx+size].reshape(self.n_inputs, self.n_memberships)
            idx += size

            # consequents
            size = self.n_rules * self.n_outputs
            p['consequents'] = flat[idx:idx+size].reshape(self.n_rules, self.n_outputs)

            return p

        flat_params = flatten(self.params)

        # Define loss function
        def loss(flat):
            p = unflatten(flat)
            total_loss = 0
            for i in range(len(X_train)):
                pred = self.forward(X_train[i], p)
                target = y_train[i]
                total_loss += np.sum((pred - target) ** 2)
            return total_loss / len(X_train)

        # Gradient descent
        grad_fn = grad(loss)
        history = []

        for epoch in range(epochs):
            g = grad_fn(flat_params)
            flat_params -= learning_rate * g

            if epoch % 100 == 0:
                current_loss = float(loss(flat_params))
                history.append(current_loss)
                if verbose:
                    print(f"Epoch {epoch}, Loss: {current_loss:.6f}")

        self.params = unflatten(flat_params)

        if verbose:
            print(f"Final loss: {history[-1]:.6f}")

        return history

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input data, shape (n_samples, n_inputs)

        Returns:
            Predictions, shape (n_samples, n_outputs)
        """
        predictions = []
        for i in range(len(X)):
            pred = self.forward(X[i])
            predictions.append(pred)
        return np.array(predictions)

    def count_parameters(self):
        """Count total number of parameters."""
        mf_params = self.n_inputs * self.n_memberships * 2  # centers and widths
        consequent_params = self.n_rules * self.n_outputs
        return mf_params + consequent_params


if __name__ == "__main__":
    # Test ANFIS on simple XOR-like problem
    print("Testing ANFIS on XOR-like problem")
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

    # Initialize ANFIS
    anfis = SimplifiedANFIS(n_inputs=2, n_outputs=1, n_memberships=3)
    print(f"Total parameters: {anfis.count_parameters()}")

    # Train
    history = anfis.train(X_train, y_train, epochs=600, learning_rate=0.5)

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

    predictions = anfis.predict(X_test)

    for i in range(len(X_test)):
        print(f"Input: {X_test[i]} -> Predicted: {predictions[i][0]:.3f}, Target: {y_test[i][0]:.3f}")

    # Compute test MSE
    test_mse = np.mean((predictions - y_test) ** 2)
    print(f"\nTest MSE: {test_mse:.6f}")
