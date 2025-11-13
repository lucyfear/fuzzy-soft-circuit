"""
Pure fuzzy soft circuits - no semantic labels, just indices and learning.
"""

import autograd.numpy as np
from autograd import grad


class PureFuzzyCircuit:
    """
    A truly semantic-free fuzzy circuit.
    Everything is just indices and learned parameters.
    """

    def __init__(self, n_inputs, n_outputs, n_memberships, n_rules):
        """
        Initialize with dimensions only.
        No semantic meaning assigned.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_memberships = n_memberships
        self.n_rules = n_rules

        # Initialize random parameters
        self.params = self._init_params()

    def _init_params(self):
        """Initialize all parameters randomly."""
        n_features = self.n_inputs * self.n_memberships

        # Random initialization - no semantic structure
        params = {
            # Membership functions: just curves, no meaning
            'mf_centers': np.random.rand(self.n_inputs, self.n_memberships),
            'mf_widths': np.random.rand(self.n_inputs, self.n_memberships) * 0.5 + 0.1,

            # Rule parameters: just numbers
            'rule_weights': np.random.randn(self.n_rules, n_features),
            'rule_activations': np.random.randn(self.n_rules),
            'rule_outputs': np.random.randn(self.n_rules, self.n_outputs),

            # Output mixing: just coefficients
            'output_mix': np.random.randn(self.n_outputs, self.n_rules)
        }

        return params

    def forward(self, x, params=None):
        """
        Forward pass - pure computation, no semantics.

        x: array of shape (n_inputs,) with values in [0, 1]
        Returns: array of shape (n_outputs,) with values in [0, 1]
        """
        if params is None:
            params = self.params

        # Step 1: Compute membership values (just Gaussian curves)
        memberships = []
        for i in range(self.n_inputs):
            for j in range(self.n_memberships):
                center = params['mf_centers'][i, j]
                width = params['mf_widths'][i, j]
                value = np.exp(-((x[i] - center) / width) ** 2)
                memberships.append(value)
        memberships = np.array(memberships)

        # Step 2: Compute rule activations (just math)
        rule_values = []
        for r in range(self.n_rules):
            # Weighted combination of membership values
            weights = 1 / (1 + np.exp(-params['rule_weights'][r]))
            activation = np.prod(memberships ** weights + (1 - weights))

            # Gate this rule
            gate = 1 / (1 + np.exp(-params['rule_activations'][r]))
            rule_values.append(activation * gate)
        rule_values = np.array(rule_values)

        # Step 3: Compute outputs (weighted sum)
        outputs = []
        for o in range(self.n_outputs):
            mix = 1 / (1 + np.exp(-params['output_mix'][o]))
            consequents = 1 / (1 + np.exp(-params['rule_outputs'][:, o]))

            weighted_sum = np.sum(rule_values * mix * consequents)
            normalizer = np.sum(rule_values * mix) + 1e-10

            outputs.append(weighted_sum / normalizer)

        return np.array(outputs)

    def learn(self, data, epochs=1000, lr=0.1):
        """
        Learn from data. No assumptions about what it means.

        data: list of (input, output) pairs
        """
        # Flatten parameters
        def flatten(p):
            return np.concatenate([v.flatten() for v in p.values()])

        def unflatten(flat):
            p = {}
            idx = 0
            for key, shape in [(k, v.shape) for k, v in self.params.items()]:
                size = np.prod(shape)
                p[key] = flat[idx:idx+size].reshape(shape)
                idx += size
            return p

        flat_params = flatten(self.params)

        # Define loss (just MSE, no semantic meaning)
        def loss(flat):
            p = unflatten(flat)
            error = 0
            for x, y in data:
                pred = self.forward(x, p)
                error += np.sum((pred - y) ** 2)
            return error / len(data)

        # Optimize
        grad_fn = grad(loss)

        for epoch in range(epochs):
            g = grad_fn(flat_params)
            flat_params -= lr * g

            if epoch % 200 == 0:
                print(f"Epoch {epoch}: loss = {loss(flat_params):.4f}")

        self.params = unflatten(flat_params)
        return self.params

    def describe_state(self):
        """
        Describe what was learned - just the numbers, no interpretation.
        """
        print("\nLearned Parameters Summary:")
        print("-" * 40)

        # Membership functions
        print(f"Membership Functions:")
        for i in range(self.n_inputs):
            print(f"  Input {i}:")
            for j in range(self.n_memberships):
                c = self.params['mf_centers'][i, j]
                w = self.params['mf_widths'][i, j]
                print(f"    MF{j}: center={c:.3f}, width={w:.3f}")

        # Active rules (just which indices have high activation)
        print(f"\nActive Rules (indices with gate > 0.5):")
        for r in range(self.n_rules):
            gate = 1 / (1 + np.exp(-self.params['rule_activations'][r]))
            if gate > 0.5:
                print(f"  Rule {r}: activation={gate:.3f}")

                # Which membership functions matter for this rule
                weights = 1 / (1 + np.exp(-self.params['rule_weights'][r]))
                important = np.where(weights > 0.5)[0]
                if len(important) > 0:
                    print(f"    Uses membership indices: {important.tolist()}")


# Example: Learn a function without any semantic labels
def demo_pure_fuzzy():
    """
    Demo: Learn a 2D->1D function without any semantic meaning.
    We don't call it temperature, humidity, or anything.
    Just input[0], input[1] -> output[0].
    """
    print("=" * 60)
    print("PURE FUZZY LEARNING - NO SEMANTICS")
    print("=" * 60)

    # Create circuit - just numbers
    circuit = PureFuzzyCircuit(
        n_inputs=2,      # Two inputs (indices 0, 1)
        n_outputs=1,     # One output (index 0)
        n_memberships=3, # 3 curves per input
        n_rules=6        # 6 potential patterns to discover
    )

    # Training data - just input/output pairs
    # No meaning assigned - could be anything!
    data = [
        ([0.1, 0.1], [0.2]),
        ([0.1, 0.9], [0.8]),
        ([0.9, 0.1], [0.8]),
        ([0.9, 0.9], [0.2]),
        ([0.5, 0.5], [0.5]),
        ([0.3, 0.7], [0.6]),
        ([0.7, 0.3], [0.6]),
    ]

    print("\nTraining on data...")
    circuit.learn(data, epochs=600, lr=0.5)

    print("\nLearned behavior:")
    test_points = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ]

    for point in test_points:
        output = circuit.forward(point)
        print(f"  {point} -> {output[0]:.3f}")

    # Describe what was learned (no semantic labels!)
    circuit.describe_state()

    print("\n" + "=" * 60)
    print("Notice: No 'HIGH', 'LOW', 'temperature', etc.")
    print("Just learned a mapping from R^2 -> R")
    print("The system discovered which patterns matter")
    print("=" * 60)


if __name__ == "__main__":
    demo_pure_fuzzy()