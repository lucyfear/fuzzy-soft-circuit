"""
Fuzzy Soft Circuits - Automatic Rule Learning

Key concepts:
- Variables are indices (0, 1, 2, ...)
- Membership functions are learned automatically
- Rules are discovered, not specified
- Everything is differentiable
"""

import autograd.numpy as np
from autograd import grad
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt


def sigmoid(x, temperature=1.0):
    """Sigmoid with temperature control."""
    return 1 / (1 + np.exp(-temperature * x))


class LearnableMembershipFunction:
    """Learnable fuzzy membership function."""

    def __init__(self, n_functions=3):
        """
        Initialize learnable membership functions.

        Args:
            n_functions: Number of membership functions (e.g., 3 for LOW/MED/HIGH)
        """
        self.n_functions = n_functions
        # Initialize parameters for triangular membership functions
        # Each function needs center and width
        self.params = np.random.randn(n_functions, 2)
        self.params[:, 1] = np.abs(self.params[:, 1]) + 0.5  # Positive width

    def __call__(self, x, params=None):
        """
        Compute membership values for input x.

        Args:
            x: Input value (normalized to [0, 1])
            params: Optional parameters to use instead of self.params

        Returns:
            Array of membership values, one per function
        """
        if params is None:
            params = self.params

        memberships = np.zeros(self.n_functions)
        for i in range(self.n_functions):
            center, width = params[i]
            # Gaussian membership function
            memberships[i] = np.exp(-((x - center) / width) ** 2)

        return memberships


class FuzzySoftCircuit:
    """
    Fuzzy soft circuit that automatically learns:
    1. Membership functions for each variable
    2. Rules through soft gates
    3. Which rules are active (IF conditions)
    """

    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 n_memberships: int = 3,
                 n_rules: int = 10):
        """
        Initialize fuzzy soft circuit.

        Args:
            n_inputs: Number of input variables
            n_outputs: Number of output variables
            n_memberships: Number of membership functions per variable
            n_rules: Number of potential rules to learn
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_memberships = n_memberships
        self.n_rules = n_rules

        # Total features after fuzzification
        self.n_fuzzy_features = n_inputs * n_memberships

        # Initialize parameters
        self.init_params()

    def init_params(self):
        """Initialize all learnable parameters."""
        params = {}

        # Membership function parameters for inputs
        # Shape: (n_inputs, n_memberships, 2) - center and width for each
        params['input_mf'] = np.random.randn(self.n_inputs, self.n_memberships, 2)
        params['input_mf'][:, :, 1] = np.abs(params['input_mf'][:, :, 1]) + 0.5

        # Spread centers evenly initially
        for i in range(self.n_inputs):
            for j in range(self.n_memberships):
                params['input_mf'][i, j, 0] = j / (self.n_memberships - 1) if self.n_memberships > 1 else 0.5

        # Rule antecedent parameters (which fuzzy features activate each rule)
        # This is like learning the IF part of rules
        # Shape: (n_rules, n_fuzzy_features)
        params['rule_antecedents'] = np.random.randn(self.n_rules, self.n_fuzzy_features) * 0.1

        # Rule activation switches (soft IF - determines if rule is relevant)
        # Shape: (n_rules,)
        params['rule_switches'] = np.random.randn(self.n_rules)

        # Rule consequent parameters (what each rule outputs)
        # Shape: (n_rules, n_outputs)
        params['rule_consequents'] = np.random.randn(self.n_rules, self.n_outputs)

        # Output combination weights (how to combine rules for each output)
        # Shape: (n_outputs, n_rules)
        params['output_weights'] = np.random.randn(self.n_outputs, self.n_rules) * 0.1

        self.params = params

    def fuzzify(self, inputs, params):
        """
        Convert crisp inputs to fuzzy membership values.

        Args:
            inputs: Array of input values, shape (n_inputs,)
            params: Parameter dictionary

        Returns:
            Flattened array of all membership values
        """
        fuzzy_values = []

        for i in range(self.n_inputs):
            x = inputs[i]
            # Get membership values for this input
            for j in range(self.n_memberships):
                center, width = params['input_mf'][i, j]
                membership = np.exp(-((x - center) / width) ** 2)
                fuzzy_values.append(membership)

        return np.array(fuzzy_values)

    def evaluate_rules(self, fuzzy_inputs, params):
        """
        Evaluate all rules given fuzzy inputs.

        This is where the magic happens - rules are discovered through
        learnable soft gates, not hardcoded.

        Args:
            fuzzy_inputs: Fuzzified input values
            params: Parameter dictionary

        Returns:
            Rule activation levels
        """
        rule_activations = []

        for r in range(self.n_rules):
            # Compute antecedent activation (soft AND of relevant features)
            # This learns WHICH fuzzy features matter for this rule
            relevance = sigmoid(params['rule_antecedents'][r])

            # Soft AND: product of relevant fuzzy inputs
            antecedent_activation = np.prod(
                fuzzy_inputs ** relevance + (1 - relevance)
            )

            # Apply rule switch (soft IF - is this rule active at all?)
            rule_switch = sigmoid(params['rule_switches'][r])

            # Final rule activation
            activation = antecedent_activation * rule_switch
            rule_activations.append(activation)

        return np.array(rule_activations)

    def forward(self, inputs, params=None):
        """
        Forward pass through the fuzzy soft circuit.

        Args:
            inputs: Input values (crisp), shape (n_inputs,)
            params: Optional parameter dictionary

        Returns:
            Output values, shape (n_outputs,)
        """
        if params is None:
            params = self.params

        # 1. Fuzzify inputs
        fuzzy_inputs = self.fuzzify(inputs, params)

        # 2. Evaluate rules
        rule_activations = self.evaluate_rules(fuzzy_inputs, params)

        # 3. Compute outputs (weighted combination of rule consequents)
        outputs = []
        for o in range(self.n_outputs):
            # Each output is a weighted sum of rule consequents
            output_val = 0
            total_weight = 0

            for r in range(self.n_rules):
                weight = sigmoid(params['output_weights'][o, r])
                consequent = sigmoid(params['rule_consequents'][r, o])

                output_val += rule_activations[r] * weight * consequent
                total_weight += rule_activations[r] * weight + 1e-10

            # Normalize
            outputs.append(output_val / total_weight)

        return np.array(outputs)

    def extract_rules(self, params=None, threshold=0.3, var_names=None):
        """
        Extract interpretable rules from learned parameters.

        Args:
            params: Parameter dictionary
            threshold: Activation threshold for considering a feature relevant
            var_names: Optional names for variables

        Returns:
            List of human-readable rules
        """
        if params is None:
            params = self.params

        if var_names is None:
            var_names = [f"var_{i}" for i in range(self.n_inputs)]

        membership_names = ['LOW', 'MED', 'HIGH'][:self.n_memberships]

        rules = []

        for r in range(self.n_rules):
            # Check if rule is active
            if sigmoid(params['rule_switches'][r]) < threshold:
                continue

            # Build antecedent
            antecedent_parts = []
            relevances = sigmoid(params['rule_antecedents'][r])

            for i in range(self.n_inputs):
                for j in range(self.n_memberships):
                    idx = i * self.n_memberships + j
                    if relevances[idx] > threshold:
                        antecedent_parts.append(
                            f"{var_names[i]} is {membership_names[j]}"
                        )

            if not antecedent_parts:
                continue

            # Build consequent
            consequent_parts = []
            for o in range(self.n_outputs):
                weight = sigmoid(params['output_weights'][o, r])
                if weight > threshold:
                    value = sigmoid(params['rule_consequents'][r, o])
                    level = 'HIGH' if value > 0.7 else 'LOW' if value < 0.3 else 'MED'
                    consequent_parts.append(
                        f"output_{o} is {level}"
                    )

            if antecedent_parts and consequent_parts:
                rule = f"IF {' AND '.join(antecedent_parts)} THEN {' AND '.join(consequent_parts)}"
                rules.append((rule, sigmoid(params['rule_switches'][r])))

        return rules

    def visualize_memberships(self, params=None):
        """Visualize learned membership functions."""
        if params is None:
            params = self.params

        fig, axes = plt.subplots(1, self.n_inputs, figsize=(4*self.n_inputs, 3))
        if self.n_inputs == 1:
            axes = [axes]

        x = np.linspace(0, 1, 100)
        membership_names = ['LOW', 'MED', 'HIGH'][:self.n_memberships]

        for i in range(self.n_inputs):
            for j in range(self.n_memberships):
                center, width = params['input_mf'][i, j]
                y = np.exp(-((x - center) / width) ** 2)
                axes[i].plot(x, y, label=membership_names[j])

            axes[i].set_title(f'Variable {i}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Membership')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def flatten_params(params_dict):
    """Flatten parameter dictionary to 1D array for optimization."""
    flat = []
    shapes = {}
    for key, value in params_dict.items():
        shapes[key] = value.shape
        flat.extend(value.flatten())
    return np.array(flat), shapes


def unflatten_params(flat_params, shapes):
    """Unflatten 1D array back to parameter dictionary."""
    params = {}
    idx = 0
    for key, shape in shapes.items():
        size = np.prod(shape)
        params[key] = flat_params[idx:idx+size].reshape(shape)
        idx += size
    return params


def train_fuzzy_circuit(circuit, data, epochs=1000, learning_rate=0.1):
    """
    Train a fuzzy soft circuit on data.

    Args:
        circuit: FuzzySoftCircuit instance
        data: List of (input, output) tuples
        epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        Trained parameters
    """
    # Flatten parameters for optimization
    flat_params, shapes = flatten_params(circuit.params)

    def loss(flat_params):
        params = unflatten_params(flat_params, shapes)
        total_loss = 0
        for inputs, targets in data:
            outputs = circuit.forward(inputs, params)
            total_loss += np.sum((outputs - targets) ** 2)
        return total_loss / len(data)

    # Gradient descent
    grad_loss = grad(loss)

    for epoch in range(epochs):
        gradients = grad_loss(flat_params)
        flat_params -= learning_rate * gradients

        if epoch % 100 == 0:
            current_loss = loss(flat_params)
            print(f"Epoch {epoch}, Loss: {current_loss:.4f}")

    # Unflatten and return
    return unflatten_params(flat_params, shapes)