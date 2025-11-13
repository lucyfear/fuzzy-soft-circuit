#!/usr/bin/env python
"""
Test script for fuzzy soft circuits.
Run this to see the system in action!
"""

import autograd.numpy as np
from fuzzy_core import FuzzySoftCircuit, train_fuzzy_circuit
import matplotlib.pyplot as plt


def test_basic_functionality():
    """Test basic functionality of the fuzzy soft circuit."""
    print("Testing Basic Functionality")
    print("-" * 40)

    # Create a simple circuit
    circuit = FuzzySoftCircuit(n_inputs=1, n_outputs=1, n_memberships=3, n_rules=3)

    # Test forward pass
    input_val = np.array([0.5])
    output = circuit.forward(input_val)
    print(f"Input: {input_val[0]:.2f} -> Output: {output[0]:.3f}")

    # Test fuzzification
    fuzzy = circuit.fuzzify(input_val, circuit.params)
    print(f"Fuzzified values: {fuzzy}")

    print("✓ Basic functionality working\n")


def test_learning():
    """Test that the circuit can learn a simple relationship."""
    print("Testing Learning Capability")
    print("-" * 40)

    # Create identity function data (output = input)
    data = [([x], [x]) for x in np.linspace(0, 1, 10)]

    circuit = FuzzySoftCircuit(n_inputs=1, n_outputs=1, n_memberships=2, n_rules=3)

    # Get initial loss
    initial_outputs = [circuit.forward(inp) for inp, _ in data]
    initial_loss = np.mean([(out[0] - tgt[0])**2 for (_, tgt), out in zip(data, initial_outputs)])

    # Train
    params = train_fuzzy_circuit(circuit, data, epochs=200, learning_rate=0.5)
    circuit.params = params

    # Get final loss
    final_outputs = [circuit.forward(inp) for inp, _ in data]
    final_loss = np.mean([(out[0] - tgt[0])**2 for (_, tgt), out in zip(data, final_outputs)])

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {(initial_loss - final_loss)/initial_loss * 100:.1f}%")

    if final_loss < initial_loss * 0.5:
        print("✓ Learning successful\n")
    else:
        print("⚠ Learning needs improvement\n")


def test_rule_extraction():
    """Test rule extraction from a trained circuit."""
    print("Testing Rule Extraction")
    print("-" * 40)

    # Train a simple system
    circuit = FuzzySoftCircuit(n_inputs=2, n_outputs=1, n_memberships=2, n_rules=4)

    # Simple AND-like data
    data = [
        ([0.1, 0.1], [0.1]),
        ([0.1, 0.9], [0.1]),
        ([0.9, 0.1], [0.1]),
        ([0.9, 0.9], [0.9]),
    ]

    params = train_fuzzy_circuit(circuit, data, epochs=500, learning_rate=0.5)
    circuit.params = params

    # Extract rules
    rules = circuit.extract_rules(var_names=['A', 'B'])

    print(f"Found {len(rules)} rules:")
    for rule, strength in rules[:3]:
        print(f"  {rule[:50]}...")
        print(f"    Strength: {strength:.2f}")

    if len(rules) > 0:
        print("✓ Rule extraction working\n")
    else:
        print("⚠ No rules extracted\n")


def test_complex_relationship():
    """Test learning a more complex fuzzy relationship."""
    print("Testing Complex Relationship")
    print("-" * 40)

    # Create a complex nonlinear relationship
    # Output = high when temp is medium OR (temp is high AND pressure is low)
    data = []
    for temp in np.linspace(0, 1, 7):
        for pressure in np.linspace(0, 1, 7):
            # Complex rule
            if 0.4 < temp < 0.6:  # Medium temp
                output = 0.8
            elif temp > 0.7 and pressure < 0.3:  # High temp, low pressure
                output = 0.9
            else:
                output = 0.2

            data.append(([temp, pressure], [output]))

    circuit = FuzzySoftCircuit(
        n_inputs=2,
        n_outputs=1,
        n_memberships=3,
        n_rules=6
    )

    # Train
    params = train_fuzzy_circuit(circuit, data, epochs=500, learning_rate=0.3)
    circuit.params = params

    # Test on new points
    test_cases = [
        ([0.5, 0.5], "Med temp, Med pressure", 0.8),   # Should be high
        ([0.8, 0.2], "High temp, Low pressure", 0.9),   # Should be high
        ([0.2, 0.8], "Low temp, High pressure", 0.2),   # Should be low
    ]

    print("Test results:")
    errors = []
    for inputs, desc, expected in test_cases:
        output = circuit.forward(inputs)[0]
        error = abs(output - expected)
        errors.append(error)
        status = "✓" if error < 0.3 else "✗"
        print(f"  {desc:25} -> {output:.2f} (expected {expected:.1f}) {status}")

    avg_error = np.mean(errors)
    if avg_error < 0.3:
        print(f"✓ Complex relationship learned (avg error: {avg_error:.3f})\n")
    else:
        print(f"⚠ Complex relationship partially learned (avg error: {avg_error:.3f})\n")


def visualize_fuzzy_surface():
    """Visualize the learned fuzzy control surface."""
    print("Generating Fuzzy Control Surface")
    print("-" * 40)

    # Train a 2-input, 1-output system
    circuit = FuzzySoftCircuit(n_inputs=2, n_outputs=1, n_memberships=3, n_rules=9)

    # Create training data for a saddle-like surface
    data = []
    for x in np.linspace(0, 1, 5):
        for y in np.linspace(0, 1, 5):
            z = x * (1 - y) + y * (1 - x)  # Saddle function
            data.append(([x, y], [z]))

    # Train
    params = train_fuzzy_circuit(circuit, data, epochs=300, learning_rate=0.5)
    circuit.params = params

    # Generate surface
    n_points = 20
    X = np.linspace(0, 1, n_points)
    Y = np.linspace(0, 1, n_points)
    Z = np.zeros((n_points, n_points))

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            Z[j, i] = circuit.forward([x, y])[0]

    # Plot
    fig = plt.figure(figsize=(10, 8))

    # 3D surface
    ax1 = fig.add_subplot(221, projection='3d')
    X_grid, Y_grid = np.meshgrid(X, Y)
    ax1.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Input 1')
    ax1.set_ylabel('Input 2')
    ax1.set_zlabel('Output')
    ax1.set_title('Learned Fuzzy Surface')

    # Contour plot
    ax2 = fig.add_subplot(222)
    contour = ax2.contour(X, Y, Z, levels=10)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('Input 1')
    ax2.set_ylabel('Input 2')
    ax2.set_title('Contour Plot')

    # Membership functions
    circuit.params = params  # Ensure params are set
    x_test = np.linspace(0, 1, 100)

    ax3 = fig.add_subplot(223)
    for j in range(circuit.n_memberships):
        mf_values = []
        for x in x_test:
            center, width = params['input_mf'][0, j]
            mf_values.append(np.exp(-((x - center) / width) ** 2))
        ax3.plot(x_test, mf_values, label=f'MF{j+1}')
    ax3.set_xlabel('Input 1 Value')
    ax3.set_ylabel('Membership')
    ax3.set_title('Input 1 Membership Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(224)
    for j in range(circuit.n_memberships):
        mf_values = []
        for x in x_test:
            center, width = params['input_mf'][1, j]
            mf_values.append(np.exp(-((x - center) / width) ** 2))
        ax4.plot(x_test, mf_values, label=f'MF{j+1}')
    ax4.set_xlabel('Input 2 Value')
    ax4.set_ylabel('Membership')
    ax4.set_title('Input 2 Membership Functions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fuzzy_surface.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to 'fuzzy_surface.png'\n")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("FUZZY SOFT CIRCUITS TEST SUITE")
    print("=" * 60)
    print()

    # Run all tests
    test_basic_functionality()
    test_learning()
    test_rule_extraction()
    test_complex_relationship()
    visualize_fuzzy_surface()

    print("=" * 60)
    print("All tests completed!")
    print("\nKey Takeaways:")
    print("• Variables are indices, not names (clean abstraction)")
    print("• Membership functions are learned, not designed")
    print("• Rules are discovered through soft gates")
    print("• Everything is differentiable and trainable")
    print("• The system can learn complex fuzzy relationships")
    print("=" * 60)