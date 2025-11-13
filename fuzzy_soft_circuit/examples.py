"""
Examples demonstrating automatic fuzzy rule learning.
"""

import autograd.numpy as np
from fuzzy_core import FuzzySoftCircuit, train_fuzzy_circuit
import matplotlib.pyplot as plt


def example_1_simple_control():
    """
    Simple example: Learn fuzzy control for a single input/output system.
    Input: temperature (0-1 normalized)
    Output: fan speed (0-1 normalized)
    """
    print("=" * 60)
    print("Example 1: Simple Temperature -> Fan Speed Control")
    print("=" * 60)

    # Create training data
    # Rule we want to learn: Low temp -> Low fan, High temp -> High fan
    data = [
        # (temp, fan_speed)
        ([0.1], [0.1]),  # Cold -> Slow fan
        ([0.2], [0.15]),
        ([0.3], [0.2]),
        ([0.5], [0.5]),  # Medium -> Medium fan
        ([0.7], [0.8]),
        ([0.8], [0.85]),  # Hot -> Fast fan
        ([0.9], [0.9]),
    ]

    # Create circuit
    circuit = FuzzySoftCircuit(
        n_inputs=1,
        n_outputs=1,
        n_memberships=3,  # LOW, MED, HIGH
        n_rules=5
    )

    # Train
    print("Training...")
    params = train_fuzzy_circuit(circuit, data, epochs=500, learning_rate=0.5)
    circuit.params = params

    # Test
    print("\nTesting:")
    test_inputs = [0.0, 0.25, 0.5, 0.75, 1.0]
    for temp in test_inputs:
        output = circuit.forward([temp])
        print(f"  Temp: {temp:.2f} -> Fan: {output[0]:.2f}")

    # Extract rules
    print("\nDiscovered Rules:")
    rules = circuit.extract_rules(var_names=['temp'])
    for rule, strength in rules:
        print(f"  {rule} (strength: {strength:.2f})")

    # Visualize membership functions
    circuit.visualize_memberships()
    plt.suptitle('Learned Membership Functions - Simple Control')
    plt.show()


def example_2_multi_input():
    """
    Multi-input example: Temperature and Humidity -> Fan Speed
    This is the classic fuzzy control example.
    """
    print("\n" + "=" * 60)
    print("Example 2: Temperature + Humidity -> Fan Speed")
    print("=" * 60)

    # Create training data based on human knowledge:
    # Hot + Dry -> Fast fan
    # Hot + Humid -> Very fast fan
    # Cold + Any -> Slow fan
    # Medium temp + Medium humidity -> Medium fan
    data = [
        # ([temp, humidity], [fan_speed])
        # Cold conditions
        ([0.2, 0.3], [0.1]),   # Cold + Dry -> Slow
        ([0.2, 0.7], [0.15]),  # Cold + Humid -> Slow

        # Medium conditions
        ([0.5, 0.5], [0.5]),   # Medium + Medium -> Medium
        ([0.5, 0.2], [0.4]),   # Medium + Dry -> Medium-Low
        ([0.5, 0.8], [0.6]),   # Medium + Humid -> Medium-High

        # Hot conditions
        ([0.8, 0.2], [0.7]),   # Hot + Dry -> Fast
        ([0.8, 0.8], [0.95]),  # Hot + Humid -> Very Fast
        ([0.9, 0.5], [0.8]),   # Very Hot + Medium -> Fast
    ]

    # Create circuit
    circuit = FuzzySoftCircuit(
        n_inputs=2,
        n_outputs=1,
        n_memberships=3,
        n_rules=9  # Potentially 3x3 combinations
    )

    # Train
    print("Training...")
    params = train_fuzzy_circuit(circuit, data, epochs=1000, learning_rate=0.3)
    circuit.params = params

    # Test on a grid
    print("\nTesting on grid:")
    print("Temp \\ Humidity:  0.2   0.5   0.8")
    for temp in [0.2, 0.5, 0.8]:
        outputs = []
        for humidity in [0.2, 0.5, 0.8]:
            output = circuit.forward([temp, humidity])
            outputs.append(output[0])
        print(f"  {temp:.1f}:            {outputs[0]:.2f}  {outputs[1]:.2f}  {outputs[2]:.2f}")

    # Extract rules
    print("\nDiscovered Rules:")
    rules = circuit.extract_rules(var_names=['temp', 'humidity'])
    for rule, strength in rules[:5]:  # Show top 5 rules
        print(f"  {rule}")
        print(f"    (strength: {strength:.2f})")

    # Visualize membership functions
    circuit.visualize_memberships()
    plt.suptitle('Learned Membership Functions - Multi-Input Control')
    plt.show()


def example_3_logic_discovery():
    """
    Example showing how the system can discover logical relationships.
    Learn: XOR-like fuzzy relationship
    """
    print("\n" + "=" * 60)
    print("Example 3: Discovering XOR-like Fuzzy Logic")
    print("=" * 60)

    # Create XOR-like fuzzy data
    # Output is high when inputs are different (one high, one low)
    data = [
        # ([input1, input2], [output])
        ([0.1, 0.1], [0.1]),   # Low, Low -> Low
        ([0.1, 0.9], [0.9]),   # Low, High -> High
        ([0.9, 0.1], [0.9]),   # High, Low -> High
        ([0.9, 0.9], [0.1]),   # High, High -> Low
        # Add some intermediate points
        ([0.5, 0.1], [0.5]),
        ([0.1, 0.5], [0.5]),
        ([0.5, 0.5], [0.3]),
        ([0.5, 0.9], [0.7]),
        ([0.9, 0.5], [0.7]),
    ]

    # Create circuit with fewer memberships to force abstraction
    circuit = FuzzySoftCircuit(
        n_inputs=2,
        n_outputs=1,
        n_memberships=2,  # Just LOW/HIGH
        n_rules=6
    )

    # Train
    print("Training...")
    params = train_fuzzy_circuit(circuit, data, epochs=1500, learning_rate=0.5)
    circuit.params = params

    # Test
    print("\nTesting (XOR-like behavior):")
    test_cases = [
        ([0.0, 0.0], "Low, Low"),
        ([0.0, 1.0], "Low, High"),
        ([1.0, 0.0], "High, Low"),
        ([1.0, 1.0], "High, High"),
        ([0.5, 0.5], "Med, Med"),
    ]

    for inputs, desc in test_cases:
        output = circuit.forward(inputs)
        print(f"  {desc:12} -> {output[0]:.3f}")

    # Extract rules
    print("\nDiscovered Rules (should find XOR-like pattern):")
    rules = circuit.extract_rules(var_names=['A', 'B'], threshold=0.4)
    for rule, strength in rules:
        print(f"  {rule}")
        print(f"    (strength: {strength:.2f})")


def example_4_mapping():
    """
    Example showing the variable mapping concept.
    We define a mapping once, then work with indices.
    """
    print("\n" + "=" * 60)
    print("Example 4: Variable Mapping System")
    print("=" * 60)

    # Define variable mapping
    var_mapping = {
        'temperature': 0,
        'pressure': 1,
        'flow_rate': 2,
        'valve_position': 0,  # output
    }

    # Reverse mapping for interpretability
    input_names = ['temperature', 'pressure', 'flow_rate']
    output_names = ['valve_position']

    print("Variable Mapping:")
    print(f"  Inputs:  {input_names}")
    print(f"  Outputs: {output_names}")

    # Create data using indices only
    data = [
        # Complex industrial control rule to learn
        ([0.2, 0.8, 0.3], [0.6]),  # Low temp, High pressure, Low flow -> Medium valve
        ([0.8, 0.2, 0.7], [0.3]),  # High temp, Low pressure, High flow -> Low valve
        ([0.5, 0.5, 0.5], [0.5]),  # All medium -> Medium valve
        ([0.9, 0.9, 0.1], [0.9]),  # High temp, High pressure, Low flow -> High valve
        ([0.1, 0.1, 0.9], [0.1]),  # Low temp, Low pressure, High flow -> Low valve
    ]

    # Create circuit
    circuit = FuzzySoftCircuit(
        n_inputs=3,
        n_outputs=1,
        n_memberships=3,
        n_rules=10
    )

    # Train
    print("\nTraining industrial controller...")
    params = train_fuzzy_circuit(circuit, data, epochs=800, learning_rate=0.3)
    circuit.params = params

    # Extract rules with meaningful names
    print("\nDiscovered Industrial Control Rules:")
    rules = circuit.extract_rules(var_names=input_names, threshold=0.35)
    for rule, strength in rules[:5]:
        # Replace output indices with names
        rule = rule.replace('output_0', output_names[0])
        print(f"  {rule}")
        print(f"    (strength: {strength:.2f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run all examples
    example_1_simple_control()
    example_2_multi_input()
    example_3_logic_discovery()
    example_4_mapping()

    print("\nAll examples completed!")
    print("\nKey insights:")
    print("1. Variables are just indices (0, 1, 2, ...)")
    print("2. Membership functions (LOW/MED/HIGH) are learned automatically")
    print("3. Rules are discovered through soft gates, not hardcoded")
    print("4. The 'IF' is a learnable switch that determines rule relevance")
    print("5. Everything is differentiable and trainable via gradient descent")