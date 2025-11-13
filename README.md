# Fuzzy Soft Circuits: Automatic Fuzzy Rule Discovery

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Learn fuzzy logic systems from data without expert knowledge.**

This project implements a novel approach to fuzzy logic where the system automatically discovers membership functions and rules as pure numerical patterns—no predefined semantic concepts like "HIGH" or "LOW" required.

## Key Innovation

Traditional fuzzy logic requires experts to manually define:
- Membership functions ("HIGH temperature = above 30°C")
- Fuzzy rules ("IF temperature is HIGH AND humidity is LOW THEN fan speed is HIGH")

**Fuzzy Soft Circuits eliminate this requirement.** The system learns:
1. **Membership Functions**: Gaussian curves at optimal positions (no predefined meaning)
2. **Rule Structures**: Which combinations of membership indices activate together
3. **Rule Relevance**: Which discovered patterns actually matter

All through gradient-based optimization on training data.

## Installation

```bash
# Clone the repository
git clone https://github.com/queelius/fuzzy-soft-circuit.git
cd fuzzy-soft-circuit

# Install dependencies
pip install autograd numpy scipy

# Optional: for visualization and benchmarks
pip install matplotlib seaborn scikit-learn pandas
```

## Quick Start

```python
from fuzzy_soft_circuit import FuzzySoftCircuit, train_fuzzy_circuit

# Create fuzzy controller - no semantic labels!
controller = FuzzySoftCircuit(
    n_inputs=2,      # Two input dimensions (e.g., temperature, humidity)
    n_outputs=1,     # One output dimension (e.g., fan speed)
    n_memberships=3, # 3 membership functions per input (indices 0, 1, 2)
    n_rules=10       # Discover up to 10 rules
)

# Training data - pure numerical patterns
data = [
    ([0.9, 0.2], [0.8]),  # Pattern: high dim0, low dim1 → high output
    ([0.3, 0.8], [0.3]),  # Pattern: low dim0, high dim1 → low output
    ([0.5, 0.5], [0.5]),  # Pattern: medium values → medium output
    # ... more examples
]

# Train - discovers patterns automatically
params = train_fuzzy_circuit(controller, data, epochs=1000)

# Make predictions
output = controller.forward([0.85, 0.25], params)
print(f"Prediction: {output}")

# Extract learned rules (optional semantic interpretation)
rules = controller.extract_rules(params)
for i, rule in enumerate(rules):
    print(f"Rule {i}: {rule}")
```

## How It Works

### Index-Based Architecture

Instead of semantic labels, the system uses indices:
- `membership_0`, `membership_1`, `membership_2` (not "LOW", "MEDIUM", "HIGH")
- System learns that `input_0_membership_2` activates around value 0.9
- Humans can interpret post-hoc: "membership_2 seems to represent 'high' values"

### Three Learning Components

**1. Fuzzification**: Learnable Gaussian membership functions
```python
membership_i(x) = exp(-((x - center_i) / width_i)²)
```
Centers and widths are learned from data.

**2. Rule Discovery**: Soft AND gates with learnable switches
```python
rule_activation = soft_AND(fuzzy_inputs) * sigmoid(rule_switch)
```
The switch parameter learns whether a rule pattern exists at all.

**3. Defuzzification**: Weighted combination of rule outputs
```python
output = Σ(rule_activation * rule_consequent) / Σ(rule_activation)
```

### Example: What the System Learns

```
Rule 3: input_0_membership_2 (0.8) AND input_1_membership_0 (0.9) → output_0_membership_2 (0.7)
```

This is a pure numerical pattern. A human might interpret it as:
> "When temperature is high (membership_2 ≈ 0.9) and humidity is low (membership_0 ≈ 0.2), set fan speed high (membership_2 ≈ 0.8)"

But the system only knows indices and learned parameters.

## Core Features

- **Pure Numerical Learning**: No hardcoded semantic concepts
- **Automatic Membership Discovery**: Learns Gaussian curves at optimal positions
- **Rule Pattern Discovery**: Finds which membership index combinations matter
- **Index-Based Architecture**: Works with `membership_0`, `membership_1`, etc.
- **Post-hoc Interpretation**: Humans can assign meaning after training
- **End-to-End Differentiable**: Uses autograd for gradient-based learning
- **Interpretable**: Can extract human-readable rules from trained models

## Advantages Over Classical Fuzzy Logic

1. **No Expert Knowledge Required**: Learns from data, not manual specifications
2. **Adaptive Memberships**: Discovers optimal fuzzy sets for the problem
3. **Rule Discovery**: Finds rules you didn't know existed
4. **End-to-end Differentiable**: Use modern optimization techniques
5. **Domain-Agnostic**: Same code works for any fuzzy control problem

## Benchmarks

The `benchmarks/` directory contains comprehensive experiments comparing Fuzzy Soft Circuits against:
- **ANFIS** (Adaptive Neuro-Fuzzy Inference System) - classical baseline
- **MLP** (Multi-Layer Perceptron) - neural network baseline

Across 5 standard datasets with statistical validation.

```bash
# Run full benchmark suite
cd benchmarks && python run_experiments.py

# Quick test on reduced dataset
cd benchmarks && python run_quick_test.py

# Generate figures for papers
cd benchmarks && python generate_example_figures.py
```

## Project Structure

```
fuzzy-soft-circuit/
├── fuzzy_soft_circuit/     # Core implementation
│   ├── fuzzy_core.py      # FuzzySoftCircuit class
│   ├── fuzzy_pure.py      # Pure numerical training functions
│   └── examples.py        # Usage examples
│
├── benchmarks/             # Experimental framework
│   ├── run_experiments.py # Main benchmark script
│   ├── datasets/          # Dataset loaders
│   ├── baselines/         # ANFIS and MLP implementations
│   └── analysis/          # Statistical analysis and visualization
│
├── tests/                  # Test suite (pytest)
│   └── test_fuzzy.py      # Comprehensive tests
│
├── papers/                 # Research paper (LaTeX)
│   └── fuzzy/
│       ├── paper.tex      # Paper source
│       └── paper.pdf      # Compiled paper
│
└── docs/                   # Documentation
```

## Research Paper

**[Automatic Fuzzy Rule Discovery Through Differentiable Soft Circuits](papers/fuzzy/paper.pdf)**
- Introduces pure numerical approach to fuzzy logic learning
- Demonstrates automatic rule discovery without expert knowledge
- Provides comprehensive benchmarks against classical methods

## Use Cases

- **Control Systems**: Temperature control, motor control, HVAC systems
- **Classification**: Problems with fuzzy boundaries between classes
- **Decision Making**: Multi-criteria fuzzy decisions
- **Pattern Recognition**: Discovering interpretable patterns in data
- **Time Series**: Fuzzy temporal pattern discovery

## Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fuzzy_soft_circuit --cov-report=html
```

## Mathematical Foundation

The system minimizes mean squared error:
```
L = Σ ||output(inputs; θ) - target||²
```

Where θ includes:
- Membership function parameters (centers, widths)
- Rule antecedent weights (feature relevance)
- Rule switches (rule existence)
- Rule consequents (outputs)
- Output combination weights

Everything is differentiable, enabling gradient-based learning through autograd.

## Contributing

We welcome contributions! Areas of interest:
- New defuzzification methods
- Alternative membership function shapes
- Performance optimizations
- Applications to new domains
- Visualization tools

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{fuzzy_soft_circuit,
  title = {Fuzzy Soft Circuits: Automatic Fuzzy Rule Discovery},
  author = {Alexander Towell},
  year = {2024},
  url = {https://github.com/queelius/fuzzy-soft-circuit}
}
```

## Related Work

This project is part of the "Soft Circuits" research program on making logic differentiable:
- **[Soft Circuits](https://github.com/queelius/soft-circuit)**: Differentiable Boolean logic circuits

---

*"Fuzzy logic, learned not specified."*
