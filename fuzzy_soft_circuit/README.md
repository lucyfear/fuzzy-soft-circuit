# Fuzzy Soft Circuits

An extension of soft circuits to fuzzy logic with automatic rule discovery.

## Key Concepts

### Variable Mapping
Variables are mapped to indices, not names:
- Temperature → 0
- Humidity → 1
- Fan Speed → 2

This keeps the implementation clean and general.

### Automatic Learning
The system learns THREE things automatically:
1. **Membership Functions**: Gaussian curves at learned positions (indices 0, 1, 2...) with no predefined meaning
2. **Rule Structure**: Which combinations of membership indices matter?
3. **Rule Relevance**: Which rules are actually useful?

### The Architecture

```
Input (crisp) → Fuzzification → Rule Evaluation → Defuzzification → Output
      ↓              ↓                ↓                  ↓
   [0.75]      [HIGH: 0.8]    [Rule fires: 0.7]    [Fan: 0.85]
              [MED: 0.2]
              [LOW: 0.0]
```

## How It Works

### 1. Fuzzification Layer
Each input variable gets `n_memberships` learnable membership functions:
```python
# Learned automatically (no semantic labels):
membership_0(x) = exp(-((x - 0.8) / 0.2)²)  # Learns to center at 0.8
membership_1(x) = exp(-((x - 0.5) / 0.3)²)  # Learns to center at 0.5
membership_2(x) = exp(-((x - 0.2) / 0.2)²)  # Learns to center at 0.2

# Post-hoc interpretation: "membership_0 seems to capture high values"
```

### 2. Rule Discovery
Instead of hardcoding rules with semantic labels, the system learns pure numerical patterns:
- **Rule Antecedents**: Soft gates discover which membership indices matter (e.g., "input_0_membership_2 AND input_1_membership_0")
- **Rule Switches**: Learn if a rule pattern is relevant at all
- **Rule Consequents**: What membership indices the rule activates for outputs

### 3. Key Innovation: Soft IF
The "IF" in fuzzy rules becomes a learnable switch:
```python
rule_activation = soft_AND(fuzzy_inputs) * sigmoid(rule_switch)
```
This lets the circuit learn which rules exist, not just their parameters.

## Usage Example

```python
from fuzzy_core import FuzzySoftCircuit, train_fuzzy_circuit

# Create circuit - just specify dimensions
circuit = FuzzySoftCircuit(
    n_inputs=2,      # e.g., temp and humidity
    n_outputs=1,     # e.g., fan speed
    n_memberships=3, # LOW, MED, HIGH per variable
    n_rules=10       # Max potential rules to discover
)

# Training data - pure numerical patterns
data = [
    ([0.8, 0.2], [0.7]),  # Pattern A
    ([0.2, 0.7], [0.2]),  # Pattern B
    # ... more examples
]
# System learns: when input_0 activates membership_2 and
# input_1 activates membership_0, output tends to be high

# Train - discovers rules automatically
params = train_fuzzy_circuit(circuit, data)

# Extract learned rules (optional semantic interpretation)
rules = circuit.extract_numerical_rules()
# Returns: "Rule 3: input_0_mem_2 (0.8) AND input_1_mem_0 (0.9) → output_0_mem_2 (0.7)"

# Can optionally map to semantics for human understanding:
# "Looks like Rule 3 learned: high temp + low humidity → high fan speed"
```

## Design Decisions

### Why Indices Instead of Names?
- **Generality**: Same code works for any domain
- **Efficiency**: No string manipulation in core loops
- **Mapping**: Easy to map between domains at boundaries

### Why Learnable Membership Functions?
- **Adaptability**: The system discovers meaningful value ranges without preconceptions
- **No Expert Needed**: System discovers what numerical patterns matter
- **End-to-end Learning**: Gradients flow through entire system
- **Pure Numerical**: No built-in semantic labels - just indices and learned parameters

### Why Soft Switches for IF?
- **Rule Discovery**: Learn which rules exist, not just parameters
- **Sparsity**: Many potential rules, few actually needed
- **Interpretability**: Active rules have high switch values

## Advantages Over Classical Fuzzy Logic

1. **No Expert Knowledge Required**: Learns from data, not manual rule specification
2. **Adaptive Memberships**: Discovers optimal fuzzy sets for the problem
3. **Rule Discovery**: Finds rules you didn't know existed
4. **End-to-end Differentiable**: Use modern optimization techniques
5. **Interpretable**: Can extract human-readable rules

## Experiments to Try

1. **Control Systems**: Temperature control, robot navigation
2. **Classification**: Fuzzy boundaries between classes
3. **Decision Making**: Multi-criteria fuzzy decisions
4. **Logic Discovery**: Learn unknown fuzzy relationships

## Mathematical Foundation

The system minimizes:
```
L = Σ ||output(inputs; θ) - target||²
```

Where θ includes:
- Membership function parameters (centers, widths)
- Rule antecedent weights (feature relevance)
- Rule switches (rule existence)
- Rule consequents (outputs)
- Output combination weights

Everything is differentiable, enabling gradient-based learning.

## Future Extensions

1. **Type-2 Fuzzy**: Uncertainty about membership functions
2. **Recurrent Rules**: Rules that depend on history
3. **Hierarchical Rules**: Rules of rules
4. **Adaptive Operators**: Learn optimal t-norms/s-norms
5. **Online Learning**: Adapt rules as new data arrives