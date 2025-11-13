# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **Fuzzy Soft Circuits** - a novel approach to fuzzy logic where the system automatically discovers membership functions and rules as pure numerical patterns without expert knowledge. Unlike traditional fuzzy logic that requires semantic labels like "HIGH" or "LOW", this system works with indices (`membership_0`, `membership_1`, etc.) and learns optimal parameters through gradient descent.

## Development Commands

### Installation
```bash
# Install package in development mode
pip install -e .

# Install with visualization tools
pip install -e ".[viz]"

# Install with development and benchmark tools
pip install -e ".[dev,benchmarks]"
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=fuzzy_soft_circuit --cov-report=html

# Run a specific test file
pytest tests/test_fuzzy.py -v

# Run a single test
pytest tests/test_fuzzy.py::TestFuzzySoftCircuit::test_forward -v
```

### Benchmarks
```bash
# Run full benchmark suite (comparing against ANFIS and MLP)
cd benchmarks && python run_experiments.py

# Run quick test on reduced dataset
cd benchmarks && python run_quick_test.py

# Run with reduced epochs for fast iteration
cd benchmarks && python run_reduced_benchmark.py

# Generate example figures for paper
cd benchmarks && python generate_example_figures.py
```

### Documentation
```bash
# Compile LaTeX paper
cd papers/fuzzy && pdflatex paper.tex && pdflatex paper.tex

# Clean LaTeX artifacts
cd papers/fuzzy && rm -f *.aux *.log *.out *.bbl *.blg
```

## Code Architecture

### Core Abstraction (fuzzy_soft_circuit/fuzzy_core.py)

**Key Design Philosophy**: Index-based, not name-based
- Uses `membership_0`, `membership_1`, `membership_2` (not "HIGH", "LOW", "MEDIUM")
- Semantic interpretation is post-hoc by humans
- System learns pure numerical patterns

**FuzzySoftCircuit Class**:

The main class that implements the fuzzy logic system with three key components:

1. **Fuzzification Layer**
   - Each input variable gets `n_memberships` learnable Gaussian membership functions
   - Parameters: centers and widths learned via gradient descent
   - `membership_i(x) = exp(-((x - center_i) / width_i)²)`

2. **Rule Layer**
   - Soft AND gates compute antecedent activations
   - Learnable switches determine if a rule pattern exists: `activation * sigmoid(rule_switch)`
   - Rule parameters include antecedent weights and consequent outputs

3. **Defuzzification**
   - Weighted combination: `output = Σ(rule_activation * consequent) / Σ(rule_activation)`
   - Produces crisp output from fuzzy rule activations

**Parameter Structure**:
```python
params = {
    'membership_centers': (n_inputs, n_memberships),  # Centers of Gaussians
    'membership_widths': (n_inputs, n_memberships),   # Widths of Gaussians
    'rule_antecedents': (n_rules, n_inputs * n_memberships),  # Rule input weights
    'rule_switches': (n_rules,),  # Whether rule exists
    'rule_consequents': (n_rules, n_outputs * n_memberships)  # Rule outputs
}
```

### Training System (fuzzy_soft_circuit/fuzzy_pure.py)

**Training Functions**:
- `train_fuzzy_circuit(circuit, data, epochs, learning_rate)`: Main training loop
- Uses autograd for automatic differentiation
- MSE loss: `L = Σ ||output - target||²`

**Parameter Management**:
- `flatten_params(params_dict)`: Converts dict to 1D array for optimizer
- `unflatten_params(flat_array, shapes)`: Reconstructs dict from 1D array
- Handles heterogeneous parameter shapes (centers, widths, switches, etc.)

### Benchmark Architecture

Located in `benchmarks/`:

**Datasets** (`datasets/dataset_loader.py`):
- Loads 5 standard fuzzy logic benchmarks
- Handles train/test splitting with fixed random seeds for reproducibility
- Normalizes all inputs to [0, 1] range
- Returns `BenchmarkDataset` objects with metadata

**Baselines**:
- `baselines/anfis.py`: Simplified ANFIS (Adaptive Neuro-Fuzzy Inference System)
- `baselines/mlp_baseline.py`: Multi-layer perceptron for comparison

**Experimental Framework** (`run_experiments.py`):
- Runs 10-fold cross-validation for statistical rigor
- Compares Fuzzy Soft Circuits against ANFIS and MLP
- Saves results as JSON with full statistics
- Generates results for research paper

**Analysis** (`analysis/`):
- `statistical_analysis.py`: Hypothesis testing, significance tests, Wilcoxon signed-rank
- `visualizations.py`: Plots for papers (loss curves, comparisons, rule visualizations)

## Important Implementation Details

### Autograd Usage
- **Always use** `autograd.numpy as np` (not regular numpy) for differentiable operations
- Avoid in-place operations that break autograd tape
- Use `grad()` to compute gradients with respect to flattened parameters

### Data Format
```python
# Training data: list of (input, output) tuples
data = [
    ([0.9, 0.2], [0.8]),  # Inputs and outputs normalized to [0, 1]
    ([0.3, 0.8], [0.3]),
    # ... more examples
]
```

### Membership Function Initialization
- Centers: Uniformly distributed across [0, 1] range
- Widths: Small positive values (typically 0.1-0.3)
- Ensures coverage of input space and smooth gradients

### Rule Switch Interpretation
- `sigmoid(switch) > 0.5`: Rule is active and relevant
- `sigmoid(switch) < 0.5`: Rule effectively disabled
- During training, unnecessary rules naturally get suppressed

### Soft AND Implementation
- Uses product t-norm: `AND(a, b) = a * b`
- Alternative: minimum t-norm `min(a, b)` (not differentiable everywhere)
- Product provides smooth gradients for learning

## Common Workflows

### Adding a New Fuzzy Operator
1. Define operator function in `fuzzy_core.py`
2. Ensure it's differentiable (use autograd-compatible ops)
3. Test gradient flow with simple examples
4. Add to rule computation in `forward()` method

### Adding a New Dataset
1. Add loader function to `benchmarks/datasets/dataset_loader.py`
2. Follow existing pattern: return `BenchmarkDataset` object
3. Ensure normalization to [0, 1]
4. Add to `load_all_datasets()` function
5. Update benchmark scripts to include new dataset

### Running Experiments
1. Configure hyperparameters in `run_experiments.py`
2. Set random seed for reproducibility (default: 42)
3. Run experiments: `python run_experiments.py`
4. Results saved to `benchmarks/results/`
5. Generate figures: `python generate_example_figures.py`

### Extracting Learned Rules
```python
# After training, extract interpretable rules
params = train_fuzzy_circuit(controller, data)
rules = controller.extract_rules(params)

# Each rule shows:
# - Which input membership indices are involved
# - Activation strength (switch value)
# - Consequent output
```

## Testing Strategy

Tests are in `tests/test_fuzzy.py`:
- `TestLearnableMembershipFunction`: Membership function learning
- `TestFuzzySoftCircuit`: Full system forward pass
- `TestTraining`: Gradient-based optimization
- `TestBenchmarks`: Dataset loading and experimental setup

**Coverage expectations**: Aim for >70% coverage of core fuzzy logic code

## Key Research Insights

**Novel Contributions**:
1. **Automatic Rule Discovery**: System discovers which rules exist, not just their parameters
2. **Index-Based Architecture**: No semantic labels required during training
3. **End-to-End Differentiable**: Entire fuzzy system trained via gradient descent
4. **Post-hoc Interpretation**: Humans assign meaning after system learns patterns

**Advantages Over ANFIS**:
- No need to pre-specify rule structure
- Discovers sparse rule sets automatically
- More interpretable learned rules

**Advantages Over MLPs**:
- Interpretable logic structure
- Fewer parameters for equivalent performance
- Can extract human-readable rules

## Project Structure

```
fuzzy-soft-circuit/
├── fuzzy_soft_circuit/     # Core module
│   ├── fuzzy_core.py      # FuzzySoftCircuit implementation
│   ├── fuzzy_pure.py      # Pure numerical training functions
│   └── examples.py        # Usage examples
│
├── benchmarks/             # Experimental framework
│   ├── run_experiments.py # Main benchmark script
│   ├── datasets/          # Dataset loaders
│   ├── baselines/         # ANFIS and MLP implementations
│   └── analysis/          # Statistical analysis and plots
│
├── tests/                  # Test suite (pytest)
├── papers/fuzzy/          # Research paper (LaTeX)
└── docs/                   # Additional documentation
```

## Dependencies

**Core**: `numpy`, `autograd`, `scipy`
**Visualization**: `matplotlib`, `seaborn`
**Benchmarks**: `scikit-learn`, `pandas`
**Testing**: `pytest`, `pytest-cov`

## Related Projects

This project is part of the "Soft Circuits" research program:
- **[Soft Circuits](https://github.com/queelius/soft-circuit)**: Differentiable Boolean logic circuits

Both share the philosophy of making logic differentiable and learnable through gradient descent while maintaining interpretability.
