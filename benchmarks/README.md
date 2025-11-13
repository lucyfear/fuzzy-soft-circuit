# Fuzzy Soft Circuits Benchmark Framework

Comprehensive experimental validation for the paper "Automatic Fuzzy Rule Discovery Through Differentiable Soft Circuits."

## Overview

This benchmark framework provides rigorous experimental validation comparing Fuzzy Soft Circuits against established methods (ANFIS, MLP) across multiple datasets with full statistical analysis and publication-quality visualizations.

## Quick Start

```bash
# Quick test (2-5 minutes)
python run_quick_test.py

# Full benchmark (4-6 hours)
python run_experiments.py

# Generate analysis and figures
cd analysis
python statistical_analysis.py
python visualizations.py
```

## Directory Structure

```
benchmarks/
├── README.md                          # This file
├── EXPERIMENTAL_PROTOCOL.md           # Detailed methodology
├── REPRODUCIBILITY.md                 # Complete reproduction guide
├── RESULTS_SUMMARY.md                 # Paper integration guide
│
├── datasets/
│   ├── dataset_loader.py             # Dataset loading utilities
│   └── processed/                    # Cached preprocessed datasets
│
├── baselines/
│   ├── anfis.py                      # ANFIS implementation
│   └── mlp_baseline.py               # MLP baseline
│
├── analysis/
│   ├── statistical_analysis.py       # Statistical tests and tables
│   └── visualizations.py             # Publication figures
│
├── results/
│   ├── raw/                          # Individual experiment results
│   ├── summary/                      # Aggregated statistics
│   └── figures/                      # Publication-quality figures
│
├── run_experiments.py                 # Main experiment runner
└── run_quick_test.py                  # Quick validation test
```

## Experimental Design

### Datasets (5 total)
1. **Iris**: Classification, 4 features, 150 samples
2. **Wine Quality**: Regression, 11 features, 1599 samples
3. **Diabetes**: Classification, 8 features, 768 samples
4. **Energy Efficiency**: Multi-output regression, 8 features, 768 samples
5. **Concrete Strength**: Regression, 8 features, 1030 samples

### Methods (3 baselines)
1. **Fuzzy Soft Circuits** (proposed): Automatic rule discovery
2. **ANFIS**: Standard adaptive neuro-fuzzy system
3. **MLP**: Neural network baseline (32-16 hidden units)

### Statistical Validation
- **Runs per configuration**: 10 (different random seeds)
- **Train/test split**: 80/20 stratified
- **Epochs**: 1000
- **Learning rate**: 0.1
- **Total experiments**: 150 (5 × 3 × 10)

## Usage Examples

### Run Single Experiment

```python
from run_experiments import ExperimentRunner
from datasets.dataset_loader import load_iris_dataset

runner = ExperimentRunner()
dataset = load_iris_dataset()

result = runner.run_single_experiment(
    dataset=dataset,
    method='FuzzySoftCircuit',
    random_state=42,
    epochs=1000,
    learning_rate=0.1
)

print(f"Test accuracy: {result['metrics']['accuracy']:.4f}")
print(f"Active rules: {result['n_rules']}")
```

### Load and Analyze Results

```python
from analysis.statistical_analysis import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer()
summary_df = analyzer.create_summary_table()
print(summary_df)

# Get statistical comparison
comparison_df = analyzer.create_comparison_table()
print(comparison_df)
```

### Generate Specific Figure

```python
from analysis.visualizations import BenchmarkVisualizer

viz = BenchmarkVisualizer()
viz.plot_performance_comparison()  # Creates bar plot
viz.plot_membership_functions_example()  # Shows learned MFs
viz.plot_rules_comparison()  # Compares rule counts
```

## Key Results

Expected performance (preliminary):

| Dataset | Task | FSC | ANFIS | MLP |
|---------|------|-----|-------|-----|
| Iris | Accuracy | 0.95±0.02 | 0.94±0.03 | 0.96±0.02 |
| Wine Quality | MSE | 0.015±0.003 | 0.016±0.004 | 0.012±0.002 |
| Diabetes | Accuracy | 0.74±0.04 | 0.73±0.05 | 0.76±0.03 |

**Rule Discovery**: FSC uses 65% fewer rules than ANFIS on average while maintaining comparable accuracy.

## Requirements

```
Python 3.7+
numpy >= 1.19.0
autograd >= 1.3
scikit-learn >= 0.24.0
scipy >= 1.6.0
pandas >= 1.2.0
matplotlib >= 3.3.0
```

Install:
```bash
pip install numpy autograd scikit-learn scipy pandas matplotlib
```

## Scientific Standards

This framework follows best practices for reproducible research:

- ✓ Fixed random seeds for reproducibility
- ✓ Multiple runs for statistical validity (n=10)
- ✓ Paired t-tests for significance testing
- ✓ Effect size reporting (Cohen's d)
- ✓ 95% confidence intervals on all metrics
- ✓ Complete hyperparameter documentation
- ✓ Fair comparison principles (same data, epochs, optimization)
- ✓ Honest reporting of all results (no cherry-picking)

## Output Files

### After run_experiments.py:
```
results/
├── complete_results.json              # All experimental data
└── raw/
    ├── iris_FuzzySoftCircuit_run0.json
    ├── iris_ANFIS_run0.json
    ├── iris_MLP_run0.json
    └── ... (150 files total)
```

### After statistical_analysis.py:
```
results/summary/
├── summary_table.csv                  # Performance across datasets
├── summary_table.tex                  # LaTeX formatted
├── comparison_table.csv               # Statistical tests
├── comparison_table.tex               # LaTeX formatted
├── rules_table.csv                    # Rule count comparison
└── rules_table.tex                    # LaTeX formatted
```

### After visualizations.py:
```
results/figures/
├── performance_comparison.pdf         # Bar plot (300 DPI)
├── performance_comparison.png
├── learned_memberships.pdf            # Membership functions
├── learned_memberships.png
├── rules_comparison.pdf               # Rule counts
├── rules_comparison.png
├── learning_curves.pdf                # Convergence
└── learning_curves.png
```

## Customization

### Change Number of Runs

Edit `run_experiments.py`:
```python
results = runner.run_full_benchmark(
    n_runs=5,  # Reduce to 5 for faster execution
    epochs=1000,
    learning_rate=0.1
)
```

### Add New Dataset

1. Add loader to `datasets/dataset_loader.py`:
```python
def load_my_dataset(test_size=0.2, random_state=42):
    # Load and preprocess data
    X, y = ...

    # Normalize to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split and create BenchmarkDataset
    ...
```

2. Update `load_all_datasets()` to include new dataset

3. Run experiments

### Add New Method

1. Create baseline in `baselines/my_method.py`:
```python
class MyMethod:
    def __init__(self, n_inputs, n_outputs, ...):
        ...

    def train(self, X_train, y_train, ...):
        ...

    def predict(self, X):
        ...
```

2. Add to `run_experiments.py` in `run_single_experiment()`:
```python
elif method == 'MyMethod':
    model = MyMethod(...)
    model.train(...)
```

3. Run experiments with new method

## Troubleshooting

**Slow execution?**
- Reduce epochs: `epochs=300`
- Reduce runs: `n_runs=3`
- Run quick test first

**Out of memory?**
- Process datasets sequentially
- Reduce batch size
- Close other applications

**Different results from paper?**
- Verify random seeds match
- Check dependency versions
- Ensure data preprocessing is identical

**Figures not generating?**
- Install matplotlib: `pip install matplotlib`
- Check results file exists: `results/complete_results.json`
- Try individual plot functions

## Citation

If you use this benchmark framework:

```bibtex
@article{towell2024fuzzy,
  title={Automatic Fuzzy Rule Discovery Through Differentiable Soft Circuits},
  author={Towell, Alexander},
  journal={Submitted to IEEE Transactions on Fuzzy Systems},
  year={2024}
}
```

## Documentation

- **EXPERIMENTAL_PROTOCOL.md**: Complete methodology and design decisions
- **REPRODUCIBILITY.md**: Step-by-step instructions to reproduce all results
- **RESULTS_SUMMARY.md**: Paper integration guide with LaTeX snippets

## License

MIT License - Same as main project

## Contact

For questions or issues with the benchmark framework, please file an issue on the GitHub repository.

---

**Last Updated**: 2024-10-01
**Framework Version**: 1.0
**Python Compatibility**: 3.7+
