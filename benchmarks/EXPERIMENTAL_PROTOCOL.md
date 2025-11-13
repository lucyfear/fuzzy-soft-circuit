# Experimental Protocol for Fuzzy Soft Circuits Validation

## Objective
Provide rigorous experimental validation of the fuzzy soft circuits approach for publication at FUZZ-IEEE or IEEE Transactions on Fuzzy Systems. The goal is to demonstrate that automatic rule discovery through differentiable programming can compete with established methods while maintaining interpretability.

## Scientific Standards

### Reproducibility Requirements
- All random seeds documented and fixed
- Complete dependency versions recorded
- Raw data and intermediate results saved
- All hyperparameters explicitly documented

### Statistical Validity
- Multiple random initializations (n=10) per configuration
- Report mean, standard deviation, and 95% confidence intervals
- Paired t-tests for statistical significance (p < 0.05)
- Effect size reporting using Cohen's d

### Fair Comparison Principles
- Identical train/test splits across all methods
- Same number of parameters where possible (document if not)
- Equivalent computational budgets (same number of epochs/iterations)
- No method-specific hyperparameter tuning (use standard configurations)
- All baselines implemented with best practices

## Benchmark Datasets

Selected from UCI Machine Learning Repository and fuzzy systems literature:

### 1. Iris Classification (Fisher, 1936)
- **Type**: Multi-class classification (3 classes)
- **Inputs**: 4 (sepal length, sepal width, petal length, petal width)
- **Samples**: 150
- **Split**: 80/20 stratified train/test
- **Metric**: Classification accuracy
- **Justification**: Classic benchmark, interpretable features, standard in fuzzy systems literature

### 2. Wine Quality (Cortez et al., 2009)
- **Type**: Regression
- **Inputs**: 11 physicochemical properties
- **Samples**: 1599 (red wine)
- **Split**: 80/20 train/test
- **Metric**: MSE, MAE
- **Justification**: Real-world problem, fuzzy rules can be interpretable to domain experts

### 3. Diabetes (Pima Indians Diabetes)
- **Type**: Binary classification
- **Inputs**: 8 medical measurements
- **Samples**: 768
- **Split**: 80/20 stratified train/test
- **Metric**: Accuracy, F1-score
- **Justification**: Medical domain where interpretability is crucial

### 4. Energy Efficiency (Tsanas & Xifara, 2012)
- **Type**: Multi-output regression
- **Inputs**: 8 building parameters
- **Outputs**: 2 (heating load, cooling load)
- **Samples**: 768
- **Split**: 80/20 train/test
- **Metric**: MSE per output
- **Justification**: Engineering domain, multi-output fuzzy system application

### 5. Concrete Compressive Strength (Yeh, 1998)
- **Type**: Regression
- **Inputs**: 8 mixture components
- **Samples**: 1030
- **Split**: 80/20 train/test
- **Metric**: MSE, R²
- **Justification**: Civil engineering, demonstrates scalability to higher dimensions

## Baseline Methods

### 1. Fuzzy Soft Circuits (Proposed Method)
**Configuration:**
- Membership functions: 3 per input (Gaussian)
- Number of potential rules: 2 × n_inputs × n_memberships (allow discovery)
- Learning rate: 0.1 (tuned via validation set)
- Epochs: 1000
- Optimizer: Gradient descent with autograd
- Random seed: Varied across 10 runs

**Implementation:** Use existing FuzzySoftCircuit class from `/home/spinoza/github/beta/soft-circuit/fuzzy_soft_circuit/fuzzy_core.py`

### 2. ANFIS (Adaptive Neuro-Fuzzy Inference System)
**Configuration:**
- Membership functions: 3 per input (Gaussian)
- Rule structure: Grid partitioning (3^n_inputs rules for n_inputs variables)
- Hybrid learning: Gradient descent + least squares
- Epochs: 1000
- Note: For high-dimensional inputs (>4), use subtractive clustering to reduce rules

**Implementation:** Use Python anfis library or implement simplified version following Jang (1993)

**Fairness Note:** ANFIS requires predefined rule structure. For n=4 inputs, grid partitioning gives 3^4=81 rules. For fairness, document rule count differences clearly.

### 3. Multi-Layer Perceptron (Standard Neural Network)
**Configuration:**
- Architecture: [n_inputs, 32, 16, n_outputs]
- Activation: ReLU for hidden layers, sigmoid/linear for output
- Learning rate: 0.001
- Optimizer: Adam
- Epochs: 1000
- Regularization: None (to match fuzzy methods)

**Implementation:** Use scikit-learn MLPRegressor/MLPClassifier

**Fairness Note:** Neural network is black-box, no interpretability. Include to show accuracy/interpretability tradeoff.

## Experimental Procedure

### Phase 1: Data Preparation
1. Load each dataset from UCI repository
2. Normalize inputs to [0, 1] range (required for fuzzy methods)
3. For classification: one-hot encode outputs for fuzzy methods
4. Create fixed train/test splits with stratification where applicable
5. Save preprocessed data and split indices for reproducibility

### Phase 2: Hyperparameter Selection
- Use 20% of training data as validation set
- Grid search for learning rates: {0.01, 0.05, 0.1, 0.5}
- Select based on validation performance
- **Critical**: Use same hyperparameters for all 10 runs to avoid overfitting

### Phase 3: Training
For each dataset and each method:
1. Run 10 training sessions with different random seeds (0-9)
2. Log training loss every 100 epochs
3. Save final trained model
4. Record training time

### Phase 4: Evaluation
For each trained model:
1. Compute test set performance metrics
2. For fuzzy methods: Extract learned rules and count active rules
3. For Fuzzy Soft Circuits: Visualize learned membership functions
4. Save all predictions for error analysis

### Phase 5: Statistical Analysis
1. Compute mean and standard deviation across 10 runs
2. Perform paired t-tests between methods
3. Calculate 95% confidence intervals
4. Report Cohen's d effect sizes
5. Create summary tables and learning curves

## Metrics and Reporting

### Performance Metrics
- **Regression**: MSE (primary), MAE (secondary), R²
- **Classification**: Accuracy (primary), F1-score (secondary)

### Interpretability Metrics
- Number of active rules (threshold γ > 0.3)
- Average antecedent length (number of conditions per rule)
- Membership function overlap analysis

### Computational Metrics
- Training time (seconds)
- Model parameters count
- Inference time per sample

### Statistical Reporting Format
```
Method: X.XX ± Y.YY (95% CI: [A.AA, B.BB])
```

## Threats to Validity and Mitigation

### Internal Validity
**Threat**: Implementation bugs favor one method
**Mitigation**: Use established libraries where possible, extensive unit testing

**Threat**: Hyperparameter tuning favors one method
**Mitigation**: Use standard configurations, document all choices, limit tuning

### External Validity
**Threat**: Selected datasets not representative of fuzzy system applications
**Mitigation**: Include diverse domains (classification, regression, medical, engineering)

**Threat**: Small datasets may not show scaling behavior
**Mitigation**: Include range of dataset sizes (150 to 1599 samples)

### Construct Validity
**Threat**: Metrics don't capture interpretability advantage
**Mitigation**: Report both accuracy metrics and interpretability metrics

**Threat**: Number of rules not comparable across methods
**Mitigation**: Document rule counts clearly, discuss in limitations

## Expected Outcomes and Honest Reporting

### Success Criteria
The paper will be considered publication-ready if:
1. Fuzzy Soft Circuits achieve competitive performance (within 5% of best method on ≥3/5 datasets)
2. Statistical significance demonstrated where claims are made
3. Interpretability advantages documented quantitatively
4. Limitations honestly reported

### Anticipated Results
Based on the XOR experiment (MSE 0.012):
- **Hypothesis 1**: Fuzzy Soft Circuits will match ANFIS on small datasets where grid partitioning is feasible
- **Hypothesis 2**: Fuzzy Soft Circuits will outperform ANFIS on higher-dimensional datasets (n>4) where grid partitioning explodes
- **Hypothesis 3**: Neural networks will achieve lowest error on some datasets, but without interpretability
- **Hypothesis 4**: Fuzzy methods will discover fewer rules than grid partitioning predicts

### Honest Reporting Commitment
- If neural networks consistently outperform fuzzy methods, report it
- If ANFIS outperforms on some datasets, discuss why
- If results are mixed, present nuanced analysis
- Include all datasets in paper, not just favorable ones
- Report any failure modes or convergence issues observed

## Timeline and Computational Requirements

### Estimated Computation Time
- Data preparation: 2 hours (manual)
- Experiments: ~4-6 hours on modern CPU
  - 5 datasets × 3 methods × 10 runs × ~2 min per run = 300 minutes
- Analysis and visualization: 2 hours
- **Total**: ~8-10 hours

### Computational Resources
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8GB minimum
- Storage: 1GB for data and results
- No GPU required (datasets are small)

## Deliverables

1. **Code**:
   - `/benchmarks/run_experiments.py` - Main execution script
   - `/benchmarks/baselines/` - ANFIS and MLP implementations
   - `/benchmarks/datasets/` - Data loading and preprocessing
   - `/benchmarks/analysis/` - Statistical analysis and visualization

2. **Results**:
   - `/benchmarks/results/raw/` - Individual run results (CSV)
   - `/benchmarks/results/summary/` - Aggregated statistics (CSV)
   - `/benchmarks/results/figures/` - Publication-quality figures (PDF, 300 DPI)

3. **Documentation**:
   - `/benchmarks/RESULTS.md` - Summary of findings
   - `/benchmarks/REPRODUCIBILITY.md` - Instructions to reproduce
   - LaTeX tables and figure captions for paper integration

## References

- Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
