"""
Main experimental framework for fuzzy soft circuits benchmark study.

Runs comprehensive experiments comparing:
1. Fuzzy Soft Circuits (proposed method)
2. ANFIS (standard fuzzy baseline)
3. MLP (neural network baseline)

Across 5 benchmark datasets with statistical validation.
"""

import sys
import os
sys.path.insert(0, '/home/spinoza/github/beta/soft-circuit')
sys.path.insert(0, '/home/spinoza/github/beta/soft-circuit/benchmarks')

import autograd.numpy as np
import time
import json
from pathlib import Path

# Import datasets
from datasets.dataset_loader import load_all_datasets, BenchmarkDataset

# Import baselines
from baselines.anfis import SimplifiedANFIS
from baselines.mlp_baseline import MLPBaseline

# Import fuzzy soft circuit
from fuzzy_soft_circuit.fuzzy_core import FuzzySoftCircuit, flatten_params, unflatten_params
from autograd import grad


class ExperimentRunner:
    """Manages benchmark experiments with statistical rigor."""

    def __init__(self, output_dir='/home/spinoza/github/beta/soft-circuit/benchmarks/results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / 'raw'
        self.raw_dir.mkdir(exist_ok=True)

    def train_fuzzy_soft_circuit(self, dataset, epochs=1000, learning_rate=0.1,
                                 n_memberships=3, n_rules=None, random_state=42):
        """Train FuzzySoftCircuit on a dataset."""
        np.random.seed(random_state)

        # Determine number of rules if not specified
        if n_rules is None:
            # Use 2 * n_inputs * n_memberships as heuristic
            n_rules = 2 * dataset.n_features * n_memberships

        # Handle multi-output
        n_outputs = dataset.y_train.shape[1] if len(dataset.y_train.shape) > 1 else 1

        # Initialize circuit
        circuit = FuzzySoftCircuit(
            n_inputs=dataset.n_features,
            n_outputs=n_outputs,
            n_memberships=n_memberships,
            n_rules=n_rules
        )

        # Prepare data
        data = [(dataset.X_train[i], dataset.y_train[i])
                for i in range(len(dataset.X_train))]

        # Flatten parameters
        flat_params, shapes = flatten_params(circuit.params)

        def loss(flat):
            params = unflatten_params(flat, shapes)
            total_loss = 0
            for inputs, targets in data:
                outputs = circuit.forward(inputs, params)
                if len(outputs.shape) == 0:
                    outputs = np.array([outputs])
                if len(targets.shape) == 0:
                    targets = np.array([targets])
                total_loss += np.sum((outputs - targets) ** 2)
            return total_loss / len(data)

        grad_fn = grad(loss)

        # Training loop
        start_time = time.time()
        history = []

        for epoch in range(epochs):
            current_loss = loss(flat_params)
            history.append(current_loss)

            g = grad_fn(flat_params)
            flat_params -= learning_rate * g

            if epoch % 200 == 0:
                print(f"  Epoch {epoch}, Loss: {current_loss:.6f}")

        train_time = time.time() - start_time

        # Update circuit parameters
        circuit.params = unflatten_params(flat_params, shapes)

        return circuit, history, train_time

    def evaluate_model(self, model, dataset, method_name):
        """Evaluate a trained model on test data."""
        if method_name == 'FuzzySoftCircuit':
            # Predict on test set
            predictions = []
            for i in range(len(dataset.X_test)):
                pred = model.forward(dataset.X_test[i])
                if len(pred.shape) == 0:
                    pred = np.array([pred])
                predictions.append(pred)
            predictions = np.array(predictions)
        else:
            # ANFIS or MLP
            predictions = model.predict(dataset.X_test)

        # Ensure correct shape
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        if len(dataset.y_test.shape) == 1:
            y_test = dataset.y_test.reshape(-1, 1)
        else:
            y_test = dataset.y_test

        # Compute metrics
        if dataset.task_type == 'regression':
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            # R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }
        else:  # classification
            # Convert one-hot to class labels for accuracy
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_test, axis=1)

            accuracy = np.mean(pred_labels == true_labels)

            # F1 score (macro average for multi-class)
            n_classes = y_test.shape[1]
            f1_scores = []
            for c in range(n_classes):
                tp = np.sum((pred_labels == c) & (true_labels == c))
                fp = np.sum((pred_labels == c) & (true_labels != c))
                fn = np.sum((pred_labels != c) & (true_labels == c))

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                f1_scores.append(f1)

            f1_macro = np.mean(f1_scores)

            metrics = {
                'accuracy': float(accuracy),
                'f1_score': float(f1_macro)
            }

        return metrics, predictions

    def count_active_rules(self, circuit, threshold=0.3):
        """Count active rules in FuzzySoftCircuit."""
        count = 0
        for r in range(circuit.n_rules):
            switch_val = 1 / (1 + np.exp(-circuit.params['rule_switches'][r]))
            if switch_val > threshold:
                count += 1
        return count

    def run_single_experiment(self, dataset, method, random_state=42,
                            epochs=1000, learning_rate=0.1):
        """Run a single experiment: train and evaluate one method on one dataset."""
        print(f"\n{'='*60}")
        print(f"Method: {method}, Dataset: {dataset.name}, Seed: {random_state}")
        print(f"{'='*60}")

        results = {
            'dataset': dataset.name,
            'method': method,
            'random_state': random_state,
            'task_type': dataset.task_type
        }

        try:
            if method == 'FuzzySoftCircuit':
                model, history, train_time = self.train_fuzzy_soft_circuit(
                    dataset, epochs=epochs, learning_rate=learning_rate,
                    random_state=random_state
                )
                results['n_rules'] = self.count_active_rules(model)
                results['n_params'] = sum(v.size for v in model.params.values())

            elif method == 'ANFIS':
                n_outputs = dataset.y_train.shape[1] if len(dataset.y_train.shape) > 1 else 1

                # Limit rules for high-dimensional datasets
                max_rules = 100 if dataset.n_features > 4 else None

                model = SimplifiedANFIS(
                    n_inputs=dataset.n_features,
                    n_outputs=n_outputs,
                    n_memberships=3,
                    max_rules=max_rules
                )

                start_time = time.time()
                history = model.train(
                    dataset.X_train, dataset.y_train,
                    epochs=epochs, learning_rate=learning_rate,
                    verbose=True
                )
                train_time = time.time() - start_time

                results['n_rules'] = model.n_rules
                results['n_params'] = model.count_parameters()

            elif method == 'MLP':
                n_outputs = dataset.y_train.shape[1] if len(dataset.y_train.shape) > 1 else 1

                model = MLPBaseline(
                    n_inputs=dataset.n_features,
                    n_outputs=n_outputs,
                    task_type=dataset.task_type,
                    hidden_layers=(32, 16),
                    random_state=random_state
                )

                train_time = model.train(dataset.X_train, dataset.y_train, verbose=True)
                history = model.get_loss_curve()

                results['n_rules'] = None  # N/A for MLP
                results['n_params'] = model.count_parameters()

            else:
                raise ValueError(f"Unknown method: {method}")

            # Evaluate
            metrics, predictions = self.evaluate_model(model, dataset, method)
            results['metrics'] = metrics
            results['train_time'] = train_time
            results['converged'] = True

            print(f"\nResults:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")
            print(f"  Train time: {train_time:.2f}s")
            if results['n_rules'] is not None:
                print(f"  Active rules: {results['n_rules']}")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results['converged'] = False
            results['error'] = str(e)

        return results

    def run_full_benchmark(self, n_runs=10, epochs=1000, learning_rate=0.1):
        """Run complete benchmark across all datasets and methods."""
        print("\n" + "="*80)
        print("FUZZY SOFT CIRCUITS BENCHMARK STUDY")
        print("="*80)

        # Load datasets
        print("\nLoading datasets...")
        datasets = load_all_datasets(test_size=0.2, random_state=42)

        methods = ['FuzzySoftCircuit', 'ANFIS', 'MLP']

        all_results = []

        for dataset_name, dataset in datasets.items():
            for method in methods:
                for run in range(n_runs):
                    random_state = run  # Different seed for each run

                    result = self.run_single_experiment(
                        dataset=dataset,
                        method=method,
                        random_state=random_state,
                        epochs=epochs,
                        learning_rate=learning_rate
                    )

                    all_results.append(result)

                    # Save individual result
                    result_file = self.raw_dir / f"{dataset_name}_{method}_run{run}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)

        # Save complete results
        complete_file = self.output_dir / 'complete_results.json'
        with open(complete_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)

        return all_results


def main():
    """Main entry point for benchmarks."""
    runner = ExperimentRunner()

    # Run full benchmark with statistical validation
    # Using 10 runs per configuration for statistical significance
    results = runner.run_full_benchmark(
        n_runs=10,
        epochs=1000,
        learning_rate=0.1
    )

    print(f"\n\nTotal experiments completed: {len(results)}")
    print(f"Results directory: {runner.output_dir}")


if __name__ == "__main__":
    main()
