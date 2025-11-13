"""
Quick test of benchmark framework with reduced scope.

Runs 2 datasets × 3 methods × 2 runs for rapid validation.
Full experiments use 5 datasets × 3 methods × 10 runs.
"""

import sys
sys.path.insert(0, '/home/spinoza/github/beta/soft-circuit')
sys.path.insert(0, '/home/spinoza/github/beta/soft-circuit/benchmarks')

from run_experiments import ExperimentRunner
from datasets.dataset_loader import load_iris_dataset, load_wine_quality_dataset

def main():
    """Run quick test with reduced scope."""
    print("\n" + "="*80)
    print("QUICK BENCHMARK TEST")
    print("Running: 2 datasets × 3 methods × 2 runs")
    print("="*80 + "\n")

    runner = ExperimentRunner()

    # Load only 2 datasets
    datasets = {
        'iris': load_iris_dataset(test_size=0.2, random_state=42),
        'wine_quality': load_wine_quality_dataset(test_size=0.2, random_state=42)
    }

    methods = ['FuzzySoftCircuit', 'ANFIS', 'MLP']
    all_results = []

    for dataset_name, dataset in datasets.items():
        for method in methods:
            for run in range(2):  # Only 2 runs for quick test
                print(f"\n{'='*60}")
                print(f"Dataset: {dataset_name}, Method: {method}, Run: {run}")
                print(f"{'='*60}")

                result = runner.run_single_experiment(
                    dataset=dataset,
                    method=method,
                    random_state=run,
                    epochs=200,  # Reduced epochs for quick test
                    learning_rate=0.5  # Higher learning rate for faster convergence
                )

                all_results.append(result)

    print("\n" + "="*80)
    print("QUICK TEST COMPLETE")
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: {runner.output_dir}")
    print("="*80)

    # Print summary
    print("\nSummary:")
    for result in all_results:
        if result.get('converged', False):
            dataset = result['dataset']
            method = result['method']
            metrics = result['metrics']
            print(f"{dataset:15} {method:20} -> {metrics}")


if __name__ == "__main__":
    main()
