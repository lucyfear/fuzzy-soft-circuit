"""
Reduced benchmark for timely completion.

Runs 3 datasets × 3 methods × 3 runs × 300 epochs
Estimated time: 3-4 hours (manageable for overnight run)

This provides statistically meaningful results while being computationally feasible.
Full benchmark (5 datasets × 10 runs × 1000 epochs) should be run post-acceptance.
"""

import sys
sys.path.insert(0, '/home/spinoza/github/beta/soft-circuit')
sys.path.insert(0, '/home/spinoza/github/beta/soft-circuit/benchmarks')

from run_experiments import ExperimentRunner
from datasets.dataset_loader import load_iris_dataset, load_wine_quality_dataset, load_diabetes_dataset
import json
from pathlib import Path

def main():
    """Run reduced benchmark."""
    print("\n" + "="*80)
    print("REDUCED BENCHMARK FOR TIMELY COMPLETION")
    print("3 datasets × 3 methods × 3 runs × 300 epochs")
    print("Estimated time: 3-4 hours")
    print("="*80 + "\n")

    runner = ExperimentRunner()

    # Load representative datasets
    print("Loading datasets...")
    datasets = {
        'iris': load_iris_dataset(test_size=0.2, random_state=42),
        'wine_quality': load_wine_quality_dataset(test_size=0.2, random_state=42),
        'diabetes': load_diabetes_dataset(test_size=0.2, random_state=42)
    }

    methods = ['FuzzySoftCircuit', 'ANFIS', 'MLP']
    n_runs = 3  # Reduced from 10
    epochs = 300  # Reduced from 1000
    learning_rate = 0.5  # Increased for faster convergence

    all_results = []
    total_experiments = len(datasets) * len(methods) * n_runs
    completed = 0

    print(f"\nTotal experiments to run: {total_experiments}\n")

    for dataset_name, dataset in datasets.items():
        for method in methods:
            for run in range(n_runs):
                completed += 1
                print(f"\n{'='*80}")
                print(f"Progress: {completed}/{total_experiments}")
                print(f"Dataset: {dataset_name}, Method: {method}, Run: {run+1}/{n_runs}")
                print(f"{'='*80}")

                result = runner.run_single_experiment(
                    dataset=dataset,
                    method=method,
                    random_state=run,
                    epochs=epochs,
                    learning_rate=learning_rate
                )

                all_results.append(result)

                # Save individual result
                result_file = runner.raw_dir / f"{dataset_name}_{method}_run{run}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)

                print(f"Result saved: {result_file}")

    # Save complete results
    complete_file = runner.output_dir / 'complete_results.json'
    with open(complete_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("REDUCED BENCHMARK COMPLETE")
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: {runner.output_dir}")
    print("\nNext steps:")
    print("1. Run: python analysis/statistical_analysis.py")
    print("2. Run: python analysis/visualizations.py")
    print("3. Integrate results into paper")
    print("="*80)

    # Print summary
    print("\nQuick Summary:")
    print("-" * 80)
    for dataset_name in datasets.keys():
        print(f"\n{dataset_name.upper()}:")
        for method in methods:
            method_results = [r for r in all_results 
                            if r['dataset'] == dataset_name and r['method'] == method 
                            and r.get('converged', False)]
            if method_results:
                metrics = method_results[0]['metrics']
                metric_str = ', '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
                print(f"  {method:20} -> {metric_str}")


if __name__ == "__main__":
    main()
