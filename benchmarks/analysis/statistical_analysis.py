"""
Statistical analysis of benchmark results.

Computes:
- Mean and standard deviation across runs
- 95% confidence intervals
- Paired t-tests for significance
- Cohen's d effect sizes
- Summary tables for paper
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats
import pandas as pd


class BenchmarkAnalyzer:
    """Analyze benchmark results with statistical rigor."""

    def __init__(self, results_dir='/home/spinoza/github/beta/soft-circuit/benchmarks/results'):
        self.results_dir = Path(results_dir)
        self.load_results()

    def load_results(self):
        """Load all experimental results."""
        results_file = self.results_dir / 'complete_results.json'

        if not results_file.exists():
            raise FileNotFoundError(f"Results not found: {results_file}")

        with open(results_file, 'r') as f:
            self.all_results = json.load(f)

        print(f"Loaded {len(self.all_results)} experimental results")

    def extract_metrics(self, dataset_name, method, metric_name):
        """Extract a specific metric across all runs."""
        values = []

        for result in self.all_results:
            if (result['dataset'] == dataset_name and
                result['method'] == method and
                result.get('converged', False)):

                metric_value = result['metrics'].get(metric_name)
                if metric_value is not None:
                    values.append(metric_value)

        return np.array(values)

    def compute_statistics(self, values):
        """Compute mean, std, and 95% CI."""
        if len(values) == 0:
            return None

        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample std
        n = len(values)

        # 95% confidence interval using t-distribution
        ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))

        return {
            'mean': mean,
            'std': std,
            'ci_lower': ci_95[0],
            'ci_upper': ci_95[1],
            'n': n
        }

    def paired_t_test(self, values1, values2):
        """Perform paired t-test between two methods."""
        if len(values1) != len(values2) or len(values1) == 0:
            return None

        t_stat, p_value = stats.ttest_rel(values1, values2)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    def cohens_d(self, values1, values2):
        """Compute Cohen's d effect size."""
        if len(values1) == 0 or len(values2) == 0:
            return None

        mean1, mean2 = np.mean(values1), np.mean(values2)
        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)

        # Pooled standard deviation
        n1, n2 = len(values1), len(values2)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))

        d = (mean1 - mean2) / pooled_std

        # Interpret effect size
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            'd': d,
            'interpretation': interpretation
        }

    def create_summary_table(self, primary_metric='mse'):
        """Create summary table of all results."""
        datasets = list(set(r['dataset'] for r in self.all_results))
        methods = ['FuzzySoftCircuit', 'ANFIS', 'MLP']

        summary_data = []

        for dataset in sorted(datasets):
            # Determine primary metric based on task type
            task_type = None
            for r in self.all_results:
                if r['dataset'] == dataset:
                    task_type = r['task_type']
                    break

            if task_type == 'classification':
                metric = 'accuracy'
            else:
                metric = primary_metric

            row = {'Dataset': dataset}

            for method in methods:
                values = self.extract_metrics(dataset, method, metric)
                stats_dict = self.compute_statistics(values)

                if stats_dict:
                    # Format: mean ± std (CI)
                    formatted = (f"{stats_dict['mean']:.4f} ± {stats_dict['std']:.4f} "
                               f"[{stats_dict['ci_lower']:.4f}, {stats_dict['ci_upper']:.4f}]")
                    row[method] = formatted
                else:
                    row[method] = "N/A"

            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        return df

    def create_comparison_table(self):
        """Create table comparing methods with statistical tests."""
        datasets = list(set(r['dataset'] for r in self.all_results))

        comparison_data = []

        for dataset in sorted(datasets):
            # Determine metric
            task_type = None
            for r in self.all_results:
                if r['dataset'] == dataset:
                    task_type = r['task_type']
                    break

            metric = 'accuracy' if task_type == 'classification' else 'mse'

            # Extract values
            fsc_values = self.extract_metrics(dataset, 'FuzzySoftCircuit', metric)
            anfis_values = self.extract_metrics(dataset, 'ANFIS', metric)
            mlp_values = self.extract_metrics(dataset, 'MLP', metric)

            if len(fsc_values) == 0 or len(anfis_values) == 0:
                continue

            # Compute statistics
            fsc_stats = self.compute_statistics(fsc_values)
            anfis_stats = self.compute_statistics(anfis_values)

            # Statistical tests: FSC vs ANFIS
            t_test = self.paired_t_test(fsc_values, anfis_values)
            effect_size = self.cohens_d(fsc_values, anfis_values)

            row = {
                'Dataset': dataset,
                'Metric': metric.upper(),
                'FSC_mean': fsc_stats['mean'],
                'ANFIS_mean': anfis_stats['mean'],
                'Difference': fsc_stats['mean'] - anfis_stats['mean'],
                'p_value': t_test['p_value'] if t_test else None,
                'Significant': t_test['significant'] if t_test else None,
                'Cohens_d': effect_size['d'] if effect_size else None,
                'Effect_size': effect_size['interpretation'] if effect_size else None
            }

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df

    def create_rule_count_table(self):
        """Compare number of rules discovered/used by fuzzy methods."""
        datasets = list(set(r['dataset'] for r in self.all_results))

        rule_data = []

        for dataset in sorted(datasets):
            row = {'Dataset': dataset}

            for method in ['FuzzySoftCircuit', 'ANFIS']:
                rules = []
                for result in self.all_results:
                    if (result['dataset'] == dataset and
                        result['method'] == method and
                        result.get('converged', False)):
                        n_rules = result.get('n_rules')
                        if n_rules is not None:
                            rules.append(n_rules)

                if rules:
                    mean_rules = np.mean(rules)
                    row[f'{method}_rules'] = f"{mean_rules:.1f}"
                else:
                    row[f'{method}_rules'] = "N/A"

            rule_data.append(row)

        df = pd.DataFrame(rule_data)
        return df

    def generate_latex_table(self, df, caption, label):
        """Convert DataFrame to LaTeX table format."""
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{" + caption + "}\n"
        latex += "\\label{" + label + "}\n"

        # Table format
        n_cols = len(df.columns)
        latex += "\\begin{tabular}{" + "l" + "c" * (n_cols - 1) + "}\n"
        latex += "\\hline\n"

        # Header
        latex += " & ".join(df.columns) + " \\\\\n"
        latex += "\\hline\n"

        # Data rows
        for _, row in df.iterrows():
            latex += " & ".join(str(val) for val in row) + " \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex

    def generate_all_tables(self):
        """Generate all summary tables and LaTeX code."""
        output_dir = self.results_dir / 'summary'
        output_dir.mkdir(exist_ok=True)

        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("="*80)

        # Summary table
        print("\n1. Summary of Results (Mean ± Std [95% CI])")
        print("-" * 80)
        summary_df = self.create_summary_table()
        print(summary_df.to_string(index=False))

        summary_df.to_csv(output_dir / 'summary_table.csv', index=False)

        latex_summary = self.generate_latex_table(
            summary_df,
            caption="Performance comparison across benchmark datasets. "
                   "Values show mean ± standard deviation with 95\\% confidence intervals "
                   "from 10 independent runs. MSE reported for regression tasks, "
                   "accuracy for classification.",
            label="tab:summary"
        )

        with open(output_dir / 'summary_table.tex', 'w') as f:
            f.write(latex_summary)

        # Comparison table
        print("\n2. Statistical Comparison: FuzzySoftCircuit vs ANFIS")
        print("-" * 80)
        comparison_df = self.create_comparison_table()
        print(comparison_df.to_string(index=False))

        comparison_df.to_csv(output_dir / 'comparison_table.csv', index=False)

        latex_comparison = self.generate_latex_table(
            comparison_df,
            caption="Statistical comparison between Fuzzy Soft Circuits and ANFIS. "
                   "Paired t-tests with Cohen's d effect sizes. "
                   "p < 0.05 indicates statistically significant difference.",
            label="tab:comparison"
        )

        with open(output_dir / 'comparison_table.tex', 'w') as f:
            f.write(latex_comparison)

        # Rule count table
        print("\n3. Number of Rules Used")
        print("-" * 80)
        rules_df = self.create_rule_count_table()
        print(rules_df.to_string(index=False))

        rules_df.to_csv(output_dir / 'rules_table.csv', index=False)

        latex_rules = self.generate_latex_table(
            rules_df,
            caption="Average number of active rules discovered (Fuzzy Soft Circuits) "
                   "vs. predefined (ANFIS). Lower rule counts indicate more compact "
                   "and interpretable models.",
            label="tab:rules"
        )

        with open(output_dir / 'rules_table.tex', 'w') as f:
            f.write(latex_rules)

        print("\n" + "="*80)
        print(f"Tables saved to: {output_dir}")
        print("="*80)

        return summary_df, comparison_df, rules_df


def main():
    """Run statistical analysis on benchmark results."""
    analyzer = BenchmarkAnalyzer()
    analyzer.generate_all_tables()


if __name__ == "__main__":
    main()
