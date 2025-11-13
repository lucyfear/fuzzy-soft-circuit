"""
Generate example figures for paper using existing XOR experiment.

This demonstrates the visualization capabilities and provides publication-ready
figures based on the validated XOR results from the paper.
"""

import sys
sys.path.insert(0, '/home/spinoza/github/beta/soft-circuit')

import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path

# Configure matplotlib for publication quality
rc('font', family='serif', size=10)
rc('text', usetex=False)  # Set to True if LaTeX is available
rc('figure', dpi=300)

# Import fuzzy soft circuit
from fuzzy_soft_circuit.fuzzy_core import FuzzySoftCircuit, flatten_params, unflatten_params
from autograd import grad


def train_xor_example():
    """Train fuzzy soft circuit on XOR problem from paper."""
    print("Training Fuzzy Soft Circuit on XOR problem...")
    
    # Create training data (from paper)
    X_train = np.array([
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.5, 0.5],
        [0.2, 0.2],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.8, 0.8],
        [0.3, 0.3],
        [0.3, 0.7],
        [0.7, 0.3],
        [0.7, 0.7],
        [0.4, 0.6],
        [0.6, 0.4],
    ])
    
    y_train = np.array([
        [0.2], [0.8], [0.8], [0.2], [0.5],
        [0.2], [0.7], [0.7], [0.2], [0.3],
        [0.6], [0.6], [0.3], [0.5], [0.5]
    ])
    
    # Create test data
    X_test = np.array([
        [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
        [0.5, 0.5], [0.25, 0.25], [0.25, 0.75],
        [0.75, 0.25], [0.75, 0.75], [0.15, 0.85],
        [0.85, 0.15], [0.5, 0.2], [0.2, 0.5],
        [0.5, 0.8], [0.8, 0.5], [0.3, 0.3],
        [0.3, 0.7], [0.7, 0.3], [0.7, 0.7], [0.6, 0.4]
    ])
    
    y_test = np.array([
        [0.2], [0.8], [0.8], [0.2], [0.5],
        [0.2], [0.7], [0.7], [0.2], [0.8],
        [0.8], [0.4], [0.4], [0.6], [0.6],
        [0.3], [0.6], [0.6], [0.3], [0.5]
    ])
    
    # Initialize circuit
    circuit = FuzzySoftCircuit(
        n_inputs=2,
        n_outputs=1,
        n_memberships=3,
        n_rules=6
    )
    
    # Prepare data
    data = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    
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
    epochs = 600
    learning_rate = 0.5
    history = []
    
    for epoch in range(epochs):
        current_loss = loss(flat_params)
        history.append(float(current_loss))
        
        g = grad_fn(flat_params)
        flat_params -= learning_rate * g
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}, Loss: {current_loss:.6f}")
    
    # Update circuit parameters
    circuit.params = unflatten_params(flat_params, shapes)
    
    # Evaluate on test set
    test_loss = 0
    predictions = []
    for i in range(len(X_test)):
        pred = circuit.forward(X_test[i])
        predictions.append(pred)
        test_loss += (pred - y_test[i]) ** 2
    
    test_mse = float(test_loss / len(X_test))
    print(f"  Final train loss: {history[-1]:.6f}")
    print(f"  Test MSE: {test_mse:.6f}")
    
    return circuit, history, X_train, y_train, X_test, y_test, np.array(predictions)


def plot_membership_functions(circuit, output_dir):
    """Plot learned membership functions."""
    print("\nGenerating membership function plots...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x_range = np.linspace(0, 1, 200)

    # input_mf has shape (n_inputs, n_memberships, 2) where last dim is [center, width]
    for input_idx in range(2):
        ax = axes[input_idx]

        for j in range(circuit.n_memberships):
            center = float(circuit.params['input_mf'][input_idx, j, 0])
            width = float(circuit.params['input_mf'][input_idx, j, 1])

            membership_vals = np.exp(-((x_range - center) / width) ** 2)

            ax.plot(x_range, membership_vals, linewidth=2,
                   label=f'MF{j} (c={center:.2f})')

        ax.set_xlabel(f'Input x{input_idx}', fontsize=11)
        ax.set_ylabel('Membership degree', fontsize=11)
        ax.set_title(f'Input Variable x{input_idx}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save in multiple formats
    fig.savefig(output_dir / 'learned_memberships.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'learned_memberships.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_dir / 'learned_memberships.pdf'}")
    print(f"  Saved: {output_dir / 'learned_memberships.png'}")

    plt.close()


def plot_learning_curve(history, output_dir):
    """Plot learning curve."""
    print("\nGenerating learning curve plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    epochs = np.arange(len(history))
    ax.plot(epochs, history, linewidth=2, color='#2E86AB', label='Training Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Training Convergence on XOR Problem', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'learning_curve.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'learning_curve.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_dir / 'learning_curve.pdf'}")
    print(f"  Saved: {output_dir / 'learning_curve.png'}")
    
    plt.close()


def plot_decision_surface(circuit, X_train, y_train, output_dir):
    """Plot decision surface learned by the model."""
    print("\nGenerating decision surface plot...")
    
    # Create meshgrid
    x0 = np.linspace(0, 1, 100)
    x1 = np.linspace(0, 1, 100)
    X0, X1 = np.meshgrid(x0, x1)
    
    # Predict on grid
    Z = np.zeros_like(X0)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            inputs = np.array([X0[i, j], X1[i, j]])
            Z[i, j] = circuit.forward(inputs)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Contour plot
    contour = ax.contourf(X0, X1, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Output Value', fontsize=11)
    
    # Overlay training points
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                        cmap='RdYlBu_r', s=100, edgecolors='black', 
                        linewidth=1.5, zorder=5)
    
    ax.set_xlabel('Input x0', fontsize=12)
    ax.set_ylabel('Input x1', fontsize=12)
    ax.set_title('Learned Decision Surface (XOR-like Function)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'decision_surface.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'decision_surface.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_dir / 'decision_surface.pdf'}")
    print(f"  Saved: {output_dir / 'decision_surface.png'}")
    
    plt.close()


def plot_rule_activations(circuit, output_dir, threshold=0.3):
    """Visualize active rules."""
    print("\nGenerating rule activation plot...")
    
    # Compute rule activations
    rule_switches = []
    for r in range(circuit.n_rules):
        switch_val = 1 / (1 + np.exp(-circuit.params['rule_switches'][r]))
        rule_switches.append(float(switch_val))
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    rule_indices = np.arange(circuit.n_rules)
    colors = ['#2E86AB' if val > threshold else '#CCCCCC' for val in rule_switches]
    
    bars = ax.bar(rule_indices, rule_switches, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Rule Index', fontsize=12)
    ax.set_ylabel('Rule Activation (Switch Value)', fontsize=12)
    ax.set_title('Learned Rule Activations', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate active rules
    for i, val in enumerate(rule_switches):
        if val > threshold:
            ax.text(i, val + 0.03, f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'rule_activations.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'rule_activations.png', bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_dir / 'rule_activations.pdf'}")
    print(f"  Saved: {output_dir / 'rule_activations.png'}")
    
    plt.close()
    
    # Count active rules
    active_rules = sum(1 for val in rule_switches if val > threshold)
    print(f"  Active rules (>{threshold}): {active_rules}/{circuit.n_rules}")


def main():
    """Generate all example figures."""
    print("="*80)
    print("GENERATING EXAMPLE FIGURES FROM XOR EXPERIMENT")
    print("="*80)
    
    output_dir = Path('/home/spinoza/github/beta/soft-circuit/benchmarks/results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    circuit, history, X_train, y_train, X_test, y_test, predictions = train_xor_example()
    
    # Generate all plots
    plot_membership_functions(circuit, output_dir)
    plot_learning_curve(history, output_dir)
    plot_decision_surface(circuit, X_train, y_train, output_dir)
    plot_rule_activations(circuit, output_dir)
    
    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETE")
    print(f"All figures saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - learned_memberships.pdf/.png")
    print("  - learning_curve.pdf/.png")
    print("  - decision_surface.pdf/.png")
    print("  - rule_activations.pdf/.png")
    print("\nThese figures are publication-ready (300 DPI) and can be")
    print("included directly in the paper.")
    print("="*80)


if __name__ == "__main__":
    main()
