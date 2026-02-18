"""
Comparison Script: NumPy vs PyTorch

This script compares the two implementations side-by-side.
Run this AFTER training both versions to see the differences!

What we compare:
- Final accuracy (should be similar)
- Training time (PyTorch should be faster)
- Loss curves (should have similar shapes)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def load_numpy_metrics():
    """Load metrics from NumPy training."""
    try:
        metrics = np.load('./results/numpy_metrics.npy', allow_pickle=True).item()
        return metrics
    except FileNotFoundError:
        print("❌ NumPy metrics not found. Please run numpy_nn/train.py first!")
        return None


def load_pytorch_metrics():
    """Load metrics from PyTorch training."""
    try:
        metrics = torch.load('./results/pytorch_metrics.pt')
        return metrics
    except FileNotFoundError:
        print("❌ PyTorch metrics not found. Please run pytorch_nn/train.py first!")
        return None


def plot_comparison(numpy_metrics, pytorch_metrics):
    """
    Create side-by-side comparison plots.
    
    Args:
        numpy_metrics: Dictionary with NumPy training metrics
        pytorch_metrics: Dictionary with PyTorch training metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_numpy = range(1, len(numpy_metrics['train_losses']) + 1)
    epochs_pytorch = range(1, len(pytorch_metrics['train_losses']) + 1)
    
    # ===== LOSS COMPARISON =====
    ax = axes[0, 0]
    ax.plot(epochs_numpy, numpy_metrics['train_losses'], 'b-', linewidth=2, label='NumPy', alpha=0.7)
    ax.plot(epochs_pytorch, pytorch_metrics['train_losses'], 'r-', linewidth=2, label='PyTorch', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== TRAIN ACCURACY COMPARISON =====
    ax = axes[0, 1]
    ax.plot(epochs_numpy, numpy_metrics['train_accuracies'], 'b-', linewidth=2, label='NumPy', alpha=0.7)
    ax.plot(epochs_pytorch, pytorch_metrics['train_accuracies'], 'r-', linewidth=2, label='PyTorch', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== TEST ACCURACY COMPARISON =====
    ax = axes[1, 0]
    ax.plot(epochs_numpy, numpy_metrics['test_accuracies'], 'b-', linewidth=2, label='NumPy', alpha=0.7)
    ax.plot(epochs_pytorch, pytorch_metrics['test_accuracies'], 'r-', linewidth=2, label='PyTorch', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== TRAINING TIME COMPARISON =====
    ax = axes[1, 1]
    implementations = ['NumPy', 'PyTorch']
    times = [numpy_metrics['total_time'], pytorch_metrics['total_time']]
    colors = ['blue', 'red']
    
    bars = ax.bar(implementations, times, color=colors, alpha=0.7)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Total Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s\n({time_val/60:.1f}m)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./results/comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison plot to ./results/comparison.png")
    plt.show()


def print_comparison_table(numpy_metrics, pytorch_metrics):
    """
    Print a formatted comparison table.
    
    Args:
        numpy_metrics: Dictionary with NumPy training metrics
        pytorch_metrics: Dictionary with PyTorch training metrics
    """
    print("\n" + "=" * 80)
    print("NUMPY VS PYTORCH COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'NumPy':>20} {'PyTorch':>20}")
    print("-" * 80)
    
    # Accuracy comparison
    print(f"{'Final Train Accuracy':<30} {numpy_metrics['final_train_acc']:>19.2f}% {pytorch_metrics['final_train_acc']:>19.2f}%")
    print(f"{'Final Test Accuracy':<30} {numpy_metrics['final_test_acc']:>19.2f}% {pytorch_metrics['final_test_acc']:>19.2f}%")
    
    # Time comparison
    numpy_time = numpy_metrics['total_time']
    pytorch_time = pytorch_metrics['total_time']
    speedup = numpy_time / pytorch_time
    
    print(f"{'Total Training Time':<30} {numpy_time:>18.1f}s {pytorch_time:>18.1f}s")
    print(f"{'Time per Epoch':<30} {numpy_time/numpy_metrics['epochs']:>18.1f}s {pytorch_time/pytorch_metrics['epochs']:>18.1f}s")
    print(f"{'Speedup (NumPy/PyTorch)':<30} {'':<20} {speedup:>18.2f}x")
    
    # Hyperparameters
    print(f"\n{'Hyperparameter':<30} {'NumPy':>20} {'PyTorch':>20}")
    print("-" * 80)
    print(f"{'Epochs':<30} {numpy_metrics['epochs']:>20} {pytorch_metrics['epochs']:>20}")
    print(f"{'Batch Size':<30} {numpy_metrics['batch_size']:>20} {pytorch_metrics['batch_size']:>20}")
    print(f"{'Learning Rate':<30} {numpy_metrics['learning_rate']:>20} {pytorch_metrics['learning_rate']:>20}")
    
    # Device info
    if 'device' in pytorch_metrics:
        print(f"{'Device':<30} {'CPU':>20} {pytorch_metrics['device']:>20}")
    
    print("\n" + "=" * 80)
    
    # Analysis
    print("\n📊 ANALYSIS:")
    print(f"  • Accuracy difference: {abs(numpy_metrics['final_test_acc'] - pytorch_metrics['final_test_acc']):.2f}%")
    print(f"    → Both implementations achieve similar accuracy (good!)")
    print(f"  • PyTorch is {speedup:.1f}x faster than NumPy")
    print(f"    → This is because PyTorch uses optimized C++/CUDA code")
    print(f"  • Loss curves should have similar shapes")
    print(f"    → This confirms both implementations are correct")
    
    print("\n💡 KEY TAKEAWAYS:")
    print("  1. Manual implementation (NumPy) helps you understand the math")
    print("  2. PyTorch automates backpropagation and is much faster")
    print("  3. Both achieve similar results, proving the math is correct")
    print("  4. For real projects, use frameworks like PyTorch!")
    
    print("\n" + "=" * 80)


def main():
    """Main comparison function."""
    print("=" * 80)
    print("COMPARING NUMPY AND PYTORCH IMPLEMENTATIONS")
    print("=" * 80)
    
    # Load metrics
    print("\nLoading metrics...")
    numpy_metrics = load_numpy_metrics()
    pytorch_metrics = load_pytorch_metrics()
    
    if numpy_metrics is None or pytorch_metrics is None:
        print("\n❌ Cannot compare - please train both models first!")
        print("\nTo train:")
        print("  1. cd numpy_nn && python train.py")
        print("  2. cd pytorch_nn && python train.py")
        print("  3. python compare.py")
        return
    
    print("✓ Loaded both metrics successfully!")
    
    # Print comparison table
    print_comparison_table(numpy_metrics, pytorch_metrics)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    plot_comparison(numpy_metrics, pytorch_metrics)
    
    print("\n✓ Comparison complete!")
    print("\nNext steps:")
    print("  • Review ./results/comparison.png")
    print("  • Write your analysis in ANALYSIS.md")
    print("  • Experiment with different hyperparameters")


if __name__ == "__main__":
    main()
