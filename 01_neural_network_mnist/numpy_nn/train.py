"""
Training Script for NumPy Neural Network

This script trains the from-scratch neural network on MNIST.
You'll see the network learn in real-time as loss decreases and accuracy increases!

Training Process:
1. Load MNIST data
2. Initialize neural network
3. For each epoch:
   - Process data in mini-batches
   - Forward pass → compute loss → backward pass → update weights
   - Track metrics (loss, accuracy)
4. Save results and visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import os

# Add parent directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_mnist_numpy, get_batch_iterator
from neural_network import NeuralNetwork
from utils import compute_accuracy


def plot_training_history(train_losses, train_accs, test_accs, save_path='../results/numpy_training.png'):
    """
    Visualize training progress.
    
    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_accs: List of test accuracies per epoch
        save_path: Where to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, train_accs, 'g-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, test_accs, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training plot to {save_path}")
    plt.close()


def train_network(epochs=20, batch_size=64, learning_rate=0.01):
    """
    Main training function.
    
    Args:
        epochs: Number of times to iterate through the entire dataset
        batch_size: Number of samples per batch
        learning_rate: Step size for gradient descent
    
    Returns:
        nn: Trained neural network
        history: Dictionary with training metrics
    """
    print("=" * 70)
    print("TRAINING NEURAL NETWORK FROM SCRATCH (NumPy)")
    print("=" * 70)
    
    # ===== LOAD DATA =====
    print("\n[1/4] Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist_numpy()
    
    # ===== INITIALIZE NETWORK =====
    print("\n[2/4] Initializing neural network...")
    nn = NeuralNetwork(input_size=784, hidden1_size=128, hidden2_size=64, output_size=10)
    
    # ===== TRAINING LOOP =====
    print(f"\n[3/4] Training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batches per epoch: {len(x_train) // batch_size}")
    print()
    
    # Metrics tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Shuffle training data each epoch
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        batch_losses = []
        
        # Progress bar for batches
        num_batches = len(x_train) // batch_size
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        
        for batch_idx in pbar:
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Training step: forward → loss → backward → update
            loss = nn.train_step(x_batch, y_batch, learning_rate)
            batch_losses.append(loss)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Compute epoch metrics
        avg_loss = np.mean(batch_losses)
        train_acc = nn.evaluate(x_train, y_train)
        test_acc = nn.evaluate(x_test, y_test)
        
        # Store metrics
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"  Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # ===== FINAL RESULTS =====
    print(f"\n[4/4] Training complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Average time per epoch: {total_time/epochs:.1f}s")
    print(f"  Final train accuracy: {train_accuracies[-1]:.2f}%")
    print(f"  Final test accuracy: {test_accuracies[-1]:.2f}%")
    
    # Check for overfitting
    overfit_gap = train_accuracies[-1] - test_accuracies[-1]
    if overfit_gap > 5:
        print(f"  ⚠ Warning: Possible overfitting (gap: {overfit_gap:.2f}%)")
    else:
        print(f"  ✓ Good generalization (gap: {overfit_gap:.2f}%)")
    
    # ===== SAVE RESULTS =====
    print("\nSaving results...")
    os.makedirs('./results', exist_ok=True)
    
    # Save plot
    plot_training_history(train_losses, train_accuracies, test_accuracies)
    
    # Save metrics to file
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'final_train_acc': train_accuracies[-1],
        'final_test_acc': test_accuracies[-1],
        'total_time': total_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    
    np.save('./results/numpy_metrics.npy', metrics)
    print("✓ Saved metrics to ./results/numpy_metrics.npy")
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    
    return nn, metrics


if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    
    # Train the network
    network, history = train_network(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    print("\n📊 You can now:")
    print("  1. Check ../results/numpy_training.png for visualizations")
    print("  2. Run the PyTorch version to compare")
    print("  3. Experiment with different hyperparameters")
