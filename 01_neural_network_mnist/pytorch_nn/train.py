"""
Training Script for PyTorch Neural Network

This script trains the PyTorch version of our neural network.
Compare this with the NumPy training script to see what PyTorch automates!

Key Differences from NumPy Version:
- loss.backward() does ALL the backpropagation automatically
- optimizer.step() updates weights automatically
- DataLoader handles batching
- Much faster (especially on GPU)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_mnist_pytorch
from model import NeuralNetworkPyTorch


def plot_training_history(train_losses, train_accs, test_accs, save_path='./results/pytorch_training.png'):
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
    ax1.set_title('Training Loss Over Time (PyTorch)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, train_accs, 'g-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, test_accs, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Over Time (PyTorch)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training plot to {save_path}")
    plt.close()


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        criterion: Loss function
        device: Device to run on (CPU or GPU)
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy percentage
    """
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't compute gradients during evaluation
        for x_batch, y_batch in data_loader:
            # Move to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_network(epochs=20, batch_size=64, learning_rate=0.01):
    """
    Main training function.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
    
    Returns:
        model: Trained model
        history: Training metrics
    """
    print("=" * 70)
    print("TRAINING NEURAL NETWORK WITH PYTORCH")
    print("=" * 70)
    
    # ===== SETUP DEVICE =====
    # PyTorch can use GPU automatically if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: Training on CPU (no GPU detected)")
    
    # ===== LOAD DATA =====
    print("\n[1/4] Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_pytorch(batch_size=batch_size)
    
    # ===== CREATE MODEL =====
    print("\n[2/4] Creating model...")
    model = NeuralNetworkPyTorch(input_size=784, hidden1_size=128, hidden2_size=64, output_size=10)
    model = model.to(device)  # Move model to GPU if available
    
    # ===== SETUP TRAINING =====
    print("\n[3/4] Setting up training...")
    
    # Loss function
    # CrossEntropyLoss combines softmax + cross-entropy (numerically stable)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    # SGD (Stochastic Gradient Descent) - same as our manual implementation
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    print(f"  Loss function: CrossEntropyLoss")
    print(f"  Optimizer: SGD (lr={learning_rate})")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # ===== TRAINING LOOP =====
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()  # Set to training mode
        
        batch_losses = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        
        for x_batch, y_batch in pbar:
            # Move data to device (GPU if available)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # ===== FORWARD PASS =====
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # ===== BACKWARD PASS =====
            # This is where PyTorch's magic happens!
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients (automatic backprop!)
            optimizer.step()       # Update weights (automatic gradient descent!)
            
            # Track loss
            batch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate after each epoch
        avg_loss = sum(batch_losses) / len(batch_losses)
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
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
    
    # Save model
    torch.save(model.state_dict(), './results/pytorch_model.pth')
    print("✓ Saved model to ./results/pytorch_model.pth")
    
    # Save metrics
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'final_train_acc': train_accuracies[-1],
        'final_test_acc': test_accuracies[-1],
        'total_time': total_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device)
    }
    
    torch.save(metrics, './results/pytorch_metrics.pt')
    print("✓ Saved metrics to ./results/pytorch_metrics.pt")
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    
    return model, metrics


if __name__ == "__main__":
    # Hyperparameters (reduced epochs for faster CPU training)
    EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    
    # Train the network
    model, history = train_network(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    print("\n📊 You can now:")
    print("  1. Check ../results/pytorch_training.png for visualizations")
    print("  2. Run compare.py to see NumPy vs PyTorch comparison")
    print("  3. Load the saved model for inference")
