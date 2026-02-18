"""
Data Loading Utilities for MNIST Dataset

This module handles downloading, preprocessing, and loading the MNIST dataset
for both NumPy (from-scratch) and PyTorch implementations.

MNIST Dataset:
- 60,000 training images of handwritten digits (0-9)
- 10,000 test images
- Each image is 28x28 pixels (grayscale)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def load_mnist_pytorch(batch_size=64, data_dir='./data'):
    """
    Load MNIST dataset for PyTorch training.
    
    Args:
        batch_size (int): Number of samples per batch
        data_dir (str): Directory to store/load MNIST data
    
    Returns:
        train_loader: PyTorch DataLoader for training set
        test_loader: PyTorch DataLoader for test set
    """
    # Define transformation: Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to tensor and scales to [0, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders for batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=0
    )
    
    print(f"✓ Loaded MNIST dataset:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, test_loader


def load_mnist_numpy(data_dir='./data'):
    """
    Load MNIST dataset as NumPy arrays for from-scratch implementation.
    
    Args:
        data_dir (str): Directory to store/load MNIST data
    
    Returns:
        x_train: Training images (60000, 784) - normalized to [0, 1]
        y_train: Training labels (60000,) - integers 0-9
        x_test: Test images (10000, 784) - normalized to [0, 1]
        y_test: Test labels (10000,) - integers 0-9
    """
    # First, download using torchvision (easiest way)
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Convert to NumPy arrays
    # Training data
    x_train = train_dataset.data.numpy().astype(np.float32)  # (60000, 28, 28)
    x_train = x_train.reshape(-1, 784)  # Flatten to (60000, 784)
    x_train = x_train / 255.0  # Normalize to [0, 1]
    
    y_train = train_dataset.targets.numpy()  # (60000,)
    
    # Test data
    x_test = test_dataset.data.numpy().astype(np.float32)  # (10000, 28, 28)
    x_test = x_test.reshape(-1, 784)  # Flatten to (10000, 784)
    x_test = x_test / 255.0  # Normalize to [0, 1]
    
    y_test = test_dataset.targets.numpy()  # (10000,)
    
    print(f"✓ Loaded MNIST as NumPy arrays:")
    print(f"  x_train shape: {x_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Pixel value range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    
    return x_train, y_train, x_test, y_test


def visualize_samples(x_data, y_data, num_samples=10, title="MNIST Samples"):
    """
    Visualize random samples from the dataset.
    
    Args:
        x_data: Image data (N, 784) or (N, 28, 28)
        y_data: Labels (N,)
        num_samples: Number of samples to display
        title: Plot title
    """
    # Reshape if flattened
    if len(x_data.shape) == 2:
        images = x_data.reshape(-1, 28, 28)
    else:
        images = x_data
    
    # Select random samples
    indices = np.random.choice(len(x_data), num_samples, replace=False)
    
    # Create plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx], cmap='gray')
        axes[i].set_title(f'Label: {y_data[idx]}')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig('./results/mnist_samples.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved sample visualization to ./results/mnist_samples.png")
    plt.show()


def get_batch_iterator(x, y, batch_size=64, shuffle=True):
    """
    Create a batch iterator for NumPy arrays (manual DataLoader).
    
    Args:
        x: Input data (N, 784)
        y: Labels (N,)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
    
    Yields:
        (batch_x, batch_y): Batches of data
    """
    n_samples = len(x)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield x[batch_indices], y[batch_indices]


# Test the data loader if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Loader")
    print("=" * 60)
    
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    
    # Test NumPy loading
    print("\n1. Loading NumPy arrays...")
    x_train, y_train, x_test, y_test = load_mnist_numpy()
    
    # Test PyTorch loading
    print("\n2. Loading PyTorch DataLoaders...")
    train_loader, test_loader = load_mnist_pytorch(batch_size=64)
    
    # Visualize samples
    print("\n3. Visualizing samples...")
    visualize_samples(x_train, y_train, num_samples=10)
    
    # Test batch iterator
    print("\n4. Testing batch iterator...")
    batch_count = 0
    for batch_x, batch_y in get_batch_iterator(x_train, y_train, batch_size=64):
        batch_count += 1
        if batch_count == 1:
            print(f"  First batch shape: {batch_x.shape}, {batch_y.shape}")
    print(f"  Total batches: {batch_count}")
    
    print("\n" + "=" * 60)
    print("✓ All data loading tests passed!")
    print("=" * 60)
