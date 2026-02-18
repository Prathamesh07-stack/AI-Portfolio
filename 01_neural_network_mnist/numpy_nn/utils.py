"""
Helper Utilities for NumPy Neural Network

This module contains activation functions, loss functions, and other utilities
needed for the from-scratch neural network implementation.

Key Concepts:
- Activation functions add non-linearity to the network
- Loss functions measure how wrong our predictions are
- These are the building blocks of neural networks
"""

import numpy as np


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

def relu(z):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Formula: f(x) = max(0, x)
    
    Why ReLU?
    - Simple and fast to compute
    - Helps with vanishing gradient problem
    - Introduces non-linearity (without it, network is just linear regression!)
    
    Args:
        z: Input values (any shape)
    
    Returns:
        Activated values (same shape as input)
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU for backpropagation.
    
    Formula: f'(x) = 1 if x > 0, else 0
    
    Why we need this:
    - During backpropagation, we need to know how the activation affects gradients
    - If input was negative (ReLU output 0), gradient doesn't flow back
    - If input was positive, gradient flows through unchanged
    
    Args:
        z: Input values (pre-activation)
    
    Returns:
        Derivative values (1 for positive, 0 for negative)
    """
    return (z > 0).astype(float)


def softmax(z):
    """
    Softmax activation for output layer.
    
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    Why Softmax?
    - Converts raw scores to probabilities (sum to 1)
    - Larger values get higher probabilities
    - Used for multi-class classification
    
    Example:
        Input: [2.0, 1.0, 0.1]
        Output: [0.659, 0.242, 0.099]  (probabilities that sum to 1)
    
    Args:
        z: Input logits (batch_size, num_classes)
    
    Returns:
        Probabilities (same shape, each row sums to 1)
    """
    # Numerical stability trick: subtract max to prevent overflow
    # This doesn't change the result but prevents exp() from exploding
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss for classification.
    
    Formula: L = -sum(y_true * log(y_pred))
    
    Why Cross-Entropy?
    - Measures difference between predicted and true probability distributions
    - Penalizes confident wrong predictions heavily
    - Works well with softmax output
    
    Example:
        True label: 7 (one-hot: [0,0,0,0,0,0,0,1,0,0])
        Prediction: [0.01, 0.01, ..., 0.90, ...]
        Loss: -log(0.90) = 0.105 (low loss, good!)
        
        If prediction was [0.01, 0.01, ..., 0.10, ...] (wrong!)
        Loss: -log(0.10) = 2.303 (high loss, bad!)
    
    Args:
        y_pred: Predicted probabilities (batch_size, num_classes)
        y_true: True labels as one-hot vectors (batch_size, num_classes)
    
    Returns:
        Average loss across the batch (scalar)
    """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross-entropy
    loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    return loss


def cross_entropy_loss_with_logits(logits, y_true):
    """
    Numerically stable cross-entropy that combines softmax + loss.
    
    This is more stable than computing softmax then loss separately.
    Used during training for better numerical precision.
    
    Args:
        logits: Raw network outputs (batch_size, num_classes)
        y_true: True labels as one-hot vectors (batch_size, num_classes)
    
    Returns:
        Average loss across the batch (scalar)
    """
    # Numerical stability
    logits_max = np.max(logits, axis=1, keepdims=True)
    logits_stable = logits - logits_max
    
    # Log-sum-exp trick
    log_sum_exp = np.log(np.sum(np.exp(logits_stable), axis=1, keepdims=True))
    
    # Cross-entropy
    loss = -np.sum(y_true * (logits_stable - log_sum_exp)) / logits.shape[0]
    return loss


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Example:
        Input: [7, 2, 1]
        Output: [[0,0,0,0,0,0,0,1,0,0],
                 [0,0,1,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0,0,0]]
    
    Args:
        labels: Integer labels (batch_size,)
        num_classes: Number of classes (10 for MNIST)
    
    Returns:
        One-hot encoded matrix (batch_size, num_classes)
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted probabilities (N, num_classes) or class indices (N,)
        labels: True labels as integers (N,)
    
    Returns:
        Accuracy as percentage (0-100)
    """
    # If predictions are probabilities, get the class with highest probability
    if len(predictions.shape) == 2:
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predicted_classes = predictions
    
    accuracy = np.mean(predicted_classes == labels) * 100
    return accuracy


def initialize_weights(input_size, output_size, method='he'):
    """
    Initialize weights for a layer.
    
    Why initialization matters:
    - Random initialization breaks symmetry (all neurons learn different features)
    - Proper scaling prevents vanishing/exploding gradients
    
    Methods:
    - 'he': He initialization for ReLU (sqrt(2/input_size))
    - 'xavier': Xavier initialization for tanh/sigmoid (sqrt(1/input_size))
    
    Args:
        input_size: Number of input neurons
        output_size: Number of output neurons
        method: Initialization method
    
    Returns:
        Weight matrix (input_size, output_size)
    """
    if method == 'he':
        # He initialization: good for ReLU activations
        std = np.sqrt(2.0 / input_size)
    elif method == 'xavier':
        # Xavier initialization: good for tanh/sigmoid
        std = np.sqrt(1.0 / input_size)
    else:
        std = 0.01
    
    weights = np.random.randn(input_size, output_size) * std
    return weights


def shuffle_data(x, y):
    """
    Shuffle data while maintaining x-y correspondence.
    
    Args:
        x: Input data (N, features)
        y: Labels (N,)
    
    Returns:
        Shuffled x and y
    """
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return x[indices], y[indices]


# Test utilities if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Test ReLU
    print("\n1. Testing ReLU:")
    test_input = np.array([-2, -1, 0, 1, 2])
    print(f"   Input: {test_input}")
    print(f"   ReLU output: {relu(test_input)}")
    print(f"   ReLU derivative: {relu_derivative(test_input)}")
    
    # Test Softmax
    print("\n2. Testing Softmax:")
    test_logits = np.array([[2.0, 1.0, 0.1]])
    print(f"   Input logits: {test_logits}")
    probs = softmax(test_logits)
    print(f"   Softmax output: {probs}")
    print(f"   Sum of probabilities: {np.sum(probs)}")
    
    # Test One-Hot Encoding
    print("\n3. Testing One-Hot Encoding:")
    test_labels = np.array([7, 2, 1])
    print(f"   Labels: {test_labels}")
    one_hot = one_hot_encode(test_labels)
    print(f"   One-hot shape: {one_hot.shape}")
    print(f"   One-hot:\n{one_hot}")
    
    # Test Accuracy
    print("\n4. Testing Accuracy:")
    predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    true_labels = np.array([1, 0, 1])
    acc = compute_accuracy(predictions, true_labels)
    print(f"   Predictions: {np.argmax(predictions, axis=1)}")
    print(f"   True labels: {true_labels}")
    print(f"   Accuracy: {acc:.2f}%")
    
    # Test Weight Initialization
    print("\n5. Testing Weight Initialization:")
    weights = initialize_weights(784, 128, method='he')
    print(f"   Weight shape: {weights.shape}")
    print(f"   Weight mean: {np.mean(weights):.6f}")
    print(f"   Weight std: {np.std(weights):.6f}")
    
    print("\n" + "=" * 60)
    print("✓ All utility tests passed!")
    print("=" * 60)
