"""
Neural Network Implementation from Scratch (NumPy Only)

This is the CORE of the project - a complete neural network built without
any deep learning frameworks. Every operation is explicit so you can see
exactly how neural networks work.

Architecture: 784 → 128 → 64 → 10
- Input: 784 neurons (28x28 flattened image)
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 64 neurons with ReLU activation  
- Output: 10 neurons with Softmax (probabilities for digits 0-9)

Key Concepts Implemented:
1. Forward Propagation: How data flows through the network
2. Backpropagation: How gradients flow backward to update weights
3. Gradient Descent: How we use gradients to improve the model
"""

import numpy as np
from utils import (
    relu, relu_derivative, softmax,
    cross_entropy_loss, one_hot_encode,
    compute_accuracy, initialize_weights
)


class NeuralNetwork:
    """
    A 3-layer neural network for MNIST digit classification.
    
    This class implements everything from scratch:
    - Weight initialization
    - Forward propagation
    - Loss computation
    - Backpropagation (gradient computation)
    - Parameter updates (learning)
    """
    
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        """
        Initialize the neural network with random weights.
        
        Args:
            input_size: Number of input features (784 for MNIST)
            hidden1_size: Number of neurons in first hidden layer
            hidden2_size: Number of neurons in second hidden layer
            output_size: Number of output classes (10 for digits 0-9)
        """
        print("Initializing Neural Network...")
        print(f"  Architecture: {input_size} → {hidden1_size} → {hidden2_size} → {output_size}")
        
        # Initialize weights using He initialization (good for ReLU)
        # Why He initialization? It prevents gradients from vanishing or exploding
        self.W1 = initialize_weights(input_size, hidden1_size, method='he')
        self.b1 = np.zeros((1, hidden1_size))  # Biases start at zero
        
        self.W2 = initialize_weights(hidden1_size, hidden2_size, method='he')
        self.b2 = np.zeros((1, hidden2_size))
        
        self.W3 = initialize_weights(hidden2_size, output_size, method='he')
        self.b3 = np.zeros((1, output_size))
        
        # Cache for storing intermediate values during forward pass
        # We need these for backpropagation!
        self.cache = {}
        
        print(f"  Total parameters: {self._count_parameters():,}")
        print("✓ Network initialized!\n")
    
    def _count_parameters(self):
        """Count total number of trainable parameters."""
        return (self.W1.size + self.b1.size + 
                self.W2.size + self.b2.size + 
                self.W3.size + self.b3.size)
    
    def forward(self, X):
        """
        Forward propagation: compute predictions from inputs.
        
        This is how the network makes predictions:
        1. Multiply inputs by weights, add bias
        2. Apply activation function (ReLU)
        3. Repeat for each layer
        4. Final layer uses Softmax to get probabilities
        
        Args:
            X: Input data (batch_size, 784)
        
        Returns:
            predictions: Probabilities for each class (batch_size, 10)
        """
        # ===== LAYER 1: Input → Hidden Layer 1 =====
        # Linear transformation: z = X @ W + b
        # Think of this as: each neuron computes a weighted sum of inputs
        self.cache['X'] = X
        self.cache['z1'] = X @ self.W1 + self.b1  # (batch, 784) @ (784, 128) = (batch, 128)
        
        # Activation: a = ReLU(z)
        # ReLU adds non-linearity - without it, the network is just linear regression!
        self.cache['a1'] = relu(self.cache['z1'])  # (batch, 128)
        
        # ===== LAYER 2: Hidden Layer 1 → Hidden Layer 2 =====
        self.cache['z2'] = self.cache['a1'] @ self.W2 + self.b2  # (batch, 128) @ (128, 64) = (batch, 64)
        self.cache['a2'] = relu(self.cache['z2'])  # (batch, 64)
        
        # ===== LAYER 3: Hidden Layer 2 → Output =====
        self.cache['z3'] = self.cache['a2'] @ self.W3 + self.b3  # (batch, 64) @ (64, 10) = (batch, 10)
        
        # Softmax: convert raw scores to probabilities
        # Output: 10 probabilities that sum to 1
        self.cache['predictions'] = softmax(self.cache['z3'])  # (batch, 10)
        
        return self.cache['predictions']
    
    def compute_loss(self, y_true):
        """
        Compute cross-entropy loss.
        
        Loss measures how wrong our predictions are:
        - Low loss = good predictions
        - High loss = bad predictions
        
        Args:
            y_true: True labels (batch_size,) as integers
        
        Returns:
            loss: Average cross-entropy loss (scalar)
        """
        # Convert integer labels to one-hot encoding
        # Example: label 7 → [0,0,0,0,0,0,0,1,0,0]
        y_true_one_hot = one_hot_encode(y_true, num_classes=10)
        
        # Compute cross-entropy loss
        loss = cross_entropy_loss(self.cache['predictions'], y_true_one_hot)
        
        return loss
    
    def backward(self, y_true, learning_rate=0.01):
        """
        Backpropagation: compute gradients and update weights.
        
        This is THE KEY ALGORITHM in neural networks!
        
        How it works:
        1. Start from the output layer
        2. Compute how much each weight contributed to the error (gradient)
        3. Work backwards through the network using the chain rule
        4. Update weights in the opposite direction of the gradient
        
        Why it works:
        - Gradients point in the direction of increasing loss
        - Moving opposite to gradients decreases loss
        - This is gradient descent!
        
        Args:
            y_true: True labels (batch_size,)
            learning_rate: How big of a step to take (hyperparameter)
        """
        batch_size = y_true.shape[0]
        y_true_one_hot = one_hot_encode(y_true, num_classes=10)
        
        # ===== OUTPUT LAYER GRADIENT =====
        # For softmax + cross-entropy, the gradient is simply: (predicted - true)
        # This is a beautiful mathematical result from calculus!
        dz3 = self.cache['predictions'] - y_true_one_hot  # (batch, 10)
        
        # Gradient for W3 and b3
        # Chain rule: dL/dW3 = a2^T @ dL/dz3
        # Intuition: how much does changing W3 affect the loss?
        dW3 = (self.cache['a2'].T @ dz3) / batch_size  # (64, 10)
        db3 = np.sum(dz3, axis=0, keepdims=True) / batch_size  # (1, 10)
        
        # ===== HIDDEN LAYER 2 GRADIENT =====
        # Propagate gradient backwards through W3
        # Chain rule: dL/da2 = dL/dz3 @ W3^T
        da2 = dz3 @ self.W3.T  # (batch, 64)
        
        # Apply ReLU derivative
        # Gradient only flows through neurons that were active (z2 > 0)
        dz2 = da2 * relu_derivative(self.cache['z2'])  # (batch, 64)
        
        # Gradient for W2 and b2
        dW2 = (self.cache['a1'].T @ dz2) / batch_size  # (128, 64)
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size  # (1, 64)
        
        # ===== HIDDEN LAYER 1 GRADIENT =====
        # Propagate gradient backwards through W2
        da1 = dz2 @ self.W2.T  # (batch, 128)
        
        # Apply ReLU derivative
        dz1 = da1 * relu_derivative(self.cache['z1'])  # (batch, 128)
        
        # Gradient for W1 and b1
        dW1 = (self.cache['X'].T @ dz1) / batch_size  # (784, 128)
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size  # (1, 128)
        
        # ===== UPDATE WEIGHTS (GRADIENT DESCENT) =====
        # Formula: W_new = W_old - learning_rate * gradient
        # This moves weights in the direction that reduces loss
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train_step(self, X_batch, y_batch, learning_rate=0.01):
        """
        Perform one training step: forward → loss → backward.
        
        Args:
            X_batch: Input batch (batch_size, 784)
            y_batch: Label batch (batch_size,)
            learning_rate: Learning rate for gradient descent
        
        Returns:
            loss: Loss for this batch
        """
        # Forward pass: make predictions
        predictions = self.forward(X_batch)
        
        # Compute loss: how wrong are we?
        loss = self.compute_loss(y_batch)
        
        # Backward pass: compute gradients and update weights
        self.backward(y_batch, learning_rate)
        
        return loss
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input data (N, 784)
        
        Returns:
            predicted_classes: Predicted digit for each sample (N,)
        """
        probabilities = self.forward(X)
        predicted_classes = np.argmax(probabilities, axis=1)
        return predicted_classes
    
    def evaluate(self, X, y):
        """
        Evaluate accuracy on a dataset.
        
        Args:
            X: Input data (N, 784)
            y: True labels (N,)
        
        Returns:
            accuracy: Percentage of correct predictions
        """
        predictions = self.predict(X)
        accuracy = compute_accuracy(predictions, y)
        return accuracy


# Test the neural network if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Neural Network")
    print("=" * 60)
    
    # Create a small test network
    print("\n1. Creating network...")
    nn = NeuralNetwork(input_size=784, hidden1_size=128, hidden2_size=64, output_size=10)
    
    # Test forward pass
    print("2. Testing forward pass...")
    test_input = np.random.randn(5, 784)  # 5 samples
    predictions = nn.forward(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {predictions.shape}")
    print(f"   Output probabilities sum: {predictions.sum(axis=1)}")  # Should be all 1.0
    print(f"   Sample prediction: {predictions[0]}")
    
    # Test loss computation
    print("\n3. Testing loss computation...")
    test_labels = np.array([7, 2, 1, 0, 4])
    loss = nn.compute_loss(test_labels)
    print(f"   Loss: {loss:.4f}")
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    old_W1 = nn.W1.copy()
    nn.backward(test_labels, learning_rate=0.01)
    print(f"   Weights changed: {not np.array_equal(old_W1, nn.W1)}")
    
    # Test training step
    print("\n5. Testing training step...")
    loss = nn.train_step(test_input, test_labels, learning_rate=0.01)
    print(f"   Training loss: {loss:.4f}")
    
    # Test prediction
    print("\n6. Testing prediction...")
    predicted_classes = nn.predict(test_input)
    print(f"   Predicted classes: {predicted_classes}")
    print(f"   True classes: {test_labels}")
    
    print("\n" + "=" * 60)
    print("✓ All neural network tests passed!")
    print("=" * 60)
