"""
PyTorch Neural Network Model

This is the SAME neural network as the NumPy version, but using PyTorch.
Compare this code with the NumPy version to see what PyTorch automates!

Key Differences:
- PyTorch handles backpropagation automatically (autograd)
- Built-in optimizers (no manual gradient descent)
- GPU support with minimal code changes
- Much faster training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetworkPyTorch(nn.Module):
    """
    Neural network using PyTorch.
    
    Architecture: 784 → 128 → 64 → 10 (same as NumPy version)
    
    What PyTorch provides:
    - Automatic gradient computation (no manual backprop!)
    - Optimized operations (faster than NumPy)
    - GPU support (just call .to('cuda'))
    - Built-in layers and activations
    """
    
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        """
        Initialize the network.
        
        In PyTorch, we define layers in __init__ and use them in forward().
        PyTorch automatically tracks parameters for us!
        """
        super(NeuralNetworkPyTorch, self).__init__()
        
        # Define layers
        # nn.Linear creates a fully connected layer (does W @ x + b)
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        
        # PyTorch initializes weights automatically using a good default
        # (similar to our He initialization)
        
        print(f"PyTorch Network initialized: {input_size} → {hidden1_size} → {hidden2_size} → {output_size}")
        print(f"Total parameters: {self._count_parameters():,}")
    
    def _count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """
        Forward pass.
        
        Compare this to the NumPy version - it's much simpler!
        PyTorch handles all the matrix operations and caching for backprop.
        
        Args:
            x: Input tensor (batch_size, 784)
        
        Returns:
            Output logits (batch_size, 10)
        """
        # Layer 1: Linear → ReLU
        x = self.fc1(x)  # Same as: x @ W1 + b1
        x = F.relu(x)    # Same as: np.maximum(0, x)
        
        # Layer 2: Linear → ReLU
        x = self.fc2(x)
        x = F.relu(x)
        
        # Layer 3: Linear (no activation here)
        # We don't apply softmax because CrossEntropyLoss does it internally
        x = self.fc3(x)
        
        return x  # Return logits (raw scores)
    
    def predict(self, x):
        """
        Make predictions.
        
        Args:
            x: Input tensor (batch_size, 784)
        
        Returns:
            Predicted classes (batch_size,)
        """
        with torch.no_grad():  # Don't compute gradients for prediction
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


# Alternative: Using nn.Sequential (even simpler!)
def create_sequential_model(input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
    """
    Create the same network using nn.Sequential.
    
    This is the most concise way to define a simple feedforward network.
    Compare this to our 200+ lines of NumPy code!
    """
    model = nn.Sequential(
        nn.Linear(input_size, hidden1_size),
        nn.ReLU(),
        nn.Linear(hidden1_size, hidden2_size),
        nn.ReLU(),
        nn.Linear(hidden2_size, output_size)
    )
    return model


# Test the model if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch Model")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating model...")
    model = NeuralNetworkPyTorch()
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    test_input = torch.randn(5, 784)  # 5 samples
    output = model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Sample output (logits): {output[0]}")
    
    # Test prediction
    print("\n3. Testing prediction...")
    predictions = model.predict(test_input)
    print(f"   Predictions: {predictions}")
    
    # Test Sequential version
    print("\n4. Testing Sequential model...")
    seq_model = create_sequential_model()
    seq_output = seq_model(test_input)
    print(f"   Sequential output shape: {seq_output.shape}")
    
    # Check if GPU is available
    print("\n5. Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
        model_gpu = model.to('cuda')
        test_input_gpu = test_input.to('cuda')
        output_gpu = model_gpu(test_input_gpu)
        print(f"   ✓ Model runs on GPU!")
    else:
        print(f"   ℹ No GPU available, will use CPU")
    
    print("\n" + "=" * 60)
    print("✓ All PyTorch model tests passed!")
    print("=" * 60)
