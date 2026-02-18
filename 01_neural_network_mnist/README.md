# Neural Network from Scratch (MNIST)

## 🎯 Goal
Understand how neural networks actually learn by:
1. Building a neural network from scratch using only NumPy
2. Building the same network using PyTorch
3. Comparing performance, speed, and understanding what PyTorch automates

## 📚 What You'll Learn
- How neural networks make predictions (forward propagation)
- How neural networks learn (backpropagation)
- What gradients are and how they guide learning
- How loss functions measure error
- What PyTorch automates (autograd, GPU support, optimizers)

## 🏗️ Architecture
```
Input Layer:  784 neurons (28×28 flattened image)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 64 neurons (ReLU activation)
Output Layer: 10 neurons (Softmax - probabilities for digits 0-9)
```

## 📂 Project Structure
```
project_1_mnist_nn/
├── data/                    # MNIST dataset (auto-downloaded)
├── numpy_nn/               # From-scratch implementation
│   ├── neural_network.py  # Core NN class
│   ├── train.py           # Training loop
│   └── utils.py           # Helper functions
├── pytorch_nn/            # PyTorch implementation
│   ├── model.py          # PyTorch model
│   └── train.py          # PyTorch training loop
├── results/              # Plots and metrics
├── data_loader.py        # Data loading utilities
├── compare.py            # Comparison script
├── requirements.txt      # Dependencies
├── README.md            # This file
└── ANALYSIS.md          # Your findings (created after training)
```

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run NumPy Implementation (From Scratch)
```bash
cd numpy_nn
python train.py
```
Expected: ~95-97% accuracy, ~5-10 minutes on CPU

### 3. Run PyTorch Implementation
```bash
cd pytorch_nn
python train.py
```
Expected: ~97-98% accuracy, ~2-3 minutes on CPU

### 4. Compare Results
```bash
python compare.py
```
Generates comparison plots and metrics

## 📊 Expected Results
- **NumPy Accuracy**: 95-97%
- **PyTorch Accuracy**: 97-98%
- **Speed Difference**: PyTorch ~3-5× faster on CPU
- **Learning**: Both should show smooth loss curves and increasing accuracy

## 🎓 Key Concepts Covered
1. **Forward Propagation**: How data flows through the network
2. **Activation Functions**: ReLU and Softmax
3. **Loss Function**: Cross-entropy for classification
4. **Backpropagation**: Computing gradients using chain rule
5. **Gradient Descent**: Updating weights to minimize loss
6. **Mini-batch Training**: Processing data in batches
7. **Overfitting**: Monitoring train vs test accuracy

## 📖 Next Steps
After completing this project:
- Experiment with different architectures (more layers, different sizes)
- Try different learning rates and batch sizes
- Add regularization techniques (dropout, L2)
- Move to convolutional neural networks (CNNs) for better image performance
