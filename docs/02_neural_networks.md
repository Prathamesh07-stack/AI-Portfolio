# Neural Networks: A Deep Dive

A neural network is a computational model loosely inspired by the structure of the human brain. It consists of layers of interconnected nodes (neurons) that process information by learning from data.

## Biological Inspiration

The brain contains roughly 86 billion neurons, each connected to thousands of others via synapses. When a neuron receives enough input signals, it "fires" and sends a signal to connected neurons. Neural networks mimic this with weighted connections and activation functions.

## Architecture of a Neural Network

### Layers
Every feedforward neural network has:
1. **Input Layer:** Receives raw data (e.g., pixel values, word embeddings).
2. **Hidden Layers:** Intermediate layers that learn abstract representations. More layers = deeper network = more complex patterns.
3. **Output Layer:** Produces the final prediction (e.g., a class probability or a number).

### Neurons
Each neuron computes:
```
output = activation(sum(weights * inputs) + bias)
```

### Activation Functions
Activation functions introduce non-linearity, which is essential for learning complex patterns:
- **ReLU (Rectified Linear Unit):** f(x) = max(0, x). Most commonly used in hidden layers.
- **Sigmoid:** f(x) = 1/(1+e^-x). Outputs 0–1, used for binary classification.
- **Softmax:** Converts logits to probabilities that sum to 1, used for multi-class output.
- **Tanh:** Similar to sigmoid but outputs -1 to 1.

## Training: Backpropagation

Training a neural network means adjusting weights to minimize a loss function.

### Forward Pass
Input data passes through each layer, producing an output.

### Loss Calculation
The loss (error) measures how far the output is from the correct answer. Common loss functions:
- **Mean Squared Error (MSE):** For regression tasks.
- **Cross-Entropy Loss:** For classification tasks.

### Backward Pass (Backpropagation)
The gradient of the loss with respect to each weight is computed using the chain rule of calculus. Weights are then updated via:
```
weight = weight - learning_rate * gradient
```

### Gradient Descent Variants
- **Batch Gradient Descent:** Uses entire dataset per update. Slow but stable.
- **Stochastic Gradient Descent (SGD):** Uses one sample per update. Fast but noisy.
- **Mini-Batch Gradient Descent:** Uses small batches (32–256 samples). Best of both worlds.
- **Adam Optimizer:** Adaptive learning rates per parameter. Most popular today.

## Overfitting and Regularization

Overfitting occurs when a model memorizes training data instead of generalizing.

Prevention techniques:
- **Dropout:** Randomly zeroes neuron outputs during training, forcing redundancy.
- **L2 Regularization (Weight Decay):** Penalizes large weights.
- **Early Stopping:** Stop training when validation loss stops improving.
- **Data Augmentation:** Artificially expand training set.

## Deep Learning

When a network has many hidden layers (typically ≥3), it is called a **deep** neural network. Deep learning has revolutionized:
- **Image recognition** (CNNs)
- **Sequential data** (RNNs, LSTMs)
- **Language understanding** (Transformers)

The key insight is that deeper networks learn hierarchical features: early layers detect edges, middle layers detect shapes, and later layers detect semantic concepts.

## Famous Neural Network Architectures

| Architecture | Year | Application |
|---|---|---|
| LeNet | 1989 | Handwritten digits |
| AlexNet | 2012 | Image classification |
| VGG | 2014 | Deep image features |
| ResNet | 2015 | Very deep nets with skip connections |
| LSTM | 1997 | Sequence modeling |
| Transformer | 2017 | Language tasks (and now everything) |
