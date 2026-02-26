# The Transformer Architecture

The Transformer is a neural network architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. It has become the dominant architecture for NLP and increasingly for vision, audio, and multimodal tasks.

## Why Transformers?

Before Transformers, sequence models like RNNs and LSTMs were standard for NLP. They processed tokens one at a time, which made them:
- **Slow to train** (sequential, hard to parallelize)
- **Poor at long-range dependencies** (information from early tokens fades)

Transformers process the entire sequence in parallel using attention mechanisms, solving both problems.

## The Self-Attention Mechanism

Self-attention allows each token to "look at" every other token in the sequence to determine which are most relevant.

### Queries, Keys, and Values
For each token, three vectors are computed:
- **Query (Q):** What am I looking for?
- **Key (K):** What do I represent?
- **Value (V):** What information do I carry?

The attention score between tokens i and j is:
```
score(i, j) = softmax(Q_i · K_j / sqrt(d_k))
```

The output for token i is a weighted sum of all Value vectors:
```
output_i = sum(score(i, j) * V_j for all j)
```

### Multi-Head Attention
Instead of one attention computation, Transformers use multiple "heads" in parallel. Each head learns different relationships (e.g., one head tracks subject-verb agreement, another tracks coreference). Outputs are concatenated and projected.

## Transformer Architecture

A full Transformer has two components:

### Encoder
Processes the input sequence. Used in models like BERT.
- Stack of N identical layers
- Each layer: Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm

### Decoder
Generates the output sequence. Used in models like GPT.
- Like encoder but with masked self-attention (can't look at future tokens) and cross-attention to encoder outputs.

## Positional Encoding

Since Transformers process tokens in parallel (not sequentially), they need positional information added to embeddings. The original paper uses sine/cosine functions:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Key Properties

- **Parallelizable**: The entire sequence is processed at once — GPUs love this.
- **Long-range dependencies**: Direct path between any two tokens regardless of distance.
- **Scalable**: Performance improves predictably with scale (data, parameters, compute).

## Variants of Transformers

| Model | Type | Key Innovation |
|---|---|---|
| BERT | Encoder-only | Bidirectional pre-training (masked language modeling) |
| GPT series | Decoder-only | Autoregressive language modeling at scale |
| T5 | Encoder-Decoder | Text-to-text framing for all NLP tasks |
| BART | Encoder-Decoder | Denoising autoencoder for seq2seq |
| ViT | Encoder | Patches of images treated as tokens |
| Whisper | Encoder-Decoder | Speech recognition |
| CLIP | Dual encoder | Connects text and images |

## Computational Complexity

Standard self-attention has O(n²) time and memory complexity where n is sequence length. This is why processing very long documents is expensive. Solutions include:
- **Sparse Attention** (Longformer, BigBird)
- **Linear Attention** approximations
- **Flash Attention** (IO-aware exact attention)
