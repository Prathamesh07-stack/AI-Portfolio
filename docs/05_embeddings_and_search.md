# Vector Embeddings and Semantic Search

## What Are Embeddings?

An embedding is a dense numerical vector that represents the meaning of text in a high-dimensional space. The key property: **semantically similar texts are close together in the embedding space**.

For example:
- "The cat sat on the mat" and "A feline rested on the rug" → vectors that are very close
- "The cat sat on the mat" and "Stock markets crashed today" → vectors that are far apart

## From Words to Sentences

### Word Embeddings (Early Approach)
Early methods like Word2Vec and GloVe created one vector per word. Problems:
- No context: "bank" (financial) and "bank" (river) get the same vector.
- No sentence-level meaning.

### Sentence Embeddings (Modern Approach)
Models like `sentence-transformers` create one vector per sentence or passage. These capture full semantic meaning including context.

The popular model `all-MiniLM-L6-v2`:
- 384-dimensional output vectors
- Trained on over 1 billion sentence pairs
- Only 22M parameters — fast and efficient
- Works great for semantic similarity and retrieval tasks

## How Embeddings Are Generated

A sentence transformer passes text through a BERT-like encoder and pools the token embeddings (usually mean pooling) to produce a single fixed-size vector:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("What is machine learning?")
# Returns a numpy array of shape (384,)
```

## Cosine Similarity

The similarity between two embeddings is measured with cosine similarity:

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

Score ranges from -1 to 1:
- **1.0:** Identical meaning
- **0.8–0.9:** Very similar (same topic, different phrasing)
- **0.5–0.7:** Related but distinct
- **< 0.3:** Unrelated

Cosine similarity is preferred over Euclidean distance because it measures angle (direction) rather than magnitude, making it robust to variations in text length.

## Semantic Search vs. Keyword Search

| Property | Keyword Search (BM25) | Semantic Search (Embeddings) |
|---|---|---|
| Matching | Exact word overlap | Meaning/concept match |
| Synonyms | Fails | Handles naturally |
| Paraphrasing | Fails | Handles naturally |
| Speed | Very fast (inverted index) | Fast (vector index) |
| Multilingual | Requires same language | Can work across languages |

Both have strengths. Hybrid search (combining both) often performs best.

## Nearest-Neighbor Search

For retrieval, we need to find the k vectors most similar to a query vector. Exact search computes similarity against every stored vector — this is:
- Perfectly accurate
- Too slow for very large databases (millions of vectors)

**Approximate Nearest Neighbor (ANN)** algorithms find nearly the closest vectors much faster. Popular libraries:
- **FAISS** (Facebook AI Similarity Search)
- **Annoy** (Spotify)
- **ScaNN** (Google)
- **HNSWlib** (hierarchical navigable small world graphs)

## FAISS Deep Dive

FAISS is a library from Meta AI for efficient similarity search.

Common index types:
- **IndexFlatL2:** Exact search, Euclidean distance. Baseline accuracy, slow at scale.
- **IndexFlatIP:** Exact search, inner product (cosine if normalized). What we use.
- **IndexIVFFlat:** Inverted file index — clusters vectors, only searches nearest clusters. Much faster at scale, slight accuracy tradeoff.
- **IndexHNSWFlat:** Graph-based ANN — fast, accurate, good for production.

For this project with a small document set, `IndexFlatIP` is perfect: exact and fast.

```python
import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatIP(dimension)

# Normalize embeddings for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index.add(embeddings)

# Search
query_vec = query_vec / np.linalg.norm(query_vec)
scores, indices = index.search(query_vec.reshape(1, -1), k=5)
```
