# Vector Databases

A vector database is a specialized database designed to store, index, and search high-dimensional vectors (embeddings) efficiently. It is the core storage layer in any RAG system.

## Why Not a Regular Database?

Traditional databases (SQL, NoSQL) are designed for exact lookups:
- "Find user with id = 42"
- "Find all products with price < 100"

Vector search requires **approximate nearest neighbor (ANN)** queries:
- "Find the 5 vectors most similar to this query vector"

No traditional index (B-tree, hash) supports this efficiently.

## FAISS (Facebook AI Similarity Search)

FAISS, developed by Meta AI, is an open-source library for efficient similarity search of dense vectors.

### Key Features
- Written in C++ with Python bindings
- Supports both CPU and GPU
- Handles billions of vectors
- Multiple index types for different accuracy/speed tradeoffs

### Index Types

**IndexFlatL2** — Exact Euclidean distance. No compression.
```python
index = faiss.IndexFlatL2(384)
```

**IndexFlatIP** — Exact inner product (cosine if vectors are normalized). What we use for RAG.
```python
index = faiss.IndexFlatIP(384)
faiss.normalize_L2(embeddings)  # Required for cosine similarity
index.add(embeddings)
```

**IndexIVFFlat** — Inverted file index. Clusters vectors into Voronoi cells. Searches only nearby clusters. 10-100x faster than Flat, slight accuracy loss.
```python
quantizer = faiss.IndexFlatL2(384)
index = faiss.IndexIVFFlat(quantizer, 384, n_clusters)
index.train(embeddings)  # Must train first
index.add(embeddings)
```

**IndexHNSWFlat** — Hierarchical Navigable Small World graph. Very fast ANN. Good production choice.

### Saving and Loading
```python
faiss.write_index(index, "vector_store/index.faiss")
index = faiss.read_index("vector_store/index.faiss")
```

## Chroma

Chroma is a newer, developer-friendly vector database purpose-built for LLM applications.

### Key Differences from FAISS
- Full database (not just an index): stores vectors + metadata + text together
- Persistent by default (SQLite backend)
- Simple Python API
- Built-in embedding models (can auto-embed text)

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.create_collection("docs")

collection.add(
    documents=["Text of chunk 1", "Text of chunk 2"],
    metadatas=[{"doc_id": "doc1"}, {"doc_id": "doc2"}],
    ids=["chunk_0", "chunk_1"]
)

results = collection.query(query_texts=["my question"], n_results=5)
```

### When to Use Chroma
- Prototyping and development
- Small to medium datasets (< 1M vectors)
- When you want automatic embedding + storage in one tool

### When to Use FAISS
- You want full control over embeddings
- Very large datasets
- Custom index types
- Performance-critical production systems

## Other Vector Databases

| Database | Type | Scale | Notable Feature |
|---|---|---|---|
| Pinecone | Cloud-managed | Billions | Fully managed, no ops |
| Weaviate | Open-source | Millions | GraphQL API, hybrid search |
| Qdrant | Open-source | Millions | Filters + payload alongside vectors |
| Milvus | Open-source | Billions | Kubernetes-native, enterprise |
| pgvector | PostgreSQL extension | Millions | No separate infra — lives in Postgres |

## Metadata Filtering

Real-world RAG often needs filtered search: "Find the top 5 chunks from documents tagged 'legal' and created after 2023."

FAISS does not support metadata filtering natively — you'd filter the results post-search. Qdrant, Weaviate, and Chroma all support pre-filtering at index time.

## Performance Benchmark (approximate)

| Index | 1M vectors, d=128, top-5 | Accuracy |
|---|---|---|
| IndexFlatL2 | ~50ms | 100% |
| IndexIVFFlat (nlist=1024) | ~2ms | 97% |
| IndexHNSWFlat | ~1ms | 98% |
| Pinecone (cloud) | ~30ms network | 99% |
