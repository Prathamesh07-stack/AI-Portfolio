"""
Step 3: Embeddings + FAISS Vector Store
Embeds chunks using sentence-transformers and stores them in a
FAISS IndexFlatIP (cosine similarity via normalized inner product).
"""

import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ingest import load_documents
from chunker import chunk_documents

VECTOR_STORE_DIR = "vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, 22M params, fast & excellent for retrieval


def get_embedding_model():
    """Load the sentence-transformers model (downloads on first run ~90MB)."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    return model


def embed_chunks(chunks: list[dict], model) -> np.ndarray:
    """
    Embed all chunk texts into a float32 numpy matrix.

    Returns:
        np.ndarray of shape (n_chunks, embedding_dim)
    """
    texts = [c['text'] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize for cosine sim via inner product
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS IndexFlatIP (exact inner product search).
    Since embeddings are L2-normalized, inner product == cosine similarity.
    """
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index, chunks: list[dict], label: str = "default"):
    """Save FAISS index and metadata to disk."""
    import faiss
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    index_path = os.path.join(VECTOR_STORE_DIR, f"index_{label}.faiss")
    meta_path = os.path.join(VECTOR_STORE_DIR, f"metadata_{label}.json")

    faiss.write_index(index, index_path)

    # Save metadata (everything except full text to keep it compact)
    metadata = [
        {
            'chunk_id':    c['chunk_id'],
            'doc_id':      c['doc_id'],
            'filename':    c['filename'],
            'chunk_idx':   c['chunk_idx'],
            'text':        c['text'],
            'word_count':  c['word_count'],
            'chunk_size_config': c['chunk_size_config'],
            'overlap_config':    c['overlap_config'],
        }
        for c in chunks
    ]
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {index_path}")
    print(f"  Saved: {meta_path}")
    return index_path, meta_path


def load_index(label: str = "default"):
    """Load FAISS index and metadata from disk."""
    import faiss
    index_path = os.path.join(VECTOR_STORE_DIR, f"index_{label}.faiss")
    meta_path = os.path.join(VECTOR_STORE_DIR, f"metadata_{label}.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}. Run embedder.py first.")

    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return index, metadata


def build_and_save(chunk_config: dict, model):
    """Build and save index for one chunk size configuration."""
    label = chunk_config['label']
    size = chunk_config['size']
    overlap = chunk_config['overlap']

    print(f"\n--- Building index for: {label} (size={size}, overlap={overlap}) ---")
    docs = load_documents("../docs")
    chunks = chunk_documents(docs, size=size, overlap=overlap)
    print(f"  Chunks: {len(chunks)}")

    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    save_index(index, chunks, label=label)
    return index, chunks


def main():
    """Build FAISS indices for all chunk size configurations."""
    print("=" * 60)
    print("STEP 3: Embeddings + Vector Store")
    print("=" * 60)

    model = get_embedding_model()

    configs = [
        {'label': 'small',  'size': 200, 'overlap': 30},
        {'label': 'medium', 'size': 500, 'overlap': 75},
    ]

    for config in configs:
        build_and_save(config, model)

    print("\n✅ All indices built and saved.")


if __name__ == "__main__":
    main()
