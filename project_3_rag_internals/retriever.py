"""
Step 4: Retrieval
Given a query, embed it with the same model, search the FAISS index,
and return the top-k most similar chunks with scores.
"""

import textwrap
import numpy as np


def retrieve(
    query: str,
    index,
    metadata: list[dict],
    model,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top-k chunks most similar to the query.

    Args:
        query:    Natural language query string.
        index:    Loaded FAISS index (IndexFlatIP).
        metadata: List of chunk metadata dicts (same order as index vectors).
        model:    SentenceTransformer model instance.
        top_k:    Number of results to return.

    Returns:
        List of result dicts ranked by score (highest first):
        {rank, score, doc_id, filename, chunk_idx, snippet}
    """
    # Embed and normalize the query (same as index: normalized → cosine sim)
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # FAISS search: returns (scores, indices) arrays of shape (1, top_k)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        chunk_meta = metadata[idx]
        results.append({
            'rank':      rank,
            'score':     float(score),
            'doc_id':    chunk_meta['doc_id'],
            'filename':  chunk_meta['filename'],
            'chunk_idx': chunk_meta['chunk_idx'],
            'chunk_id':  chunk_meta['chunk_id'],
            'text':      chunk_meta['text'],
            'snippet':   chunk_meta['text'][:300],  # first 300 chars for display
        })

    return results


def print_results(query: str, results: list[dict]):
    """Pretty-print retrieval results."""
    print(f"\n{'='*65}")
    print(f"QUERY: {query}")
    print(f"{'='*65}")
    for r in results:
        print(f"\n  Rank #{r['rank']}  |  Score: {r['score']:.4f}  |  {r['doc_id']}  (chunk {r['chunk_idx']})")
        print(f"  " + "-" * 60)
        wrapped = textwrap.fill(r['snippet'], width=60, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)
        if len(r['text']) > 300:
            print(f"  ... [{len(r['text'].split())} words total]")
    print()


def build_context(results: list[dict], max_chunks: int = 3) -> str:
    """Combine top chunks into a context string for the LLM prompt."""
    context_parts = []
    for r in results[:max_chunks]:
        context_parts.append(f"[Source: {r['doc_id']}, chunk {r['chunk_idx']}]\n{r['text']}")
    return "\n\n---\n\n".join(context_parts)


def main():
    """Run retrieval demo with a set of test queries."""
    from embedder import load_index, get_embedding_model

    print("=" * 65)
    print("STEP 4: Retrieval Demo")
    print("=" * 65)

    # Load medium-chunk index by default
    index, metadata = load_index("medium")
    model = get_embedding_model()

    test_queries = [
        "What is the transformer architecture and how does attention work?",
        "How does RAG reduce hallucinations?",
        "What is cosine similarity in vector search?",
        "What are the risks of fine-tuning instead of using RAG?",
        "Explain backpropagation and gradient descent",
    ]

    for query in test_queries:
        results = retrieve(query, index, metadata, model, top_k=3)
        print_results(query, results)

    print("✅ Retrieval demo complete.")


if __name__ == "__main__":
    main()
