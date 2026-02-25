"""
Step 2: Text Chunking
Splits documents into overlapping word-level chunks.
Experiment with different sizes to see impact on retrieval quality.
"""

from ingest import load_documents


def chunk(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text:    The input text to split.
        size:    Maximum number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= size:
        return [text]

    chunks = []
    step = size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")

    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        if end == len(words):
            break
        start += step

    return chunks


def chunk_documents(
    documents: list[dict],
    size: int = 400,
    overlap: int = 50,
) -> list[dict]:
    """
    Chunk all documents and return flat list of chunk dicts.

    Returns:
        List of dicts: {
            'chunk_id': str,       # unique ID: doc_id__chunk_N
            'doc_id': str,
            'filename': str,
            'chunk_idx': int,
            'text': str,
            'word_count': int,
            'chunk_size_config': int,
            'overlap_config': int,
        }
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk(doc['text'], size=size, overlap=overlap)
        for idx, chunk_text in enumerate(doc_chunks):
            all_chunks.append({
                'chunk_id': f"{doc['doc_id']}__chunk_{idx}",
                'doc_id': doc['doc_id'],
                'filename': doc['filename'],
                'chunk_idx': idx,
                'text': chunk_text,
                'word_count': len(chunk_text.split()),
                'chunk_size_config': size,
                'overlap_config': overlap,
            })
    return all_chunks


def main():
    """Run chunking experiments with different sizes."""
    print("=" * 60)
    print("STEP 2: Chunking Experiments")
    print("=" * 60)

    docs = load_documents("docs")

    configs = [
        ("SMALL",  200, 30),
        ("MEDIUM", 500, 75),
    ]

    for label, size, overlap in configs:
        chunks = chunk_documents(docs, size=size, overlap=overlap)
        total = len(chunks)
        avg_words = sum(c['word_count'] for c in chunks) / total if total else 0

        print(f"\n[{label}] size={size} words, overlap={overlap} words")
        print(f"  Total chunks : {total}")
        print(f"  Avg words/chunk: {avg_words:.0f}")
        print()

        # Per-document breakdown
        from collections import defaultdict
        per_doc = defaultdict(int)
        for c in chunks:
            per_doc[c['doc_id']] += 1

        for doc_id, count in sorted(per_doc.items()):
            print(f"    {doc_id:<35} → {count} chunks")

    print("\n✅ Chunking experiments complete.")
    return chunks


if __name__ == "__main__":
    main()
