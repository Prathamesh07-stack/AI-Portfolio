# Chunking Strategies for RAG

One of the most important — and underappreciated — decisions in building a RAG system is how to split documents into chunks. Chunking strategy directly affects retrieval quality, answer accuracy, and hallucination rates.

## Why Chunk at All?

- **Embedding models have token limits** (typically 256–512 tokens). Longer texts must be split.
- **Precision:** Smaller chunks → more targeted retrieval. You retrieve exactly the relevant paragraph.
- **Context window:** LLMs have token limits. Retrieving full documents would overflow the context.
- **Signal-to-noise:** Feeding the LLM a 5-page document to answer a one-line question adds irrelevant noise.

## Fixed-Size Chunking

The simplest approach: split text every N words or N tokens.

```python
def chunk(text, size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = words[i:i + size]
        chunks.append(" ".join(chunk))
        if i + size >= len(words):
            break
    return chunks
```

**Overlap** ensures that sentences split at a boundary appear in both chunks, preventing information loss.

### Chunk Size Trade-offs

| Chunk Size | Pros | Cons |
|---|---|---|
| Small (100–300 words) | Precise retrieval, low noise | May split related ideas, more chunks to manage |
| Medium (300–600 words) | Good balance of precision and context | Still may split at awkward boundaries |
| Large (600–1000 words) | Full context preserved | Dilutes retrieval signal, may overflow context |

## Sentence-Aware Chunking

Instead of cutting at word N, cut at sentence boundaries. This avoids cutting mid-sentence.

```python
import re

def sentence_chunk(text, max_words=400, overlap_sentences=2):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Group sentences until max_words reached, then start new chunk
    ...
```

## Semantic Chunking

Advanced: use embedding similarity to detect topic shifts. Split when consecutive sentences' embeddings diverge significantly. This creates coherent, topically consistent chunks — but is computationally expensive.

## Parent-Child Chunking

- Store **small chunks** (e.g., 100 words) in the index for precise retrieval.
- But when retrieved, return the **parent chunk** (e.g., 500 words) to the LLM for more context.
- Best of both worlds.

## Chunk Overlap

Overlap is critical when using fixed-size chunking. Without overlap:
- "The transformer model, introduced in 2017, [CHUNK BREAK] revolutionized NLP." 
- The connection is lost.

With 50-word overlap:
- Chunk 1: "...The transformer model, introduced in 2017"
- Chunk 2: "...introduced in 2017, revolutionized NLP..."
- Both chunks carry the full context.

## Recommended Settings for This Project

| Experiment | Chunk Size | Overlap |
|---|---|---|
| Small | 200 words | 30 words |
| Medium | 500 words | 75 words |
| Large | 800 words | 100 words |

## Metadata Tracking

Every chunk should store:
```json
{
  "doc_id": "04_rag_explained",
  "chunk_idx": 2,
  "text": "...",
  "word_count": 312,
  "chunk_size_config": 400,
  "overlap_config": 50
}
```

This allows you to trace every retrieved chunk back to its source document.

## Impact on RAG Performance

- **Too small:** Risk of retrieving fragments without enough context. The LLM says "I don't know."
- **Too large:** The retrieved chunk contains the answer but also a lot of noise. Hallucination risk increases.
- **Optimal:** Depends on the document type and question style. Always experiment.
