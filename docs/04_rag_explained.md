# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that enhances large language model (LLM) responses by retrieving relevant information from an external knowledge base before generating an answer. It was introduced in a 2020 paper by Lewis et al. from Facebook AI Research.

## The Problem RAG Solves

Large language models are trained on data up to a specific cutoff date. They cannot:
- Access private or internal documents
- Know about events after their training cutoff
- Always reliably recall specific facts from training (they sometimes hallucinate)

RAG addresses all three by giving the model a direct view into a curated knowledge base at inference time.

## How RAG Works

The RAG pipeline has two major phases:

### 1. Indexing (Offline)
1. **Collect documents** — internal wikis, PDFs, databases, etc.
2. **Chunk documents** — split into smaller passages (e.g., 300–600 words each).
3. **Embed chunks** — convert each chunk into a dense vector using an embedding model.
4. **Store in vector database** — index vectors for fast similarity search.

### 2. Inference (Online)
1. **User asks a question.**
2. **Embed the query** — convert query to a vector using the same embedding model.
3. **Retrieve top-k chunks** — find the most semantically similar chunks via vector search.
4. **Build a prompt** — combine the retrieved chunks with the original question.
5. **Generate answer** — LLM uses the provided context to answer accurately.

## RAG Prompt Template

A typical RAG prompt looks like:
```
You are a helpful assistant. Answer the user's question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
---
[RETRIEVED CHUNK 1]
[RETRIEVED CHUNK 2]
[RETRIEVED CHUNK 3]
---

Question: [USER QUESTION]
Answer:
```

## Why RAG Works Better Than Pure LLM

| Property | Pure LLM | RAG |
|---|---|---|
| Knowledge | Fixed (training cutoff) | Dynamic (live knowledge base) |
| Private data | No | Yes |
| Grounding | Prone to hallucination | Anchored to retrieved facts |
| Transparency | Hard to verify | Source chunks are visible |
| Cost | Cheap (no retrieval) | Small overhead for retrieval |

## RAG Limitations

Even with RAG, failures can occur:
- **Retrieval failure:** The right chunk isn't retrieved (wrong query embedding, poor chunking).
- **Context too long:** Too many chunks dilute the relevant information.
- **Retrieval + hallucination:** Model retrieves correct chunks but still invents details.
- **Query-chunk mismatch:** Query phrasing differs significantly from document phrasing.
- **Chunk boundary issues:** A key sentence is split across two chunks.

## Advanced RAG Techniques

- **HyDE (Hypothetical Document Embeddings):** Generate a hypothetical answer, embed it, then retrieve similar chunks.
- **Re-ranking:** Use a cross-encoder to re-rank retrieved chunks by relevance.
- **Query expansion:** Rephrase query multiple ways and retrieve for each.
- **Multi-hop retrieval:** Retrieve iteratively, using each step's output to refine the next query.
- **Parent-child chunking:** Store small chunks for retrieval but pass parent (larger) chunks to LLM.

## Real-World Applications

- Enterprise Q&A systems over internal documentation
- Customer support bots grounded in product manuals
- Medical knowledge bases with up-to-date clinical guidelines
- Legal research assistants
- Coding assistants with codebase context
