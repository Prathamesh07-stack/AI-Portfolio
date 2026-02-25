# Project 3: RAG from Scratch – Internals 🔍

Build a minimal but **real** Retrieval-Augmented Generation (RAG) system from scratch to understand exactly how retrieval improves LLM answers — and where hallucinations still happen.

---

## 🎯 Goal

Build every layer of a RAG pipeline by hand:
- Document ingestion & normalization
- Chunking with configurable size and overlap
- Semantic embeddings using `sentence-transformers`
- Vector indexing with **FAISS** (IndexFlatIP, cosine similarity)
- Top-k retrieval with similarity scores
- LLM answering **with** and **without** context (via Ollama)
- Hallucination analysis across chunk sizes

---

## 🧠 Concepts Covered

| Concept | Where |
|---|---|
| Vector embeddings | `embedder.py` |
| Cosine similarity | `embedder.py`, `retriever.py` |
| Nearest-neighbor search (FAISS) | `embedder.py`, `retriever.py` |
| Chunking strategies (size, overlap) | `chunker.py` |
| Full RAG pipeline | `rag_pipeline.py` |
| Hallucination behavior | `RAG_ANALYSIS.md` |

---

## 📂 Project Structure

```
project_3_rag_internals/
├── docs/                     # 10 knowledge-base documents
│   ├── 01_what_is_ai.md
│   ├── 02_neural_networks.md
│   ├── 03_transformers.md
│   ├── 04_rag_explained.md
│   ├── 05_embeddings_and_search.md
│   ├── 06_chunking_strategies.md
│   ├── 07_hallucinations.md
│   ├── 08_llm_internals.md
│   ├── 09_vector_databases.md
│   └── 10_fine_tuning_vs_rag.md
├── vector_store/             # FAISS indices + metadata (auto-created)
├── eval_results/             # Evaluation JSON outputs (auto-created)
├── ingest.py                 # Step 1: Load & normalize documents
├── chunker.py                # Step 2: Chunk with size + overlap
├── embedder.py               # Step 3: Embed + build FAISS index
├── retriever.py              # Step 4: Semantic retrieval
├── rag_pipeline.py           # Step 5: RAG vs No-RAG evaluation
├── eval_questions.json       # 15 evaluation questions
├── RAG_ANALYSIS.md           # Analysis report (write after experiments)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup virtual environment

```bash
cd project_3_rag_internals
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run each step in order

```bash
# Step 1 – Ingest documents
python ingest.py

# Step 2 – Chunking experiments
python chunker.py

# Step 3 – Build embedding indices (downloads model ~90MB first time)
python embedder.py

# Step 4 – Retrieval demo
python retriever.py

# Step 5 – RAG vs No-RAG evaluation (needs Ollama running)
python rag_pipeline.py
```

### 3. (Optional) Setup Ollama for real LLM answers

```bash
# Install Ollama from https://ollama.com
brew install ollama             # macOS
ollama pull llama3.2            # or: ollama pull mistral
ollama serve                    # start the local server
```

If Ollama is not running, `rag_pipeline.py` will run but show a friendly error message instead of real LLM answers. All retrieval still works.

---

## 📊 Key Experiments

### Chunking Impact
Run `chunker.py` — observe how chunk count changes with size 200 vs 500 words.

### Retrieval Quality
Run `retriever.py` — see which chunks FAISS retrieves for each query and their similarity scores.

### RAG vs No-RAG
Run `rag_pipeline.py` — compare LLM answers on:
- Questions **answerable from docs** (expect RAG to win)
- Questions **NOT in docs** (expect both to struggle, but differently)

### Hallucination Analysis
After running the pipeline:
1. Open `eval_results/results_*.json`
2. Fill in the `labels` section for each question (correct/hallucinated)
3. Compare small vs medium chunk results
4. Write findings in `RAG_ANALYSIS.md`

---

## 🔑 Key Takeaways

- **RAG = retrieval + generation.** The model is only as good as what it retrieves.
- **Chunk size matters.** Too small → fragmented context. Too large → noisy context.
- **Cosine similarity** works because it measures direction, not magnitude.
- **FAISS IndexFlatIP** on normalized vectors = exact cosine similarity. Perfect for small datasets.
- **Hallucinations happen** even with RAG when retrieval fails or the model blends retrieved text with training knowledge.

---

*Part of the AI Portfolio – Project 3 | Tech: sentence-transformers, FAISS, Ollama*
