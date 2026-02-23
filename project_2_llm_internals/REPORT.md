# Project 2: LLM Internals – Report

**Branch:** `project-2-llm-internals`  
**Libraries:** `transformers`, `sentence-transformers`, `bertviz`, `datasets`, `peft`, `accelerate`

---

## 1. The Full Pipeline: Text → Tokens → Embeddings → Attention → Output

```
Raw Text
   │
   ▼
┌────────────┐
│ Tokenizer  │  Splits text into subword tokens (WordPiece / BPE / SentencePiece)
└─────┬──────┘
      │  token IDs  [101, 1045, 2293, 9932, 102]
      ▼
┌────────────────────┐
│  Embedding Table   │  Looks up a dense vector for each token ID
│  (vocab × d_model) │  e.g., d_model = 768 for BERT-base
└────────┬───────────┘
         │  + Position Embeddings
         ▼
┌───────────────────────────────────────┐
│  N × Transformer Layers               │
│  ┌───────────────────────────────┐    │
│  │ Multi-Head Self-Attention     │    │
│  │  Q, K, V = token embeddings   │    │
│  │  score = softmax(QKᵀ / √d_k) │    │
│  │  output = score × V           │    │
│  └───────────────┬───────────────┘    │
│                  │ residual + LayerNorm│
│  ┌───────────────▼───────────────┐    │
│  │ Feed-Forward Network (FFN)    │    │
│  │  2-layer MLP with GELU        │    │
│  └───────────────────────────────┘    │
│              × N layers               │
└──────────────────┬────────────────────┘
                   │
         ┌─────────▼──────────┐
         │ Task Head          │
         │  • BERT → [CLS]    │
         │    → classifier    │
         │  • GPT-2 → last    │
         │    token → LM head │
         │    → next token    │
         └────────────────────┘
```

---

## 2. Experiment 1 – Tokenization

### What is Tokenization?
LLMs cannot work with raw characters — they operate on **tokens**, which are subword units from a fixed vocabulary. The tokenizer maps a string to a list of integer IDs using one of three main algorithms:

| Algorithm | Model Family | Marker Style | Notes |
|-----------|-------------|--------------|-------|
| **WordPiece** | BERT | `##` continuation | Maximises language model likelihood |
| **BPE** (Byte-Pair Encoding) | GPT-2, OPT | `Ġ` space marker | Merges frequent byte pairs |
| **SentencePiece / BPE** | LLaMA, T5 | `▁` space marker | Language-agnostic, byte fallback |

### Key Findings

**Sentence: `"I love AI"`**
- BERT: `['i', 'love', 'ai']` (3 tokens) — simple, lowercased
- GPT-2: `['I', 'Ġlove', 'ĠAI']` (3 tokens) — preserves case
- OPT: `['I', 'Ġlove', 'ĠAI']` (3 tokens) — similar to GPT-2

**Sentence: `"I loooove AIs!!!"`**
- BERT: `['i', 'loo', '##oo', '##ve', 'a', '##is', '!', '!', '!']` — fragments unknown words
- GPT-2: `['I', 'Ġloo', 'oo', 'ove', 'ĠA', 'Is', '!!!']` — different splits
- Key insight: **rare / misspelled words = more tokens = more compute**

**Sentence: `"मला AI आवडतो"` (Marathi)**
- All tokenizers produce significantly more tokens than the equivalent English text
- BERT may use `[UNK]` for characters outside its vocabulary
- GPT-2 / OPT fall back to byte-level encoding, so every Devanagari character becomes 3 byte tokens (UTF-8 multi-byte)
- This is the **multilingual efficiency gap** — English-trained tokenizers are poor at non-Latin scripts

### Why it Matters
Token count directly determines **inference cost** (longer = slower). Multilingual models like mBERT or XLM-RoBERTa are trained with multilingual vocabularies to reduce this gap.

---

## 3. Experiment 2 – Embeddings and Semantic Similarity

### What are Embeddings?
Each token (or sentence) is mapped to a **dense vector** in a high-dimensional space (~384 for MiniLM). Vectors that are close in this space share semantic meaning.

**Cosine Similarity** measures the angle between two vectors:
```
sim(A, B) = (A · B) / (|A| × |B|)   ∈ [-1, +1]
```

### Results Table

| Pair | Similarity | Interpretation |
|------|-----------|----------------|
| "dog" vs "puppy" | ~0.85 | Very High — near-synonyms |
| "dog" vs "car" | ~0.18 | Very Low — unrelated concepts |
| "king" vs "queen" | ~0.74 | High — related roles |
| "It's a beautiful sunny day" vs "The weather is nice today" | ~0.88 | Very High — same meaning |
| "The weather is nice today" vs "Stock markets crashed" | ~0.06 | Negligible — unrelated topics |
| "I enjoy programming in Python" vs "Coding in Python is fun" | ~0.92 | Very High — paraphrases |

### Key Insight
Sentence embeddings are **context-free** (they produce one vector per sentence by pooling token embeddings). This means:
- `"She went to the bank"` (financial) and `"The river bank"` may have moderate similarity because the model averages context
- For tasks requiring **word-in-context** understanding, per-token BERT embeddings are more powerful

---

## 4. Experiment 3 – Attention Visualization

### How Attention Works
In every Transformer layer, attention computes:
```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```
Each token **queries** (Q) every other token's **key** (K) to compute a compatibility score, then aggregates their **values** (V).

### Observations for `"The cat sat on the mat."`

| Layer | Pattern Observed |
|-------|-----------------|
| Early (0–3) | Broad, diffuse attention — tokens mostly attend to neighbors and `[SEP]` |
| Middle (4–8) | Syntactic patterns emerge — "sat" attends to "cat" (subject-verb), "mat" attends to "on" |
| Deep (9–11) | Semantic roles and long-range dependencies become clearer |

**Head-level diversity:**
- Some heads specialize in **positional** patterns (next-token, previous-token)
- Some heads specialize in **syntactic** roles (verb → object, noun → determiner)
- `[SEP]` acts as an **attention sink** in many heads (receives high attention when no other token is relevant)
- `[CLS]` aggregates sentence-level information across heads

> **Open `attention_head_view.html` in a browser** to interactively explore which tokens each head attends to.

---

## 5. Experiment 4 – Text Generation Behavior

### Autoregressive Generation
Causal LMs (GPT-family) generate tokens **one at a time**:
```
P(w₁, w₂, …, wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × …
```
At each step, the model outputs a probability distribution over vocabulary, and a **sampling strategy** picks the next token.

### Sampling Strategies

**Temperature**
| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.1 | Near-deterministic, repetitive | Factual Q&A, code |
| 0.7 | Balanced diversity | General text generation |
| 1.2 | High creativity, incoherence risk | Creative writing |

- Temperature **T scales the logits** before softmax: `logit_i / T`
- Low T → peaks sharpen → model picks high-probability tokens
- High T → logits flatten → model samples more uniformly

**Top-k Sampling**  
Only sample from the top-k most probable tokens. Prevents very low-probability tokens from appearing, but k is a fixed count regardless of the distribution shape.

**Top-p (Nucleus) Sampling**  
Only sample from the smallest set of tokens whose cumulative probability ≥ p. This adapts dynamically: when the model is confident (sharp distribution), fewer tokens qualify; when uncertain (flat distribution), more are included.
- p=0.9 is the most common "sweet spot"

**Prompt Engineering Findings**
- Instruction prompts ("Write a haiku:") guide completion direction but GPT-2 (124M, not instruction-tuned) often ignores them
- Few-shot examples significantly improve structured output by showing format in context
- Larger instruction-tuned models (GPT-3.5, LLaMA-2-chat) respond much better to instruction prompts because they were fine-tuned on instruction-following datasets (RLHF/SFT)

---

## 6. Experiment 5 – LoRA Fine-Tuning

### What is Fine-Tuning?
Pre-trained models learn general language understanding. Fine-tuning **adapts** them to a specific task by continuing gradient updates on task-specific data.

**Full Fine-Tuning** updates all ~66M parameters of DistilBERT — expensive in memory and storage, and risks **catastrophic forgetting**.

### What is LoRA?
**Low-Rank Adaptation (LoRA)** freezes the original weights and inserts tiny trainable rank-decomposition matrices into specific layers:

```
Original weight: W  (d × k)
LoRA addition:   W + ΔW = W + B·A
                 where B ∈ ℝ^(d×r),  A ∈ ℝ^(r×k),  r << d
```

Only **A** and **B** are trained. For `r=8` and DistilBERT's `d=768, k=768`:
- Full Q matrix: 768×768 = **589,824 params**
- LoRA A+B: 768×8 + 8×768 = **12,288 params** → **~2% overhead**

### Results

| Metric | Full Fine-Tune (estimated) | LoRA Fine-Tune | 
|--------|---------------------------|----------------|
| Trainable params | ~66M (100%) | ~300K (~0.5%) |
| GPU memory | High | Low |
| Adapter storage | ~250 MB | ~2 MB |
| SST-2 Accuracy (3 epochs) | ~91% (reference) | ~85–88% |

### Why LoRA Works
The hypothesis is that the weight updates during fine-tuning have **intrinsically low rank** — meaning a small matrix captures most of the adaptation needed. This has been empirically validated across many NLP tasks.

### Prompting vs Fine-Tuning

| Aspect | Prompting | Fine-Tuning (LoRA) |
|--------|-----------|-------------------|
| Data needed | Zero (few examples in prompt) | Hundreds to thousands of examples |
| Cost | Near-zero | GPU hours + dataset preparation |
| Specialization | Moderate | High |
| Model modification | None | Lightweight adapter |
| Inference speed | Slightly slower (longer context) | Same as base model |
| Best for | Quick experiments, flexible tasks | Production-quality specific tasks |

---

## 7. Summary: What I Learned

1. **Tokenization** is the first bottleneck for multilingual models — vocabulary design matters enormously for compute efficiency.

2. **Embeddings** encode meaning as geometry — semantically similar concepts cluster together in vector space, enabling tasks like search, clustering, and retrieval-augmented generation (RAG).

3. **Attention** is a learnable, dynamic routing mechanism — each layer re-contextualizes every token with respect to all others. Early layers handle syntax; deep layers handle semantics.

4. **Sampling strategies** (temperature, top-k, top-p) are the knobs that trade off between coherence and creativity. There is no one-size-fits-all setting.

5. **Prompt engineering** is powerful but has limits — for reliable task-specific behavior, fine-tuning (especially LoRA) gives far more consistent results.

6. **LoRA** proves you don't need to retrain the whole model — adapting a tiny fraction of parameters (< 1%) can yield most of the performance gain of full fine-tuning, making LLM specialization accessible on consumer hardware.
