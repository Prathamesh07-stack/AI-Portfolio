# AI Projects Portfolio 🧠

My journey learning Artificial Intelligence from the ground up. This repository contains hands-on implementations of core AI concepts, starting from scratch and moving to advanced frameworks.

---

## 📂 Projects

### [01. Neural Network from Scratch (MNIST)](./01_neural_network_mnist/)
**Goal:** Understand the math behind neural networks by building one without frameworks.
- **Concepts:** Backpropagation, Gradient Descent, Activation Functions, Matrix Math.
- **Tech Stack:** NumPy (Manual), PyTorch (Comparison).
- **Result:** ~97% accuracy on handwritten digits.

---

### [02. LLM Internals from Scratch](./project_2_llm_internals/)
**Goal:** Understand exactly how LLMs work by running five experiments from the inside.
- **Concepts:** Tokenization, Embeddings, Attention, Text Generation, LoRA Fine-Tuning.
- **Tech Stack:** HuggingFace Transformers, sentence-transformers, PEFT.
- **Result:** Trained a LoRA adapter, visualized attention heads, compared base vs fine-tuned outputs.

---

### [03. RAG from Scratch – Internals](./project_3_rag_internals/)
**Goal:** Build a full Retrieval-Augmented Generation pipeline to understand how retrieval improves LLM answers.
- **Concepts:** Vector Embeddings, Cosine Similarity, FAISS Vector DB, Chunking, Hallucination Analysis.
- **Tech Stack:** sentence-transformers (`all-MiniLM-L6-v2`), FAISS, Ollama (local LLM).
- **Result:** Built end-to-end RAG system, compared RAG vs no-RAG on 15 questions across 2 chunk sizes.

---

## 🚀 Upcoming Projects
4. **Convolutional Neural Networks (CNNs)** - Custom Image Classifier
5. **Natural Language Processing (NLP)** - Sentiment Analysis Bot

---

*Created by Prathamesh as part of the AI Learning Path.*
