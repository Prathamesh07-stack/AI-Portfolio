# Large Language Models: Internals and Scaling

## What Is a Large Language Model?

A Large Language Model (LLM) is a neural network — almost always Transformer-based — trained on massive amounts of text to predict the next token. "Large" refers to the number of parameters: modern LLMs range from 1 billion to over 1 trillion parameters.

## Pre-training

LLMs are pre-trained on a simple self-supervised objective:

**Autoregressive language modeling:** Given tokens t₁, t₂, ..., tₙ₋₁, predict tₙ.

No human labels are needed — the text itself is the supervision signal. This is why LLMs can be trained on vast internet-scale datasets.

Training data includes:
- Web pages (Common Crawl)
- Books and academic papers
- Wikipedia
- Code repositories (GitHub)
- News articles, forums

## Tokenization

Before processing text, LLMs convert it to tokens using a tokenizer (typically BPE — Byte Pair Encoding).

- "unhappiness" → ["un", "happ", "iness"] → [3492, 5678, 9010]
- Tokens are roughly 4 characters on average in English
- GPT-4 has a vocabulary of ~100,000 tokens

Token efficiency matters: "Hello world" might be 2 tokens or 4 depending on the tokenizer.

## Context Window

The context window is the maximum number of tokens an LLM can process at once. It includes:
- The system prompt
- Conversation history
- Retrieved chunks (in RAG)
- The user's question

| Model | Context Window (tokens) |
|---|---|
| GPT-3 | 4,096 |
| GPT-4 | 8K–128K |
| Claude 3.5 | 200,000 |
| Llama 3 | 8K–128K |
| Gemini 1.5 | 1,000,000 |

Longer context = more expensive (quadratic attention cost).

## Instruction Tuning and RLHF

Raw pre-trained models generate text continuations — they don't "answer questions." To make them helpful assistants:

### Supervised Fine-Tuning (SFT)
Fine-tune on (instruction, response) pairs: "Explain black holes in simple terms" → [ideal explanation]. The model learns to follow instructions.

### RLHF (Reinforcement Learning from Human Feedback)
1. Generate multiple candidate responses.
2. Humans rank them by quality.
3. Train a **reward model** to predict human preferences.
4. Fine-tune the LLM using RL (PPO) to maximize reward model score.

This makes models more helpful, harmless, and honest (HHH).

## Emergent Abilities

As models scale, unexpected capabilities "emerge" at certain sizes:
- Multi-step reasoning
- Code generation
- Arithmetic
- Chain-of-thought reasoning

These abilities aren't explicitly trained — they arise from scale.

## Key LLMs

| Model | Developer | Parameters | Open Source |
|---|---|---|---|
| GPT-4 | OpenAI | Unknown | No |
| Claude 3.5 Sonnet | Anthropic | Unknown | No |
| Gemini 1.5 Pro | Google | Unknown | No |
| Llama 3 70B | Meta | 70B | Yes |
| Mistral 7B | Mistral AI | 7B | Yes |
| Phi-3 Mini | Microsoft | 3.8B | Yes |

## Inference and Temperature

At generation time, the model produces a probability distribution over all tokens. **Temperature** controls randomness:
- **Temperature = 0:** Always pick the highest probability token (deterministic).
- **Temperature = 1:** Sample proportionally to probabilities (default, creative).
- **Temperature > 1:** More random. Often nonsensical.

For RAG evaluation, temperature = 0 gives reproducible results.

## Why LLMs Still Need RAG

Even the largest LLMs:
- Have knowledge cutoffs
- Cannot access private data
- Hallucinate specific facts
- Cannot update their knowledge without retraining

RAG adds a real-time, updatable knowledge layer on top of a static model.
