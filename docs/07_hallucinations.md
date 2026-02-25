# LLM Hallucinations: What They Are and Why They Happen

## What Is a Hallucination?

In AI, a **hallucination** is when a language model generates text that is factually incorrect, made up, or not grounded in any real source — but stated with full confidence as if it were true.

Examples of hallucinations:
- Citing a scientific paper that doesn't exist
- Stating that a historical event happened on the wrong date
- Inventing a company's product features that were never announced
- Describing a law that was never passed

The model doesn't "know" it's wrong. It generates the most statistically likely next token given its training, regardless of factuality.

## Why Do LLMs Hallucinate?

### 1. Training on the Internet
LLMs are trained on internet text, which contains errors, contradictions, fiction, and speculation alongside facts. The model learns statistical patterns, not truth.

### 2. Probability, Not Lookup
LLMs don't store facts in a database — they compress patterns into billions of parameters. When asked about a specific fact they saw rarely, they may generate a plausible-sounding but wrong answer.

### 3. No Grounding Mechanism
Base LLMs have no external memory. They can't "check" an answer. They generate text that follows the statistical pattern of correct-sounding text.

### 4. Training to Sound Helpful
Models are RLHF-tuned to give confident, fluent answers. This can make them more likely to fabricate than to admit ignorance.

### 5. Confabulation Under Uncertainty
When a model encounters a question it's uncertain about, it often generates a confident-sounding blend of related true information — creating a plausible but false answer.

## Types of Hallucinations

| Type | Description | Example |
|---|---|---|
| Factual | Incorrect real-world fact | "Einstein won the Nobel Prize in 1925" (it was 1921) |
| Attribution | Cites wrong source | "As Turing wrote in his 1952 paper on AI..." |
| Entity | Invents person/organization | "The CEO of X Inc., John Smith..." (John Smith doesn't exist) |
| Fabricated detail | Adds false specifics | "The law passed with 72% of the vote" (invented number) |
| Outdated knowledge | Treat old info as current | "The latest version of GPT is 4" (post-cutoff) |

## Hallucination Rates in Practice

Research shows hallucination rates vary widely:
- **Simple factual questions:** 5–15% hallucination rate
- **Obscure facts:** 30–50%+
- **Opinion or reasoning tasks:** Much lower
- **Code generation:** Mostly correct structure, but APIs/functions may not exist

## How RAG Reduces Hallucinations

RAG reduces hallucinations by:
1. **Grounding** the model's response in retrieved text from a trusted source.
2. **Restricting** the answer with explicit instruction: "Answer ONLY from the context provided."
3. **Transparency:** The retrieved chunks are visible — you can verify the source.

However, RAG doesn't eliminate hallucinations:
- The model may still add details not in the context.
- If retrieval fails (wrong chunks recovered), the model hallucinates as if there's no context.
- Models sometimes blend context with training knowledge.

## Measuring Hallucinations

To measure hallucination in this project:
- **Manual labeling:** For each question, mark the answer as Correct / Hallucinated / Refused.
- **Grounding check:** Does the answer contain claims traceable to retrieved chunks?
- Compare raw LLM answers vs RAG answers on the same questions.

## Mitigation Strategies Beyond RAG

- **Temperature = 0:** Deterministic output, no creative invention.
- **Self-consistency:** Ask the same question N times, take majority answer.
- **Fact verification chains:** Ask the model to verify each claim in its answer.
- **Citations requirement:** Force the model to cite the source chunk for each statement.
