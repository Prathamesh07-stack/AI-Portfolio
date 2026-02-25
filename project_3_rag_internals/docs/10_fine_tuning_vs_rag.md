# Fine-Tuning and Parameter-Efficient Training

## What Is Fine-Tuning?

Fine-tuning means taking a pre-trained model and continuing training on a smaller, domain-specific dataset. The goal: adapt the model's general knowledge to a specific task or domain.

Examples:
- Fine-tune GPT on customer support logs → better customer support bot
- Fine-tune BERT on medical records → better medical NLP
- Fine-tune code LLM on your codebase → better internal code assistant

## Full Fine-Tuning vs. PEFT

### Full Fine-Tuning
Update all parameters. Most effective but:
- Requires GPU memory to store model + gradients + optimizer states (8–16x model size)
- Expensive for large models (100B+ parameters)
- Risk of "catastrophic forgetting" — old knowledge overwritten

### Parameter-Efficient Fine-Tuning (PEFT)
Only update a small fraction of parameters. Much cheaper with competitive results.

## LoRA (Low-Rank Adaptation)

LoRA (Hu et al., 2021) is the most popular PEFT method.

### Key Idea
Instead of updating weight matrix W directly (size d×d), add a low-rank update:

```
W' = W + ΔW = W + B × A
```

where:
- A is d×r (r << d)
- B is r×d
- r is the "rank" — a hyperparameter (typically 4–64)

A and B together require far fewer parameters than updating W directly.

### Why It Works
The hypothesis: weight updates for specific tasks lie in a low-dimensional subspace. The full rank isn't necessary.

### LoRA Hyperparameters
- **rank (r):** Number of dimensions in the low-rank update. Smaller = fewer parameters. Larger = more expressive.
- **alpha (α):** Scaling factor (α/r scales the update). Usually set to r or 2×r.
- **target_modules:** Which attention layers to apply LoRA to (usually query, value projections).

### QLoRA
Quantized LoRA — loads the base model in 4-bit quantized format before applying LoRA adapters. Enables fine-tuning 7B models on a 16GB GPU.

## When to Fine-Tune vs. Use RAG

| Scenario | Use RAG | Use Fine-Tuning |
|---|---|---|
| New knowledge (facts) | ✅ | ❌ (expensive, doesn't stick well) |
| Updated knowledge | ✅ | ❌ (retraining needed) |
| Style/tone adaptation | ❌ | ✅ |
| Domain terminology | ❌ | ✅ |
| Task format (classification, extraction) | ❌ | ✅ |
| Private data reasoning | ✅ (no training) | ✅ (but risky privacy-wise) |

## Fine-Tuning vs. RAG: Common Misconception

People often think fine-tuning "bakes in" facts. Research shows this is unreliable:
- Models fine-tuned on facts still hallucinate those facts ~30% of the time
- RAG retrieves the exact text, leaving less room for invention

**Best Practice:** Use RAG for facts, fine-tuning for behavior/style.

## The Combined Approach

Many production systems combine both:
1. **Fine-tune** the model on domain format, instructions, and style.
2. **Add RAG** for dynamic, factual, up-to-date knowledge retrieval.

This gives the best of both: domain expertise + factual grounding.

## Evaluating Fine-Tuned Models

- **ROUGE scores** for summarization tasks
- **BLEU** for translation
- **F1** for extraction
- **Human evaluation** for open-ended generation
- **HellaSwag, MMLU** benchmarks for general capability
