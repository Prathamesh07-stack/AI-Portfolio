"""
Experiment 4 – Text Generation Behavior
Use GPT-2 to explore how temperature, top-k, and top-p affect
randomness, creativity, and coherence of generated text.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
MODEL_NAME   = "gpt2"
MAX_NEW_TOKENS = 60
REPEAT_PER_CONFIG = 2   # How many completions to generate per config


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def generate(model, tokenizer, prompt: str, label: str, **kwargs) -> None:
    """Tokenise prompt and generate text, printing the result."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            **kwargs,
        )
    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    generated  = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\n  [{label}]")
    print(f"  PROMPT    : {prompt!r}")
    print(f"  GENERATED : {generated!r}")


def main() -> None:
    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    # ── SECTION 1 : Temperature effect ──────────────────────────────────
    section("EFFECT OF TEMPERATURE  (top_k=50  top_p=0.95)")
    prompt = "The future of artificial intelligence is"

    configs = [
        ("temperature=0.1  (very conservative)", dict(temperature=0.1,  top_k=50, top_p=0.95)),
        ("temperature=0.7  (balanced)",          dict(temperature=0.7,  top_k=50, top_p=0.95)),
        ("temperature=1.2  (very creative)",     dict(temperature=1.2,  top_k=50, top_p=0.95)),
    ]

    for label, kwargs in configs:
        for i in range(REPEAT_PER_CONFIG):
            generate(model, tokenizer, prompt, f"{label} | run {i+1}", **kwargs)

    # ── SECTION 2 : Top-k effect ─────────────────────────────────────────
    section("EFFECT OF TOP-K  (temperature=0.7  top_p=1.0)")
    prompt = "In the distant future, robots will"

    for k in [1, 10, 50, 200]:
        generate(model, tokenizer, prompt, f"top_k={k}", temperature=0.7, top_k=k, top_p=1.0)

    # ── SECTION 3 : Top-p (nucleus sampling) effect ──────────────────────
    section("EFFECT OF TOP-P / NUCLEUS SAMPLING  (temperature=0.7  top_k=0)")
    prompt = "Scientists recently discovered that"

    for p in [0.1, 0.5, 0.9, 1.0]:
        generate(model, tokenizer, prompt, f"top_p={p}", temperature=0.7, top_k=0, top_p=p)

    # ── SECTION 4 : Prompt engineering – instruction style ───────────────
    section("PROMPT ENGINEERING – Instruction Style")
    prompts = [
        "Write a haiku about the ocean:\n",
        "List three benefits of exercise:\n1.",
        "Translate 'I love programming' to French:\n",
        "Q: What is the capital of France?\nA:",
    ]
    for prompt in prompts:
        generate(model, tokenizer, prompt, "instruction prompt", temperature=0.7, top_k=50, top_p=0.9)

    # ── SECTION 5 : Few-shot prompting ───────────────────────────────────
    section("PROMPT ENGINEERING – Few-Shot Examples")
    few_shot_prompt = (
        "Sentiment: positive\nReview: I love this product!\n\n"
        "Sentiment: negative\nReview: This is the worst purchase I've made.\n\n"
        "Sentiment: positive\nReview: Absolutely amazing experience!\n\n"
        "Sentiment:"
    )
    for i in range(REPEAT_PER_CONFIG):
        generate(model, tokenizer, few_shot_prompt, f"few-shot | run {i+1}", temperature=0.5, top_k=10, top_p=0.9)

    # ── Observations ─────────────────────────────────────────────────────
    section("KEY OBSERVATIONS")
    print("""
  Temperature
  ───────────
  • Low temperature (0.1): nearly deterministic, repetitive, "safe" word choices.
  • Medium temperature (0.7): balanced – readable and varied.
  • High temperature (1.2): creative but prone to grammatical errors and
    incoherent tangents. Beyond 1.5 outputs often become gibberish.

  Top-k
  ─────
  • top_k=1 is greedy decoding – always picks the single most likely token.
    Very repetitive ("the the the…").
  • Larger k lets the model sample from a broader vocabulary, improving
    naturalness but also randomness.

  Top-p (Nucleus Sampling)
  ────────────────────────
  • p=0.1: only the top 10% probability mass is sampled → very conservative.
  • p=0.9: the "sweet spot" for most tasks – good diversity without chaos.
  • p=1.0: sample from the entire distribution (essentially no top-p filter).

  Prompt Engineering
  ──────────────────
  • Instruction-style prompts (e.g., "Write a haiku:") guide the model
    even without fine-tuning, but GPT-2 (small, 124 M params) is not
    instruction-tuned so results are inconsistent.
  • Few-shot prompts dramatically improve structured output quality.
""")


if __name__ == "__main__":
    main()
