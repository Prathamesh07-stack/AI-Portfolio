"""
Experiment 2 – Embedding Similarity
Use sentence-transformers (all-MiniLM-L6-v2) to compute cosine similarity
between word pairs and sentence pairs and display the results in a table.
"""

from sentence_transformers import SentenceTransformer, util
import torch


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────
# Pairs to compare
# ─────────────────────────────────────────
WORD_PAIRS = [
    ("dog", "puppy"),
    ("dog", "car"),
    ("king", "queen"),
    ("bank", "river"),
    ("apple", "orange"),
    ("python", "snake"),
]

SENTENCE_PAIRS = [
    (
        "The weather is nice today.",
        "It's a beautiful sunny day.",
    ),
    (
        "The weather is nice today.",
        "Stock markets crashed yesterday.",
    ),
    (
        "I enjoy programming in Python.",
        "Coding in Python is fun for me.",
    ),
    (
        "I enjoy programming in Python.",
        "The Eiffel Tower is in Paris.",
    ),
    (
        "She went to the bank to deposit money.",
        "The river bank was covered in mud.",
    ),
]


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def cosine_sim(model: SentenceTransformer, a: str, b: str) -> float:
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b).item())


def bar(score: float, width: int = 20) -> str:
    """Simple ASCII progress bar for the score."""
    filled = round(score * width)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def print_table(title: str, pairs: list[tuple[str, str]], model: SentenceTransformer) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Text A':<35} {'Text B':<35} {'Score':>6}  Visual")
    print(f"  {'-' * 35} {'-' * 35} {'-' * 6}  {'─' * 22}")
    for a, b in pairs:
        score = cosine_sim(model, a, b)
        a_disp = (a[:33] + "…") if len(a) > 35 else a
        b_disp = (b[:33] + "…") if len(b) > 35 else b
        print(f"  {a_disp:<35} {b_disp:<35} {score:>6.4f}  {bar(score)}")


def interpret(score: float) -> str:
    if score > 0.85:
        return "Very High – nearly identical meaning"
    elif score > 0.65:
        return "High – semantically similar"
    elif score > 0.40:
        return "Moderate – somewhat related"
    elif score > 0.20:
        return "Low – loosely related"
    else:
        return "Very Low – unrelated"


def main() -> None:
    print(f"\nLoading model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    print_table("WORD-LEVEL COSINE SIMILARITY", WORD_PAIRS, model)
    print_table("SENTENCE-LEVEL COSINE SIMILARITY", SENTENCE_PAIRS, model)

    # ── Interpretation section ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  INTERPRETATION")
    print(f"{'=' * 70}")
    print("""
  Cosine similarity ranges from -1 (opposite) to +1 (identical direction).

  Key takeaways:
  • "dog" / "puppy"  → high score: model learned they refer to the same animal.
  • "dog" / "car"    → low score: semantically unrelated concepts.
  • "king" / "queen" → high-ish: related role, but not identical meaning.
  • Sentences about the same topic cluster much closer than unrelated ones.
  • Context matters: "bank" + money ≠ "bank" + river — the model partially
    captures polysemy but embeddings are context-free (sentence-level pooled),
    so subtle disambiguation is limited compared to per-token representations.
""")

    # ── Embedding shape demo ─────────────────────────────────────────────
    print(f"{'=' * 70}")
    print("  EMBEDDING VECTOR SHAPE DEMO")
    print(f"{'=' * 70}")
    sample_emb = model.encode("Hello world", convert_to_tensor=True)
    print(f"\n  model.encode('Hello world') → tensor of shape {tuple(sample_emb.shape)}")
    print(f"  First 10 values: {sample_emb[:10].tolist()}\n")


if __name__ == "__main__":
    main()
