"""
Experiment 3 – Attention Visualization
Load BERT with output_attentions=True and use bertviz to generate
interactive HTML files showing head-level and model-level attention.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view, model_view


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
SENTENCE   = "The cat sat on the mat."
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def save_html(view_html: str, filename: str) -> str:
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(view_html)
    return path


def main() -> None:
    # ── Load model & tokenizer ──────────────────────────────────────────
    section("Loading Model")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Sentence : '{SENTENCE}'")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()

    # ── Tokenize ────────────────────────────────────────────────────────
    inputs  = tokenizer(SENTENCE, return_tensors="pt")
    tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    section("Tokens")
    print(f"  {tokens}")

    # ── Forward pass ────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions   # tuple: (layer, batch, heads, seq, seq)

    # ── Attention tensor info ────────────────────────────────────────────
    section("Attention Tensor Shape")
    print(f"  Number of layers  : {len(attentions)}")
    print(f"  Per-layer shape   : {tuple(attentions[0].shape)}")
    print(f"  → (batch=1, heads={attentions[0].shape[1]}, seq_len={attentions[0].shape[2]}, seq_len={attentions[0].shape[3]})")
    print(f"  Sentence tokens   : {tokens}")
    print(f"  Sequence length   : {len(tokens)} (includes [CLS] and [SEP])")

    # ── bertviz: Head View ───────────────────────────────────────────────
    section("Generating Head View HTML")
    try:
        html_head = head_view(attentions, tokens, html_action="return")
        path_head = save_html(html_head.data, "attention_head_view.html")
        print(f"  ✓ Saved: {path_head}")
    except Exception as e:
        print(f"  ⚠ head_view failed: {e}")

    # ── bertviz: Model View ──────────────────────────────────────────────
    section("Generating Model View HTML")
    try:
        html_model = model_view(attentions, tokens, html_action="return")
        path_model = save_html(html_model.data, "attention_model_view.html")
        print(f"  ✓ Saved: {path_model}")
    except Exception as e:
        print(f"  ⚠ model_view failed: {e}")

    # ── Quantitative observations ────────────────────────────────────────
    section("Key Attention Observations")

    # Compute per-layer mean attention matrix (averaged over heads & batch)
    for layer_idx, attn_layer in enumerate(attentions):
        # attn_layer: (1, num_heads, seq, seq)
        mean_attn = attn_layer[0].mean(dim=0)  # (seq, seq)
        # Which token does [CLS] (index 0) attend to most?
        top_idx = mean_attn[0].argmax().item()
        print(f"  Layer {layer_idx:2d} | [CLS] attends most to → '{tokens[top_idx]}' (avg over heads)")

    print("""
  General patterns to look for in the HTML viewer:
  ─────────────────────────────────────────────────
  • Early layers  : broad / diffuse attention (attending to nearby words).
  • Middle layers : syntactic patterns (subject ↔ verb, adjective → noun).
  • Deep layers   : semantic / task-specific attention.
  • [CLS] token   : aggregates sentence-level representation.
  • [SEP] token   : often receives a lot of attention as a "no-op" sink.
  • "sat" likely attends strongly to "cat" (subject-verb relationship).
  • "mat" likely attends to "on" and "the" (prepositional phrase structure).

  Open the saved HTML files in a browser to interactively explore heads.
""")


if __name__ == "__main__":
    main()
