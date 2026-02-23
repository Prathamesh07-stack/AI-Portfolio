"""
Experiment 1 – Tokenization
Compare how bert-base-uncased, gpt2, and facebook/opt-125m tokenizers
split the same text into different subword tokens.
"""

from transformers import AutoTokenizer

# ─────────────────────────────────────────
# Models / tokenizers to compare
# ─────────────────────────────────────────
TOKENIZER_NAMES = [
    "bert-base-uncased",
    "gpt2",
    "facebook/opt-125m",  # BPE tokenizer (same family as LLaMA) – no gated access needed
]

# ─────────────────────────────────────────
# Test sentences
# ─────────────────────────────────────────
SENTENCES = {
    "English (simple)":      "I love AI",
    "English (exaggerated)": "I loooove AIs!!!",
    "Marathi":               "मला AI आवडतो",   # "I like AI" in Marathi
}

# ─────────────────────────────────────────
# Header helper
# ─────────────────────────────────────────
def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> None:
    # Load all tokenizers
    tokenizers = {}
    for name in TOKENIZER_NAMES:
        print(f"Loading tokenizer: {name} …")
        tokenizers[name] = AutoTokenizer.from_pretrained(name)

    # ── Vocabulary sizes ────────────────────────────────────────────────
    section("Vocabulary Sizes")
    for name, tok in tokenizers.items():
        vs = tok.vocab_size
        print(f"  {name:<30} vocab_size = {vs:,}")

    # ── Per-sentence breakdown ──────────────────────────────────────────
    for sent_label, sentence in SENTENCES.items():
        section(f"Sentence: '{sentence}'  [{sent_label}]")
        for name, tok in tokenizers.items():
            encoding = tok(sentence, add_special_tokens=False)
            ids      = encoding["input_ids"]
            tokens   = tok.convert_ids_to_tokens(ids)

            print(f"\n  ── Tokenizer: {name}")
            print(f"     Tokens  : {tokens}")
            print(f"     IDs     : {ids}")
            print(f"     Count   : {len(tokens)} token(s)")

    # ── Observations ────────────────────────────────────────────────────
    section("Key Observations")
    print("""
  1. BERT (WordPiece) uses '##' prefixes to mark continuation sub-tokens.
     Rare / misspelled words are split aggressively (e.g., 'loooove' → many pieces).

  2. GPT-2 (BPE) adds a leading Ġ (space marker) before non-first tokens.
     Numbers and punctuation are split differently from BERT.

  3. OPT-125m (BPE, same family as LLaMA) behaves similarly to GPT-2 BPE
     but may produce slightly different splits depending on its vocab.

  4. For Marathi (non-Latin script), all tokenizers fall back to very fine-grained
     Unicode character or byte-level splits, producing many more tokens than the
     equivalent English text – illustrating the multilingual efficiency gap.
""")


if __name__ == "__main__":
    main()
