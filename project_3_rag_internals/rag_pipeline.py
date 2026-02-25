"""
Step 5: RAG Pipeline – RAG vs No-RAG Evaluation
Compares LLM answers with and without retrieved context.
Uses Ollama (local, free) as the LLM backend with graceful fallback.
"""

import json
import os
import sys
import time
import textwrap
from datetime import datetime

from embedder import load_index, get_embedding_model
from retriever import retrieve, build_context, print_results

EVAL_QUESTIONS_FILE = "eval_questions.json"
RESULTS_DIR = "eval_results"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # Change to your preferred model (mistral, phi3, etc.)


# ── Prompt Templates ─────────────────────────────────────────────────────────

def build_no_rag_prompt(question: str) -> str:
    """Pure LLM prompt — no external context provided."""
    return f"""You are a helpful AI assistant. Answer the following question to the best of your knowledge.
Be specific and concise. If you don't know, say "I don't know."

Question: {question}

Answer:"""


def build_rag_prompt(question: str, context: str) -> str:
    """RAG prompt — model must answer strictly from provided context."""
    return f"""You are a helpful AI assistant. Answer the question using ONLY the context provided below.
Do NOT use any knowledge outside of this context.
If the answer is not in the context, say "The context does not contain enough information to answer this question."

Context:
---
{context}
---

Question: {question}

Answer:"""


# ── LLM Backend (Ollama) ─────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 120) -> str:
    """
    Call Ollama local API.
    Returns the model's response string, or an error message if unavailable.
    """
    try:
        import requests
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,        # deterministic for reproducibility
                "num_predict": 300,      # max tokens in response
            }
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        else:
            return f"[OLLAMA ERROR: HTTP {resp.status_code}]"
    except Exception as e:
        # Ollama not running — return mock response so evaluation still runs
        return (
            f"[OLLAMA NOT AVAILABLE: {e}]\n"
            f"To enable real LLM answers:\n"
            f"  1. Install Ollama from https://ollama.com\n"
            f"  2. Run: ollama pull {model}\n"
            f"  3. Run: ollama serve\n"
            f"  4. Then re-run this script."
        )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_questions(
    questions: list[dict],
    index,
    metadata: list[dict],
    embedding_model,
    top_k: int = 3,
    index_label: str = "medium",
) -> list[dict]:
    """
    For each question: get no-RAG answer and RAG answer.

    Returns list of result dicts with both answers.
    """
    results = []

    for i, q in enumerate(questions, start=1):
        question = q['question']
        expected_in_docs = q.get('answerable_from_docs', True)
        topic = q.get('topic', '')

        print(f"\n[{i}/{len(questions)}] {question}")
        print(f"  Topic: {topic} | In docs: {expected_in_docs}")

        # ── No-RAG ──
        no_rag_prompt = build_no_rag_prompt(question)
        print("  Calling LLM (no-RAG)...", end=" ", flush=True)
        t0 = time.time()
        no_rag_answer = call_ollama(no_rag_prompt)
        no_rag_time = round(time.time() - t0, 2)
        print(f"done ({no_rag_time}s)")

        # ── Retrieval ──
        retrieved = retrieve(question, index, metadata, embedding_model, top_k=top_k)
        context = build_context(retrieved, max_chunks=top_k)

        # ── RAG ──
        rag_prompt = build_rag_prompt(question, context)
        print("  Calling LLM (RAG)...", end=" ", flush=True)
        t0 = time.time()
        rag_answer = call_ollama(rag_prompt)
        rag_time = round(time.time() - t0, 2)
        print(f"done ({rag_time}s)")

        result = {
            'question_id':         i,
            'question':            question,
            'topic':               topic,
            'answerable_from_docs': expected_in_docs,
            'no_rag_answer':       no_rag_answer,
            'rag_answer':          rag_answer,
            'retrieved_chunks':    [
                {
                    'rank':     r['rank'],
                    'score':    round(r['score'], 4),
                    'doc_id':   r['doc_id'],
                    'chunk_idx': r['chunk_idx'],
                    'snippet':  r['snippet'],
                }
                for r in retrieved
            ],
            'index_label':         index_label,
            'top_k':               top_k,
            'timing': {
                'no_rag_seconds': no_rag_time,
                'rag_seconds':    rag_time,
            },
            # Manual labeling fields (fill in after reviewing results)
            'labels': {
                'no_rag_correct':      None,
                'no_rag_hallucinated': None,
                'rag_correct':         None,
                'rag_hallucinated':    None,
            },
        }
        results.append(result)

    return results


def print_comparison(results: list[dict]):
    """Print a readable side-by-side comparison of results."""
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON: No-RAG vs RAG")
    print("=" * 70)

    for r in results:
        print(f"\nQ{r['question_id']}: {r['question']}")
        print(f"  [In docs: {r['answerable_from_docs']}]")
        print()

        print("  NO-RAG ANSWER:")
        for line in textwrap.wrap(r['no_rag_answer'][:600], width=60):
            print(f"    {line}")

        print()
        print("  RAG ANSWER:")
        for line in textwrap.wrap(r['rag_answer'][:600], width=60):
            print(f"    {line}")

        print()
        print("  TOP RETRIEVED SOURCES:")
        for chunk in r['retrieved_chunks']:
            print(f"    #{chunk['rank']} (score={chunk['score']:.3f}): {chunk['doc_id']} chunk {chunk['chunk_idx']}")

        print("\n  " + "-" * 65)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 5: RAG vs No-RAG Evaluation")
    print("=" * 70)

    # Load eval questions
    with open(EVAL_QUESTIONS_FILE, 'r') as f:
        questions = json.load(f)
    print(f"\nLoaded {len(questions)} evaluation questions.")

    # Load embedding model
    embedding_model = get_embedding_model()

    # Run evaluation for each chunk size index
    configs = ['small', 'medium']

    for config_label in configs:
        print(f"\n{'═'*70}")
        print(f"  Evaluating with index: {config_label.upper()}")
        print(f"{'═'*70}")

        try:
            index, metadata = load_index(config_label)
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping {config_label}: {e}")
            continue

        results = evaluate_questions(
            questions=questions,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            top_k=3,
            index_label=config_label,
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(RESULTS_DIR, f"results_{config_label}_{timestamp}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  ✅ Results saved: {out_file}")

        # Print comparison
        print_comparison(results)

    print("\n🏁 Evaluation complete for all configs.")
    print(f"   Results saved in: {RESULTS_DIR}/")
    print("   Open the JSON files and fill in the 'labels' fields to track accuracy/hallucination.")


if __name__ == "__main__":
    main()
