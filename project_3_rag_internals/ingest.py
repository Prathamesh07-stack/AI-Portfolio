"""
Step 1: Document Ingestion
Loads all .md and .txt files from the docs/ folder,
normalizes whitespace, and returns structured document dicts.
"""

import os
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    """Clean and normalize raw text from documents."""
    # Remove markdown headers (keep the text, lose the #)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove markdown bold/italic markers
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    # Remove inline code backticks
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    # Remove markdown table separators
    text = re.sub(r'\|[-:]+\|', '', text)
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace 3+ newlines with 2 newlines (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    # Remove empty lines caused by header removal
    text = '\n'.join(line for line in lines if line)
    return text.strip()


def load_documents(docs_dir: str = "../docs") -> list[dict]:
    """
    Load all .md and .txt files from docs_dir.

    Returns:
        List of dicts: {
            'doc_id': str,        # filename without extension
            'filename': str,      # full filename
            'filepath': str,      # full path
            'raw_text': str,      # original text
            'text': str,          # normalized text
            'word_count': int,    # word count of normalized text
        }
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents = []
    supported_extensions = ['.md', '.txt']

    for filepath in sorted(docs_path.iterdir()):
        if filepath.suffix.lower() not in supported_extensions:
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        normalized = normalize_text(raw_text)
        word_count = len(normalized.split())

        doc = {
            'doc_id': filepath.stem,
            'filename': filepath.name,
            'filepath': str(filepath),
            'raw_text': raw_text,
            'text': normalized,
            'word_count': word_count,
        }
        documents.append(doc)

    return documents


def main():
    """Run ingestion and print stats."""
    print("=" * 60)
    print("STEP 1: Document Ingestion")
    print("=" * 60)

    docs = load_documents("../docs")

    print(f"\nLoaded {len(docs)} documents:\n")
    total_words = 0
    for doc in docs:
        print(f"  [{doc['doc_id']}]")
        print(f"    File      : {doc['filename']}")
        print(f"    Words     : {doc['word_count']:,}")
        print()
        total_words += doc['word_count']

    print(f"Total words across all docs: {total_words:,}")
    print("\n✅ Ingestion complete.")
    return docs


if __name__ == "__main__":
    main()
