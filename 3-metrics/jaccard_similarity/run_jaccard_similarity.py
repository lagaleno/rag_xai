import os
import re
import pandas as pd
from tqdm.auto import tqdm

# Import functions from utils file
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]  # 3-experiments/
sys.path.append(str(EXPERIMENTS_ROOT))
from utils import build_examples, flatten_examples

# ================== CONFIGURAÇÕES ==================

# Path to the JSONL dataset with explanations
JSONL_FILE = "../../1-creating_dataset/explainrag_hotpot_llama.jsonl"

# Outputs
CSV_OUT = "jaccard_similarity_results.csv"
SUMMARY_OUT = "jaccard_similarity_summary_by_label.csv"

# ===================================================


def tokenize(text):
    """
    Simple tokenization:
    - lowercases
    - keeps only alphanumeric 'words'
    - returns a set of tokens
    """
    text = text.lower()
    tokens = re.findall(r"\w+", text)
    return set(tokens)


def jaccard_similarity(text_a, text_b):
    """
    Jaccard similarity between two texts.
    If both sets are empty, returns 0.0.
    """
    set_a = tokenize(text_a)
    set_b = tokenize(text_b)

    if not set_a and not set_b:
        return 0.0

    inter = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.0

    return len(inter) / len(union)


def main():
    if not os.path.exists(JSONL_FILE):
        raise FileNotFoundError(f"JSONL file not found: {JSONL_FILE}")

    print(f"📥 Loading dataset from: {JSONL_FILE}")
    examples = build_examples(JSONL_FILE)
    print(f"Total examples loaded: {len(examples)}")

    if not examples:
        print("No examples found. Check the JSONL format.")
        return

    # Flatten examples into chunk–explanation pairs
    rows = flatten_examples(examples)
    print(f"Total chunk–explanation pairs: {len(rows)}")

    if not rows:
        print("No chunk–explanation pairs found. Nothing to evaluate.")
        return

    # Compute Jaccard similarity for each pair
    jaccards = []
    print("📏 Computing Jaccard similarities...")
    for item in tqdm(rows):
        sim = jaccard_similarity(item["chunk_text"], item["explanation_text"])
        jaccards.append(sim)

    # Build DataFrame
    df = pd.DataFrame(rows)
    df["jaccard_similarity"] = jaccards

    # Save detailed results
    df.to_csv(CSV_OUT, index=False)
    print(f"✅ Jaccard similarity results saved to: {CSV_OUT}")

    # Summary by label
    summary = df.groupby("label")["jaccard_similarity"].agg(["mean", "std", "count"])
    summary.to_csv(SUMMARY_OUT)
    print(f"✅ Summary by label saved to: {SUMMARY_OUT}")
    print("\n📊 Jaccard similarity summary by label:")
    print(summary)


if __name__ == "__main__":
    main()