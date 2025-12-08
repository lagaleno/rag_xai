import os
import json

import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
# Import functions from utils file
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
METRICS_ROOT = THIS_FILE.parents[1]  # 3-metrics/
sys.path.append(str(METRICS_ROOT))
from utils import build_examples, flatten_examples

# ==== IMPORT PROVENANCE ====
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from provenance import ProvenanceDB  # noqa: E402


# ================== CONFIGURAÇÕES ==================

PROJECT_ROOT = THIS_FILE.parents[2]

# Caminho para o dataset JSONL com as explicações
JSONL_FILE = PROJECT_ROOT / "1-creating_dataset" / "explainrag_hotpot_llama.jsonl"

# Saídas
CSV_OUT = PROJECT_ROOT / "3-metrics" / "cosine_similarity" / "cosine_similarity_results.csv"
SUMMARY_OUT = PROJECT_ROOT / "3-metrics" / "cosine_similarity" / "cosine_similarity_summary_by_label.csv"

# Modelo de embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ===================================================


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

    # Load embedding model
    print(f"🔢 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build aligned lists of chunks and explanations
    chunks = []
    explanations = []
    for item in rows:
        chunks.append(item["chunk_text"])
        explanations.append(item["explanation_text"])

    print("🧮 Encoding chunks...")
    emb_chunks = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)

    print("🧮 Encoding explanations...")
    emb_expls = model.encode(explanations, convert_to_tensor=True, show_progress_bar=True)

    # Cosine similarity between chunk[i] and explanation[i]
    print("📏 Computing cosine similarities...")
    sim_matrix = util.cos_sim(emb_chunks, emb_expls)
    sims = sim_matrix.diagonal().cpu().tolist()

    # Build DataFrame
    df = pd.DataFrame(rows)
    df["cosine_similarity"] = sims

    # Save detailed results
    df.to_csv(CSV_OUT, index=False)
    print(f"✅ Cosine similarity results saved to: {CSV_OUT}")

    # Summary by label
    summary = df.groupby("label")["cosine_similarity"].agg(["mean", "std", "count"])
    summary.to_csv(SUMMARY_OUT)
    print(f"✅ Summary by label saved to: {SUMMARY_OUT}")
    print("\n📊 Cosine similarity summary by label:")
    print(summary)

    # ========= PROVENIÊNCIA: salvar cada linha =========

    experiment_id_env = os.getenv("EXPERIMENT_ID")
    if experiment_id_env is None:
        print("⚠️ EXPERIMENT_ID not found in environment. Skipping Cosine provenance.")
        return

    experiment_id = int(experiment_id_env)
    prov = ProvenanceDB()

    inserted = 0
    for _, row in df.iterrows():
        sample_id = row.get("dataset_id")
        label = row.get("label", "")
        cosine = row.get("cosine_similarity", 0.0)

        metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "chunk_len": len(str(row.get("chunk_text", ""))),
            "expl_len": len(str(row.get("explanation_text", ""))),
        }

        prov.insert_cosine_result(
            experiment_id=experiment_id,
            sample_id=str(sample_id),
            label=str(label),
            cosine=float(cosine),
            metadata=metadata,
        )
        inserted += 1

    print(f"🧮 Cosine results registrados no banco: {inserted} linhas.")
    prov.close()

if __name__ == "__main__":
    main()