import os
import json

import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
import shutil
# Import functions from utils file
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
METRICS_ROOT = THIS_FILE.parents[1]  # 3-metrics/
sys.path.append(str(METRICS_ROOT))
from utils import build_examples, flatten_examples

# ==== IMPORT PROVENANCE ====
PROJECT_ROOT = THIS_FILE.parents[2]
RECORDS_ROOT = PROJECT_ROOT / "records"
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

    # ============ Proveniência (cosine_similarity) ============

    experiment_id_env = os.environ.get("EXPERIMENT_ID")
    xai_dataset_id_env = os.environ.get("XAI_DATASET_ID")

    if experiment_id_env is not None and xai_dataset_id_env is not None:
        try:
            experiment_id = int(experiment_id_env)
            xai_dataset_id = int(xai_dataset_id_env)

            # Caminho em records/experiments/{experiment_id}/cosine/...
            records_dir = RECORDS_ROOT / "experiments" / str(experiment_id) / "cosine"
            records_dir.mkdir(parents=True, exist_ok=True)

            # Vamos copiar o CSV de resultados detalhados
            cosine_records_rel = f"records/experiments/{experiment_id}/cosine/{CSV_OUT.name}"
            cosine_records_abs = PROJECT_ROOT / cosine_records_rel

            shutil.copy2(CSV_OUT, cosine_records_abs)

            prov = ProvenanceDB()
            prov.insert_cosine_similarity_run(
                experiment_id=experiment_id,
                xai_dataset_id=xai_dataset_id,
                embedding=EMBEDDING_MODEL,
                path=cosine_records_rel,  # path relativo
            )
            prov.close()
            print(f"💾 Cosine similarity registrada no banco para experiment_id={experiment_id}")
        except Exception as e:
            print(f"⚠️ Erro ao registrar cosine_similarity no banco: {e}")
    else:
        print("⚠️ EXPERIMENT_ID ou XAI_DATASET_ID não definido no ambiente; pulando registro de cosine_similarity.")

if __name__ == "__main__":
    main()