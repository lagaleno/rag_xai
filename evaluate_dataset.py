import json
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util

# ================== CONFIGURAÇÕES ==================

# Caminho do seu dataset JSONL com as explicações
# Estrutura esperada de cada linha:
# {
#   "id": "...",
#   "chunk": {"text": "...", ...},
#   "explanations": [
#       {"text": "...", "label": "correct"},
#       {"text": "...", "label": "incomplete"},
#       {"text": "...", "label": "incorrect"}
#   ],
#   ...
# }
JSONL_FILE = "explainrag_hotpot_llama.jsonl"

# Saídas
CSV_OUT = "explanations_sentencewise_embeddings_metrics.csv"
SUMMARY_OUT = "explanations_sentencewise_embeddings_summary_by_label.csv"
PLOT_F1_BOX = "emb_f1_by_label_boxplot.png"
PLOT_PREC_BOX = "emb_precision_by_label_boxplot.png"
PLOT_REC_BOX = "emb_recall_by_label_boxplot.png"

# Modelo de embeddings (bom, pequeno, rápido)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Threshold de similaridade para considerar que uma sentença está "coberta"
# (0 a 1). 0.7–0.8 costuma ser um bom ponto de partida.
SIM_THRESHOLD = 0.75

# ====================================================


def split_into_sentences(text: str):
    """
    Split simples em sentenças usando pontuação (. ! ?).
    Não é perfeito, mas é suficiente para nosso cenário.
    """
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def compute_sentencewise_prf(chunk: str, explanation: str,
                             model: SentenceTransformer,
                             threshold: float = 0.75):
    """
    Calcula precision, recall, f1 com base em cobertura de sentenças via embeddings.

    - chunk: texto de origem
    - explanation: explicação gerada
    - threshold: similaridade mínima para considerar que há cobertura
    """

    chunk_sents = split_into_sentences(chunk)
    expl_sents = split_into_sentences(explanation)

    # Casos borda
    if not chunk_sents or not expl_sents:
        return 0.0, 0.0, 0.0, len(chunk_sents), len(expl_sents)

    # Embeddings por sentença
    emb_chunk = model.encode(chunk_sents, convert_to_tensor=True)
    emb_expl = model.encode(expl_sents, convert_to_tensor=True)

    # Matriz de similaridade (len_chunk x len_expl)
    sim_matrix = util.cos_sim(emb_chunk, emb_expl)  # tensor

    # Cobertura de sentenças do chunk:
    # para cada sentença do chunk, verifica se alguma da explicação bate >= threshold
    chunk_covered = (sim_matrix >= threshold).any(dim=1).cpu().numpy()

    # Ancoragem de sentenças da explicação:
    # para cada sentença da explicação, verifica se alguma do chunk bate >= threshold
    expl_grounded = (sim_matrix >= threshold).any(dim=0).cpu().numpy()

    num_chunk = len(chunk_sents)
    num_expl = len(expl_sents)

    covered_chunk = chunk_covered.sum()
    grounded_expl = expl_grounded.sum()

    # Precision: fração de sentenças da explicação que estão ancoradas no chunk
    if num_expl == 0:
        precision = 0.0
    else:
        precision = grounded_expl / num_expl

    # Recall: fração de sentenças do chunk cobertas pela explicação
    if num_chunk == 0:
        recall = 0.0
    else:
        recall = covered_chunk / num_chunk

    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1), num_chunk, num_expl


def load_explanations(jsonl_path):
    """
    Lê o JSONL e extrai:
    - id da instância
    - índice da explicação dentro da instância
    - label (correct / incomplete / incorrect)
    - texto da explicação
    - chunk_text
    """
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            ex_id = ex.get("id", "")

            chunk_text = ex.get("chunk", {}).get("text", "")
            if not chunk_text:
                continue

            for idx, exp in enumerate(ex.get("explanations", [])):
                label = exp.get("label", "")
                text = exp.get("text", "").strip()
                if not text:
                    continue

                records.append({
                    "example_id": ex_id,
                    "exp_index": idx,
                    "label": label,
                    "chunk_text": chunk_text,
                    "explanation_text": text
                })

    return records


def main():
    if not os.path.exists(JSONL_FILE):
        raise FileNotFoundError(f"Arquivo JSONL não encontrado: {JSONL_FILE}")

    print(f"📥 Lendo dataset de: {JSONL_FILE}")
    records = load_explanations(JSONL_FILE)
    print(f"Total de explicações carregadas: {len(records)}")

    if not records:
        print("Nenhuma explicação encontrada. Verifique o arquivo JSONL.")
        return

    print(f"🔢 Carregando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    precisions = []
    recalls = []
    f1s = []
    chunk_lens = []
    expl_lens = []

    for rec in tqdm(records, desc="Calculando métricas sentence-wise (embeddings)"):
        p, r, f1, n_chunk, n_expl = compute_sentencewise_prf(
            rec["chunk_text"],
            rec["explanation_text"],
            model,
            threshold=SIM_THRESHOLD
        )
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        chunk_lens.append(n_chunk)
        expl_lens.append(n_expl)

    # DataFrame com resultados
    df = pd.DataFrame(records)
    df["precision"] = precisions
    df["recall"] = recalls
    df["f1"] = f1s
    df["chunk_num_sentences"] = chunk_lens
    df["expl_num_sentences"] = expl_lens

    # Salva CSV detalhado
    df.to_csv(CSV_OUT, index=False)
    print(f"✅ Resultados detalhados salvos em: {CSV_OUT}")

    # Resumo por label
    summary = df.groupby("label")[["precision", "recall", "f1"]].agg(["mean", "std", "count"])
    summary.to_csv(SUMMARY_OUT)
    print(f"✅ Resumo por label salvo em: {SUMMARY_OUT}")
    print("\n📊 Resumo por label (média / desvio padrão / n):")
    print(summary)

    # ============ Gráficos ============

    # F1 por label
    plt.figure(figsize=(6, 4))
    df.boxplot(column="f1", by="label")
    plt.title("Sentence-wise F1 (embeddings) por label de explicação")
    plt.suptitle("")  # remove título automático do pandas
    plt.xlabel("Label")
    plt.ylabel("F1 (cobertura de sentenças)")
    plt.tight_layout()
    plt.savefig(PLOT_F1_BOX)
    plt.close()
    print(f"📈 Gráfico salvo: {PLOT_F1_BOX}")

    # Precision por label
    plt.figure(figsize=(6, 4))
    df.boxplot(column="precision", by="label")
    plt.title("Sentence-wise Precision (embeddings) por label de explicação")
    plt.suptitle("")
    plt.xlabel("Label")
    plt.ylabel("Precision (sentenças da explicação ancoradas)")
    plt.tight_layout()
    plt.savefig(PLOT_PREC_BOX)
    plt.close()
    print(f"📈 Gráfico salvo: {PLOT_PREC_BOX}")

    # Recall por label
    plt.figure(figsize=(6, 4))
    df.boxplot(column="recall", by="label")
    plt.title("Sentence-wise Recall (embeddings) por label de explicação")
    plt.suptitle("")
    plt.xlabel("Label")
    plt.ylabel("Recall (sentenças do chunk cobertas)")
    plt.tight_layout()
    plt.savefig(PLOT_REC_BOX)
    plt.close()
    print(f"📈 Gráfico salvo: {PLOT_REC_BOX}")

    print("\n✅ Avaliação concluída.")


if __name__ == "__main__":
    main()
