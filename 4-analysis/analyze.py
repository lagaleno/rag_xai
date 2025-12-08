from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd

# ===================== CONFIG =====================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # sobe até a raiz do projeto

# Arquivos de entrada
COSINE_RESULTS = PROJECT_ROOT / "3-metrics" / "cosine_similarity" / "cosine_similarity_results.csv"
LOGIC_SUMMARY_LONG = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "logical_metrics_summary_long.csv"
LLM_JUDGE_SUMMARY = PROJECT_ROOT / "3-metrics" / "llm_judge" / "llm_judge_summary.csv"

# Identificador do experimento (apenas para nomear figuras/tabelas)
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID") or "noid"

# Saídas
FIGURES_DIR = THIS_FILE.parent / "figures" / EXPERIMENT_ID
TABLES_DIR = THIS_FILE.parent / "tables" / EXPERIMENT_ID

# Labels padrão (para ordenar e garantir consistência)
LABELS = ["correct", "incomplete", "incorrect"]

# ==================================================


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ===================== COSINE SIMILARITY =====================

def load_cosine_results() -> pd.DataFrame:
    """
    Lê cosine_similarity_results.csv
    Formato esperado: dataset_id,label,chunk_text,explanation_text,cosine_similarity
    """
    if not COSINE_RESULTS.exists():
        print(f"⚠️ Cosine results not found at {COSINE_RESULTS}")
        return pd.DataFrame()
    df = pd.read_csv(COSINE_RESULTS)
    return df


def plot_cosine_boxplot(df: pd.DataFrame):
    """
    Boxplot da distribuição de cosine_similarity por label.
    Dá uma visão geral de como a similaridade se comporta por tipo de explicação.
    """
    if df.empty:
        print("⚠️ Empty cosine DataFrame, skipping cosine boxplot.")
        return

    if "label" not in df.columns or "cosine_similarity" not in df.columns:
        print("⚠️ Columns 'label' or 'cosine_similarity' not found in cosine results.")
        return

    labels_present = sorted(df["label"].unique())
    data = [df[df["label"] == lab]["cosine_similarity"].dropna().values for lab in labels_present]

    plt.figure()
    plt.boxplot(data, labels=labels_present)
    plt.ylabel("Cosine similarity")
    plt.title(f"Cosine similarity distribution by label (Exp: {EXPERIMENT_ID})")
    plt.tight_layout()

    out = FIGURES_DIR / f"cosine_boxplot_ex{EXPERIMENT_ID}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved cosine boxplot: {out}")


# ===================== CONFUSÃO E ACURÁCIA (MÉTRICA GERAL) =====================

def load_confusion_long(path: Path) -> pd.DataFrame:
    """
    Lê um CSV no formato "long":
      true_label,pred_label,count

    Retorna:
      - DataFrame long
    """
    if not path.exists():
        print(f"⚠️ Confusion-long CSV not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    expected_cols = {"true_label", "pred_label", "count"}
    if not expected_cols.issubset(df.columns):
        print(f"⚠️ Missing columns in {path}: expected {expected_cols}, got {set(df.columns)}")
        return pd.DataFrame()
    return df


def pivot_confusion(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DF long (true_label, pred_label, count)
    e devolve uma matriz (wide) com:
      - index = true_label
      - columns = pred_label
    Garantindo presença de LABELS nas linhas/colunas (preenchendo com 0).
    """
    if df_long.empty:
        return pd.DataFrame()

    conf = df_long.pivot(
        index="true_label",
        columns="pred_label",
        values="count",
    )

    # garante as labels padrão (se existirem outras, mantemos também)
    all_rows = sorted(set(conf.index).union(LABELS))
    all_cols = sorted(set(conf.columns).union(LABELS))

    conf = conf.reindex(index=all_rows, columns=all_cols, fill_value=0)

    # restringe às 3 labels de interesse, se estiverem presentes
    conf = conf.reindex(index=LABELS, columns=LABELS, fill_value=0)

    return conf


def plot_confusion_matrix(confusion_df: pd.DataFrame, title: str, out_path: Path):
    """
    Plota matriz de confusão usando apenas matplotlib.

    confusion_df deve ter:
      - index: true labels
      - columns: predicted labels
    """
    if confusion_df.empty:
        print("⚠️ Empty confusion matrix, skipping plot.")
        return

    true_labels = confusion_df.index.tolist()
    pred_labels = confusion_df.columns.tolist()
    matrix = confusion_df.values

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)  # sem cmap explícito (usa padrão)
    plt.colorbar()
    plt.xticks(range(len(pred_labels)), pred_labels, rotation=45, ha="right")
    plt.yticks(range(len(true_labels)), true_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    # valores nas células
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved confusion matrix plot: {out_path}")


def compute_accuracy_by_label(confusion_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir da matriz de confusão (true x pred), calcula acurácia por label:

      accuracy(label) = conf[label,label] / sum(conf[label,*])

    Retorna DF com colunas: label, accuracy, total
    """
    rows = []
    for lab in LABELS:
        if lab not in confusion_df.index:
            rows.append({"label": lab, "accuracy": 0.0, "total": 0})
            continue
        row = confusion_df.loc[lab]
        total = row.sum()
        hit = row.get(lab, 0)
        acc = float(hit) / float(total) if total > 0 else 0.0
        rows.append({"label": lab, "accuracy": acc, "total": int(total)})

    return pd.DataFrame(rows)


def plot_accuracy_bar(acc_df: pd.DataFrame, title: str, out_path: Path):
    """
    Gráfico de barras da acurácia por label.
    """
    if acc_df.empty:
        print("⚠️ Empty accuracy DataFrame, skipping plot.")
        return

    labels = acc_df["label"].tolist()
    values = acc_df["accuracy"].tolist()

    x = range(len(labels))
    plt.figure()
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved accuracy barplot: {out_path}")


# ===================== LÓGICA vs LLM-JUDGE (ACURÁCIA) =====================

def plot_logic_vs_judge_accuracy(logic_acc: pd.DataFrame, judge_acc: pd.DataFrame):
    """
    Compara as acurácias por label da métrica lógica e do LLM-judge
    num gráfico de barras agrupadas.
    """
    if logic_acc.empty or judge_acc.empty:
        print("⚠️ Empty accuracy DF for logic or judge, skipping comparison plot.")
        return

    # garantir que ambas tenham as mesmas labels na mesma ordem
    merged = pd.merge(
        logic_acc[["label", "accuracy"]],
        judge_acc[["label", "accuracy"]],
        on="label",
        how="inner",
        suffixes=("_logic", "_judge"),
    )

    if merged.empty:
        print("⚠️ No overlapping labels between logic and judge accuracy.")
        return

    labels = merged["label"].tolist()
    acc_logic = merged["accuracy_logic"].tolist()
    acc_judge = merged["accuracy_judge"].tolist()

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], acc_logic, width, label="Logic metric")
    plt.bar([i + width / 2 for i in x], acc_judge, width, label="LLM judge")

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Logic metric vs LLM-judge – Accuracy by label (Exp: {EXPERIMENT_ID})")
    plt.legend()
    plt.tight_layout()

    out = FIGURES_DIR / f"logic_vs_judge_accuracy_ex{EXPERIMENT_ID}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved logic vs judge accuracy comparison: {out}")


# ===================== MAIN =====================

def main():
    print("=== Starting analysis ===")
    ensure_dirs()

    # ----- 1) Cosine similarity -----
    cosine_df = load_cosine_results()
    if not cosine_df.empty:
        plot_cosine_boxplot(cosine_df)

    # ----- 2) Logic metric (dataset vs lógica) -----
    logic_long = load_confusion_long(LOGIC_SUMMARY_LONG)
    logic_conf = pivot_confusion(logic_long)
    if not logic_conf.empty:
        # salvar matriz wide
        logic_conf_csv = TABLES_DIR / f"logic_confusion_ex{EXPERIMENT_ID}.csv"
        logic_conf.to_csv(logic_conf_csv)
        print(f"✅ Saved logic confusion matrix CSV: {logic_conf_csv}")
        print("\n📊 Logic confusion matrix:")
        print(logic_conf)

        # figura da matriz
        logic_conf_fig = FIGURES_DIR / f"logic_confusion_ex{EXPERIMENT_ID}.png"
        plot_confusion_matrix(logic_conf, "Logic metric – Confusion matrix", logic_conf_fig)

        # acurácia por label
        logic_acc_df = compute_accuracy_by_label(logic_conf)
        logic_acc_csv = TABLES_DIR / f"logic_accuracy_by_label_ex{EXPERIMENT_ID}.csv"
        logic_acc_df.to_csv(logic_acc_csv, index=False)
        print("\n📈 Logic accuracy by label:")
        print(logic_acc_df)
        print(f"✅ Saved logic accuracy CSV: {logic_acc_csv}")

        logic_acc_fig = FIGURES_DIR / f"logic_accuracy_by_label_ex{EXPERIMENT_ID}.png"
        plot_accuracy_bar(
            logic_acc_df,
            f"Logic metric – Accuracy by label (Exp: {EXPERIMENT_ID})",
            logic_acc_fig,
        )
    else:
        logic_acc_df = pd.DataFrame()

    # ----- 3) LLM-judge (dataset vs juiz) -----
    judge_long = load_confusion_long(LLM_JUDGE_SUMMARY)
    judge_conf = pivot_confusion(judge_long)
    if not judge_conf.empty:
        judge_conf_csv = TABLES_DIR / f"llm_judge_confusion_ex{EXPERIMENT_ID}.csv"
        judge_conf.to_csv(judge_conf_csv)
        print(f"\n✅ Saved LLM judge confusion matrix CSV: {judge_conf_csv}")
        print("\n📊 LLM judge confusion matrix:")
        print(judge_conf)

        judge_conf_fig = FIGURES_DIR / f"llm_judge_confusion_ex{EXPERIMENT_ID}.png"
        plot_confusion_matrix(judge_conf, "LLM-judge – Confusion matrix", judge_conf_fig)

        judge_acc_df = compute_accuracy_by_label(judge_conf)
        judge_acc_csv = TABLES_DIR / f"llm_judge_accuracy_by_label_ex{EXPERIMENT_ID}.csv"
        judge_acc_df.to_csv(judge_acc_csv, index=False)
        print("\n📈 LLM judge accuracy by label:")
        print(judge_acc_df)
        print(f"✅ Saved LLM judge accuracy CSV: {judge_acc_csv}")

        judge_acc_fig = FIGURES_DIR / f"llm_judge_accuracy_by_label_ex{EXPERIMENT_ID}.png"
        plot_accuracy_bar(
            judge_acc_df,
            f"LLM-judge – Accuracy by label (Exp: {EXPERIMENT_ID})",
            judge_acc_fig,
        )
    else:
        judge_acc_df = pd.DataFrame()

    # ----- 4) Comparação lógica vs LLM-judge -----
    if not logic_acc_df.empty and not judge_acc_df.empty:
        plot_logic_vs_judge_accuracy(logic_acc_df, judge_acc_df)

    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
