import glob
from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd

# ===================== CONFIG =====================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # sobe de 5-experiment/ para a raiz do projeto

# Cosine
COSINE_RESULTS = PROJECT_ROOT / "5-experiment" / "cosine_similarity_results.csv"
COSINE_SUMMARY = PROJECT_ROOT / "5-experiment" / "cosine_similarity_summary_by_label.csv"

# Jaccard
JACCARD_RESULTS = PROJECT_ROOT / "5-experiment" / "jaccard_similarity_results.csv"
JACCARD_SUMMARY = PROJECT_ROOT / "5-experiment" / "jaccard_similarity_summary_by_label.csv"

# Logic (vários trials) - summary por trial
LOGIC_SUMMARY_PATTERN = str(
    PROJECT_ROOT
    / "5-experiment"
    / "logical_summary_results_trials_out"
    / "logical_metrics_summary_results_trial*.csv"
)

# Output
EXPERIMENT_ID = str(os.getenv("EXPERIMENT_ID")) or 28
FIGURES_DIR = THIS_FILE.parent / "figures" / EXPERIMENT_ID
OUTPUT_TABLE = THIS_FILE.parent / "table" / EXPERIMENT_ID



# ==================================================


def ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def ensure_table_dir():
    OUTPUT_TABLE.mkdir(parents=True, exist_ok=True)


# ------------ LOADERS DE SUMMARY ------------

def load_cosine_summary() -> pd.DataFrame:
    """
    Lê cosine_similarity_summary_by_label.csv
    Formato esperado: label,mean,std,count
    """
    if not COSINE_SUMMARY.exists():
        print(f"⚠️ No cosine summary at {COSINE_SUMMARY}")
        return pd.DataFrame()

    df = pd.read_csv(COSINE_SUMMARY)
    df["metric"] = "cosine_similarity"
    # Garantir colunas na ordem padrão
    return df[["metric", "label", "mean", "std", "count"]]


def load_jaccard_summary() -> pd.DataFrame:
    """
    Lê jaccard_similarity_summary_by_label.csv
    Formato esperado: label,mean,std,count
    """
    if not JACCARD_SUMMARY.exists():
        print(f"⚠️ No jaccard summary at {JACCARD_SUMMARY}")
        return pd.DataFrame()

    df = pd.read_csv(JACCARD_SUMMARY)
    df["metric"] = "jaccard_similarity"
    return df[["metric", "label", "mean", "std", "count"]]


def load_logic_summary() -> pd.DataFrame:
    """
    Lê TODOS os logical_metrics_summary_results_trial*.csv e
    agrega entre trials.

    Formato de cada arquivo:
      explanation_label,mean,std,count

    Aqui:
      - mean = F1 médio daquele trial para aquele label
      - std e count são internos ao trial (não usamos diretamente)
    """
    files = sorted(glob.glob(LOGIC_SUMMARY_PATTERN))
    if not files:
        print(f"⚠️ No logical summary trial files found with pattern: {LOGIC_SUMMARY_PATTERN}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["trial"] = Path(f).stem 
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    grouped = (
        df_all
        .groupby("explanation_label")
        .agg(
            {
                "mean": "mean",
                "std": "mean",
                "count": "mean"
            }
        )
        .reset_index()
        .rename(columns={"explanation_label": "label"})
    )

    grouped["metric"] = "logic_f1"
    return grouped[["metric", "label", "mean", "std", "count"]]


# ------------ PLOTS ------------

def plot_bar(summary: pd.DataFrame, metric_name: str, file_name: str):
    """
    summary: DataFrame com colunas [label, mean, std]
    """
    if summary.empty:
        return

    labels = summary["label"].tolist()
    means = summary["mean"].tolist()
    stds = summary["std"].tolist()

    x = range(len(labels))
    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} by label")
    plt.tight_layout()

    out = FIGURES_DIR / file_name
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_box(results_path: Path, value_col: str, label_col: str, title: str, file_name: str):
    """
    Boxplot da distribuição dos valores por label
    (usado para cosine/jaccard que têm CSV 'completo').
    """
    if not results_path.exists():
        print(f"⚠️ Results file not found at {results_path}, skipping {file_name}.")
        return

    df = pd.read_csv(results_path)
    if value_col not in df.columns or label_col not in df.columns:
        print(f"⚠️ Columns {value_col} or {label_col} not in {results_path.name}, skipping {file_name}.")
        return

    labels = sorted(df[label_col].unique())
    data = [df[df[label_col] == lab][value_col].dropna().values for lab in labels]

    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()

    out = FIGURES_DIR / file_name
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

def plot_metric_comparison_per_label(wide_df: pd.DataFrame):
    """
    Cria 3 gráficos: correct / incomplete / incorrect
    Comparando cosine, jaccard e logic (lado a lado)
    """
    labels = wide_df["label"].tolist()

    for label in labels:
        row = wide_df[wide_df["label"] == label].iloc[0]

        metrics = ["Cosine Similarity", "Jaccard Similarity", "Logical F1"]
        means = [
            row["cosine_similarity_mean"],
            row["jaccard_similarity_mean"],
            row["logic_f1_mean"],
        ]
        stds = [
            row["cosine_similarity_std"],
            row["jaccard_similarity_std"],
            row["logic_f1_std"],
        ]

        plt.figure(figsize=(6, 4))
        x = range(len(metrics))
        plt.bar(x, means, yerr=stds, capsize=6)
        plt.xticks(x, metrics, rotation=20)
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title(f"Metric comparison – {label}")
        plt.tight_layout()

        out = FIGURES_DIR / f"{label}_metric_comparison_ex{EXPERIMENT_ID}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved per-label comparison: {out}")

def plot_overall_metric_comparison(wide_df: pd.DataFrame):
    """
    Gera um único gráfico com:
    - 3 grupos (correct, incomplete, incorrect)
    - 3 barras por grupo (cosine, jaccard, logic)
    """

    labels = wide_df["label"].tolist()
    metrics = ["cosine", "jaccard", "logic"]

    # extrair arrays de valores
    cosine = wide_df["cosine_similarity_mean"].tolist()
    jaccard = wide_df["jaccard_similarity_mean"].tolist()
    logic = wide_df["logic_f1_mean"].tolist()

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar([i - width for i in x], cosine, width, label="Cosine")
    plt.bar(x, jaccard, width, label="Jaccard")
    plt.bar([i + width for i in x], logic, width, label="Logic F1")

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Comparison of all metrics across labels")
    plt.legend()
    plt.tight_layout()

    out = FIGURES_DIR / f"overall_metric_comparison_ex{EXPERIMENT_ID}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved overall comparison: {out}")

# ------------ MAIN ------------

def main():
    print("=== Starting metrics analysis ===")
    ensure_figures_dir()
    ensure_table_dir()

    # 1) Carregar summaries
    cosine_summary = load_cosine_summary()
    jaccard_summary = load_jaccard_summary()
    logic_summary = load_logic_summary()


    summaries = [s for s in [jaccard_summary, cosine_summary, logic_summary] if not s.empty]


    if not summaries:
        print("⚠️ No metric summaries found, aborting analysis.")
        return

    # 2) Montar tabela comparativa geral
    comparison = pd.concat(summaries, ignore_index=True)
    comparison.to_csv(OUTPUT_TABLE/f"comparison_exp{EXPERIMENT_ID}", index=False)
    print(f"\nSaved comparison table at: {OUTPUT_TABLE}")

    # === Pivot para formato wide (label como linhas, métricas como colunas) ===

    # Remover count
    comparison_no_count = comparison.drop(columns=["count"])

    # Criar colunas combinadas metric_mean e metric_std
    comparison_no_count["metric_mean"] = comparison_no_count["metric"] + "_mean"
    comparison_no_count["metric_std"]  = comparison_no_count["metric"] + "_std"

    # Pivotar mean
    mean_pivot = comparison_no_count.pivot(
        index="label",
        columns="metric_mean",
        values="mean"
    )

    # Pivotar std
    std_pivot = comparison_no_count.pivot(
        index="label",
        columns="metric_std",
        values="std"
    )

    # Combinar tudo no formato solicitado
    final_table = pd.concat([mean_pivot, std_pivot], axis=1)

    # Ordenar colunas alfabeticamente (opcional)
    final_table = final_table.reindex(sorted(final_table.columns), axis=1)

    # Salvar em CSV
    final_pivot_path = OUTPUT_TABLE / f"metrics_comparison_wide_ex{EXPERIMENT_ID}.csv"
    final_table.to_csv(final_pivot_path)
    print(f"\nSaved wide comparison table at: {final_pivot_path}")

    print("\n=== Wide Comparison Table ===")
    print(final_table)

    # 3) Gráficos de barras
    if not jaccard_summary.empty:
        plot_bar(jaccard_summary, "Jaccard Similarity", f"jaccard_by_label_bar_ex{EXPERIMENT_ID}.png")

    if not cosine_summary.empty:
        plot_bar(cosine_summary, "Cosine Similarity", f"cosine_by_label_bar_ex{EXPERIMENT_ID}.png")

    if not logic_summary.empty:
        plot_bar(logic_summary, "Logical F1 (across trials)", f"logic_f1_by_label_bar_ex{EXPERIMENT_ID}.png")

    # 4) Boxplots (usando resultados completos para cosine/jaccard)
    if JACCARD_RESULTS.exists():
        plot_box(
            JACCARD_RESULTS,
            value_col="jaccard_similarity",
            label_col="label",
            title="Jaccard similarity distribution by label",
            file_name=f"jaccard_boxplot_ex{EXPERIMENT_ID}.png",
        )

    if COSINE_RESULTS.exists():
        plot_box(
            COSINE_RESULTS,
            value_col="cosine_similarity",
            label_col="label",
            title="Cosine similarity distribution by label",
            file_name=f"cosine_boxplot_ex{EXPERIMENT_ID}.png",
        )

    # 5) Gráfico de comparação geral
    wide_path = OUTPUT_TABLE / f"metrics_comparison_wide_ex{EXPERIMENT_ID}.csv"
    wide_df = pd.read_csv(wide_path)

    plot_metric_comparison_per_label(wide_df)
    plot_overall_metric_comparison(wide_df)

    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
