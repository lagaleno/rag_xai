from pathlib import Path
import importlib.util
import os
import sys
import pandas as pd

# THIS_FILE = .../projeto/5-experiment/main_experiment.py
THIS_FILE = Path(__file__).resolve()

# PROJECT_ROOT = .../projeto
PROJECT_ROOT = THIS_FILE.parent.parent

# garantir que o Python enxergue a raiz do projeto para importar 'provenance'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print("THIS_FILE:", THIS_FILE)
print("PROJECT_ROOT:", PROJECT_ROOT)

from provenance import ProvenanceDB  # noqa: E402

# ===== CAMINHO DOS SCRIPTS =====
GET_HOTPOTQA_SCRIPT = PROJECT_ROOT / "0-utils" / "get_hotpotqa.py"
CREATE_DATASET_SCRIPT = PROJECT_ROOT / "1-creating_dataset" / "create_dataset.py"
VALIDATE_SCRIPT = PROJECT_ROOT / "2-validating_dataset" / "validate_dataset.py"

COSINE_SCRIPT = PROJECT_ROOT / "3-metrics" / "cosine_similarity" / "run_cosine_similarity.py"

DEFINE_PREDICATES_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "01_define_predicate_schema.py"
DEFINE_LOGICALRULES_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "02_define_logical_rules.py"
EXTRACT_FACTS_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "03_extract_facts_llm.py"
LOGIC_METRIC_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "04_inference_metric_prototype.py"

LLM_JUDGE_SCRIPT = PROJECT_ROOT / "3-metrics" / "llm_judge" / "llm_judge_classification.py"

ANALYSIS_SCRIPT = PROJECT_ROOT / "4-analysis" / "analyze.py"

# ===== CONFIGURAÇÕES BÁSICAS =====

HOTPOT_CSV = PROJECT_ROOT / "0-utils" / "hotpotqa_train.csv"
HOTPOT_SAMPLE_CSV = PROJECT_ROOT / "1-creating_dataset" / "hotpot_sample.csv"
EXPLAINRAG_DATASET_CSV = PROJECT_ROOT / "1-creating_dataset" / "explainrag_hotpot_llama_summary.csv"
EXPLAINRAG_DATASET_JSONL = PROJECT_ROOT / "1-creating_dataset" / "explainrag_hotpot_llama.jsonl"

# Saídas de métricas (usadas para resumo no banco)
COSINE_SUMMARY_CSV = PROJECT_ROOT / "3-metrics" / "cosine_similarity" / "cosine_similarity_summary_by_label.csv"

# ============================================================


def run_script(path: Path, func_name: str | None = "main"):
    """
    Carrega um script .py e executa a função func_name se existir.
    Se func_name=None, só importar o módulo já executa o código de topo.
    """
    print(f"\n>>> Running script: {path}")
    if not path.exists():
        print(f"⚠️ Script not found at {path}, skipping.")
        return

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # executa o arquivo

    if func_name and hasattr(module, func_name):
        result = getattr(module, func_name)()
        return result


def main():
    # ==============================
    # 1) MARCAR INÍCIO DO EXPERIMENTO NO BANCO
    # ==============================
    prov = ProvenanceDB()

    experiment_id = prov.create_experiment(
        hotpot_path=str(HOTPOT_SAMPLE_CSV),
        seed=0,
        n_samples=0,
    )

    print(f"🧪 Experiment started with id={experiment_id}")

    # deixar o experiment_id disponível pros outros scripts via variável de ambiente
    os.environ["EXPERIMENT_ID"] = str(experiment_id)
    print("======== Set up the environment ========")

    # ==============================
    # 2) HOTPOTQA: BAIXAR/CARREGAR APENAS SE PRECISAR
    # ==============================
    if HOTPOT_CSV.exists():
        print(f"📄 HotpotQA CSV already exists at {HOTPOT_CSV}, skipping get_hotpotqa.py")
    else:
        print(f"⬇️ HotpotQA CSV not found. Running: {GET_HOTPOTQA_SCRIPT}")
        run_script(GET_HOTPOTQA_SCRIPT)

    # ==============================
    # 3) DATASET DE EXPLICAÇÕES: GERAR APENAS SE PRECISAR
    # ==============================
    if EXPLAINRAG_DATASET_JSONL.exists():
        print(f"📄 ExplainRAG dataset JSONL already exists at {EXPLAINRAG_DATASET_JSONL}, skipping create_dataset.py")
    else:
        print(f"🧾 ExplainRAG dataset JSONL not found. Running: {CREATE_DATASET_SCRIPT}")
        run_script(CREATE_DATASET_SCRIPT)

    # ==============================
    # 4) VALIDAR DATASET
    # ==============================
    print(f"🧮 Validating dataset...")
    is_valid = run_script(VALIDATE_SCRIPT)

    if not is_valid:
        print("\n❌ Explainability dataset is not valid. Delete the dataset and try generating a new one.")
        prov.close()
        print("\n✅ Main experiment finished (aborted due to invalid dataset).")
        return

    print("✅ Dataset is valid. Proceeding with metrics.")

    # ==============================
    # 5) SIMILARIDADE DE COSSENO
    # ==============================
    print("\n======== Cosine Similarity Metric ========")
    if COSINE_SCRIPT.exists():
        run_script(COSINE_SCRIPT)
    else:
        print(f"⚠️ Cosine script not found at {COSINE_SCRIPT}")

    # ==============================
    # 6) MÉTRICA LÓGICA: CRIAR REGISTRO E CONFIG
    # ==============================
    logic_metric_id = prov.create_logic_metric(
        experiment_id=experiment_id,
        num_trials=1,          # mantemos 1 só por compatibilidade,
        predicate_config={},   # será atualizado pelo script 01
        rules_config={},       # será atualizado pelo script 02
        facts_config={},       # será atualizado pelo script 03
    )

    os.environ["LOGIC_METRIC_ID"] = str(logic_metric_id)
    print(f"🧠 logic_metric criado com ID={logic_metric_id}")

    # 6.1) Extrair o esquema de predicados
    print("\n======== First-Order Logic – Predicate Schema ========")
    if DEFINE_PREDICATES_SCRIPT.exists():
        run_script(DEFINE_PREDICATES_SCRIPT)
    else:
        print(f"⚠️ Define Predicate script not found at {DEFINE_PREDICATES_SCRIPT}")

    # 6.2) Extrair as regras lógicas
    print("\n======== First-Order Logic – Logical Rules ========")
    if DEFINE_LOGICALRULES_SCRIPT.exists():
        run_script(DEFINE_LOGICALRULES_SCRIPT)
    else:
        print(f"⚠️ Define Logical Rules script not found at {DEFINE_LOGICALRULES_SCRIPT}")

    # 6.3) Extrair fatos com LLM
    print("\n======== First-Order Logic – Fact Extraction ========")
    if EXTRACT_FACTS_SCRIPT.exists():
        run_script(EXTRACT_FACTS_SCRIPT)
    else:
        print(f"⚠️ Fact extraction script not found at {EXTRACT_FACTS_SCRIPT}")

    # 6.4) Rodar métrica lógica (1 vez, sem trials)
    print("\n======== First-Order Logic – Metric Computation ========")
    if LOGIC_METRIC_SCRIPT.exists():
        run_script(LOGIC_METRIC_SCRIPT)
    else:
        print(f"⚠️ Logic metric script not found at {LOGIC_METRIC_SCRIPT}")

    # ==============================
    # 7) LLM-AS-JUDGE – CLASSIFICAÇÃO BASELINE
    # ==============================
    print("\n======== LLM-as-Judge – Classification Baseline ========")
    if LLM_JUDGE_SCRIPT.exists():
        run_script(LLM_JUDGE_SCRIPT)
    else:
        print(f"⚠️ LLM judge script not found at {LLM_JUDGE_SCRIPT}")

    # ==============================
    # 8) RESUMO PARA PROVENIÊNCIA
    # ==============================

    # ---------- COSINE ----------
    if COSINE_SUMMARY_CSV.exists():
        cos_df = pd.read_csv(COSINE_SUMMARY_CSV, index_col="label")
        cosine_summary = {}
        for label in cos_df.index:
            row = cos_df.loc[label]
            cosine_summary[label] = {
                "mean": float(row["mean"]),
                "std": float(0.0 if pd.isna(row["std"]) else row["std"]),
                "count": int(row["count"]),
            }
    else:
        cosine_summary = None
        print(f"⚠️ Cosine summary CSV not found at {COSINE_SUMMARY_CSV}")

    # ---------- LOGIC ----------
    # Agora a métrica lógica é avaliada principalmente via matriz de confusão
    # e comparação com o LLM-judge. Podemos registrar apenas que existe uma
    # configuração/execução, mas não calculamos F1 médio aqui.
    logic_summary = None

    prov.update_experiment_summaries(
        experiment_id=experiment_id,
        cosine_summary=cosine_summary,
        logic_summary=logic_summary,
    )
    prov.close()

    print("📊 Overview de métricas salvo na tabela experiment.")

    # Se quiser rodar a análise automática, descomente:
    # run_script(ANALYSIS_SCRIPT)

    print("\n✅ Main experiment finished.")


if __name__ == "__main__":
    main()
