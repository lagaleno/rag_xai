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
    
    run_script(CREATE_DATASET_SCRIPT)
    
    # ==============================
    # RECUPERAR hotpot_sample_id DO BANCO
    # ==============================

    hotpot_sample_id = prov.get_or_create_hotpot_sample(
        n_sample=int(os.getenv("HOTPOT_N_SAMPLES")),
        seed=int(os.getenv("HOTPOT_SEED")),
    )

    # ==============================
    # CRIAR EXPERIMENTO COM ESSES IDs
    # ==============================

    experiment_id = prov.create_experiment(
        hotpot_sample_id=hotpot_sample_id,
        xai_dataset_id=os.getenv("XAI_DATASET_ID"),
    )

    print(f"🧪 Experiment started with id={experiment_id}")
    os.environ["EXPERIMENT_ID"] = str(experiment_id)
    
    os.environ["HOTPOT_SAMPLE_ID"] = str(hotpot_sample_id)

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

    # Se quiser rodar a análise automática, descomente:
    run_script(ANALYSIS_SCRIPT)

    print("\n✅ Main experiment finished.")


if __name__ == "__main__":
    main()
