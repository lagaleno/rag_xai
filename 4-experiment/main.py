from pathlib import Path
import importlib.util
import shutil

# THIS_FILE = .../projeto/4-experiment/main_experiment.py
THIS_FILE = Path(__file__).resolve()

# PROJECT_ROOT = .../projeto
PROJECT_ROOT = THIS_FILE.parent.parent

print("THIS_FILE:", THIS_FILE)
print("PROJECT_ROOT:", PROJECT_ROOT)

# ===== CAMINHO DOS SCRIPTS =====
JACCARD_SCRIPT = PROJECT_ROOT / "3-metrics" / "jaccard_similarity" / "run_jaccard_similarity.py"
COSINE_SCRIPT = PROJECT_ROOT / "3-metrics" / "cosine_similarity" / "run_cosine_similarity.py"

DEFINE_PREDICATES_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "01_define_predicate_schema.py"
DEFINE_LOGICALRULES_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "02_define_logical_rules.py"
EXTRACT_FACTS_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "03_extract_facts_llm.py"
LOGIC_METRIC_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "04_inference_metric_prototype.py"

# ===== CONFIGURAÇÕES BÁSICAS =====

HOTPOT_CSV = PROJECT_ROOT / "0-utils" / "hotpotqa_train.csv"
EXPLAINRAG_DATASET = PROJECT_ROOT / "2-validating_dataset" / "explicability_dataset.csv"

FACTS_JSONL = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "facts_extracted_llm.jsonl"
LOGIC_RESULTS_CSV = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "logical_metrics_results.csv"

N_TRIALS = 1  # número de trials da métrica lógica

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
        getattr(module, func_name)()


def main():
    print("======== ExplainRAG Main Experiment (from 4-experiment) ========")

    # 1) Similaridade: Jaccard
    if JACCARD_SCRIPT.exists():
        run_script(JACCARD_SCRIPT)  # assume que o script já roda tudo no topo ou em main()
    else:
        print(f"⚠️ Jaccard script not found at {JACCARD_SCRIPT}")

    # 2) Similaridade: Cosine
    if COSINE_SCRIPT.exists():
        run_script(COSINE_SCRIPT)
    else:
        print(f"⚠️ Cosine script not found at {COSINE_SCRIPT}")

    # 3) Métrica de Lógica de primeira ordem
    # 3.1) Extrair o esquema de predicados 

    # 3.2) Extrair as regras lógicas 

    # 3.3) Trials de lógica -> extrair os datos e realizar o cálculo da métria em cada trial
    for trial in range(1, N_TRIALS + 1):
        print(f"\n========== Logic metric trial {trial}/{N_TRIALS} ==========")

        # 3.1) Extrair fatos
        run_script(EXTRACT_FACTS_SCRIPT)

        # salvar cópia dos fatos deste trial (opcional)
        if FACTS_JSONL.exists():
            facts_trial = FACTS_JSONL.with_name(
                FACTS_JSONL.stem + f"_trial{trial}" + FACTS_JSONL.suffix
            )
            shutil.copy2(FACTS_JSONL, facts_trial)
            print(f"Saved facts for trial {trial}: {facts_trial}")
        else:
            print(f"⚠️ Facts file not found at {FACTS_JSONL}")

        # 3.2) Rodar métrica lógica
        run_script(LOGIC_METRIC_SCRIPT)

        if LOGIC_RESULTS_CSV.exists():
            results_trial = LOGIC_RESULTS_CSV.with_name(
                LOGIC_RESULTS_CSV.stem + f"_trial{trial}" + LOGIC_RESULTS_CSV.suffix
            )
            shutil.copy2(LOGIC_RESULTS_CSV, results_trial)
            print(f"Saved logic results for trial {trial}: {results_trial}")
        else:
            print(f"⚠️ Logic results file not found at {LOGIC_RESULTS_CSV}")

    print("\n✅ Main experiment finished.")


if __name__ == "__main__":
    main()
