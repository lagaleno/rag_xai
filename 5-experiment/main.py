from pathlib import Path
import importlib.util
import os
import shutil
import sys
import pandas as pd 
# pega todos os arquivos de summary por trial
from glob import glob


# THIS_FILE = .../projeto/5-experiment/main_experiment.py
THIS_FILE = Path(__file__).resolve()

# PROJECT_ROOT = .../projeto
PROJECT_ROOT = THIS_FILE.parent.parent

# garantir que o Python enxergue a raiz do projeto para importar 'provenance'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print("THIS_FILE:", THIS_FILE)
print("PROJECT_ROOT:", PROJECT_ROOT)

from provenance import ProvenanceDB  

# ===== CAMINHO DOS SCRIPTS =====
GET_HOTPOTQA_SCRIPT = PROJECT_ROOT / "0-utils" / "get_hotpotqa.py"
CREATE_DATASET_SCRIPT = PROJECT_ROOT / "1-creating_dataset" / "create_dataset.py"
VALIDATE_SCRIPT = PROJECT_ROOT / "2-validating_dataset" / "validate_dataset.py"

JACCARD_SCRIPT = PROJECT_ROOT / "3-metrics" / "jaccard_similarity" / "run_jaccard_similarity.py"
COSINE_SCRIPT = PROJECT_ROOT / "3-metrics" / "cosine_similarity" / "run_cosine_similarity.py"

DEFINE_PREDICATES_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "01_define_predicate_schema.py"
DEFINE_LOGICALRULES_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "02_define_logical_rules.py"
EXTRACT_FACTS_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "03_extract_facts_llm.py"
LOGIC_METRIC_SCRIPT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "04_inference_metric_prototype.py"

ANALYSIS_SCRIPT = PROJECT_ROOT / "4-analysis" / "analyze.py"

# ===== CONFIGURAÇÕES BÁSICAS =====

HOTPOT_CSV = PROJECT_ROOT / "0-utils" / "hotpotqa_train.csv"
EXPLAINRAG_DATASET_CSV = PROJECT_ROOT / "1-creating_dataset" / "explainrag_hotpot_llama_summary.csv"
EXPLAINRAG_DATASET_JSONL = PROJECT_ROOT / "1-creating_dataset" / "explainrag_hotpot_llama.jsonl"

FACTS_JSONL = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "facts_extracted_llm.jsonl"
TRIAL_FACTS_OUT_NAME = "facts_extracted_trial"
TRIAL_FACTS_OUT = PROJECT_ROOT / "5-experiment" / TRIAL_FACTS_OUT_NAME
LOGIC_RESULTS_CSV = PROJECT_ROOT / "5-experiment" / "logical_metrics_results.csv"
LOGIC_RESULTS_SUMMARY_CSV = PROJECT_ROOT / "5-experiment" / "logical_metrics_summary_results.csv"
TRIAL_LOGIC_RESULT_NAME = "logical_result_trials_out"
TRIAL_LOGIC_RESULT = PROJECT_ROOT / "5-experiment" / TRIAL_LOGIC_RESULT_NAME
TRIAL_LOGIC_SUMMARY_RESULT = PROJECT_ROOT / "5-experiment" / "logical_summary_results_trials_out"

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
        result = getattr(module, func_name)()
        return result
   

def main():
    # ==============================
    # 1) MARCAR INÍCIO DO EXPERIMENTO NO BANCO
    # ==============================
    prov = ProvenanceDB()

    experiment_id = prov.create_experiment(
        hotpot_path="",
        seed=0,
        n_samples=0,
    )

    print(f"🧪 Experiment started with id={experiment_id}")

    # deixar o experiment_id disponível pros outros scripts via variável de ambiente
    os.environ["EXPERIMENT_ID"] = str(experiment_id)
    print("======== Set up the enviroment ========")

    run_script(GET_HOTPOTQA_SCRIPT)

    run_script(CREATE_DATASET_SCRIPT)

    print(f"🧮 Validating Dataset")
    is_valid = run_script(VALIDATE_SCRIPT)

    if is_valid:
        print("======== ExplainRAG Main Experiment (from 5-experiment) ========")

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

        logic_metric_id = prov.create_logic_metric(
            experiment_id=experiment_id,
            num_trials=N_TRIALS,
            predicate_config={},   # será preenchido depois (script 01)
            rules_config={},       # será preenchido depois (script 02)
            facts_config={}        # será preenchido depois (script 03)
        )

        # exportamos para os scripts 01/02/03/04 usarem
        os.environ["LOGIC_METRIC_ID"] = str(logic_metric_id)

        print(f"🧠 logic_metric criado com ID={logic_metric_id}")
        # 3) Métrica de Lógica de primeira ordem
        # 3.1) Extrair o esquema de predicados 
        if DEFINE_PREDICATES_SCRIPT.exists():
            run_script(DEFINE_PREDICATES_SCRIPT)
        else:
            print(f"⚠️ Define Predicate script not found at {DEFINE_PREDICATES_SCRIPT}")
            
        # 3.2) Extrair as regras lógicas
        if DEFINE_LOGICALRULES_SCRIPT.exists():
            run_script(DEFINE_LOGICALRULES_SCRIPT)
        else:
            print(f"⚠️ Define Logical Rules not found at {DEFINE_LOGICALRULES_SCRIPT}") 

        # Prepara ambiente para armazenar arquivos 

        # Verifica se diretórios existem apra limpar o ambiente e criar os novos diretórios
        if os.path.isdir(TRIAL_FACTS_OUT):
            shutil.rmtree(TRIAL_FACTS_OUT)
        os.mkdir(TRIAL_FACTS_OUT)
        if os.path.isdir(TRIAL_LOGIC_RESULT):
            shutil.rmtree(TRIAL_LOGIC_RESULT)
        os.mkdir(TRIAL_LOGIC_RESULT)
        if os.path.isdir(TRIAL_LOGIC_SUMMARY_RESULT):
            shutil.rmtree(TRIAL_LOGIC_SUMMARY_RESULT)
        os.mkdir(TRIAL_LOGIC_SUMMARY_RESULT)

        # 3.3) Trials de lógica -> extrair os datos e realizar o cálculo da métria em cada trial
        for trial in range(1, N_TRIALS + 1):
            print(f"\n========== Logic metric trial {trial}/{N_TRIALS} ==========")
            
            os.environ["LOGIC_TRIAL_NUMBER"] = str(trial) # registra o trial no ambiente
            
            # 3.1) Extrair fatos
            run_script(EXTRACT_FACTS_SCRIPT)

            # salvar cópia dos fatos deste trial (opcional)
            if FACTS_JSONL.exists():
                facts_trial = FACTS_JSONL.with_name(
                    FACTS_JSONL.stem + f"_trial{trial}" + FACTS_JSONL.suffix
                )
                shutil.copy2(FACTS_JSONL, facts_trial)
                shutil.move(facts_trial, TRIAL_FACTS_OUT)
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
                shutil.move(results_trial, TRIAL_LOGIC_RESULT)
                print(f"Saved logic results for trial {trial}: {results_trial}")
            else:
                print(f"⚠️ Logic results file not found at {LOGIC_RESULTS_CSV}")

            if LOGIC_RESULTS_SUMMARY_CSV.exists():
                results_summary_trial = LOGIC_RESULTS_SUMMARY_CSV.with_name(
                    LOGIC_RESULTS_SUMMARY_CSV.stem + f"_trial{trial}" + LOGIC_RESULTS_SUMMARY_CSV.suffix
                )
                shutil.copy2(LOGIC_RESULTS_SUMMARY_CSV, results_summary_trial)
                shutil.move(results_summary_trial, TRIAL_LOGIC_SUMMARY_RESULT)
                print(f"Saved logic results for trial {trial}: {results_summary_trial}")
            else:
                print(f"⚠️ Logic results file not found at {TRIAL_LOGIC_SUMMARY_RESULT}")

        # Pega o resumo dos resultados das métricas para o banco de Proveniência

        # ---------- JACCARD ----------
        jaccard_summary_path = PROJECT_ROOT / "5-experiment" / "jaccard_similarity_summary_by_label.csv"
        if jaccard_summary_path.exists():
            jacc_df = pd.read_csv(jaccard_summary_path, index_col="label")
            jaccard_summary = {}
            for label in jacc_df.index:
                row = jacc_df.loc[label]
                jaccard_summary[label] = {
                    "mean": float(row["mean"]),
                    "std": float(0.0 if pd.isna(row["std"]) else row["std"]),
                    "count": int(row["count"]),
                }
        else:
            jaccard_summary = None
            print(f"⚠️ Jaccard summary CSV não encontrado em {jaccard_summary_path}")

        # ---------- COSINE ----------
        cosine_summary_path = PROJECT_ROOT / "5-experiment" / "cosine_similarity_summary_by_label.csv"
        if cosine_summary_path.exists():
            cos_df = pd.read_csv(cosine_summary_path, index_col="label")
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
            print(f"⚠️ Cosine summary CSV não encontrado em {cosine_summary_path}")

        # ---------- LOGIC (USANDO OS SUMMARIES POR TRIAL) ----------
        logic_summary = {"per_trial": {}, "overall": None}

        
        summary_files = sorted(
            TRIAL_LOGIC_SUMMARY_RESULT.glob("logical_metrics_summary_results_trial*.csv")
        )

        if summary_files:
            for path in summary_files:
                # tenta extrair o número do trial do nome do arquivo
                # ex: logical_metrics_summary_results_trial1.csv -> 1
                name = path.stem  # logical_metrics_summary_results_trial1
                # pega tudo que for dígito no final do nome
                trial_str = "".join(ch for ch in name if ch.isdigit())
                trial_num = trial_str or "1"

                df_sum = pd.read_csv(path)

                trial_dict = {}
                for _, row in df_sum.iterrows():
                    label = str(row["explanation_label"])
                    trial_dict[label] = {
                        "mean": float(row["mean"]),
                        "std": float(0.0 if pd.isna(row["std"]) else row["std"]),
                        "count": int(row["count"]),
                    }

                logic_summary["per_trial"][str(trial_num)] = trial_dict
        else:
            logic_summary = None
            print(f"⚠️ Nenhum summary de lógica por trial encontrado em {TRIAL_LOGIC_SUMMARY_RESULT}")
        # 2) overall: agrega TODOS os trials usando os resultados completos
        all_logic_files = sorted(
            TRIAL_LOGIC_RESULT.glob("logical_metrics_results_trial*.csv")
        )

        if all_logic_files:
            dfs = [pd.read_csv(p) for p in all_logic_files]
            all_logic_df = pd.concat(dfs, ignore_index=True)

            label_col = "explanation_label"

            if label_col not in all_logic_df.columns:
                print(f"⚠️ Coluna {label_col} não encontrada em logical_results. Colunas disponíveis: {list(all_logic_df.columns)}")
            else:
                group = all_logic_df.groupby(label_col)[["precision", "recall", "f1"]].agg(["mean", "std", "count"])

                LOGIC_SUMMARY_ALL = PROJECT_ROOT / "5-experiment" / "logical_metrics_summary_all_trials.csv"
                group.to_csv(LOGIC_SUMMARY_ALL)
                print(f"✅ Logic summary (todos os trials) salvo em: {LOGIC_SUMMARY_ALL}")

                overall = {}
                for label in group.index:
                    stats = group.loc[label]
                    overall[label] = {
                        "precision": {
                            "mean": float(stats[("precision", "mean")]),
                            "std": float(0.0 if pd.isna(stats[("precision", "std")]) else stats[("precision", "std")]),
                            "count": int(stats[("precision", "count")]),
                        },
                        "recall": {
                            "mean": float(stats[("recall", "mean")]),
                            "std": float(0.0 if pd.isna(stats[("recall", "std")]) else stats[("recall", "std")]),
                            "count": int(stats[("recall", "count")]),
                        },
                        "f1": {
                            "mean": float(stats[("f1", "mean")]),
                            "std": float(0.0 if pd.isna(stats[("f1", "std")]) else stats[("f1", "std")]),
                            "count": int(stats[("f1", "count")]),
                        },
                    }

                logic_summary["overall"] = overall
        else:
            print(f"⚠️ Nenhum CSV de resultados lógicos por trial encontrado em {TRIAL_LOGIC_RESULT}")
        

        # ---------- Atualizar experimento ----------
        prov.update_experiment_summaries(
            experiment_id=experiment_id,
            jaccard_summary=jaccard_summary,
            cosine_summary=cosine_summary,
            logic_summary=logic_summary,
        )
        prov.close()

        print("📊 Overview de métricas salvo na tabela experiment.")

        run_script(ANALYSIS_SCRIPT)
    
    else:
        print("\n ❌ Explainability dataset is not validy, delete and try generating a new one")

    print("\n✅ Main experiment finished.")


if __name__ == "__main__":
    main()
