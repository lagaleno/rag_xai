import requests
import os
from datasets import load_dataset
import random
import pandas as pd
from pathlib import Path
from provenance import ProvenanceDB 

THIS_FILE = Path(__file__).resolve()

# PROJECT_ROOT = .../projeto
PROJECT_ROOT = THIS_FILE.parent.parent

# Quantidade de amostras de HotpotQA
N_SAMPLES = 10

SEED = 42
random.seed(SEED)

# Arquivo de saída
HOTPOTQA_OUT = PROJECT_ROOT / "0-utils" / "hotpotqa_train.csv"

# ==========================
# CARREGAR HOTPOTQA
# ==========================

# Tenta recuperar o EXPERIMENT_ID do ambiente (setado em 5-experiment/main.py)
experiment_id_env = os.getenv("EXPERIMENT_ID")
experiment_id = int(experiment_id_env) if experiment_id_env is not None else None

prov = None
if experiment_id is not None:
    prov = ProvenanceDB()
    print(f"📌 Updating experiment id={experiment_id} with HotpotQA info...")
else:
    print("⚠️ No EXPERIMENT_ID found in environment. Running without provenance update.")

if not HOTPOTQA_OUT.exists():
    print(f"✍️ Getting HOTPOTQA from Hugginface")
    print("📥 Carregando HotpotQA...")
    try:
        ds = load_dataset("hotpot_qa", "distractor")
        print(ds)
        train = ds["train"].shuffle(seed=SEED).select(range(min(N_SAMPLES, len(ds["train"]))))
    except:
        ds = load_dataset('parquet', data_files='original_backup.parquet')
        print(ds)
        train = ds["train"].shuffle(seed=SEED).select(range(min(N_SAMPLES, len(ds["train"]))))
    finally:
        print(train)
        pd.DataFrame(train).to_csv(HOTPOTQA_OUT, index=False)
        # Atualiza o experimento no banco com as infos corretas
        if prov is not None and experiment_id is not None:
            prov.update_experiment_hotpot_info(
                experiment_id=experiment_id,
                hotpot_path=str(HOTPOTQA_OUT),
                seed=SEED,
                n_samples=N_SAMPLES,
            )
            print("✅ Experiment updated with HotpotQA info.")

            prov.close()
            print("🔌 Conexão com o banco fechada.")

else:
    print(f"📝 Using an existing version of the Hotpotqa dataset")
    if prov is not None and experiment_id is not None:
        prov.update_experiment_hotpot_info(
            experiment_id=experiment_id,
            hotpot_path=str(HOTPOTQA_OUT),
            seed=SEED,
            n_samples=N_SAMPLES,
        )
        print("✅ Experiment updated with HotpotQA info.")

        prov.close()
        print("🔌 Conexão com o banco fechada.")