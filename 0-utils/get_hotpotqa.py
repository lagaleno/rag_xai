import requests
from datasets import load_dataset
import random
import pandas as pd


# Quantidade de amostras de HotpotQA
N_SAMPLES = 10

SEED = 42
random.seed(SEED)

# Arquivo de saída
HOTPOTQA_OUT = "hotpotqa_train.csv"

# ==========================
# CARREGAR HOTPOTQA
# ==========================

print("📥 Carregando HotpotQA...")
ds = load_dataset("hotpot_qa", "distractor")
train = ds["train"].shuffle(seed=SEED).select(range(min(N_SAMPLES, len(ds["train"]))))
print(train)

pd.DataFrame(train).to_csv(HOTPOTQA_OUT, index=False)
