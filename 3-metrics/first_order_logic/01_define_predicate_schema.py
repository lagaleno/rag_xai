import json
import os
import requests
from pathlib import Path
import sys
import shutil

import pandas as pd

# Caminhos
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
RECORDS_ROOT = PROJECT_ROOT / "records"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from provenance import ProvenanceDB  # noqa: E402

# ================== CONFIGURAÇÕES ==================

# Caminho para o CSV do HotpotQA preparado
HOTPOT_CSV = PROJECT_ROOT / "1-creating_dataset" / "hotpotqa_sample.csv"

# Tamanho do batch: quantas linhas do dataset por chamada ao LLaMA
BATCH_SIZE = 10

# Modelo do LLaMA no Ollama
LLAMA_MODEL_NAME = "llama3"

# Arquivo de saída com o esquema de predicados
SCHEMA_OUT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "predicate_schema.json"

TEMPERATURE = 0.1

# ===================================================


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Carrega o dataset inteiro do HotpotQA (versão CSV preparada).
    Em seguida embaralha as linhas para dar mais variedade aos batches.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Embaralha para evitar que só o começo do dataset influencie o schema
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def build_batch_prompt(batch_rows: pd.DataFrame) -> str:
    """
    Constrói o prompt para o LLaMA propor um esquema de predicados,
    usando um batch de exemplos do HotpotQA.
    """
    examples_text = []
    for i, (_, row) in enumerate(batch_rows.iterrows(), start=1):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        context = str(row.get("context", "")).strip()

        block = (
            f"Example {i}:\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Supporting passage (chunk):\n{context}\n"
        )
        examples_text.append(block)

    examples_block = "\n\n".join(examples_text)

    prompt = f"""
You are helping to design a **logical predicate schema** for a factoid,
multi-hop question answering dataset similar to HotpotQA.

Below are several representative examples with a question, an answer,
and a supporting passage (chunk). We want to define a **small, generic
set of first-order logic predicates** that can capture the kinds of
facts involved in such questions.

The predicates should be:
- general (not tied to specific entities),
- reusable across many questions,
- limited in number (ideally between 6 and 20 in total, across all batches),
- suitable for expressing relations like location, type, authorship,
  membership, biography, and part-whole relations.

For each example, imagine what kinds of logical facts might be useful
to represent (e.g., who did what, where something is located, what type
an entity is, etc.). Then propose a compact schema of predicates.

Here are the examples:

{examples_block}

Return your answer **strictly as JSON** with the following format:

{{
  "predicates": [
    {{
      "name": "PredicateName",
      "args": ["arg1_type", "arg2_type"],
      "description": "Short natural-language description of what this predicate means."
    }},
    ...
  ]
}}

The argument types should be generic labels like "entity", "person",
"place", "work", "organization", "type", "date", etc.

Do not include any example-specific constants. Focus on the schema only.

Now propose the general predicate schema (for this batch of examples)
in the JSON format described above having ONLY JSON with no text before
or after the json schema.
"""
    return prompt.strip()


def call_llama(prompt: str, temperature: float = TEMPERATURE) -> str:
    """
    Chama o modelo LLaMA via Ollama (http://localhost:11434/api/chat)
    e retorna o texto bruto da resposta.
    """
    print("🧠 Calling LLaMA to propose predicate schema for this batch...")
    url = "http://localhost:11434/api/chat"

    data = {
        "model": LLAMA_MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt},
        ],
        "options": {
            "temperature": temperature,
        },
        "stream": False,
    }

    resp = requests.post(url, json=data)
    resp.raise_for_status()

    out = resp.json()
    text = out["message"]["content"].strip()
    # print(text)  # opcional: debug
    return text


def parse_schema(raw_output: str) -> dict:
    """
    Extrai o bloco JSON de dentro da saída do LLM.

    - Procura o primeiro '{' e o último '}'.
    - Tenta fazer json.loads nesse intervalo.
    - Se não der certo, mostra a saída bruta para depuração.
    """
    text = raw_output.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        print("⚠️ Could not find a valid JSON block in the LLaMA output.")
        print("Raw output:")
        print(text)
        raise ValueError("No JSON block found in LLaMA output.")

    json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON block. Error:")
        print(e)
        print("Extracted JSON candidate:")
        print(json_str)
        raise


def deduplicate_and_accumulate(global_predicates: dict, batch_schema: dict) -> None:
    """
    Atualiza o dicionário global de predicados com o que veio do batch.

    global_predicates:
      (name, args_tuple) -> {
          "name": ...,
          "args": [...],
          "description": ...,
          "usage_count": int
      }

    batch_schema:
      {"predicates": [ {name, args, description}, ... ]}
    """
    preds = batch_schema.get("predicates", [])
    for p in preds:
        name = p.get("name")
        args = p.get("args", [])
        desc = p.get("description", "")

        if not name or not isinstance(args, list):
            continue

        key = (name, tuple(args))

        if key not in global_predicates:
            global_predicates[key] = {
                "name": name,
                "args": args,
                "description": desc,
                "usage_count": 1,
            }
        else:
            global_predicates[key]["usage_count"] += 1


def main():
    print(f"📥 Loading full dataset from: {HOTPOT_CSV}")
    df = load_dataset(HOTPOT_CSV)

    n_rows = len(df)
    print(f"Dataset loaded with {n_rows} rows.")

    global_predicates = {}  # (name, args_tuple) -> dict

    # Loop em batches
    num_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing in {num_batches} batches of size {BATCH_SIZE}...")

    for batch_idx in range(0, n_rows, BATCH_SIZE):
        batch_rows = df.iloc[batch_idx : batch_idx + BATCH_SIZE]
        print(f"\n=== Batch {batch_idx // BATCH_SIZE + 1} / {num_batches} ===")
        print(f"Rows {batch_idx} to {batch_idx + len(batch_rows) - 1}")

        prompt = build_batch_prompt(batch_rows)
        raw_output = call_llama(prompt)
        batch_schema = parse_schema(raw_output)

        deduplicate_and_accumulate(global_predicates, batch_schema)

        print(
            f"🔁 Global predicates so far: {len(global_predicates)} "
            f"(after processing this batch)"
        )

    # Monta schema final
    final_schema = {
        "predicates": [
            {
                "name": info["name"],
                "args": info["args"],
                "description": info["description"],
                "usage_count": info["usage_count"],
            }
            for info in global_predicates.values()
        ]
    }

    # Salva JSON final
    SCHEMA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEMA_OUT, "w", encoding="utf-8") as f:
        json.dump(final_schema, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Global predicate schema saved to: {SCHEMA_OUT}")
    print("\nExtracted predicates (name(args) [usage_count]):")
    for p in final_schema["predicates"]:
        name = p.get("name")
        args = ", ".join(p.get("args", []))
        usage = p.get("usage_count", 0)
        print(f"- {name}({args})  [{usage}]")

    # ============ Proveniência (predicates) ============

    # 1) Recuperar HOTPOT_SAMPLE_ID do ambiente
    hotpot_sample_id_env = os.environ.get("HOTPOT_SAMPLE_ID")

    if hotpot_sample_id_env is None:
        print("⚠️ HOTPOT_SAMPLE_ID não definido no ambiente; pulando registro em 'predicates'.")
        return

    try:
        hotpot_sample_id = int(hotpot_sample_id_env)
    except ValueError:
        print(f"⚠️ HOTPOT_SAMPLE_ID inválido: {hotpot_sample_id_env!r}; pulando registro em 'predicates'.")
        return

    # 2) Definir caminho em records/predicates/...
    #    Aqui podemos organizar por hotpot_sample_id
    records_dir = RECORDS_ROOT / "predicates" / str(hotpot_sample_id)
    records_dir.mkdir(parents=True, exist_ok=True)

    # Nome do arquivo dentro de records
    schema_records_rel = f"records/predicates/{hotpot_sample_id}/predicate_schema.json"
    schema_records_abs = PROJECT_ROOT / schema_records_rel

    # 3) Copiar o arquivo original para a pasta records
    if SCHEMA_OUT.exists():
        shutil.copy2(SCHEMA_OUT, schema_records_abs)
    else:
        print(f"⚠️ SCHEMA_FILE não encontrado em {SCHEMA_OUT}; não será possível registrar o path corretamente.")
        return

    # 4) Registrar na tabela predicates
    try:
        prov = ProvenanceDB()
        predicate_id = prov.insert_predicates(
            hotpot_sample_id=hotpot_sample_id,
            model=LLAMA_MODEL_NAME,      
            temperature=TEMPERATURE,   
            prompt=prompt,  
            path=schema_records_rel,   
        )
        os.environ["PREDICATE_ID"] = str(predicate_id)
        prov.close()
        print(f"💾 Predicates registrados no banco com id={predicate_id} para hotpot_sample_id={hotpot_sample_id}")
    except Exception as e:
        print(f"⚠️ Erro ao registrar 'predicates' no banco: {e}")


if __name__ == "__main__":
    main()
