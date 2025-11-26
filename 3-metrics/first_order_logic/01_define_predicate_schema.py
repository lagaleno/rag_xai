import json
import os
import random
import subprocess
import requests

import pandas as pd
from pathlib import Path
import sys

# Caminhos
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from provenance import ProvenanceDB  # noqa: E402

# ================== CONFIGURAÇÕES ==================


# Caminho para o CSV do HotpotQA preparado
HOTPOT_CSV = PROJECT_ROOT / "0-utils" / "hotpotqa_train.csv"

# Número de exemplos para colocar no prompt
N_EXAMPLES = 3

# Modelo do LLaMA no Ollama
LLAMA_MODEL_NAME = "llama3"

# Arquivo de saída com o esquema de predicados
SCHEMA_OUT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "predicate_schema.json"

TEMPERATURE = 0.5

EXAMPLES_ID = [] # guardar para proveniência
# ===================================================


def load_sample_examples(csv_path: str, n: int = 3):
    """
    Carrega alguns exemplos aleatórios do CSV do HotpotQA para
    ilustrar o tipo de fato que queremos modelar com predicados.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Garante que temos pelo menos n exemplos
    if len(df) < n:
        n = len(df)

    sample = df.sample(n=n, random_state=42)
    examples = []
    for _, row in sample.iterrows():
        id = str(row.get("id", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        context = str(row.get("context", "")).strip()

        examples.append(
            {
                "question": question,
                "answer": answer,
                "context": context,
            }
        )

        EXAMPLES_ID.append(id)

    return examples


def build_prompt(examples):
    """
    Constrói o prompt para o LLaMA propor um esquema de predicados
    genérico e compacto, baseado em poucos exemplos do HotpotQA.
    """
    examples_text = []
    for i, ex in enumerate(examples, start=1):
        block = (
            f"Example {i}:\n"
            f"Question: {ex['question']}\n"
            f"Answer: {ex['answer']}\n"
            f"Supporting passage (chunk):\n{ex['context']}\n"
        )
        examples_text.append(block)

    examples_block = "\n\n".join(examples_text)

    prompt = f"""
You are helping to design a **logical predicate schema** for a factoid,
multi-hop question answering dataset similar to HotpotQA.

Below are a few representative examples with a question, an answer,
and a supporting passage (chunk). We want to define a **small, generic
set of first-order logic predicates** that can capture the kinds of
facts involved in such questions.

The predicates should be:
- general (not tied to specific entities),
- reusable across many questions,
- limited in number (ideally between 6 and 12),
- suitable for expressing relations like location, type, authorship,
  membership, biography, and part-whole relations.

For each example, imagine what kinds of logical facts might be useful
to represent (e.g., who did what, where something is located, what type
an entity is, etc.). Then propose a compact schema of predicates.

Return your answer **strictly as JSON** with the following format:

{{
  "predicates": [
    {{
      "name": "PredicateName",
      "args": ["arg1_type", "arg2_type"],
      "description": "Short natural-language description of what this predicate means."
    }},
    ...
  ],
}}

The argument types should be generic labels like "entity", "person",
"place", "work", "organization", "type", "date", etc.

Do not include any example-specific constants. Focus on the schema only.

Here are the representative examples:

{examples_block}

Now propose the predicate schema in the JSON format described above having ONLY JSON with no text before or after the json schema.
"""
    return prompt.strip()


def call_llama(prompt: str, temperature: float = TEMPERATURE) -> str:
    """
    Chama o modelo LLaMA via Ollama na linha de comando.
    Se você usar outra interface, adapte esta função.
    """
    print("🧠 Calling LLaMA to propose predicate schema...")
    # Comando básico do Ollama: `echo "prompt" | ollama run model`
    # Aqui usamos subprocess para passar o prompt via stdin.
    # result = subprocess.run(
    #     ["ollama", "run", LLAMA_MODEL_NAME],
    #     input=prompt,
    #     text=True,
    #     capture_output=True,
    # )
    """
        Chama o modelo LLaMA via Ollama (http://localhost:11434/api/chat)
        e tenta interpretar a saída como JSON.
    """
    url = "http://localhost:11434/api/chat"

    data = {
        "model": LLAMA_MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt},
        ],
        "options": {
            "temperature": temperature,
        },
        "stream": False
    }
    resp = requests.post(url, json=data)
    resp.raise_for_status()

    if not resp:
        print("❌ Error calling LLaMA:")
        print(resp.stderr)
        raise RuntimeError("LLaMA call failed")
    
    out = resp.json()

    # Ollama retorna algo como {"message": {"role": "...", "content": "..."}, ...}
    text = out["message"]["content"].strip()

    return text


def parse_schema(raw_output: str):
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

    json_str = text[start:end + 1]

    # Tenta parsear diretamente
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON block. Error:")
        print(e)
        print("Extracted JSON candidate:")
        print(json_str)
        raise

def main():

    print(f"📥 Loading sample examples from: {HOTPOT_CSV}")
    examples = load_sample_examples(HOTPOT_CSV, n=N_EXAMPLES)

    print(f"Using {len(examples)} examples to build the prompt.")
    prompt = build_prompt(examples)

    # Chama o LLaMA
    raw_output = call_llama(prompt)


    # Tenta parsear JSON
    schema = parse_schema(raw_output)

    # Salva JSON final
    with open(SCHEMA_OUT, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"✅ Predicate schema saved to: {SCHEMA_OUT}")
    print("\nExtracted predicates:")
    for pred in schema.get("predicates", []):
        print(f"- {pred.get('name')}({', '.join(pred.get('args', []))})")
    
    # ====== Proveniência ======
    logic_metric_id_env = os.getenv("LOGIC_METRIC_ID")
    if logic_metric_id_env is None:
        print("⚠️ LOGIC_METRIC_ID não encontrado no ambiente. Pulando registro de predicate_config.")
        return

    logic_metric_id = int(logic_metric_id_env)

    predicate_config = {
        "prompt": prompt,
        "list_predicates": schema["predicates"],
        "qtt_predicates": len(schema["predicates"]),
        "model": LLAMA_MODEL_NAME,
        "number_examples": N_EXAMPLES,
        "qa_examples_id": EXAMPLES_ID,
        "temperature": TEMPERATURE
    }

    prov = ProvenanceDB()
    prov.update_logic_metric_configs(
        logic_metric_id=logic_metric_id,
        predicate_config=predicate_config,
    )
    prov.close()

    print(f"🧠 predicate_config registrado no banco para logic_metric_id={logic_metric_id}")


if __name__ == "__main__":
    main()
