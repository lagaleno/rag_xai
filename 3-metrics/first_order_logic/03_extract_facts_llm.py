import ast
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import requests
import shutil


import pandas as pd
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
RECORDS_ROOT = PROJECT_ROOT / "records"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from provenance import ProvenanceDB  # noqa: E402

# ================== CONFIGURAÇÕES ==================


# Arquivo de entrada com o schema de predicados (o que você já gerou)
SCHEMA_FILE = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "predicate_schema.json"

# CSV de entrada com o dataset de explicações
# Ajuste este caminho para o seu arquivo real
INPUT_CSV = PROJECT_ROOT / "1-creating_dataset" / "explainrag_hotpot_llama_summary.csv"

# Colunas esperadas no CSV (ajuste se necessário)
COL_ID = "id"
COL_CHUNK = "chunk"

# Aqui é a coluna que contém a LISTA de explicações em formato string
# (exatamente como o exemplo que você mandou)
COL_EXPL_LIST = "explanations"  # ajuste para o nome real da coluna

# Arquivo de saída com os fatos extraídos
OUTPUT_JSONL = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "facts_extracted_llm.jsonl"

# Modelo do LLaMA no Ollama
LLAMA_MODEL_NAME = "llama3"

# Limite opcional de linhas (para teste). Se None, processa tudo.
MAX_ROWS = None  # ex.: 10 para testar, depois None

TEMPERATURE = 0.5

PROV_METRIC = {}

# ===================================================


def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_extraction_prompt(schema: Dict[str, Any], text: str, source_type: str) -> str:
    """
    Constrói o prompt para o LLaMA extrair fatos de um texto
    (chunk ou explanation), usando o schema de predicados.
    source_type: "chunk" ou "explanation" (só para contextualizar no prompt)
    """
    predicates = schema.get("predicates", [])
    pred_lines = []
    for p in predicates:
        name = p.get("name")
        args = ", ".join(p.get("args", []))
        desc = p.get("description", "")
        pred_lines.append(f"- {name}({args}): {desc}")
    predicates_block = "\n".join(pred_lines)

    prompt = f"""
You are given a {source_type} from a QA system (either a supporting passage or an explanation).
Your task is to extract factual information from this text and represent it using ONLY
the following logical predicates and their argument order:

{predicates_block}

IMPORTANT INSTRUCTIONS:
- Use ONLY the predicates listed above. Do NOT invent new predicates.
- Respect the EXACT argument order given for each predicate.
- Arguments should be short, canonical identifiers for entities (e.g., "Paris", "France", "JK_Rowling"),
  not full sentences.
- Extract only clear, factual statements that can be expressed with these predicates.
- If no facts can be extracted according to the schema, return an empty list.

Return the result STRICTLY as JSON in the following format:

{{
  "facts": [
    {{
      "predicate": "predicate_name",
      "args": ["arg1", "arg2", ...]
    }},
    ...
  ]
}}

Do NOT include any additional text before or after the JSON.

Here is the {source_type} text:

\"\"\"{text}\"\"\"
"""

    return prompt.strip()


def call_llama(prompt: str,  temperature: float = TEMPERATURE) -> str:
    """
    Chama o modelo LLaMA via Ollama.
    """
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


def parse_facts_output(raw_output: str) -> List[Dict[str, Any]]:
    """
    Extrai o JSON da saída do LLaMA e retorna a lista de facts.
    Tolerante a algum texto antes/depois (faz recorte pelo primeiro '{' e último '}').
    """
    text = raw_output.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        print("⚠️ Could not find JSON in LLaMA output. Returning empty facts.")
        print("Raw output:")
        print(text)
        return []

    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON, returning empty facts.")
        print("Error:", e)
        print("JSON candidate:")
        print(json_str)
        return []

    facts = data.get("facts", [])
    cleaned_facts = []
    for f in facts:
        pred = f.get("predicate")
        args = f.get("args", [])
        if isinstance(pred, str) and isinstance(args, list):
            cleaned_facts.append({"predicate": pred, "args": args})
    return cleaned_facts


def process_row(schema: Dict[str, Any], row: pd.Series) -> List[Dict[str, Any]]:
    """
    Para uma linha do dataset, faz:
    - extrai fatos do chunk (uma vez só),
    - lê a lista de explicações (correct/incomplete/incorrect),
    - para cada explicação, extrai fatos e gera um registro separado.

    Retorna uma lista de dicts, um por explicação.
    """
    ex_id = row[COL_ID]

    # 1) Converte a string da coluna de chunk para dicionário para obter somente o texto do chunk, ignorando a proveniência 
    #    Extrai fatos do chunk uma única vez
    print(f"   → Extracting facts from chunk for id={ex_id}")
    try:
        chunk_dict = ast.literal_eval(row[COL_CHUNK])
        chunk_text = chunk_dict["text"]
    except Exception as e:
        print(f"⚠️ Could not parse chunk dictionary for id={ex_id}: {e}")
        chunk_text = []
    prompt_chunk = build_extraction_prompt(schema, chunk_text, source_type="chunk")
    PROV_METRIC["prompt"] = prompt_chunk # armazenando para proveniência
    raw_chunk = call_llama(prompt_chunk)
    chunk_facts = parse_facts_output(raw_chunk)

    # 2) Converte a string da coluna de explicações para lista Python
    #    Exemplo de conteúdo: "[{'label': 'correct', 'text': '...'}, ...]"
    expl_list_raw = row[COL_EXPL_LIST]
    try:
        explanations = ast.literal_eval(expl_list_raw)
    except Exception as e:
        print(f"⚠️ Could not parse explanations list for id={ex_id}: {e}")
        explanations = []

    results_for_row: List[Dict[str, Any]] = []

    # 3) Para cada explicação (correct / incomplete / incorrect)
    for expl_obj in explanations:
        expl_label = expl_obj.get("label")
        expl_text = expl_obj.get("text", "")

        print(f"   → Extracting facts from explanation [{expl_label}] for id={ex_id}")
        prompt_expl = build_extraction_prompt(schema, expl_text, source_type="explanation")
        raw_expl = call_llama(prompt_expl)
        expl_facts = parse_facts_output(raw_expl)

        result = {
            "id": ex_id,
            "explanation_label": expl_label,      # correct / incomplete / incorrect
            "chunk_text": chunk_text,
            "explanation_text": expl_text,
            "chunk_facts": chunk_facts,
            "explanation_facts": expl_facts,
        }
        results_for_row.append(result)

    return results_for_row


def main():

    print(f"📥 Loading predicate schema from: {SCHEMA_FILE}")
    schema = load_schema(SCHEMA_FILE)

    print(f"📥 Loading dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    print(f"Total rows to process: {len(df)}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for idx, row in df.iterrows():

            ex_id = row[COL_ID]
            print(f"\n=== Processing row {idx+1}/{len(df)} (id={ex_id}) ===")
            try:
                row_results = process_row(schema, row)
            except Exception as e:
                print(f"⚠️ Error processing row {ex_id}: {e}")
                # registra uma entrada com erro (sem fatos)
                error_record = {
                    "id": ex_id,
                    "explanation_label": None,
                    "chunk_text": str(row[COL_CHUNK]),
                    "explanation_text": None,
                    "chunk_facts": [],
                    "explanation_facts": [],
                    "error": str(e),
                }
                f_out.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                continue

            # escreve uma linha por explicação
            for rec in row_results:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Finished. Facts saved to: {OUTPUT_JSONL}")

    # ============ Proveniência (facts) ============

    # 1) Recuperar IDs necessários do ambiente
    predicate_id_env = os.environ.get("PREDICATE_ID")
    xai_dataset_id_env = os.environ.get("XAI_DATASET_ID")

    if predicate_id_env is None or xai_dataset_id_env is None:
        print("⚠️ PREDICATE_ID ou XAI_DATASET_ID não definido no ambiente; pulando registro em 'facts'.")
        return

    try:
        predicate_id = int(predicate_id_env)
        xai_dataset_id = int(xai_dataset_id_env)
    except ValueError:
        print(f"⚠️ IDs inválidos: PREDICATE_ID={predicate_id_env!r}, XAI_DATASET_ID={xai_dataset_id_env!r}; pulando registro em 'facts'.")
        return

    # 2) Definir caminho em records/facts/...
    #    Podemos organizar por predicate_id + xai_dataset_id
    records_dir = RECORDS_ROOT / "facts" / f"p{predicate_id}_xd{xai_dataset_id}"
    records_dir.mkdir(parents=True, exist_ok=True)

    facts_records_rel = f"records/facts/p{predicate_id}_xd{xai_dataset_id}/{OUTPUT_JSONL.name}"
    facts_records_abs = PROJECT_ROOT / facts_records_rel

    # 3) Copiar o arquivo original de fatos para a pasta records
    if OUTPUT_JSONL.exists():
        shutil.copy2(OUTPUT_JSONL, facts_records_abs)
    else:
        print(f"⚠️ OUTPUT_JSONL não encontrado em {OUTPUT_JSONL}; não será possível registrar o path corretamente.")
        return

    # 4) Registrar na tabela facts
    try:
        prov = ProvenanceDB()
        facts_id = prov.insert_facts(
            predicate_id=predicate_id,
            xai_dataset_id=xai_dataset_id,
            model=LLAMA_MODEL_NAME,    
            temperature=TEMPERATURE,  
            prompt=PROV_METRIC,      
            path=facts_records_rel,   
        )
        prov.close()
        os.environ["FACTS_ID"] = str(facts_id)
        print(f"💾 Facts registrados no banco com id={facts_id} (predicate_id={predicate_id}, xai_dataset_id={xai_dataset_id})")
    except Exception as e:
        print(f"⚠️ Erro ao registrar 'facts' no banco: {e}")


if __name__ == "__main__":
    main()
