import json
from pathlib import Path
import sys
import os
import requests
import shutil

import pandas as pd

# ================== PATH SETUP ==================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
RECORDS_ROOT = PROJECT_ROOT / "records"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from provenance import ProvenanceDB  # opcional, se quiser registrar depois

# ================== CONFIGURAÇÕES ==================

# Arquivo com o dataset de explicações geradas (ajuste o caminho se necessário)
DATASET_JSONL = PROJECT_ROOT / "1-creating_dataset" / "explainrag_hotpot_llama.jsonl"

# Arquivo de saída com os rótulos do juiz
JUDGE_OUTPUT_CSV = PROJECT_ROOT / "3-metrics" / "llm_judge" / "llm_judge_labels.csv"
JUDGE_SUMMARY_OUTPUT_CSV = PROJECT_ROOT / "3-metrics" / "llm_judge" / "llm_judge_summary.csv"

JUDGE_MATRIX_CSV = PROJECT_ROOT / "3-metrics" / "llm_judge" / "llm_judge_confusion_matrix.csv"

# Modelo do LLM que atuará como juiz 
# Exemplo: se você usou "llama3" para gerar explicações, aqui pode usar outro,
# ou uma variante com mais parâmetros.
JUDGE_MODEL_NAME = "llama3.1"

# Temperatura do juiz (mais baixa para ser mais estável)
JUDGE_TEMPERATURE = 0.1

# Limite opcional de exemplos para teste (None = usar todos)
MAX_SAMPLES = None  # ou por ex. 50 para testar mais rápido

# ================== FUNÇÕES AUXILIARES ==================


def load_explanations(jsonl_path: Path, max_samples=None):
    """
    Lê o dataset explainrag_dataset.jsonl e achata em um DataFrame com:
    id, question, answer, chunk_text, explanation_label, explanation_text
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {jsonl_path}")

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            ex_id = rec.get("id")
            question = rec.get("question", "")
            answer = rec.get("answer", "")

            # ajuste aqui se o campo do chunk tiver outro nome (ex.: "context")
            chunk_text = rec.get("chunk") or rec.get("context") or ""

            explanations = rec.get("explanations", [])

            # Cada explicação vira uma linha
            for expl in explanations:
                label = expl.get("label")
                text = expl.get("text", "")

                rows.append(
                    {
                        "id": ex_id,
                        "question": question,
                        "answer": answer,
                        "chunk_text": chunk_text,
                        "explanation_label": label,      # label original do dataset
                        "explanation_text": text,
                    }
                )

    df = pd.DataFrame(rows)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    return df


def build_judge_prompt(row: dict) -> str:
    """
    Constrói o prompt para o LLM atuar como juiz da explicação.
    Ele deve classificar a explicação em:
    - correct
    - incomplete
    - incorrect
    com base no chunk (contexto), na pergunta e na resposta.
    """
    question = row["question"]
    answer = row["answer"]
    chunk = row["chunk_text"]
    explanation = row["explanation_text"]

    prompt = f"""
You are an expert judge of natural-language explanations for question answering.

Your task is to evaluate whether a given explanation is:
- "correct": the explanation is factually consistent with the chunk,
             and it mentions all key facts needed to justify the answer.
- "incomplete": the explanation is partially supported by the chunk
                and relevant to the question and answer, but it misses
                some important facts or steps.
- "incorrect": the explanation is not supported by the chunk, contradicts it,
               or is mostly irrelevant to justifying the answer.

You will receive:
- a question
- an answer
- a supporting passage (chunk)
- a candidate explanation

You MUST decide which label best fits the explanation with respect to
the chunk and the answer.

Return your judgment as JSON ONLY, with the following format:

{{
  "label": "correct" | "incomplete" | "incorrect",
  "justification": "short natural-language justification of your decision"
}}

Do NOT include any other fields. Do NOT include any text before or after the JSON.

QUESTION:
{question}

ANSWER:
{answer}

CHUNK:
{chunk}

EXPLANATION:
{explanation}
"""
    return prompt.strip()


def call_judge_llm(prompt: str, model_name: str = JUDGE_MODEL_NAME, temperature: float = JUDGE_TEMPERATURE) -> str:
    """
    Chama o modelo juiz via Ollama (http://localhost:11434/api/chat)
    e retorna o texto bruto (que deve ser um JSON).
    """
    print("🧠 Calling LLM judge...")
    url = "http://localhost:11434/api/chat"

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are an expert, strict but fair judge of natural-language explanations."},
            {"role": "user", "content": prompt},
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
    return text


def parse_judge_output(raw_output: str) -> dict:
    """
    Extrai o JSON de dentro da saída do juiz e faz json.loads.
    Se tiver texto extra, tenta pegar só o bloco entre o primeiro '{' e o último '}'.
    """
    text = raw_output.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        print("⚠️ Could not find a valid JSON block in the judge output.")
        print("Raw output:")
        print(text)
        raise ValueError("No JSON block found in judge output.")

    json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON from judge output:")
        print(e)
        print("Extracted candidate JSON:")
        print(json_str)
        raise


# ================== MAIN ==================


def main():
    print(f"📥 Loading explanations from: {DATASET_JSONL}")
    df = load_explanations(DATASET_JSONL, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(df)} (id, explanation_label, explanation_text) pairs.")

    results = []

    for idx, row in df.iterrows():
        print(f"\n=== [{idx+1}/{len(df)}] id={row['id']} label={row['explanation_label']} ===")
        prompt = build_judge_prompt(row.to_dict())
        raw_output = call_judge_llm(prompt)
        try:
            judge_res = parse_judge_output(raw_output)
            judge_label = judge_res.get("label", "").strip().lower()
            justification = judge_res.get("justification", "").strip()
        except Exception as e:
            print(f"❌ Error parsing judge output for id={row['id']}: {e}")
            judge_label = "error"
            justification = raw_output[:500]  # salva um pedaço para debug

        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "chunk_text": row["chunk_text"],
                "explanation_label": row["explanation_label"],  # label do dataset
                "explanation_text": row["explanation_text"],
                "judge_label": judge_label,                     # label do juiz
                "judge_justification": justification,
            }
        )

    out_df = pd.DataFrame(results)
    JUDGE_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(JUDGE_OUTPUT_CSV, index=False)
    print(f"\n✅ LLM judge labels saved to: {JUDGE_OUTPUT_CSV}")

    # Pequeno resumo: matriz de confusão dataset vs juiz
    print("\n📊 Confusion matrix: dataset label (rows) vs judge_label (cols):")
    confusion = pd.crosstab(out_df["explanation_label"], out_df["judge_label"])
    print(confusion)

    # 1) Salvar a matriz "largona" como está 
    
    confusion.to_csv(JUDGE_MATRIX_CSV)
    print(f"✅ Saved judge confusion matrix (wide) to: {JUDGE_MATRIX_CSV}")

    # 2) Gerar versão "long" com true_label, pred_label, count
    confusion_long = (
        confusion
        .reset_index()  # traz explanation_label para coluna
        .melt(
            id_vars="explanation_label",      # coluna que fica fixa (true)
            var_name="judge_label",           # nome da coluna das categorias preditas
            value_name="count"                # nome da coluna de contagem
        )
        .rename(columns={
            "explanation_label": "true_label",
            "judge_label": "pred_label",
        })
    )

    confusion_long.to_csv(JUDGE_SUMMARY_OUTPUT_CSV, index=False)
    print(f"✅ Saved judge confusion matrix (long) to: {JUDGE_SUMMARY_OUTPUT_CSV}")

    # ============ Proveniência (llm_judge) ============

    experiment_id_env = os.environ.get("EXPERIMENT_ID")
    xai_dataset_id_env = os.environ.get("XAI_DATASET_ID")

    if experiment_id_env is not None and xai_dataset_id_env is not None:
        try:
            experiment_id = int(experiment_id_env)
            xai_dataset_id = int(xai_dataset_id_env)

            # Pasta: records/experiments/{experiment_id}/llm_judge/
            records_dir = RECORDS_ROOT / "experiments" / str(experiment_id) / "llm_judge"
            records_dir.mkdir(parents=True, exist_ok=True)

            # Copiamos o CSV de julgamentos para dentro de records
            llm_judge_records_rel = f"records/experiments/{experiment_id}/llm_judge/{JUDGE_SUMMARY_OUTPUT_CSV.name}"
            llm_judge_records_abs = PROJECT_ROOT / llm_judge_records_rel

            shutil.copy2(JUDGE_SUMMARY_OUTPUT_CSV, llm_judge_records_abs)

            prov = ProvenanceDB()
            prov.insert_llm_judge_run(
                experiment_id=experiment_id,
                xai_dataset_id=xai_dataset_id,
                model=JUDGE_MODEL_NAME,       
                temperature=JUDGE_TEMPERATURE,   
                prompt=prompt,        
                path=llm_judge_records_rel, 
            )
            prov.close()
            print(f"💾 LLM Judge registrada no banco para experiment_id={experiment_id}")
        except Exception as e:
            print(f"⚠️ Erro ao registrar llm_judge no banco: {e}")
    else:
        print("⚠️ EXPERIMENT_ID ou XAI_DATASET_ID não definido no ambiente; pulando registro de llm_judge.")


if __name__ == "__main__":
    main()
