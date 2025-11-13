import os
import json
import time
import random
from typing import Dict, Any, List

import requests
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd

# ==========================
# 1) CONFIGURAÇÕES GERAIS
# ==========================

# modelo LLaMA no Ollama
LLAMA_MODEL = "llama3"  # troque se usar outro (ex: "llama3:8b")

# Quantidade de amostras de HotpotQA
N_SAMPLES = 10   # começa com 10 ou 30 pra validar

SEED = 42
random.seed(SEED)

# Arquivos de saída
JSONL_OUT = "explainrag_hotpot_llama.jsonl"
CSV_SUMMARY_OUT = "explainrag_hotpot_llama_summary.csv"

# ==========================
# 2) CHAMADA AO LLaMA (Ollama)
# ==========================

def call_llm_json_llama(system: str, user: str, temperature: float = 0.3) -> Dict[str, Any]:
    """
    Chama o modelo LLaMA via Ollama (http://localhost:11434/api/chat)
    e tenta interpretar a saída como JSON.
    """
    url = "http://localhost:11434/api/chat"

    data = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
        },
        "stream": False
    }

    resp = requests.post(url, json=data)
    resp.raise_for_status()
    out = resp.json()

    # Ollama retorna algo como {"message": {"role": "...", "content": "..."}, ...}
    text = out["message"]["content"].strip()

    # Tenta extrair JSON mesmo se vier com texto antes/depois
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except Exception:
            print("⚠️ Não foi possível extrair JSON bem formado.")

    json_str = text[start:end+1]
    return json.loads(json_str)

# ==========================
# 3) CARREGAR HOTPOTQA
# ==========================

print("📥 Carregando HotpotQA...")
ds = load_dataset("hotpot_qa", "distractor")
train = ds["train"].shuffle(seed=SEED).select(range(min(N_SAMPLES, len(ds["train"]))))
print(train)

# ==========================
# 4) FUNÇÃO: construir CHUNK
#    (compatível com formatos diferentes de supporting_facts)
# ==========================

def build_ctx_map(example: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Cria um dicionário: título -> lista de sentenças.
    example["context"] normalmente é: [[title, [sent1, sent2, ...]], ...]
    """
    ctx_map = {}
    for title, sents in example["context"]:
        ctx_map[title] = sents
    return ctx_map

def build_chunk_from_supporting_facts(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constrói um chunk textual a partir de supporting_facts.
    Retorna um dicionário com texto e metadados (lista de (title, sent_idx, sentence)).
    """
    # Mapear título -> lista de sentenças
    ctx_map = {}
    for idx, title in enumerate(example["context"]["title"]):
        sents = example["context"]["sentences"][idx]       # lista de sentenças
        ctx_map[title] = sents

    sf = example.get("supporting_facts", [])
    print(sf)

    pieces = []
    provenance = []
    titles = sf["title"]
    sent_ids = sf["sent_id"]
    for title, sent_idx in zip(titles, sent_ids):
        sents = ctx_map.get(title, [])
        if 0 <= sent_idx < len(sents):
            sent_text = sents[sent_idx]
            pieces.append(sent_text)
            provenance.append({
                "title": title,
                "sent_idx": sent_idx,
                "sentence": sent_text
            })


    # Fallback: se por algum motivo ficou vazio, concatene 2-3 sentenças do primeiro artigo
    if not pieces and example["context"]:
        title, sents = example["context"][0]
        for i, s in enumerate(sents[:3]):
            pieces.append(s)
            provenance.append({"title": title, "sent_idx": i, "sentence": s})

    chunk_text = " ".join(pieces).strip()
    return {"text": chunk_text, "provenance": provenance}

# ==========================
# 5) PROMPTS PARA GERAÇÃO DAS 3 EXPLICAÇÕES
# ==========================

SYSTEM_GENERATE = """You are an expert evaluator helping create a benchmark dataset of explanations for RAG-style question answering.

You will receive a question, an answer, and a supporting text.  
Your task is to generate THREE natural-language explanations of why the answer is correct.

IMPORTANT RULES:
- Do NOT mention or refer to the “chunk”, “supporting text”, “provided passage”, “context”, or any meta-information.
- Do NOT write sentences that reveal the quality of the explanation, such as: “this is incomplete”, “more information is needed”, “this does not fully explain”, “additional details are required”, “however, this lacks…”, "this is icorrect"
- All three explanations must read as fully natural to an end-user.
- You MAY use facts or quote sentences from the supporting text, but as if they were general knowledge.

DEFINITION OF THE THREE EXPLANATIONS:
1) **CORRECT** – A complete and fully supported explanation that contains all essential reasoning.
2) **INCOMPLETE** – A plausible explanation that sounds natural but deliberately omits at least one essential supporting fact.  
   - The explanation MUST NOT state or imply that something is missing.  
   - It should feel like a normal explanation but be partially insufficient.
   - The explanation MUST use FEWER supporting facts than the correct explanation.
   - The explanation MUST be SHORTER than the correct explanation (for example, about half as many details).
3) **INCORRECT** – An explanation that contains at least one unsupported or contradictory statement, but still written naturally (no meta-talk).

OUTPUT FORMAT:
Return ONLY a JSON object:
{"correct": "...", "incomplete": "...", "incorrect": "..."}

Write all explanations in **English**.
Each explanation should contain **2–6 sentences**."""

USER_GENERATE_TMPL = """[QUESTION]
{question}

[ANSWER]
{answer}

[CHUNK]
{chunk}

Return ONLY a single valid JSON object. Do not include any extra text before or after the JSON. The JSON format is:
{{"correct": "...","incomplete": "...","incorrect": "..."}}"""

# ==========================
# 6) LOOP PRINCIPAL — GERAR DATASET
# ==========================

rows_summary = []

with open(JSONL_OUT, "w", encoding="utf-8") as fout:
    for ex in tqdm(train, desc="Gerando dataset com LLaMA"):
        q = ex["question"]
        gold_answer = ex["answer"]
        chunk = build_chunk_from_supporting_facts(ex)
        chunk_text = chunk["text"]

        if not chunk_text:
            continue

        # Chama LLaMA para gerar as 3 explicações
        try:
            gen = call_llm_json_llama(
                system=SYSTEM_GENERATE,
                user=USER_GENERATE_TMPL.format(
                    question=q,
                    answer=gold_answer,
                    chunk=chunk_text
                ),
                temperature=0.4,
            )
        except Exception as e:
            print(f"⚠️ Erro ao gerar explicações para id={ex.get('_id','')} -> {e}")
            continue

        expl_correct = gen.get("correct", "").strip()
        expl_incomplete = gen.get("incomplete", "").strip()
        expl_incorrect = gen.get("incorrect", "").strip()

        rec = {
            "id": ex.get("id", ""),
            "dataset": "hotpotqa-distractor",
            "question": q,
            "answer": gold_answer,
            "chunk": {
                "text": chunk_text,
                "provenance": chunk["provenance"],
            },
            "explanations": [
                {"label": "correct", "text": expl_correct},
                {"label": "incomplete", "text": expl_incomplete},
                {"label": "incorrect", "text": expl_incorrect,},
            ],
            "meta": {
                "model_generate": LLAMA_MODEL,
                "seed": SEED,
            }
        }

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        rows_summary.append(rec)

        time.sleep(0.2)  # só pra não saturar o servidor local

# Salva CSV resumo
pd.DataFrame(rows_summary).to_csv(CSV_SUMMARY_OUT, index=False)
print(f"\n✅ Dataset salvo em: {JSONL_OUT}")
print(f"✅ CSV salvo em: {CSV_SUMMARY_OUT}")
