import json
import os

def build_examples(jsonl_path):
    """
    Lê o JSONL e monta uma lista de exemplos no formato:

    {
        "dataset_id": ...,
        "chunk": "...",
        "explanations": {
            "correct": "...",
            "incomplete": "...",
            "incorrect": "..."
        }
    }
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    examples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            dataset_id = ex.get("id", None)

            # Chunk pode vir como dict {"text": "..."} ou string
            chunk_obj = ex.get("chunk", "")
            if isinstance(chunk_obj, dict):
                chunk_text = chunk_obj.get("text", "")
            elif isinstance(chunk_obj, str):
                chunk_text = chunk_obj
            else:
                chunk_text = ""

            if not chunk_text:
                continue

            # Inicializa explicações
            exp_dict = {"correct": None, "incomplete": None, "incorrect": None}

            for explanation in ex.get("explanations", []):
                label = explanation.get("label", "").strip()
                text = explanation.get("text", "").strip()
                if label in exp_dict and text:
                    exp_dict[label] = text

            examples.append({
                "dataset_id": dataset_id,
                "chunk": chunk_text,
                "explanations": exp_dict
            })

    return examples


def flatten_examples(examples):
    """
    Transforma a lista de exemplos estruturados em pares chunk–explicação:

    [
        {
            "dataset_id": ...,
            "label": "correct/incomplete/incorrect",
            "chunk_text": "...",
            "explanation_text": "..."
        },
        ...
    ]
    """
    rows = []

    for ex in examples:
        dataset_id = ex["dataset_id"]
        chunk_text = ex["chunk"]
        exps = ex["explanations"]

        for label in ["correct", "incomplete", "incorrect"]:
            exp_text = exps.get(label)
            if exp_text:
                rows.append({
                    "dataset_id": dataset_id,
                    "label": label,
                    "chunk_text": chunk_text,
                    "explanation_text": exp_text
                })

    return rows
