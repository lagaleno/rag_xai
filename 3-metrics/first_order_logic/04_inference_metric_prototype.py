import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import shutil

import pandas as pd
import os
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
RECORDS_ROOT = PROJECT_ROOT / "records"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from provenance import ProvenanceDB

# ================== CONFIGURAÇÕES ==================

SCHEMA_FILE = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "predicate_schema.json"
RULES_FILE = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "logical_rules.json"

# Arquivo gerado pelo 03_extract_facts_llm.py
FACTS_JSONL = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "facts_extracted_llm.jsonl"

# Arquivo de saída com resultados por explicação
OUTPUT_CSV = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "logical_metrics_results.csv"
SUMMARY_OUT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "logical_metrics_summary_results.csv"
LONG_SUMMARY = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "logical_metrics_summary_long.csv"

# Tipo para um fato: (nome_predicado, (arg1, arg2, ...))
Fact = Tuple[str, Tuple[str, ...]]

# Se quiser considerar só alguns predicados como "relevantes" no closure,
# defina aqui. Se None, usa todos.
RELEVANT_PREDICATES = None
# Exemplo se quiser restringir:
# RELEVANT_PREDICATES = ["located_in", "type_of", "member_of", "founder_of"]

# Limiares para classificação lógica baseados em cobertura de fatos
THRESH_CORRECT = 0.6      # coverage >= 0.8 -> correct
THRESH_INCOMPLETE = 0.3   # 0.3 <= coverage < 0.8 -> incomplete
# coverage < 0.3 -> incorrect

# ===================================================


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def facts_from_dict_list(fact_dicts: List[Dict]) -> Set[Fact]:
    """
    Converte uma lista de dicts do tipo:
      {"predicate": "located_in", "args": ["A", "B"]}
    em um set de fatos do tipo:
      ("located_in", ("A", "B"))

    Garante que:
    - predicate é string
    - todos os args são convertidos para string
    - ignora entradas malformadas
    """
    facts: Set[Fact] = set()
    for fd in fact_dicts:
        pred = fd.get("predicate")
        raw_args = fd.get("args", [])

        if not isinstance(pred, str):
            continue
        if not isinstance(raw_args, list):
            continue

        # Força todos os argumentos a serem strings (evita dicts/listas não-hashable)
        args = [str(a) for a in raw_args]

        facts.add((pred, tuple(args)))
    return facts



def dict_list_from_facts(facts: Set[Fact]) -> List[Dict]:
    """
    Operação inversa de facts_from_dict_list, só para debug/impressão.
    """
    return [
        {"predicate": pred, "args": list(args)}
        for (pred, args) in sorted(facts)
    ]


# =============== MOTOR DE INFERÊNCIA ===============

def unify_premise_with_fact(premise: Dict, fact: Fact, env: Dict[str, str]) -> Dict[str, str] | None:
    """
    Tenta unificar uma premissa com um fato, respeitando um ambiente
    (env) pré-existente de variáveis -> constantes.

    premise: {"predicate": "located_in", "args": ["A", "B"]}
    fact:    ("located_in", ("Paris", "France"))
    env:     {"A": "Paris"}  (por exemplo)

    Retorna um novo env extendido se unificação for possível,
    ou None se falhar.
    """
    p_pred = premise["predicate"]
    p_args = premise.get("args", [])
    f_pred, f_args = fact

    if p_pred != f_pred:
        return None
    if len(p_args) != len(f_args):
        return None

    new_env = dict(env)

    for var, const in zip(p_args, f_args):
        if var in new_env:
            if new_env[var] != const:
                return None
        else:
            new_env[var] = const

    return new_env


def find_rule_matches(rule: Dict, facts: Set[Fact]) -> List[Dict[str, str]]:
    """
    Encontra todos os ambientes de variáveis (env) que satisfazem as
    premissas da regra com os fatos disponíveis.

    Retorna uma lista de dicts do tipo:
      [{"A": "Paris", "B": "France", "C": "Europe"}, ...]
    """
    premises = rule.get("premises", [])
    if not premises:
        return []

    envs: List[Dict[str, str]] = [dict()]

    for prem in premises:
        new_envs: List[Dict[str, str]] = []

        for env in envs:
            for fact in facts:
                extended = unify_premise_with_fact(prem, fact, env)
                if extended is not None:
                    new_envs.append(extended)

        if not new_envs:
            return []

        envs = new_envs

    return envs


def apply_rule_once(rule: Dict, facts: Set[Fact]) -> Set[Fact]:
    new_facts: Set[Fact] = set()

    envs = find_rule_matches(rule, facts)
    conclusion = rule.get("conclusion", {})
    c_pred = conclusion.get("predicate")
    c_args_vars = conclusion.get("args", [])

    for env in envs:
        c_args_instantiated = []
        ok = True
        for var in c_args_vars:
            if var not in env:
                ok = False
                break
            # força valor a ser string
            c_args_instantiated.append(str(env[var]))

        if not ok:
            continue

        fact_conclusion: Fact = (str(c_pred), tuple(c_args_instantiated))
        if fact_conclusion not in facts:
            new_facts.add(fact_conclusion)

    return new_facts


def forward_chaining(facts: Set[Fact], rules: List[Dict]) -> Set[Fact]:
    """
    Faz encadeamento direto até saturar (não há novos fatos).
    Retorna o closure (fatos originais + inferidos).
    """
    closure = set(facts)
    changed = True

    while changed:
        changed = False
        for rule in rules:
            inferred = apply_rule_once(rule, closure)
            if inferred:
                closure |= inferred
                changed = True

    return closure


# ========== "MÉTRICA" LÓGICA COMO EVIDÊNCIA =========

def logical_evidence(
    chunk_facts: Set[Fact],
    expl_facts: Set[Fact],
    rules: List[Dict],
    relevant_predicates: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Calcula evidências lógicas a partir de:
    - fatos do chunk,
    - fatos da explicação,
    - regras lógicas.

    Retorna:
    - closure_chunk: fatos inferidos a partir do chunk
    - relevant_closure: subconjunto de closure_chunk considerado relevante
    - intersection: fatos da explicação que batem com o closure relevante
    - coverage: proporção de fatos relevantes do chunk que a explicação cobre
    """
    closure_chunk = forward_chaining(chunk_facts, rules)

    if relevant_predicates is None:
        relevant_closure = set(closure_chunk)
    else:
        relevant_closure = {
            f for f in closure_chunk if f[0] in relevant_predicates
        }

    intersection = expl_facts & relevant_closure

    denom = len(relevant_closure) if len(relevant_closure) > 0 else 1
    coverage = len(intersection) / denom

    return {
        "closure_chunk": closure_chunk,
        "relevant_closure": relevant_closure,
        "intersection": intersection,
        "coverage": coverage,
    }


def classify_logical(evidence: Dict[str, Any]) -> str:
    """
    Usa a cobertura lógica (coverage) como medida de completude e
    devolve um rótulo: correct / incomplete / incorrect.
    """
    coverage = evidence.get("coverage", 0.0)
    if coverage >= THRESH_CORRECT:
        return "correct"
    elif coverage >= THRESH_INCOMPLETE:
        return "incomplete"
    else:
        return "incorrect"


# ===================== MAIN =====================

def main():
    print(f"📥 Loading predicate schema from: {SCHEMA_FILE}")
    schema = load_json(SCHEMA_FILE)

    print(f"📥 Loading logical rules from: {RULES_FILE}")
    rules_obj = load_json(RULES_FILE)
    rules = rules_obj.get("rules", [])

    print(f"📥 Reading facts from: {FACTS_JSONL}")
    if not FACTS_JSONL:
        raise FileNotFoundError(f"facts_extracted_llm.jsonl not found at {FACTS_JSONL}")

    results_rows = []

    with open(FACTS_JSONL, "r", encoding="utf-8") as f_in:
        for line_idx, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            ex_id = rec.get("id")
            expl_label = rec.get("explanation_label")
            chunk_facts_dicts = rec.get("chunk_facts", [])
            expl_facts_dicts = rec.get("explanation_facts", [])

            if rec.get("error"):
                print(f"⚠️ Skipping id={ex_id}, label={expl_label} due to extraction error.")
                results_rows.append(
                    {
                        "id": ex_id,
                        "explanation_label": expl_label,
                        "num_chunk_facts": 0,
                        "num_expl_facts": 0,
                        "coverage": 0.0,
                        "logic_label": "unknown",
                    }
                )
                continue

            chunk_facts = facts_from_dict_list(chunk_facts_dicts)
            expl_facts = facts_from_dict_list(expl_facts_dicts)

            evidence = logical_evidence(
                chunk_facts,
                expl_facts,
                rules,
                relevant_predicates=RELEVANT_PREDICATES,
            )

            logic_label = classify_logical(evidence)

            results_rows.append(
                {
                    "id": ex_id,
                    "explanation_label": expl_label,   # label do dataset
                    "num_chunk_facts": len(chunk_facts),
                    "num_expl_facts": len(expl_facts),
                    "coverage": evidence["coverage"],
                    "logic_label": logic_label,
                }
            )

            print(f"   Processed {line_idx} explanations...")


    # Converte para DataFrame e salva CSV
    df = pd.DataFrame(results_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Logical classification saved to: {OUTPUT_CSV}")

    # ===================== CONFUSION MATRIX =====================
    if not df.empty:
        print("\n📊 Confusion between dataset label (TRUE) and logic_label (PREDICTED):")

        # matriz padrão
        confusion = pd.crosstab(df["explanation_label"], df["logic_label"])
        print(confusion)

        # salva matriz tradicional
        confusion.to_csv(SUMMARY_OUT)
        print(f"📁 Confusion matrix saved to: {SUMMARY_OUT}")

        # ----- versão “long format” muito mais clara -----
        long_rows = []
        for true_label in confusion.index:
            for pred_label in confusion.columns:
                count = confusion.loc[true_label, pred_label]
                long_rows.append({
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(count)
                })

        long_df = pd.DataFrame(long_rows)

        
        long_df.to_csv(LONG_SUMMARY, index=False)
        print(f"📁 Long-format confusion saved to: {LONG_SUMMARY}")
        
    # ============ Proveniência (first_order_logic) ============

    # 1) Recuperar IDs do ambiente
    experiment_id_env = os.environ.get("EXPERIMENT_ID")
    xai_dataset_id_env = os.environ.get("XAI_DATASET_ID")
    predicate_id_env = os.environ.get("PREDICATE_ID")
    rules_id_env = os.environ.get("RULES_ID")
    facts_id_env = os.environ.get("FACTS_ID")

    missing = [
        name for name, value in [
            ("EXPERIMENT_ID", experiment_id_env),
            ("XAI_DATASET_ID", xai_dataset_id_env),
            ("PREDICATE_ID", predicate_id_env),
            ("RULES_ID", rules_id_env),
            ("FACTS_ID", facts_id_env),
        ] if value is None
    ]
    if missing:
        print(f"⚠️ Variáveis de ambiente faltando ({', '.join(missing)}); pulando registro em 'first_order_logic'.")
        return

    try:
        experiment_id = int(experiment_id_env)
        xai_dataset_id = int(xai_dataset_id_env)
        predicate_id = int(predicate_id_env)
        rules_id = int(rules_id_env)
        facts_id = int(facts_id_env)
    except ValueError:
        print("⚠️ IDs inválidos em ambiente; pulando registro em 'first_order_logic'.")
        return

    # 2) Definir caminho em records/experiments/{experiment_id}/fol/...
    records_dir = RECORDS_ROOT / "experiments" / str(experiment_id) / "fol"
    records_dir.mkdir(parents=True, exist_ok=True)

    fol_records_rel = f"records/experiments/{experiment_id}/fol/{OUTPUT_CSV.name}"
    fol_records_abs = PROJECT_ROOT / fol_records_rel

    if OUTPUT_CSV.exists():
        shutil.copy2(OUTPUT_CSV, fol_records_abs)
    else:
        print(f"⚠️ LOGIC_RESULTS_CSV não encontrado em {OUTPUT_CSV}; não será possível registrar o path corretamente.")
        return

    # 3) Construir dicionário de thresholds/config da métrica (se houver)
    #    Se você tiver constantes como:
    #    F1_THRESHOLD, PRECISION_THRESHOLD, RECALL_THRESHOLD, etc., coloque aqui.
    #    Caso contrário, podemos registrar um dict vazio {}.

    thresholds = {
        "correct": THRESH_CORRECT,
        "incomplete": THRESH_INCOMPLETE,
    }

    # 4) Registrar na tabela first_order_logic
    try:
        prov = ProvenanceDB()
        fol_id = prov.insert_first_order_logic_run(
            experiment_id=experiment_id,
            xai_dataset_id=xai_dataset_id,
            predicate_id=predicate_id,
            rules_id=rules_id,
            facts_id=facts_id,
            thresholds=thresholds,
            path=fol_records_rel,   
        )
        prov.close()
        print(f"💾 First-order logic run registrada no banco com id={fol_id}")
    except Exception as e:
        print(f"⚠️ Erro ao registrar 'first_order_logic' no banco: {e}")



if __name__ == "__main__":
    main()
