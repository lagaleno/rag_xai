import json
import os
import requests
from pathlib import Path
import sys
import shutil

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
RECORDS_ROOT = PROJECT_ROOT / "records"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from provenance import ProvenanceDB  # noqa: E402

# ================== CONFIGURAÇÕES ==================

SCHEMA_FILE = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "predicate_schema.json"

RULES_OUT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "logical_rules.json"
RULES_RAW_OUT = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "logical_rules_raw.json"

LLAMA_MODEL_NAME = "llama3"


TEMPERATURE = 0.5

# Ativar / desativar prints de debug na validação
DEBUG_VALIDATION = True

# ===================================================


def load_schema(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(schema: dict) -> str:
    """
    Constrói um prompt bem explícito, mostrando:
    - a lista de predicados com seus tipos,
    - templates com variáveis X1, X2... respeitando a aridade,
    - exemplos de regras VÁLIDAS usando esse esquema.
    """
    predicates = schema.get("predicates", [])

    # Lista legível de predicados com tipos
    human_lines = []
    template_lines = []
    sample_rule_premises = []
    sample_rule_conclusion = None

    for i, p in enumerate(predicates, start=1):
        name = p.get("name")
        args = p.get("args", [])
        args_str = ", ".join(args)
        human_lines.append(f"- {name}({args_str})")

        # Template com variáveis X1, X2, ...
        var_args = [f"X{j+1}" for j in range(len(args))]
        var_args_str = ", ".join(var_args)
        template_lines.append(f"- {name}({var_args_str})")

        # Guardar 2 predicados simples para um exemplo de regra
        if len(sample_rule_premises) < 2:
            sample_rule_premises.append({
                "predicate": name,
                "args": var_args,
            })
        elif sample_rule_conclusion is None:
            sample_rule_conclusion = {
                "predicate": name,
                "args": var_args,
            }

    predicates_human = "\n   ".join(human_lines)
    predicates_templates = "\n   ".join(template_lines)

    # fallback se não tivermos um exemplo de conclusão
    if sample_rule_conclusion is None and sample_rule_premises:
        sample_rule_conclusion = sample_rule_premises[0]

    # Exemplo de regra no formato JSON
    example_rule = {
        "name": "example_rule",
        "description": "Example rule using the given predicates with variables.",
        "premises": sample_rule_premises,
        "conclusion": sample_rule_conclusion,
    }

    example_rule_json = json.dumps(
        {"rules": [example_rule]},
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You will generate a small set of FIRST-ORDER LOGIC inference rules (Horn clauses)
based ONLY on the predicate schema provided below.

STRICT REQUIREMENTS YOU MUST FOLLOW:

1. You MUST use ONLY these predicates, exactly as written, with exactly these
   argument positions (same number of arguments, in the same order).
   These are the allowed predicates with their argument *types*:

   {predicates_human}

2. When writing rules, you MUST use logical variables like X1, X2, X3, ...
   instead of the type names. Here are the exact templates you MUST follow:

   {predicates_templates}

   For example, if the schema has:
     - located_in(entity, location)
   then in a rule you MUST write it as:
     - located_in(X1, X2)

   The number of arguments MUST be exactly the same as in the schema.

3. DO NOT invent additional predicates such as “typically_found_in”,
   “associated_with”, “connected_to”, etc. Use ONLY the predicates listed
   above.

4. Avoid vague rules that conclude `related_to(...)`.
   Only produce a rule with `related_to` IF the conclusion is clearly justified
   by the premises. If unsure, DO NOT use `related_to`.

5. NO tautologies.
   Do NOT produce rules where the conclusion is identical
   to one of the premises (same predicate AND same arguments).

6. Rules must be informational: they must generate NEW logical consequences,
   e.g., transitivity, inheritance, propagation, membership inference, etc.

7. Produce as many rules as needed, as long as each rule is valid, non-tautological, and logically informative.

8. Output MUST be valid JSON ONLY. No explanation, no text before or after.

JSON FORMAT TO OUTPUT:
{{
  "rules": [
    {{
      "name": "rule_name",
      "description": "Short natural-language summary",
      "premises": [
        {{"predicate": "...", "args": ["X1","X2"] }},
        ...
      ],
      "conclusion": {{
        "predicate": "...",
        "args": ["X1","X2"]
      }}
    }}
  ]
}}

Here is ONE EXAMPLE of a valid JSON structure using the schema
(THIS IS ONLY AN EXAMPLE, YOU MUST PROPOSE YOUR OWN RULES):

{example_rule_json}

Remember: your answer MUST be ONLY a JSON object with a "rules" list.
"""
    return prompt.strip()


def call_llama(prompt: str, temperature: float = TEMPERATURE) -> str:
    """
    Chama o modelo LLaMA via Ollama (http://localhost:11434/api/chat)
    e retorna o texto bruto.
    """
    print("🧠 Calling LLaMA to propose logical rules...")
    url = "http://localhost:11434/api/chat"

    data = {
        "model": LLAMA_MODEL_NAME,
        "messages": [
            # Deixamos o prompt como 'user' para alguns modelos seguirem melhor.
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


def parse_rules(raw_output: str) -> dict:
    """
    Extrai o bloco JSON de dentro da saída do LLM.
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


def validate_and_filter_rules(schema: dict, rules_obj: dict) -> dict:
    """
    Valida e filtra regras geradas pelo LLM:
    - só aceita regras cujos predicados existem no schema,
    - aridade bate com o schema,
    - descarta tautologias simples (conclusão == premissa),
    - NÃO força restrições adicionais sobre variáveis.

    Retorna um novo objeto {"rules": [...]} só com as regras válidas.
    """
    pred_arity = {
        p["name"]: len(p.get("args", []))
        for p in schema.get("predicates", [])
    }

    good_rules = []
    all_rules = rules_obj.get("rules", [])
    if DEBUG_VALIDATION:
        print(f"🔎 LLM returned {len(all_rules)} rules before validation.")

    for idx, rule in enumerate(all_rules, start=1):
        premises = rule.get("premises", [])
        conclusion = rule.get("conclusion", {})

        concl_pred = conclusion.get("predicate")
        concl_args = conclusion.get("args", [])

        reason = None  # motivo para descartar, se houver

        # 0) Precisa ter pelo menos 1 premissa
        if not premises:
            reason = "no premises provided"

        # 1) Conclusão: predicado conhecido e aridade correta
        if reason is None:
            if concl_pred not in pred_arity:
                reason = f"unknown conclusion predicate '{concl_pred}'"
            elif len(concl_args) != pred_arity[concl_pred]:
                reason = (
                    f"wrong arity for conclusion '{concl_pred}': "
                    f"got {len(concl_args)}, expected {pred_arity[concl_pred]}"
                )

        # 2) Premissas: predicados conhecidos e aridade certa
        if reason is None:
            for prem in premises:
                ppred = prem.get("predicate")
                pargs = prem.get("args", [])
                if ppred not in pred_arity:
                    reason = f"unknown premise predicate '{ppred}'"
                    break
                if len(pargs) != pred_arity[ppred]:
                    reason = (
                        f"wrong arity for premise '{ppred}': "
                        f"got {len(pargs)}, expected {pred_arity[ppred]}"
                    )
                    break

        # 3) Conclusão não pode ser idêntica a uma das premissas (tautologia óbvia)
        if reason is None:
            is_tautology = any(
                (prem.get("predicate") == concl_pred and prem.get("args", []) == concl_args)
                for prem in premises
            )
            if is_tautology:
                reason = "tautology (conclusion identical to a premise)"

        if reason is not None:
            if DEBUG_VALIDATION:
                print(f"  ❌ Rule {idx} dropped: {reason}")
            continue

        good_rules.append(rule)
        if DEBUG_VALIDATION:
            print(f"  ✅ Rule {idx} accepted: {rule.get('name')}")

    if DEBUG_VALIDATION:
        print(f"✅ {len(good_rules)} / {len(all_rules)} rules accepted after validation.")

    return {"rules": good_rules}


def main():
    print(f"📥 Loading predicate schema from: {SCHEMA_FILE}")
    schema = load_schema(SCHEMA_FILE)

    prompt = build_prompt(schema)

    raw_output = call_llama(prompt)

    # Parseia regras "cruas"
    raw_rules = parse_rules(raw_output)

    # Salva as regras cruas para inspeção
    RULES_RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RULES_RAW_OUT, "w", encoding="utf-8") as f:
        json.dump(raw_rules, f, indent=2, ensure_ascii=False)
    print(f"📁 Raw logical rules saved to: {RULES_RAW_OUT}")

    # Valida e filtra
    rules = validate_and_filter_rules(schema, raw_rules)

    if len(rules.get("rules", [])) == 0:
        print("⚠️ All rules were filtered out. Keeping RAW rules instead for inspection.")
        # como fallback, mantemos as regras cruas:
        rules = raw_rules

    # Salva JSON final de regras (após validação ou fallback)
    with open(RULES_OUT, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"✅ Logical rules saved to: {RULES_OUT}")
    print("\nExtracted rules:")
    for rule in rules.get("rules", []):
        print(f"- {rule.get('name')}: {rule.get('description')}")

    # ============ Proveniência (rules) ============

    predicate_id_env = os.environ.get("PREDICATE_ID")

    if predicate_id_env is None:
        print("⚠️ PREDICATE_ID não definido no ambiente; pulando registro em 'rules'.")
        return

    try:
        predicate_id = int(predicate_id_env)
    except ValueError:
        print(f"⚠️ PREDICATE_ID inválido: {predicate_id_env!r}; pulando registro em 'rules'.")
        return

    records_dir = RECORDS_ROOT / "rules" / str(predicate_id)
    records_dir.mkdir(parents=True, exist_ok=True)

    rules_records_rel = f"records/rules/{predicate_id}/{RULES_OUT.name}"
    rules_records_abs = PROJECT_ROOT / rules_records_rel

    if RULES_OUT.exists():
        shutil.copy2(RULES_OUT, rules_records_abs)
    else:
        print(f"⚠️ RULES_OUT não encontrado em {RULES_OUT}; não será possível registrar o path corretamente.")
        return

    try:
        prov = ProvenanceDB()
        rules_id = prov.insert_rules(
            predicate_id=predicate_id,
            model=LLAMA_MODEL_NAME,
            temperature=TEMPERATURE,
            prompt=prompt,
            path=rules_records_rel,
        )
        prov.close()
        os.environ["RULES_ID"] = str(rules_id)
        print(f"💾 Rules registradas no banco com id={rules_id} (predicate_id={predicate_id})")
    except Exception as e:
        print(f"⚠️ Erro ao registrar 'rules' no banco: {e}")


if __name__ == "__main__":
    main()
