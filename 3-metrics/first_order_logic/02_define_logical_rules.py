import json
import os
import subprocess
from pathlib import Path

# ================== CONFIGURAÇÕES ==================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

# Arquivo de entrada com o schema de predicados (o que você já gerou)
SCHEMA_FILE = PROJECT_ROOT / "3-metrics" / "first_order_logic" / "predicate_schema.json"

# Arquivo de saída com as regras sugeridas
RULES_OUT = "logical_rules.json"

# Modelo do LLaMA no Ollama
LLAMA_MODEL_NAME = "llama3"

# Quantidade alvo de regras (apenas para orientar o LLM)
TARGET_MIN_RULES = 3
TARGET_MAX_RULES = 7

# ===================================================


def load_schema(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(schema: dict) -> str:
    """
    Constrói o prompt para o LLaMA propor regras lógicas gerais
    (Horn clauses) com base APENAS no esquema de predicados.
    """
    predicates = schema.get("predicates", [])

    # Monta a listagem de predicados no formato:
    # - located_in(entity, location)
    # - type_of(entity, type)
    lines = []
    for p in predicates:
        name = p.get("name")
        args = p.get("args", [])
        args_str = ", ".join(args)
        lines.append(f"- {name}({args_str})")

    predicates_with_args = "\n   ".join(lines)

    prompt = f"""
You will generate a small set of FIRST-ORDER LOGIC inference rules (Horn clauses)
based ONLY on the predicate schema provided below.

STRICT REQUIREMENTS YOU MUST FOLLOW:

1. **Use ONLY these predicates exactly as written, with exactly these arguments
   in exactly this order. Do NOT invent new predicates. Do NOT rename predicates.**
   The allowed predicates are:

   {predicates_with_args}

2. **DO NOT invent additional predicates** such as “typically_found_in”,
   “associated_with”, “connected_to”, etc.

3. **Avoid vague rules** that conclude “related_to(...)”.  
   Only produce a rule with `related_to` IF the conclusion is clearly justified
   and unavoidable from the semantics of the premises.  
   If unsure, DO NOT use `related_to`.

4. **NO tautologies**.  
   You MUST NOT produce rules where the conclusion is identical to a premise.

5. **NO purely associative rules** such as:  
      created_by(x, a) AND created_by(y, b) → related_to(x, y)  
   These are invalid.

6. **Rules must be informational: they must generate NEW logical consequences**,  
   e.g., transitivity, inheritance, propagation, membership inference, etc.

7. You MUST produce **between 2 and 4 rules**, not more.

8. Output MUST be valid JSON ONLY. No explanation, no text before or after.

JSON FORMAT TO OUTPUT:
{{
  "rules": [
    {{
      "name": "rule_name",
      "description": "Short natural-language summary",
      "premises": [
        {{"predicate": "...", "args": ["...","..."] }},
        ...
      ],
      "conclusion": {{
        "predicate": "...",
        "args": ["...","..."]
      }}
    }}
  ]
}}
"""
    return prompt.strip()

def validate_and_filter_rules(schema: dict, rules_obj: dict) -> dict:
    """
    Valida e filtra regras geradas pelo LLM:
    - só aceita regras cujos predicados existem no schema,
    - aridade bate com o schema,
    - todas as variáveis da conclusão aparecem nas premissas,
    - conclusão não é idêntica a uma premissa (evita tautologia).

    Retorna um novo objeto {"rules": [...]} só com as regras válidas.
    """
    # Mapa: nome_predicado -> número de argumentos esperados
    pred_arity = {
        p["name"]: len(p.get("args", []))
        for p in schema.get("predicates", [])
    }

    good_rules = []
    for rule in rules_obj.get("rules", []):
        premises = rule.get("premises", [])
        conclusion = rule.get("conclusion", {})

        # 1) Verifica se conclusion tem predicado conhecido e aridade correta
        concl_pred = conclusion.get("predicate")
        concl_args = conclusion.get("args", [])

        if concl_pred not in pred_arity:
            # predicado desconhecido
            continue
        if len(concl_args) != pred_arity[concl_pred]:
            # aridade errada
            continue

        # 2) Verifica premissas: todos predicados conhecidos e aridade certa
        premises_ok = True
        for prem in premises:
            ppred = prem.get("predicate")
            pargs = prem.get("args", [])
            if ppred not in pred_arity:
                premises_ok = False
                break
            if len(pargs) != pred_arity[ppred]:
                premises_ok = False
                break
        if not premises_ok:
            continue

        # 3) Conclusão não pode ser idêntica a uma das premissas (tautologia)
        is_tautology = any(
            (prem.get("predicate") == concl_pred and prem.get("args", []) == concl_args)
            for prem in premises
        )
        if is_tautology:
            continue

        # 4) Todas as variáveis da conclusão devem aparecer em pelo menos uma premissa
        vars_in_premises = set()
        for prem in premises:
            for v in prem.get("args", []):
                vars_in_premises.add(v)

        all_vars_bound = all(v in vars_in_premises for v in concl_args)
        if not all_vars_bound:
            # tipo "type" aparecendo só na conclusão cai aqui
            continue

        # Se passou por todos os checks, consideramos a regra válida
        good_rules.append(rule)

    return {"rules": good_rules}



def call_llama(prompt: str) -> str:
    """
    Chama o modelo LLaMA via Ollama na linha de comando.
    Ajuste LLAMA_MODEL_NAME conforme seu setup.
    """
    print("🧠 Calling LLaMA to propose logical rules...")
    result = subprocess.run(
        ["ollama", "run", LLAMA_MODEL_NAME],
        input=prompt,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("❌ Error calling LLaMA:")
        print(result.stderr)
        raise RuntimeError("LLaMA call failed")

    return result.stdout


def parse_rules(raw_output: str):
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

    print(f"📥 Loading predicate schema from: {SCHEMA_FILE}")
    schema = load_schema(SCHEMA_FILE)

    prompt = build_prompt(schema)

    # Chama o LLaMA
    raw_output = call_llama(prompt)

    # Parseia regras
    raw_rules = parse_rules(raw_output)

    # Verifica se as regras geradas pela LLM estão de acordo com os predicados
    rules = validate_and_filter_rules(schema, raw_rules)


    # Salva JSON final de regras
    with open(RULES_OUT, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"✅ Logical rules saved to: {RULES_OUT}")
    print("\nExtracted rules:")
    for rule in rules.get("rules", []):
        print(f"- {rule.get('name')}: {rule.get('description')}")


if __name__ == "__main__":
    main()
