import os
import pymysql
from pymysql.cursors import DictCursor
import json
from typing import Optional, Dict, Any


class ProvenanceDB:
    def __init__(self):
        """
        Conecta ao MariaDB rodando no Docker.

        Como o MariaDB está rodando na porta 3307 no host,
        configuramos host=localhost e port=3307.
        """
        self.conn = pymysql.connect(
            host=os.getenv("PROV_DB_HOST", "localhost"),
            port=int(os.getenv("PROV_DB_PORT", "3307")),
            user=os.getenv("PROV_DB_USER", "larissa"),
            password=os.getenv("PROV_DB_PASSWORD", "1234"),
            database=os.getenv("PROV_DB_NAME", "provdb"),
            cursorclass=DictCursor,
            autocommit=True,
        )

    # ============================================================
    # hotpot_sample
    # ============================================================

    def get_or_create_hotpot_sample(self,
                                n_sample: int,
                                seed: int) -> int:
        """
        Retorna o id de um registro em hotpot_sample que tenha
        (n_sample, seed). Se não existir, cria com path vazio.
        O path será atualizado depois.
        """
        select_query = """
            SELECT id
            FROM hotpot_sample
            WHERE n_sample = %s
            AND seed = %s
            LIMIT 1
        """
        insert_query = """
            INSERT INTO hotpot_sample (n_sample, seed, path)
            VALUES (%s, %s, %s)
        """

        with self.conn.cursor() as cur:
            cur.execute(select_query, (n_sample, seed))
            row = cur.fetchone()
            if row:
                return row["id"]

            # path vazio por enquanto; será atualizado depois
            cur.execute(insert_query, (n_sample, seed, ""))
            return cur.lastrowid
    
    def update_hotpot_sample_path(self,
                              hotpot_sample_id: int,
                              path: str) -> None:
        """
        Atualiza o campo path de um hotpot_sample existente.
        Espera path relativo, por ex: 'records/hotpot_sample/1/hotpot_sample.csv'
        """
        query = "UPDATE hotpot_sample SET path = %s WHERE id = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (path, hotpot_sample_id))


    # ============================================================
    # xai_dataset
    # ============================================================

    def get_or_create_xai_dataset(self,
                              hotpot_sample_id: int,
                              prompt: str,
                              model: str,
                              temperature: float) -> int:
        """
        Retorna o id de um xai_dataset que combina:
        (hotpot_sample_id, prompt, model, temperature).
        O path será definido/atualizado depois.
        """
        select_query = """
            SELECT id
            FROM xai_dataset
            WHERE hotpot_sample_id = %s
            AND model = %s
            AND temperature = %s
            AND prompt = %s
            LIMIT 1
        """
        insert_query = """
            INSERT INTO xai_dataset (hotpot_sample_id, prompt, model, temperature, path)
            VALUES (%s, %s, %s, %s, %s)
        """

        with self.conn.cursor() as cur:
            cur.execute(
                select_query,
                (hotpot_sample_id, model, temperature, prompt),
            )
            row = cur.fetchone()
            if row:
                return row["id"]

            # path vazio por enquanto
            cur.execute(
                insert_query,
                (hotpot_sample_id, prompt, model, temperature, ""),
            )
            return cur.lastrowid
   
    def update_xai_dataset_path(self,
                            xai_dataset_id: int,
                            path: str) -> None:
        """
        Atualiza o campo path de um xai_dataset existente.
        Espera path relativo, ex: 'records/xai_dataset/3/xai_dataset.jsonl'
        """
        query = "UPDATE xai_dataset SET path = %s WHERE id = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (path, xai_dataset_id))
    
    def get_latest_xai_dataset_for_hotpot_sample(self, hotpot_sample_id: int) -> int | None:
        """
        Retorna o id do xai_dataset mais recente associado a um dado hotpot_sample_id.
        Assume que creating_dataset.py já registrou o dataset.
        """
        query = """
            SELECT id
            FROM xai_dataset
            WHERE hotpot_sample_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (hotpot_sample_id,))
            row = cur.fetchone()
            return row["id"] if row else None

        
    # ============================================================
    # experiment
    # ============================================================

    def create_experiment(self,
                          hotpot_sample_id: int,
                          xai_dataset_id: int) -> int:
        """
        Cria um novo experimento apontando para um hotpot_sample
        e um xai_dataset específicos.
        """
        query = """
            INSERT INTO experiment (hotpot_sample_id, xai_dataset_id)
            VALUES (%s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (hotpot_sample_id, xai_dataset_id))
            return cur.lastrowid

    # ============================================================
    # validity
    # ============================================================

    def insert_validity(self,
                        xai_dataset_id: int,
                        embedding: str,
                        similarity_threshold: float,
                        output: bool) -> int:
        """
        Registra o resultado da validação do dataset XAI.
        Pode haver múltiplas validações para o mesmo xai_dataset
        com embeddings/thresholds diferentes.
        """
        query = """
            INSERT INTO validity (xai_dataset_id, embedding, similarity_threshold, output)
            VALUES (%s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (xai_dataset_id, embedding, similarity_threshold, output),
            )
            return cur.lastrowid

    # ============================================================
    # cosine_similarity
    # ============================================================

    def insert_cosine_similarity_run(self,
                                     experiment_id: int,
                                     xai_dataset_id: int,
                                     embedding: str,
                                     path: str) -> int:
        """
        Registra uma execução da métrica de similaridade de cosseno
        (um arquivo CSV de resultados).
        """
        query = """
            INSERT INTO cosine_similarity (experiment_id, xai_dataset_id, embedding, path)
            VALUES (%s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (experiment_id, xai_dataset_id, embedding, path),
            )
            return cur.lastrowid
        
    # ============================================================
    # llm_judge
    # ============================================================

    def insert_llm_judge_run(self,
                             experiment_id: int,
                             xai_dataset_id: int,
                             model: str,
                             temperature: float,
                             prompt: str,
                             path: str) -> int:
        """
        Registra uma execução da métrica de LLM como juiz,
        apontando para o CSV com os julgamentos.
        """
        query = """
            INSERT INTO llm_judge (experiment_id, xai_dataset_id, model, temperature, prompt, path)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (experiment_id, xai_dataset_id, model, temperature, prompt, path),
            )
            return cur.lastrowid

    # ============================================================
    # predicates
    # ============================================================

    def insert_predicates(self,
                          hotpot_sample_id: int,
                          model: str,
                          temperature: float,
                          prompt: str,
                          path: str) -> int:
        """
        Registra uma listagem de predicados lógicos gerada a partir
        de uma amostra de HotpotQA.
        """
        query = """
            INSERT INTO predicates (hotpot_sample_id, model, temperature, prompt, path)
            VALUES (%s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (hotpot_sample_id, model, temperature, prompt, path),
            )
            return cur.lastrowid

    # ============================================================
    # rules
    # ============================================================

    def insert_rules(self,
                     predicate_id: int,
                     model: str,
                     temperature: float,
                     prompt: str,
                     path: str) -> int:
        """
        Registra o arquivo de regras lógicas derivadas dos predicados.
        """
        query = """
            INSERT INTO rules (predicate_id, model, temperature, prompt, path)
            VALUES (%s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (predicate_id, model, temperature, prompt, path),
            )
            return cur.lastrowid
        
    # ============================================================
    # facts
    # ============================================================

    def insert_facts(self,
                     predicate_id: int,
                     xai_dataset_id: int,
                     model: str,
                     temperature: float,
                     prompt: Optional[Dict[str, Any]],
                     path: str) -> int:
        """
        Registra o arquivo de fatos extraídos (para o dataset XAI),
        associados a um conjunto de predicados.
        """
        query = """
            INSERT INTO facts (predicate_id, xai_dataset_id, model, temperature, prompt, path)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (predicate_id, xai_dataset_id, model, temperature, prompt, path),
            )
            return cur.lastrowid


    # ============================================================
    # first_order_logic
    # ============================================================

    def insert_first_order_logic_run(self,
                                     experiment_id: int,
                                     xai_dataset_id: int,
                                     predicate_id: int,
                                     rules_id: int,
                                     facts_id: int,
                                     thresholds: Optional[Dict[str, Any]],
                                     path: str) -> int:
        """
        Registra uma execução da métrica de lógica de primeira ordem (FOL),
        apontando para:
        - o experimento
        - o dataset XAI
        - os predicados, regras e fatos usados
        - thresholds (JSON)
        - caminho do CSV de resultados
        """
        query = """
            INSERT INTO first_order_logic
                (experiment_id, xai_dataset_id, predicate_id, rules_id, facts_id, thresholds, path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        thresholds_json = json.dumps(thresholds or {})

        with self.conn.cursor() as cur:
            cur.execute(
                query,
                (
                    experiment_id,
                    xai_dataset_id,
                    predicate_id,
                    rules_id,
                    facts_id,
                    thresholds_json,
                    path,
                ),
            )
            return cur.lastrowid


    # ============================================================
    # CLOSE CONNECTION
    # ============================================================

    def close(self):
        self.conn.close()
