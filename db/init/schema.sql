-- Cria o banco (se ainda não existir) e seleciona
CREATE DATABASE IF NOT EXISTS provdb
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE provdb;

-- ===== LIMPEZA OPCIONAL =====
-- IMPORTANTE: isso apaga as tabelas se já existirem!
-- Se você for recriar o volume do Docker do zero, isso aqui é redundante,
-- mas não faz mal.

-- DROP TABLE IF EXISTS first_order_logic;
-- DROP TABLE IF EXISTS facts;
-- DROP TABLE IF EXISTS rules;
-- DROP TABLE IF EXISTS predicates;
-- DROP TABLE IF EXISTS llm_judge;
-- DROP TABLE IF EXISTS cosine_similarity;
-- DROP TABLE IF EXISTS validity;
-- DROP TABLE IF EXISTS experiment;
-- DROP TABLE IF EXISTS xai_dataset;
-- DROP TABLE IF EXISTS hotpot_sample;

-- =========================================================
-- 1) hotpot_sample
--    Amostra do HotpotQA usada no experimento
-- =========================================================

CREATE TABLE hotpot_sample (
    id INT AUTO_INCREMENT PRIMARY KEY,
    n_sample INT NOT NULL,                -- N_SAMPLES usado no script
    seed INT NOT NULL,                    -- SEED usado no script
    path VARCHAR(512) NOT NULL,           -- caminho do CSV amostrado
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_hotpot_sample_unique (n_sample, seed, path)
) ENGINE=InnoDB;

-- =========================================================
-- 2) xai_dataset
--    Dataset de explicabilidade gerado em creating_dataset.py
-- =========================================================

CREATE TABLE xai_dataset (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hotpot_sample_id INT NOT NULL,
    prompt TEXT NOT NULL,                 -- prompt usado para gerar explicações
    model VARCHAR(100) NOT NULL,          -- ex: 'gpt-4o-mini'
    temperature FLOAT NOT NULL,
    path VARCHAR(512) NOT NULL,           -- caminho do CSV/JSONL final do dataset XAI
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_xai_hotpot_sample
        FOREIGN KEY (hotpot_sample_id)
        REFERENCES hotpot_sample(id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
    INDEX idx_xai_dataset_unique (hotpot_sample_id, model, temperature, path(191))
) ENGINE=InnoDB;

-- =========================================================
-- 3) experiment
--    Contexto geral do experimento
-- =========================================================

CREATE TABLE experiment (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hotpot_sample_id INT NOT NULL,        -- FK → hotpot_sample.id
    xai_dataset_id INT NOT NULL,          -- FK → xai_dataset.id
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_experiment_hotpot_sample
        FOREIGN KEY (hotpot_sample_id)
        REFERENCES hotpot_sample(id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
    CONSTRAINT fk_experiment_xai_dataset
        FOREIGN KEY (xai_dataset_id)
        REFERENCES xai_dataset(id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- =========================================================
-- 4) validity
--    Validade do dataset XAI (saída do validate_dataset.py)
-- =========================================================

CREATE TABLE validity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    xai_dataset_id INT NOT NULL,          -- FK → xai_dataset.id
    embedding VARCHAR(100) NOT NULL,      -- ex: 'all-MiniLM-L6-v2'
    similarity_threshold FLOAT NOT NULL,  -- SIM_THRESHOLD
    output BOOLEAN NOT NULL,              -- resultado (True/False)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_validity_xai_dataset
        FOREIGN KEY (xai_dataset_id)
        REFERENCES xai_dataset(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- =========================================================
-- 5) cosine_similarity
--    Execuções da métrica de similaridade do cosseno
-- =========================================================

CREATE TABLE cosine_similarity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,           -- FK → experiment.id
    xai_dataset_id INT NOT NULL,          -- FK → xai_dataset.id
    embedding VARCHAR(100) NOT NULL,      -- modelo de embedding usado
    path VARCHAR(512) NOT NULL,           -- caminho do CSV de resultados
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cosine_experiment
        FOREIGN KEY (experiment_id)
        REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_cosine_xai_dataset
        FOREIGN KEY (xai_dataset_id)
        REFERENCES xai_dataset(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- =========================================================
-- 6) llm_judge
--    Execuções da métrica de LLM como juíz
-- =========================================================

CREATE TABLE llm_judge (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,           -- FK → experiment.id
    xai_dataset_id INT NOT NULL,          -- FK → xai_dataset.id
    model VARCHAR(100) NOT NULL,          -- modelo da LLM juíza
    temperature FLOAT NOT NULL,
    prompt TEXT NOT NULL,
    path VARCHAR(512) NOT NULL,           -- caminho do CSV com os julgamentos
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_llm_judge_experiment
        FOREIGN KEY (experiment_id)
        REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_llm_judge_xai_dataset
        FOREIGN KEY (xai_dataset_id)
        REFERENCES xai_dataset(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- =========================================================
-- 7) predicates
--    Predicados lógicos inferidos/definidos a partir da amostra Hotpot
-- =========================================================

CREATE TABLE predicates (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hotpot_sample_id INT NOT NULL,        -- FK → hotpot_sample.id
    model VARCHAR(100) NOT NULL,
    temperature FLOAT NOT NULL,
    prompt TEXT NOT NULL,
    path VARCHAR(512) NOT NULL,           -- caminho do arquivo/listagem de predicados
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_predicates_hotpot_sample
        FOREIGN KEY (hotpot_sample_id)
        REFERENCES hotpot_sample(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- =========================================================
-- 8) rules
--    Regras lógicas obtidas a partir dos predicados
-- =========================================================

CREATE TABLE rules (
    id INT AUTO_INCREMENT PRIMARY KEY,
    predicate_id INT NOT NULL,            -- FK → predicates.id
    model VARCHAR(100) NOT NULL,
    temperature FLOAT NOT NULL,
    prompt TEXT NOT NULL,
    path VARCHAR(512) NOT NULL,           -- caminho do arquivo de regras
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_rules_predicates
        FOREIGN KEY (predicate_id)
        REFERENCES predicates(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- =========================================================
-- 9) facts
--    Fatos extraídos a partir das explicações/chunks
-- =========================================================

CREATE TABLE facts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    predicate_id INT NOT NULL,            -- FK → predicates.id
    xai_dataset_id INT NOT NULL,          -- FK → xai_dataset.id
    model VARCHAR(100) NOT NULL,
    temperature FLOAT NOT NULL,
    prompt JSON NOT NULL,
    path VARCHAR(512) NOT NULL,           -- caminho do arquivo com os fatos extraídos
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_facts_predicates
        FOREIGN KEY (predicate_id)
        REFERENCES predicates(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_facts_xai_dataset
        FOREIGN KEY (xai_dataset_id)
        REFERENCES xai_dataset(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- =========================================================
-- 10) first_order_logic
--     Execuções da métrica de lógica de primeira ordem
-- =========================================================

CREATE TABLE first_order_logic (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,           -- FK → experiment.id
    xai_dataset_id INT NOT NULL,          -- FK → xai_dataset.id
    predicate_id INT NOT NULL,            -- FK → predicates.id
    rules_id INT NOT NULL,                -- FK → rules.id
    facts_id INT NOT NULL,                -- FK → facts.id
    thresholds JSON NULL,                 -- parâmetros/limiares usados na métrica
    path VARCHAR(512) NOT NULL,           -- caminho do CSV de resultados FOL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_fol_experiment
        FOREIGN KEY (experiment_id)
        REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_fol_xai_dataset
        FOREIGN KEY (xai_dataset_id)
        REFERENCES xai_dataset(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_fol_predicates
        FOREIGN KEY (predicate_id)
        REFERENCES predicates(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_fol_rules
        FOREIGN KEY (rules_id)
        REFERENCES rules(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_fol_facts
        FOREIGN KEY (facts_id)
        REFERENCES facts(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;
