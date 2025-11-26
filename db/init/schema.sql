-- Tabela principal de experimento
CREATE TABLE experiment (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hotpot_path VARCHAR(255),
    seed INT,
    n_samples INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Atividade de criação do dataset de explicabilidade
CREATE TABLE creation (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    hotpotqa_sample INT,
    prompt TEXT,
    model VARCHAR(100),
    temperature FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_creation_experiment
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- Linhas do dataset de explicabilidade (XAI)
CREATE TABLE xai_dataset (
    id INT AUTO_INCREMENT PRIMARY KEY,
    creation_id INT NOT NULL,
    sample_id VARCHAR(64),
    original_dataset_name VARCHAR(100),
    question TEXT,
    answer TEXT,
    chunk TEXT,
    explanation TEXT,
    meta JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_xai_creation
        FOREIGN KEY (creation_id) REFERENCES creation(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- Validação com embeddings (ou outras)
CREATE TABLE validation (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    embedding_model VARCHAR(100),
    threshold FLOAT,
    is_valid BOOLEAN,
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_validation_experiment
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- Resultados agregados de cosseno
CREATE TABLE cosine_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    similarity_correct JSON,
    similarity_incorrect JSON,
    similarity_incomplete JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_cosine_experiment
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- Resultados agregados de Jaccard
CREATE TABLE jaccard_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    similarity_correct JSON,
    similarity_incorrect JSON,
    similarity_incomplete JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_jaccard_experiment
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

CREATE TABLE logic_metric (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    num_trials INT,
    predicate_config JSON,
    rules_config JSON,
    facts_config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_logic_metric_experiment
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;

CREATE TABLE logic_result (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL,
    logic_metric_id INT NOT NULL,
    trial_number INT NOT NULL,
    sample_id VARCHAR(64),
    label VARCHAR(32),
    precision_result FLOAT,
    recall_result FLOAT,
    f1_result FLOAT,
    facts JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_logic_result_experiment
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_logic_result_metric
        FOREIGN KEY (logic_metric_id) REFERENCES logic_metric(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB;
