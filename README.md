# 📘 XAI Dataset Generation & Evaluation  
### *(with Provenance Tracking via MariaDB + Docker)*

This repository contains the full pipeline for creating, validating, and benchmarking a dataset of natural-language explanations for RAG (Retrieval-Augmented Generation) systems.  
The workflow includes:

- extracting a sample of HotpotQA  
- generating 3 explanation types per question (*correct*, *incomplete*, *incorrect*) using an LLM  
- validating explanations using embedding similarity  
- running a complete multi-metric experiment:
  - cosine similarity  
  - llm as a judge
  - logical inference (predicate schema + rules + fact extraction + inference engine)  
- aggregating and visualizing results  
- **tracking full provenance** (experiment metadata, creation events, metrics, intermediate steps) using a **MariaDB database running inside Docker**

---

# 🧭 TL;DR — How to run everything

1. **Install Python dependencies**
   ```bash
   ./install.sh
   ```

2. **Start the provenance database**
   ```bash
   docker compose up -d
   ```

3. **Run the full experiment** (recreates datasets if missing)  
   ```bash
   cd 5-experiment
   python main.py
   ```

4. **Analyze results (graphs + tables)**  
   ```bash
   cd 4-analysis
   python analyze.py
   ```

5. **(Optional) Browse provenance records**  
   Open: http://localhost:8080  
   - System: MySQL  
   - Server: mariadb  
   - User: larissa  
   - Password: 1234  
   - Database: provdb  

---

# 📦 1. Prerequisites

You need:

- Python 3.9+
- Docker & Docker Compose
- Linux or macOS recommended
- pip installed
- Internet access (for downloading HotpotQA)
- Ollama installed and running: https://ollama.com/download

LLaMA model pulled locally, e.g.:
```bash
ollama pull llama3       # or llama3.1 / llama3:instruct
```
The pipeline uses Ollama's HTTP server at: `http://localhost:11434/api/chat`
Make sure it is running: `ollama serve`
You can check available models: `ollama ls`

---

# 🐳 2. Provenance Database Setup (Docker)

This repository includes a ready-to-use `docker-compose.yml` that starts:

- **MariaDB 11** (stores provenance)
- **Adminer** (web UI for inspecting the DB)

To start the database:

```bash
docker compose up -d
```

Check that it is running:

```bash
docker ps
```

You should see `prov_db` and `prov_adminer` running.

### Database connection info

| Field       | Value      |
|-------------|------------|
| host        | localhost  |
| port        | 3307       |
| user        | larissa    |
| password    | 1234       |
| database    | provdb     |

Adminer URL: http://localhost:8080

---

# 📁 3. Project Structure

```
project_root/
│
├── provenance.py                         # Provenance API
│
├── db/
│   ├── init/schema.sql                   # Auto-loaded by Docker
│
├── 0-utils/
│   ├── get_hotpotqa.py                   # Downloads HotpotQA (only if missing)
│   └── hotpotqa_train.csv
│
├── 1-creating_dataset/
│   ├── create_dataset.py                 # Generates 3 LLM explanations per question
│   └── explainrag_hotpot_llama.jsonl
│
├── 2-validating_dataset/
│   ├── validate_dataset.py               # Embedding validation
│
├── 3-metrics/
│   ├── cosine_similarity/
│   │   └── run_cosine_similarity.py
│   │
│   ├── llm_judge/
│   │   └── run_llm_judge.py              # Baseline classifier using LLM-as-judge
│   │
│   └── first_order_logic/
│       ├── 01_define_predicate_schema.py
│       ├── 02_define_logical_rules.py
│       ├── 03_extract_facts_llm.py
│       ├── 04_inference_metric_prototype.py
│       └── logical_rules.json
│
├── 4-analysis/
│   └── analyze.py                        # Cosine distribution + confusion matrices
│
├── 5-experiment/
│   └── main.py                           # Full orchestrated experiment
│
├── records                               # directory to register provenance
├── docker-compose.yml
└── requirements.txt
└── install.sh
```

---

# 🧪 4. How to Run the Full Pipeline

## Step 0 — Start the Provenance Database (required)

```bash
docker compose up -d
```

---

## Step 1 — Run the Main Experiment

This script:

- creates a new experiment entry in the provenance DB  
- downloads HotpotQA if needed  
- generates the explanation dataset  
- validates the dataset  
- runs:
  - Cosine 
  - LLM as a Judge
  - Logical inference (predicates, rules, fact extraction, multi-trial)  
- stores all metric outputs  
- updates the database at every stage  
- Runs the analysis script `4-analysis\analyze.py` which generate summary CSVs and graphs (boxplots, bar plots, grouped comparisons)

Run:

```bash
cd 5-experiment
python main.py
```

A new row will appear in the `experiment`.

---

# 🗄 5. Provenance Logging (What gets stored?)

### Tables include:

- `experiment`  
- `hotpot_sample`  
- `xai_dataset`  
- `validity`  
- `cosine_similarity`  
- `llm_judge`  
- `first_order_logic`  
- `predicates`
- `rules`
- `facts`

Each script updates the DB through `provenance.py`.

This ensures that every experiment is:

- reproducible  
- auditable  
- traceable  

with full metadata about each stage.

---

# ❗ Troubleshooting

### MariaDB port already in use (3306)

If you see:

```
Error: Ports are not available
```

Edit `docker-compose.yml`:

```yaml
ports:
  - "3307:3306"
```

---

### Import errors (`ModuleNotFoundError: provenance`)

Always run scripts from the folder `5-experiment`:

```bash
cd 5-experiment
python main.py
```

---

### Database connection issues

Check containers:

```bash
docker ps
```

Check Adminer UI:
http://localhost:8080

---

### Matplotlib issues on macOS

```bash
brew install freetype pkg-config libpng
```

---

# 💬 Contact

For questions or suggestions, feel free to open an issue.
