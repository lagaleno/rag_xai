# 📘 XAI Dataset Generation & Evaluation  
### *(with Provenance Tracking via MariaDB + Docker)*

This repository contains the full pipeline for creating, validating, and benchmarking a dataset of natural-language explanations for RAG (Retrieval-Augmented Generation) systems.  
The workflow includes:

- extracting a sample of HotpotQA  
- generating 3 explanation types per question (*correct*, *incomplete*, *incorrect*) using an LLM  
- validating explanations using embedding similarity  
- running a complete multi-metric experiment:
  - cosine similarity  
  - Jaccard similarity  
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
   cd 4-experiment
   python main.py
   ```

4. **Analyze results (graphs + tables)**  
   ```bash
   cd 5-analysis
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

- **Python 3.9+**
- **Docker & Docker Compose**
- Linux or macOS recommended
- `pip` installed
- Internet access (for downloading HotpotQA and LLM responses)

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
├── provenance.py                     # Provenance logging API for all scripts
│
├── db/
│   ├── init/schema.sql               # Tables created automatically by Docker
│   └── data/                         # Docker-managed database files
│
├── 0-utils/
│   ├── get_hotpotqa.py               # Downloads HotpotQA and updates provenance
│   └── hotpotqa_train.csv
│
├── 1-creating_dataset/
│   └── create_dataset.py             # Generates 3 explanations per Q/A
│
├── 2-validating_dataset/
│   ├── validate_dataset.py           # Embedding-based validation
│   └── figures/
│
├── 3-metrics/
│   ├── cosine_similarity/
│   ├── jaccard_similarity/
│   └── first_order_logic/
│
├── 4-experiment/
│   └── main.py                       # Orchestrates the full experiment
│
├── 5-analysis/
│   └── analyze.py                    # Final plots + aggregated results
│
├── docker-compose.yml
├── requirements.txt
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
  - Jaccard  
  - Cosine  
  - Logical inference (predicates, rules, fact extraction, multi-trial)  
- stores all metric outputs  
- updates the database at every stage  

Run:

```bash
cd 4-experiment
python main.py
```

A new row will appear in the `experiment` table.

---

## Step 2 — Analyze Results

After the experiment is complete:

```bash
cd 5-analysis
python analyze.py
```

Outputs include:

- summary CSVs  
- graphs (boxplots, bar plots, grouped comparisons)

---

# 🗄 5. Provenance Logging (What gets stored?)

### Tables include:

- `experiment`  
- `creation`  
- `xai_dataset`  
- `validation`  
- `cosine_results`  
- `jaccard_results`  
- `logic_metric`  
- `logic_result`

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

Always run scripts from the folder `4-experiment`:

```bash
cd 4-experiment
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
