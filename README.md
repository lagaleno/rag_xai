# XAI Dataset Generation & Evaluation

This repository contains the full pipeline used to create, validate, and prepare a benchmark dataset of natural-language explanations for RAG (Retrieval-Augmented Generation) systems.  
The project includes:

- automatic extraction of HotpotQA data  
- generation of a dataset containing examples of three explanation types (*correct*, *incomplete*, *incorrect*) using an LLM  
- preliminary validation using sentence-wise embedding similarity  
- a experimentation pipeline for comparing multiple automatic explainability metrics.  The experiments include:
  - cosine similarity between each explanation and its supporting chunk (embedding-based semantic overlap)
  - Jaccard similarity (lexical overlap baseline)
  - logical-inference–based evaluation using a predicate schema, LLM-generated logical rules, fact extraction, and an inference engine
  - multi-trial execution for the logic-based metric, allowing variance estimation
  - automated result aggregation, producing per-metric summaries (mean, std, count) grouped by explanation label
  - generation of plots (bar plots, boxplots, grouped comparisons) for analysis and inclusion in the research paper

---
## TL;DR
- Make sure you have all technical Prerequisites installed, described in `requirements.txt`
- Make sure you have the file `hotpotqa_tra.csv` on the folder `0-utils`
- Create the explanation dataset by the script `1-creating_dataset/create_dataset.py` or has one dowloaded and on folder `1-creating_dataset`
- Evaluate the data set by the script `2-validating_dataset/validate_dataset.py`
- Run the experiment throygh `4-experiment/main.py` running `python main.py`
- Run the analysis script to get the graphs and aggregated csv on `5-analysis/analyze.py`
  
## 📦 1. Prerequisites

- **Python 3.9+**  
- macOS or Linux (recommended)  
- `pip` or `pip3` installed  
- (Optional) A local LLaMA runtime (Ollama / llama.cpp)

---

## 🚀 2. Installation

All dependencies are listed in `requirements.txt`.

Simply run:

```bash
./install.sh
```

The script automatically detects whether your system uses  
`pip3` (macOS) or `pip` (Linux/WSL) and installs all dependencies accordingly.

If you prefer manual installation:

```bash
pip3 install -r requirements.txt
```

---

## 📁 3. Project Structure

```
project_root/
│
├── 0-utils/
│   ├── get_hotpotqa.py        # Downloads HotpotQA and converts to CSV
│   └── hotpotqa_train.csv/    # CSV containing hotpotqa dataset
│
├── 1-creating_dataset/
│   └── create_dataset.py     # Generates 3 explanation types per Q/A
│
├── 2-validating_dataset/
│   ├── validate_dataset.py       # Sentence-wise cosine similarity validation
│   ├── figures/                  # Evaluation plots (precision, recall, F1)
│   └── metrics/                  # Evaluation metrics (precision, recall, F1)
│
├── 3-metrics/
│   ├── utils.py             # File with functions to be used in scripts of the experiment
│   ├── cosine_similarity/   # Compute the cosine similarity
│   ├── jaccard_similarity/  # Compute Jaccard similarity
│   └── first_order_logic/   # Evaluate the explenation through first order logic
│
├── 4-experiment/
│   ├── main.py                          # Script to orchestrate the computation of metrics
│   ├── logical_summary_results_trial/   # Store the results from each trial
│   ├── *.csv/                           # Store the results from each metrics in csv files
│
├── 5-analysis/
│   ├── analyze.py              # Script to perform tha analysis of the metrics result
│   ├── figure/                 # Store graph figures generated
│   ├── *.csv/                  # Store the summary results for each label in each metric
│
├── requirements.txt
├── install.sh
└── README.md
```

---

## 📝 4. Step-by-Step Usage

### **Step 0 — Prepare HotpotQA**

Download and convert the dataset:

```bash
python 0-utils/get_hotpotqa.py
```

Output:

```
0-utils/hotpotqa_train.csv
```

---

### **Step 1 — Generate the Explanation Dataset**

```bash
python 1-creating_dataset/create_dataset.py
```

Output:

```
1-creating_dataset/explainrag_hotpot_llama.jsonl
```

---

### **Step 2 — Validate the Dataset (Sanity Check)**

```bash
python 2-validating_dataset/evaluate_dataset.py
```

Outputs:

- explanations_sentencewise_embeddings_metrics.csv  
- explanations_sentecewise_embeddings_summary_by_label.csv  
- emb_f1_by_label_boxplot.png  
- emb_precision_by_label_boxplot.png  
- emb_recall_by_label_boxplot.png  

---

### **Step 3 — Metrics**
This project includes a set of metrics to analyze how different automatic metrics behave when evaluating natural-language explanations in a RAG setting. All experiments assume that the explanation dataset has already been generated, the file: `1-creating_dataset/explainrag_hotpot_llama.jsonl`.

#### Cosine Similarity
This experiment measures the global semantic similarity between each explanation and its supporting chunk using sentence embeddings and cosine similarity. To run:

```bash
python 3-metrics/cosine_similarity/run_cosine_similarity.py
```
What it does:

- Loads explainrag_dataset.jsonl
- Builds aligned pairs: (chunk, explanation, label)
- Encodes chunks and explanations with all-MiniLM-L6-v2 (SentenceTransformers)
- Computes cosine similarity between each chunk–explanation pair
  
Output:
```
3-metrics/cosine_similarity/cosine_similarity_results.csv
3-metrics/cosine_similarity/cosine_similarity_summary_by_label.csv
```
#### Jaccard Similarity
This experiment computes a token-level Jaccard similarity between each explanation and its chunk, serving as a simple lexical baseline for comparison with embedding-based methods. To run:

```bash
python 3-metrics/jaccard_similarity/run_jaccard_similarity.py
```
What it does:
- Loads explainrag_dataset.jsonl
- Reuses the same (chunk, explanation, label) pairs
- Tokenizes both texts into lowercase alphanumeric “words”
- Computes Jaccard similarity

Output:
```
3-metrics/jaccard_similarity/jaccard_similarity_results.csv
3-metrics/jaccard_similarity/jaccard_similarity_summary_by_label.csv
```

#### Logical Inference Metric (First-Order Logic)

This experiment evaluates each explanation using a symbolic reasoning engine based on first-order logic (FOL).  
The goal is to measure how much of an explanation can be logically supported or inferred from the chunk through a set of predefined predicates and inference rules.

To run:

```bash
python 3-metrics/first_order_logic/04_inference_metric_prototype.py
```
What it does:
- Loads the predicate schema and logical rules (predicate_schema.json, logical_rules.json)
- Reads the extracted factual representations (facts_extracted_llm.jsonl) produced by the LLM
- For each explanation:
    - Converts chunk and explanation facts into structured predicates
    - Performs forward-chaining (logical closure) over chunk facts
    - Computes:
        - TP: facts asserted in the explanation that are entailed by the chunk
        - FP: facts asserted in the explanation that are not supported by the chunk
        - FN: facts inferable from the chunk but missing from the explanation
    - Computes precision, recall, and F1 in this logical space
- Aggregates metrics by explanation label (correct, incomplete, incorrect)

Output:
```bash
3-metrics/first_order_logic/logical_metrics_results.csv
```
Example Columns include: `id, explanation_label, tp, fp, fn, precision, recall, f1`

#### Shared Utilities
Metrics computation reuse common helper functions defined in: `3-metrics/utils.py`

This module:
- parses the JSONL dataset,
- builds structured examples with:
    - dataset_id
    - chunk
    - explanations (correct, incomplete, incorrect)
- flattens them into chunk–explanation pairs for metric computation.

### **Step 4 — Experiment**
This part is to mainly orchestrate the running of the experiment, putting on order the different scripts descibed above to get the metrics results on csvs.  

```bash
python 4-experiment/main.py
```
The outputs repeents the outputs of the different metrics

### **Step 5 — Analysis**
Through the results obtained in the experiments this script will shocase graphs and table to assist in the analysis of the result

```bash
python 5-analysis/analyze.py
```

---

## ❗ Troubleshooting

### Pip issues  
```bash
pip3 install -r requirements.txt
```

### Matplotlib errors on macOS

```bash
brew install freetype pkg-config libpng
```

### LLaMA inference errors  
Ensure your local model server is running.

---

## 💬 Contact

For questions or contributions, please open an issue in the repository.
