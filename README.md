# XAI Dataset Generation & Evaluation

This repository contains the full pipeline used to create, validate, and prepare a benchmark dataset of natural-language explanations for RAG (Retrieval-Augmented Generation) systems.  
The project includes:

- automatic extraction of HotpotQA data  
- generation of a dataset containing examples of three explanation types (*correct*, *incomplete*, *incorrect*) using an LLM  
- preliminary validation using sentence-wise embedding similarity  
- TODO: add description of experiment

---

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
├── 3-experiments/
│   ├── utils.py/            # File with functions to be used in scripts of the experiment
│   ├── cosine_similarity/   # Compute the cosine similarity
│   ├── jaccard_similarity/  # Compute Jaccard similarity
│   └── ...
│
├── requirements.txt
├── install.sh
└── README.md
```

---

## 📝 4. Step-by-Step Usage

### **Step 1 — Prepare HotpotQA**

Download and convert the dataset:

```bash
python 0-utils/prepare_hotpot_csv.py
```

Output:

```
0-utils/hotpotqa_train.csv
```

---

### **Step 2 — Generate the Explanation Dataset**

```bash
python 1-creating_dataset/generate_explanations.py
```

Output:

```
1-creating_dataset/explainrag_hotpot_llama.jsonl
```

---

### **Step 3 — Validate the Dataset (Sanity Check)**

```bash
python 2-validating_dataset/evaluate_embeddings.py
```

Outputs:

- explanations_sentencewise_metrics.csv  
- explanations_summary_by_label.csv  
- f1_by_label_boxplot.png  
- precision_by_label_boxplot.png  
- recall_by_label_boxplot.png  

---

### **Step 4 — Experiments**
This project includes a set of experiments to analyze how different automatic metrics behave when evaluating natural-language explanations in a RAG setting. All experiments assume that the explanation dataset has already been generated, the file: `1-creating_dataset/explainrag_hotpot_llama.jsonl`.

#### Cosine Similarity
This experiment measures the global semantic similarity between each explanation and its supporting chunk using sentence embeddings and cosine similarity. To run:

```bash
python 3-experiments/cosine_similarity/run_cosine_similarity.py
```
What it does:

- Loads explainrag_dataset.jsonl
- Builds aligned pairs: (chunk, explanation, label)
- Encodes chunks and explanations with all-MiniLM-L6-v2 (SentenceTransformers)
- Computes cosine similarity between each chunk–explanation pair
  
Output:
```
3-experiments/cosine_similarity/cosine_similarity_results.csv
3-experiments/cosine_similarity/cosine_similarity_summary_by_label.csv
```
#### Jaccard Similarity
This experiment computes a token-level Jaccard similarity between each explanation and its chunk, serving as a simple lexical baseline for comparison with embedding-based methods. To run:

```bash
python 3-experiments/jaccard_similarity/run_jaccard_similarity.py
```
What it does:
- Loads explainrag_dataset.jsonl
- Reuses the same (chunk, explanation, label) pairs
- Tokenizes both texts into lowercase alphanumeric “words”
- Computes Jaccard similarity

Output:
```
3-experiments/jaccard_similarity/jaccard_similarity_results.csv
3-experiments/jaccard_similarity/jaccard_similarity_summary_by_label.csv
```

### Shared Utilities
Experiments reuse common helper functions defined in: `3-experiments/utils.py`

This module:
- parses the JSONL dataset,
- builds structured examples with:
    - dataset_id
    - chunk
    - explanations (correct, incomplete, incorrect)
- flattens them into chunk–explanation pairs for metric computation.

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
