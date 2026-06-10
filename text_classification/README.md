# Text Classification Pipeline — ICM Cluster

## Overview

End-to-end text classification pipeline with 5 feature extractors × 9 classifiers
(7 sklearn + 2 LLMs) across 4 pipeline variants: serial, optimized, parallel, and MPI.

---

## Project Structure

```
text_classification/
├── configs/
│   └── config.py               ← all seeds, paths, hyperparameters
├── utils/
│   ├── data_utils.py           ← dataset loading, cleaning, splitting
│   └── metrics.py              ← Timer, evaluate(), save_results()
├── features/
│   └── feature_extractors.py  ← BoW, TF-IDF, N-gram, Word2Vec, Doc2Vec
├── models/
│   ├── sklearn_models.py       ← LR, RF, XGB, SVM, NB, KNN, MLP
│   └── llm_models.py           ← LLaMA-2, Mistral (zero-shot)
├── parallel/
│   ├── parallel_features.py    ← joblib parallel feature extraction
│   ├── parallel_models.py      ← joblib parallel model training
│   └── pipeline_mpi.py         ← mpi4py multi-node pipeline
├── analysis/
│   └── analyse_results.py      ← plots, CSVs, speedup curves
├── slurm/
│   ├── run_serial.sh
│   ├── run_optimized.sh
│   ├── run_parallel.sh
│   ├── run_mpi.sh
│   ├── run_llm.sh              ← GPU job for LLaMA / Mistral
│   └── run_scaling.sh          ← job array for strong-scaling study
├── pipeline_serial.py          ← baseline
├── pipeline_optimized.py       ← sparse matrices + caching + LinearSVM
├── pipeline_parallel.py        ← joblib parallel features + models
├── run_all.py                  ← local convenience runner
├── setup_env.sh                ← one-time venv setup on cluster
└── requirements.txt
```

---

## Dataset

Download from: https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset

Place as: `data/train_data.txt`  (tab-separated: `label\ttext`)

---

## Quick Start (local)

```bash
# 1. Create environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run serial baseline (no LLMs)
python pipeline_serial.py --data data/train_data.txt --no-llm

# 3. Run optimized
python pipeline_optimized.py --data data/train_data.txt --no-llm

# 4. Run parallel
python pipeline_parallel.py --data data/train_data.txt --no-llm

# 5. Analyse results
python analysis/analyse_results.py --results-dir results/
```

---

## ICM Cluster Workflow

```bash
# One-time setup
bash setup_env.sh

# Pull Ollama models (run interactively, not inside a job)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
ollama pull mistral

# Submit jobs
sbatch slurm/run_serial.sh
sbatch slurm/run_optimized.sh
sbatch slurm/run_parallel.sh
sbatch slurm/run_mpi.sh
sbatch slurm/run_llm.sh          # starts ollama serve internally
sbatch slurm/run_scaling.sh      # array job for speedup study
```

---

## Feature Extractors

| Name       | Class               | Output shape       | Notes                        |
|------------|---------------------|--------------------|------------------------------|
| `bow`      | BagOfWordsExtractor | (N, 20 000)        | CountVectorizer unigrams     |
| `tfidf`    | TFIDFExtractor      | (N, 20 000)        | TF-IDF unigrams + bigrams    |
| `ngram`    | NgramExtractor      | (N, 20 000)        | TF-IDF bigrams + trigrams    |
| `word2vec` | Word2VecExtractor   | (N, 100)           | Mean-pooled Word2Vec         |
| `doc2vec`  | Doc2VecExtractor    | (N, 100)           | Paragraph Vector DM          |

---

## Models

| Name                  | Notes                                            |
|-----------------------|--------------------------------------------------|
| `logistic_regression` | L2, saga solver (fast for large sparse data)     |
| `random_forest`       | 200 trees, n_jobs=-1                             |
| `xgboost`             | 200 estimators, early stopping (optimized)       |
| `svm`                 | LinearSVC + CalibratedClassifierCV (optimized)   |
| `naive_bayes`         | ComplementNB (robust for multi-class text)       |
| `knn`                 | k=5, cosine distance                             |
| `mlp`                 | (256, 128) hidden, early stopping                |
| `llama`               | Llama-3 8B Q4 via Ollama (local, no GPU required)        |
| `mistral`             | Mistral 7B Q4 via Ollama (local, no GPU required)        |

---

## Parallelism Strategy

```
Serial:      feat1 → models → feat2 → models → … (sequential)

Optimized:   same as serial but:
               • sparse matrices (no .toarray() for BoW/TF-IDF)
               • feature caching (joblib.dump)
               • LinearSVC instead of kernel SVM
               • XGBoost early stopping

Parallel:    ┌── feat1 ──┐   ┌── model1 ──┐
             ├── feat2 ──┤   ├── model2 ──┤
             ├── feat3 ──┤ → ├── model3 ──┤  (joblib.Parallel)
             ├── feat4 ──┤   ├── model4 ──┤
             └── feat5 ──┘   └── model5 ──┘

MPI:         rank0 (master) dispatches (feat, model) pairs to ranks 1..N
             each rank handles one combination independently
             results collected at rank0
```

---

## Reproducibility

- All random seeds fixed via `RANDOM_SEED = 42` in `configs/config.py`.
- Applied to: train/test split, all sklearn models, Word2Vec, Doc2Vec, XGBoost, MLP.
- LLMs use `do_sample=False` (greedy decoding).
- Serial vs. parallel results may differ slightly due to floating-point
  non-associativity in parallel reductions (Word2Vec worker threads).
  All metrics should be within ±0.5% across variants.

---

## Environment

```
Python 3.10
scikit-learn 1.3.2
xgboost 2.0.3
gensim 4.3.2
ollama 0.2.1        (Python client — pip install ollama)
Ollama server       (https://ollama.com/download — separate install)
mpi4py 3.1.4        (MPI pipeline only)
```

See `requirements.txt` for full pinned versions.
