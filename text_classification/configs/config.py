import os

RANDOM_SEED = 42

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
MODELS_DIR    = os.path.join(BASE_DIR, "saved_models")
RAW_CSV       = os.path.join(DATA_DIR, "train_data.txt")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

TEST_SIZE = 0.20
VAL_SIZE  = 0.10

MAX_FEATURES_BOW   = 20_000
MAX_FEATURES_TFIDF = 20_000
NGRAM_RANGE_BOW    = (1, 1)
NGRAM_RANGE_TFIDF  = (1, 2)
NGRAM_RANGE_NGRAM  = (2, 3)

W2V_DIM       = 100; W2V_WINDOW = 5; W2V_MIN_COUNT = 2
W2V_EPOCHS    = 10;  W2V_WORKERS = 4
D2V_DIM       = 100; D2V_WINDOW = 5; D2V_MIN_COUNT = 2
D2V_EPOCHS    = 10;  D2V_WORKERS = 4

LR_PARAMS  = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs",
               "multi_class": "auto", "random_state": RANDOM_SEED}
RF_PARAMS  = {"n_estimators": 200, "max_depth": None,
               "n_jobs": -1, "random_state": RANDOM_SEED}
XGB_PARAMS = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
               "eval_metric": "mlogloss", "n_jobs": -1,
               "random_state": RANDOM_SEED, "verbosity": 0}
SVM_PARAMS = {"C": 1.0, "kernel": "linear", "probability": True,
               "random_state": RANDOM_SEED}
NB_PARAMS  = {"alpha": 1.0}
KNN_PARAMS = {"n_neighbors": 5, "metric": "cosine", "n_jobs": -1}
MLP_PARAMS = {"hidden_layer_sizes": (256, 128), "activation": "relu",
               "max_iter": 300, "random_state": RANDOM_SEED,
               "early_stopping": True, "validation_fraction": 0.1}

LLM_SAMPLE_SIZE      = 500
LLM_TIMEOUT          = 120
OLLAMA_BASE_URL      = "http://localhost:11434"
OLLAMA_LLAMA_MODEL   = "llama3"
OLLAMA_MISTRAL_MODEL = "mistral"

N_JOBS_FEATURES = -1
N_JOBS_MODELS   = -1
N_JOBS_CV       = 4
LOG_LEVEL       = "INFO"
