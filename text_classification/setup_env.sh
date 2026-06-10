#!/bin/bash

set -euo pipefail

ENV_NAME="tc_env"
ENV_DIR="$HOME/venvs/$ENV_NAME"

echo "=== Setting up environment: $ENV_DIR ==="

module purge
module load python/3.10
module load gcc/11.2

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

python -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('wordnet',   quiet=True)
"

python -c "
import sklearn, xgboost, gensim, ollama
print('sklearn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('gensim:',  gensim.__version__)
print('ollama:',  ollama.__version__)
print('All OK.')
"

echo "=== Environment ready: source $ENV_DIR/bin/activate ==="
