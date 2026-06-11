#!/bin/bash

VENV=~/HPAI/text_classification/venv
REQ=~/HPAI/text_classification/requirements.txt

# Load modules if available — adjust names to match your cluster
module load python/3.10  2>/dev/null || true
module load gcc/11.2     2>/dev/null || true

python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -r "$REQ"

python -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('wordnet',   quiet=True)
"

python -c "
import sklearn, xgboost, gensim, ollama, mpi4py
print('sklearn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('gensim:',  gensim.__version__)
print('ollama:',  ollama.__version__)
print('mpi4py:',  mpi4py.__version__)
print('All OK.')
"

echo "Done. Activate with: source $VENV/bin/activate"
