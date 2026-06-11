#!/bin/bash

REQ=~/HPAI/text_classification/requirements.txt

source ~/miniconda3/bin/activate myenv

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

echo "Done. Activate with: source ~/miniconda3/bin/activate myenv"
