import logging
import os
import urllib.request
import numpy as np
from typing import List, Optional
from config import (
    LR_PARAMS, RF_PARAMS, XGB_PARAMS, SVM_PARAMS,
    NB_PARAMS, KNN_PARAMS, MLP_PARAMS, RANDOM_SEED,
    LLM_SAMPLE_SIZE, LLM_TIMEOUT,
    OLLAMA_BASE_URL, OLLAMA_LLAMA_MODEL, OLLAMA_MISTRAL_MODEL,
    LABEL_NAMES,
)

logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _import_cuml():
    try:
        import cuml
        return cuml
    except ImportError:
        return None


_CUML   = _import_cuml()
USE_GPU = _CUML is not None and _gpu_available()

logger.info("Models: %s", "cuML (GPU)" if USE_GPU else "scikit-learn (CPU)")


def _to_cupy(X):
    import cupy as cp
    return cp.array(np.array(X, dtype=np.float32))


def _to_numpy(X):
    try:
        import cupy as cp
        if isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
    except ImportError:
        pass
    return np.array(X)


class LogisticRegressionModel:
    name = "logistic_regression"
    supports_sparse = True

    def __init__(self, **kwargs):
        params = {**LR_PARAMS, **kwargs}
        if USE_GPU:
            self.clf = _CUML.linear_model.LogisticRegression(
                C=params.get("C", 1.0), max_iter=params.get("max_iter", 1000), solver="qn")
        else:
            from sklearn.linear_model import LogisticRegression
            self.clf = LogisticRegression(**params)

    def fit(self, X, y):
        self.clf.fit(_to_cupy(X) if USE_GPU else np.array(X), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else np.array(X)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else np.array(X)))


class RandomForestModel:
    name = "random_forest"
    supports_sparse = True

    def __init__(self, **kwargs):
        params = {**RF_PARAMS, **kwargs}
        if USE_GPU:
            self.clf = _CUML.ensemble.RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=(params.get("max_depth") or 16),
                random_state=params.get("random_state", RANDOM_SEED))
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(_to_cupy(X) if USE_GPU else np.array(X, dtype=np.float32), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(
            _to_cupy(X) if USE_GPU else np.array(X, dtype=np.float32)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(
            _to_cupy(X) if USE_GPU else np.array(X, dtype=np.float32)))


class XGBoostModel:
    name = "xgboost"
    supports_sparse = True

    def __init__(self, **kwargs):
        import xgboost as xgb
        params = {**XGB_PARAMS, **kwargs}
        params.pop("use_label_encoder", None)
        if USE_GPU:
            params["device"] = "cuda"
            params["tree_method"] = "hist"
        self.clf = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(X, y); return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class SVMModel:
    name = "svm"
    supports_sparse = True

    def __init__(self, **kwargs):
        params = {**SVM_PARAMS, **kwargs}
        if USE_GPU:
            self.clf = _CUML.svm.SVC(
                C=params.get("C", 1.0), kernel=params.get("kernel", "linear"),
                probability=True)
        else:
            from sklearn.svm import SVC
            self.clf = SVC(**params)

    def fit(self, X, y):
        self.clf.fit(_to_cupy(X) if USE_GPU else np.array(X), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else np.array(X)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else np.array(X)))


class NaiveBayesModel:
    name = "naive_bayes"
    supports_sparse = True

    def __init__(self, **kwargs):
        self._params = {**NB_PARAMS, **kwargs}
        self.clf = None

    def _nn(self, X):
        X = np.array(X, dtype=np.float32)
        mn = X.min()
        return X - mn if mn < 0 else X

    def fit(self, X, y):
        X = self._nn(X)
        if USE_GPU:
            self.clf = _CUML.naive_bayes.ComplementNB(**self._params)
            self.clf.fit(_to_cupy(X), y)
        else:
            from sklearn.naive_bayes import ComplementNB
            self.clf = ComplementNB(**self._params)
            self.clf.fit(X, y)
        return self

    def predict(self, X):
        X = self._nn(X)
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else X))

    def predict_proba(self, X):
        X = self._nn(X)
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else X))


class KNNModel:
    name = "knn"
    supports_sparse = False

    def __init__(self, **kwargs):
        params = {**KNN_PARAMS, **kwargs}
        if USE_GPU:
            self.clf = _CUML.neighbors.KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                metric=params.get("metric", "cosine"))
        else:
            from sklearn.neighbors import KNeighborsClassifier
            self.clf = KNeighborsClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(_to_cupy(X) if USE_GPU else np.array(X), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else np.array(X)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else np.array(X)))


class MLPModel:
    name = "mlp"
    supports_sparse = False

    def __init__(self, **kwargs):
        from sklearn.neural_network import MLPClassifier
        self.clf = MLPClassifier(**{**MLP_PARAMS, **kwargs})

    def fit(self, X, y):
        self.clf.fit(np.array(X), y); return self

    def predict(self, X):
        return self.clf.predict(np.array(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(np.array(X))


ALL_SKLEARN_MODELS = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest":       RandomForestModel,
    "xgboost":             XGBoostModel,
    "svm":                 SVMModel,
    "naive_bayes":         NaiveBayesModel,
    "knn":                 KNNModel,
    "mlp":                 MLPModel,
}


_env_host = os.environ.get("OLLAMA_HOST", "")
if _env_host:
    OLLAMA_BASE_URL = _env_host if _env_host.startswith("http") else f"http://{_env_host}"

_PROMPT = (
    "Classify the following document into exactly one of these categories: {classes}.\n\n"
    "Document:\n{text}\n\n"
    "Rules:\n- Reply with ONLY the category name.\n"
    "- Do not add any explanation, punctuation, or extra words.\nCategory:"
)


def _build_prompt(text, class_names):
    return _PROMPT.format(classes=", ".join(class_names),
                          text=" ".join(text.split()[:256]))


def _parse_pred(raw, class_names):
    raw_l = raw.strip().lower()
    for c in class_names:
        if c.lower() == raw_l: return c
    for c in class_names:
        if c.lower() in raw_l: return c
    tok = raw_l.split()[0] if raw_l.split() else ""
    for c in class_names:
        if tok.startswith(c.lower()[:4]): return c
    return class_names[0]


def _check_ollama(base_url):
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
        return True
    except Exception:
        return False


class OllamaClassifier:
    def __init__(self, model_name: str,
                 sample_size: int = LLM_SAMPLE_SIZE,
                 timeout: int = LLM_TIMEOUT,
                 seed: int = RANDOM_SEED):
        self.model_name  = model_name
        self.sample_size = sample_size
        self.timeout     = timeout
        self.seed        = seed
        self.base_url    = OLLAMA_BASE_URL
        self.class_names: Optional[List[str]] = None
        self.name        = model_name.replace(":", "_")

    def fit(self, X, y, label_encoder=None):
        raw = list(label_encoder.classes_) if label_encoder else sorted(set(y))
        self.class_names = [str(LABEL_NAMES.get(int(str(c)), str(c))) if str(c).isdigit() else str(c)
                            for c in raw]
        if not _check_ollama(self.base_url):
            raise RuntimeError(
                f"Ollama not reachable at {self.base_url}.\n"
                "Start with: ollama serve  or  bash apptainer/build_and_pull.sh")
        import ollama
        try:
            ollama.Client(host=self.base_url).chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 3})
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach '{self.model_name}' — run: bash apptainer/build_and_pull.sh\n"
                f"Error: {exc}") from exc
        logger.info("'%s' ready. %d classes.", self.model_name, len(self.class_names))
        return self

    def _classify_one(self, text):
        import ollama
        resp = ollama.Client(host=self.base_url).chat(
            model=self.model_name,
            messages=[{"role": "user", "content": _build_prompt(text, self.class_names)}],
            options={"temperature": 0, "num_predict": 20, "seed": self.seed})
        return _parse_pred(resp["message"]["content"], self.class_names)

    def predict(self, X) -> np.ndarray:
        if self.class_names is None:
            raise RuntimeError("Call fit() before predict().")
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(X))
        if len(X) > self.sample_size:
            indices = rng.choice(indices, size=self.sample_size, replace=False)
            logger.warning("LLM eval sampled to %d / %d.", self.sample_size, len(X))
        pred_names = np.array([self.class_names[0]] * len(X), dtype=object)
        for k, idx in enumerate(indices):
            if k % 50 == 0:
                logger.info("  [%s] %d / %d …", self.model_name, k, len(indices))
            try:
                pred_names[idx] = self._classify_one(X[idx])
            except Exception as exc:
                logger.warning("  Ollama error on doc %d: %s", idx, exc)
        return np.array(
            [self.class_names.index(p) if p in self.class_names else 0
             for p in pred_names], dtype=int)

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(X), len(self.class_names)), dtype=np.float32)
        proba[np.arange(len(X)), preds] = 1.0
        return proba


class LlamaClassifier(OllamaClassifier):
    def __init__(self, **kwargs):
        super().__init__(model_name=OLLAMA_LLAMA_MODEL, **kwargs)
        self.name = "llama3"


class MistralClassifier(OllamaClassifier):
    def __init__(self, **kwargs):
        super().__init__(model_name=OLLAMA_MISTRAL_MODEL, **kwargs)
        self.name = "mistral"


ALL_LLM_MODELS = {"llama": LlamaClassifier, "mistral": MistralClassifier}
