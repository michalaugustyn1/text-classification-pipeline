import logging
import numpy as np
import scipy.sparse as sp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    LR_PARAMS, RF_PARAMS, XGB_PARAMS, SVM_PARAMS,
    NB_PARAMS, KNN_PARAMS, MLP_PARAMS, RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def _todense(X, dtype=None):
    arr = X.toarray() if sp.issparse(X) else np.asarray(X)
    return arr.astype(dtype) if dtype is not None else arr


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
                C=params.get("C", 1.0), max_iter=params.get("max_iter", 1000),
                solver="qn")
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
        self.clf.fit(_to_cupy(X) if USE_GPU else _todense(X, np.float32), y)
        return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(
            _to_cupy(X) if USE_GPU else _todense(X, np.float32)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(
            _to_cupy(X) if USE_GPU else _todense(X, np.float32)))


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
        X = _todense(X, np.float32)
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
        self.clf.fit(_to_cupy(X) if USE_GPU else _todense(X), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else _todense(X)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else _todense(X)))


class MLPModel:
    name = "mlp"
    supports_sparse = False

    def __init__(self, **kwargs):
        from sklearn.neural_network import MLPClassifier
        self.clf = MLPClassifier(**{**MLP_PARAMS, **kwargs})

    def fit(self, X, y):
        self.clf.fit(_todense(X), y); return self

    def predict(self, X):
        return self.clf.predict(_todense(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(_todense(X))


ALL_SKLEARN_MODELS = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest":       RandomForestModel,
    "xgboost":             XGBoostModel,
    "svm":                 SVMModel,
    "naive_bayes":         NaiveBayesModel,
    "knn":                 KNNModel,
    "mlp":                 MLPModel,
}

def get_sklearn_model(name: str, **kwargs):
    if name not in ALL_SKLEARN_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(ALL_SKLEARN_MODELS)}")
    return ALL_SKLEARN_MODELS[name](**kwargs)
