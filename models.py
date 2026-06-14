import logging
import os
import numpy as np
from typing import List, Optional
from config import (
    LR_PARAMS, RF_PARAMS, XGB_PARAMS, SVM_PARAMS,
    NB_PARAMS, KNN_PARAMS, MLP_PARAMS, RANDOM_SEED,
    LLM_SAMPLE_SIZE, HF_LLAMA_MODEL, HF_MISTRAL_MODEL, HF_BATCH_SIZE,
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
USE_GPU = _CUML is not None and _gpu_available() and os.environ.get("FORCE_CPU", "0") != "1"

logger.info("Models: %s", "cuML (GPU)" if USE_GPU else "scikit-learn (CPU)")


def _to_dense(X):
    if hasattr(X, 'toarray'):
        return X.toarray().astype(np.float32)
    try:
        import cupy as cp
        if isinstance(X, cp.ndarray):
            return cp.asnumpy(X).astype(np.float32)
    except ImportError:
        pass
    return np.asarray(X, dtype=np.float32)


def _to_cupy(X):
    import cupy as cp
    if isinstance(X, cp.ndarray):
        return X.astype(cp.float32)
    return cp.array(_to_dense(X))


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
        self.clf.fit(_to_cupy(X) if USE_GPU else _to_dense(X), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else _to_dense(X)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else _to_dense(X)))


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
        self.clf.fit(_to_cupy(X) if USE_GPU else _to_dense(X), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else _to_dense(X)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else _to_dense(X)))


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
        self.clf.fit(_to_cupy(X) if USE_GPU else _to_dense(X), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else _to_dense(X)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else _to_dense(X)))


class NaiveBayesModel:
    name = "naive_bayes"
    supports_sparse = True

    def __init__(self, **kwargs):
        self._params = {**NB_PARAMS, **kwargs}
        self.clf = None

    def _nn(self, X):
        X = _to_dense(X)
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
        self.clf.fit(_to_cupy(X) if USE_GPU else _to_dense(X), y); return self

    def predict(self, X):
        return _to_numpy(self.clf.predict(_to_cupy(X) if USE_GPU else _to_dense(X)))

    def predict_proba(self, X):
        return _to_numpy(self.clf.predict_proba(_to_cupy(X) if USE_GPU else _to_dense(X)))


class MLPModel:
    name = "mlp"
    supports_sparse = False

    def __init__(self, **kwargs):
        from sklearn.neural_network import MLPClassifier
        self.clf = MLPClassifier(**{**MLP_PARAMS, **kwargs})

    def fit(self, X, y):
        self.clf.fit(_to_dense(X), y); return self

    def predict(self, X):
        return self.clf.predict(_to_dense(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(_to_dense(X))


ALL_SKLEARN_MODELS = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest":       RandomForestModel,
    "xgboost":             XGBoostModel,
    "svm":                 SVMModel,
    "naive_bayes":         NaiveBayesModel,
    "knn":                 KNNModel,
    "mlp":                 MLPModel,
}


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


class HFClassifier:
    def __init__(self, model_id: str,
                 sample_size: int = LLM_SAMPLE_SIZE,
                 seed: int = RANDOM_SEED,
                 batch_size: int = HF_BATCH_SIZE):
        self.model_id    = model_id
        self.sample_size = sample_size
        self.seed        = seed
        self.batch_size  = batch_size
        self.class_names: Optional[List[str]] = None
        self.name        = model_id.split("/")[-1].lower()
        self._tokenizer  = None
        self._model      = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        token = os.environ.get("HF_TOKEN")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token, padding_side="left")
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16,
            device_map="auto", token=token)
        self._model.eval()
        self._model.generation_config.max_length = None
        logger.info("Loaded %s on %s", self.model_id,
                    next(self._model.parameters()).device)

    def unload(self):
        if self._model is not None:
            import torch, gc
            del self._model, self._tokenizer
            self._model = self._tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Unloaded %s", self.model_id)

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    def fit(self, X, y, label_encoder=None):
        raw = list(label_encoder.classes_) if label_encoder else sorted(set(y))
        self.class_names = [
            str(LABEL_NAMES.get(int(str(c)), str(c))) if str(c).isdigit() else str(c)
            for c in raw
        ]
        self._load()
        logger.info("'%s' ready. %d classes.", self.model_id, len(self.class_names))
        return self

    def _infer_batch(self, texts):
        import torch
        def _apply_chat(text):
            out = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": _build_prompt(text, self.class_names)}],
                add_generation_prompt=True, return_tensors="pt")
            t = out.input_ids if hasattr(out, 'input_ids') else out
            return t.squeeze(0)

        token_ids_list = [_apply_chat(t) for t in texts]
        max_len = max(x.shape[0] for x in token_ids_list)
        pad_id  = self._tokenizer.pad_token_id
        input_ids = torch.full((len(token_ids_list), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(token_ids_list):
            offset = max_len - ids.shape[0]
            input_ids[i, offset:] = ids
            attn_mask[i, offset:] = 1
        input_ids = input_ids.to(self._model.device)
        attn_mask = attn_mask.to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids, attention_mask=attn_mask,
                max_new_tokens=20, do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id)
        return [
            _parse_pred(
                self._tokenizer.decode(row[max_len:], skip_special_tokens=True),
                self.class_names)
            for row in out
        ]

    def predict(self, X) -> np.ndarray:
        if self.class_names is None:
            raise RuntimeError("Call fit() before predict().")
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(X))
        if len(X) > self.sample_size:
            indices = rng.choice(indices, size=self.sample_size, replace=False)
            logger.warning("LLM eval sampled to %d / %d.", self.sample_size, len(X))
        pred_names = np.array([self.class_names[0]] * len(X), dtype=object)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            if start % 50 == 0:
                logger.info("  [%s] %d / %d …", self.name, start, len(indices))
            try:
                for idx, pred in zip(batch_idx,
                                     self._infer_batch([X[i] for i in batch_idx])):
                    pred_names[idx] = pred
            except Exception as exc:
                logger.warning("  HF error on batch %d: %s", start, exc, exc_info=True)
        self.unload()
        return np.array(
            [self.class_names.index(p) if p in self.class_names else 0
             for p in pred_names], dtype=int)

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(X), len(self.class_names)), dtype=np.float32)
        proba[np.arange(len(X)), preds] = 1.0
        return proba


class LlamaClassifier(HFClassifier):
    def __init__(self, **kwargs):
        super().__init__(model_id=HF_LLAMA_MODEL, **kwargs)
        self.name = "llama3"


class MistralClassifier(HFClassifier):
    def __init__(self, **kwargs):
        super().__init__(model_id=HF_MISTRAL_MODEL, **kwargs)
        self.name = "mistral"


ALL_LLM_MODELS = {"llama": LlamaClassifier, "mistral": MistralClassifier}
