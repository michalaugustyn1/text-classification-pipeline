import argparse
import hashlib
import logging
import os
import sys
import time
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import (RANDOM_SEED, RESULTS_DIR, MODELS_DIR,
                              MAX_FEATURES_TFIDF, MAX_FEATURES_BOW,
                              NGRAM_RANGE_BOW, NGRAM_RANGE_TFIDF, NGRAM_RANGE_NGRAM)
from utils.data_utils import prepare_data
from utils.metrics import evaluate, save_results, timed
from features.feature_extractors import Word2VecExtractor, Doc2VecExtractor
from models.sklearn_models import RandomForestModel, XGBoostModel, NaiveBayesModel, KNNModel, MLPModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("optimized_pipeline")

CACHE_DIR = os.path.join(MODELS_DIR, "feature_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class SparseBOW:
    name = "bow"; sparse = True
    def __init__(self):
        self.vec = CountVectorizer(max_features=MAX_FEATURES_BOW, ngram_range=NGRAM_RANGE_BOW,
                                   token_pattern=r"\b[a-z][a-z0-9]{2,}\b", strip_accents="unicode")
    def fit(self, X): self.vec.fit(X); return self
    def transform(self, X): return self.vec.transform(X)
    def fit_transform(self, X): return self.vec.fit_transform(X)


class SparseTFIDF:
    name = "tfidf"; sparse = True
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=NGRAM_RANGE_TFIDF,
                                   sublinear_tf=True, token_pattern=r"\b[a-z][a-z0-9]{2,}\b",
                                   strip_accents="unicode")
    def fit(self, X): self.vec.fit(X); return self
    def transform(self, X): return self.vec.transform(X)
    def fit_transform(self, X): return self.vec.fit_transform(X)


class SparseNgram:
    name = "ngram"; sparse = True
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=NGRAM_RANGE_NGRAM,
                                   sublinear_tf=True, token_pattern=r"\b[a-z][a-z0-9]{2,}\b",
                                   strip_accents="unicode")
    def fit(self, X): self.vec.fit(X); return self
    def transform(self, X): return self.vec.transform(X)
    def fit_transform(self, X): return self.vec.fit_transform(X)


SPARSE_EXTRACTORS = {
    "bow": SparseBOW, "tfidf": SparseTFIDF, "ngram": SparseNgram,
    "word2vec": Word2VecExtractor, "doc2vec": Doc2VecExtractor,
}


class LinearSVMModel:
    name = "svm"
    def __init__(self):
        self.clf = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED), cv=3)
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X): return self.clf.predict(X)
    def predict_proba(self, X): return self.clf.predict_proba(X)


class XGBoostModelOpt:
    name = "xgboost"
    def __init__(self):
        import xgboost as xgb
        self.clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_SEED, verbosity=0)
    def fit(self, X, y, X_val=None, y_val=None):
        self.clf.fit(X, y, verbose=False)
        return self
    def predict(self, X): return self.clf.predict(X)
    def predict_proba(self, X): return self.clf.predict_proba(X)


OPTIMIZED_MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        C=1.0, max_iter=1000, solver="saga", multi_class="auto",
        random_state=RANDOM_SEED, n_jobs=-1),
    "random_forest": RandomForestModel,
    "xgboost":       XGBoostModelOpt,
    "svm":           LinearSVMModel,
    "naive_bayes":   NaiveBayesModel,
    "knn":           KNNModel,
    "mlp":           MLPModel,
}


def _cache_key(feat_name, X_train, X_test):
    h = hashlib.md5()
    for s in (feat_name, str(len(X_train)), str(len(X_test)), X_train[0]):
        h.update(s.encode())
    return h.hexdigest()[:12]


def load_or_extract(feat_name, X_train, X_val, X_test, use_cache=True):
    key   = _cache_key(feat_name, X_train, X_test)
    cpath = os.path.join(CACHE_DIR, f"{feat_name}_{key}.pkl")
    if use_cache and os.path.exists(cpath):
        logger.info("Cache hit → %s", cpath)
        return joblib.load(cpath)
    ext  = SPARSE_EXTRACTORS[feat_name]()
    data = ext.fit_transform(X_train), ext.transform(X_val), ext.transform(X_test)
    if use_cache:
        joblib.dump(data, cpath, compress=3)
    return data


def run_combination(feat_name, model_name, Xtr, Xv, Xt,
                    y_train, y_val, y_test, le, feat_time):
    result = {"feature": feat_name, "model": model_name,
              "feature_fit_transform_time": feat_time}
    model = OPTIMIZED_MODELS[model_name]()
    with timed("train", result): model.fit(Xtr, y_train)
    with timed("inference_test", result):
        y_pred = model.predict(Xt)
    m = evaluate(y_test, y_pred, le.classes_)
    result.update({"test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
                   "test_f1_weighted": m["f1_weighted"]})
    logger.info("[%s × %s]  acc=%.4f  f1=%.4f  train=%.2fs",
                feat_name, model_name, result["test_accuracy"],
                result["test_f1_macro"], result["train_time"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default=None)
    parser.add_argument("--out",      default=RESULTS_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--llm",      action="store_true")
    parser.add_argument("--features", nargs="+", default=list(SPARSE_EXTRACTORS.keys()))
    parser.add_argument("--models",   nargs="+", default=list(OPTIMIZED_MODELS.keys()))
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    from configs.config import RAW_CSV
    (X_train, y_train, X_val, y_val, X_test, y_test, le) = prepare_data(args.data or RAW_CSV)

    all_results = []
    t0 = time.perf_counter()

    for feat_name in args.features:
        ft0 = time.perf_counter()
        Xtr, Xv, Xt = load_or_extract(feat_name, X_train, X_val, X_test,
                                       use_cache=not args.no_cache)
        feat_time = time.perf_counter() - ft0
        for model_name in args.models:
            logger.info("=== %s × %s ===", feat_name, model_name)
            try:
                all_results.append(run_combination(
                    feat_name, model_name, Xtr, Xv, Xt,
                    y_train, y_val, y_test, le, feat_time))
            except Exception as exc:
                logger.error("FAILED [%s × %s]: %s", feat_name, model_name, exc)
                all_results.append({"feature": feat_name, "model": model_name, "error": str(exc)})

    if args.llm:
        from models.llm_models import ALL_LLM_MODELS
        for llm_name in ("llama", "mistral"):
            result = {"feature": "raw_text", "model": llm_name}
            llm = ALL_LLM_MODELS[llm_name]()
            with timed("train", result):          llm.fit(X_train, y_train, label_encoder=le)
            with timed("inference_test", result): y_pred = llm.predict(X_test)
            n = min(len(X_test), llm.sample_size)
            m = evaluate(y_test[:n], y_pred[:n], le.classes_)
            result.update({"test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
                           "test_f1_weighted": m["f1_weighted"]})
            all_results.append(result)

    total = time.perf_counter() - t0
    logger.info("Total: %.1f s", total)
    os.makedirs(args.out, exist_ok=True)
    save_results({"total_time": total, "runs": all_results},
                 os.path.join(args.out, "results_optimized.json"))


if __name__ == "__main__":
    main()
