import logging
import os
import sys
import time
import joblib
import numpy as np
from utils.gpu_utils import seed as gpu_seed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import N_JOBS_FEATURES, RANDOM_SEED
from features.feature_extractors import ALL_EXTRACTORS

logger = logging.getLogger(__name__)


def _extract_one(feat_name, X_train, X_val, X_test):
    gpu_seed(RANDOM_SEED)
    ext = ALL_EXTRACTORS[feat_name]()
    Xtr = ext.fit_transform(X_train)
    Xv  = ext.transform(X_val)
    Xt  = ext.transform(X_test)
    return feat_name, Xtr, Xv, Xt


def extract_all_features_parallel(X_train, X_val, X_test,
                                   feature_names=None,
                                   n_jobs=N_JOBS_FEATURES):
    if feature_names is None:
        feature_names = list(ALL_EXTRACTORS.keys())
    t0 = time.perf_counter()
    results_list = joblib.Parallel(
        n_jobs=min(n_jobs if n_jobs > 0 else len(feature_names), len(feature_names)),
        backend="multiprocessing", verbose=5)(
        joblib.delayed(_extract_one)(name, X_train, X_val, X_test)
        for name in feature_names)
    elapsed = time.perf_counter() - t0
    features = {name: (Xtr, Xv, Xt) for name, Xtr, Xv, Xt in results_list}
    logger.info("Parallel feature extraction: %.2f s", elapsed)
    return features, elapsed
