import logging
import os
import sys
import time
import joblib
import numpy as np
from utils.gpu_utils import to_numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import N_JOBS_MODELS, RANDOM_SEED
from models.sklearn_models import ALL_SKLEARN_MODELS
from utils.metrics import evaluate

logger = logging.getLogger(__name__)


def _train_one(model_name, feat_name, Xtr, Xv, Xt, y_train, y_val, y_test, label_classes):
    import time as _t
    import numpy as _np
    _np.random.seed(RANDOM_SEED)
    result = {"feature": feat_name, "model": model_name}
    model = ALL_SKLEARN_MODELS[model_name]()
    t0 = _t.perf_counter()
    model.fit(to_numpy(Xtr), y_train)
    result["train_time"] = _t.perf_counter() - t0
    t1 = _t.perf_counter()
    y_pred = model.predict(to_numpy(Xt))
    result["inference_test_time"] = _t.perf_counter() - t1
    m = evaluate(y_test, y_pred, label_classes)
    result["test_accuracy"]    = m["accuracy"]
    result["test_f1_macro"]    = m["f1_macro"]
    result["test_f1_weighted"] = m["f1_weighted"]
    return result


def train_all_models_parallel(feat_name, Xtr, Xv, Xt,
                               y_train, y_val, y_test,
                               label_classes,
                               model_names=None,
                               n_jobs=N_JOBS_MODELS):
    if model_names is None:
        model_names = list(ALL_SKLEARN_MODELS.keys())
    Xtr_np = to_numpy(Xtr).astype("float32")
    Xv_np  = to_numpy(Xv).astype("float32")
    Xt_np  = to_numpy(Xt).astype("float32")
    t0 = time.perf_counter()
    results_list = joblib.Parallel(
        n_jobs=min(n_jobs if n_jobs > 0 else len(model_names), len(model_names)),
        backend="loky", verbose=5)(
        joblib.delayed(_train_one)(
            name, feat_name, Xtr_np, Xv_np, Xt_np,
            y_train, y_val, y_test, label_classes)
        for name in model_names)
    elapsed = time.perf_counter() - t0
    logger.info("Parallel model training [%s]: %.2f s", feat_name, elapsed)
    return results_list, elapsed
