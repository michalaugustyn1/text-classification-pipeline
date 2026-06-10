import time
import json
import logging
import numpy as np
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils.gpu_utils import to_numpy, USE_GPU

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self):
        self._start = None
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


@contextmanager
def timed(label: str, results_dict: dict = None):
    t = Timer()
    with t:
        yield t
    logger.info("%s took %.4f s", label, t.elapsed)
    if results_dict is not None:
        results_dict[f"{label}_time"] = t.elapsed


def evaluate(y_true, y_pred, label_names=None):
    if label_names is not None:
        label_names = [str(n) for n in label_names]
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report":      classification_report(y_true, y_pred,
                                             target_names=label_names,
                                             zero_division=0),
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if USE_GPU:
            obj = to_numpy(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        return super().default(obj)


def save_results(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logger.info("Results saved → %s", path)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
