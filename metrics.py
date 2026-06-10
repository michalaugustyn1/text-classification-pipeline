import csv
import io
import time
import logging
import numpy as np
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, f1_score, classification_report
from gpu_utils import to_numpy, USE_GPU

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


def _scalar(v):
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.integer):  return int(v)
    if USE_GPU:
        v = to_numpy(np.array(v))
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return v.item()
    return v


def save_results(results: dict, path: str):
    runs = results.get("runs", [])
    with open(path, "w", newline="") as f:
        for k, v in results.items():
            if k == "runs":
                continue
            if isinstance(v, dict):
                for kk, vv in v.items():
                    vv = _scalar(vv)
                    f.write(f"{kk}: {vv:.4f}\n" if isinstance(vv, float) else f"{kk}: {vv}\n")
            else:
                v = _scalar(v)
                f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")
        if runs:
            all_keys = list(dict.fromkeys(k for r in runs for k in r))
            f.write("\n")
            writer = csv.DictWriter(f, fieldnames=all_keys, delimiter="\t",
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            for run in runs:
                row = {}
                for k, v in run.items():
                    v = _scalar(v)
                    row[k] = f"{v:.6f}" if isinstance(v, float) else ("" if v is None else v)
                writer.writerow(row)
    logger.info("Results saved → %s", path)


def load_results(path: str) -> tuple:
    meta, lines = {}, []
    past_meta = False
    with open(path) as f:
        for line in f:
            if not past_meta and line.strip() == "":
                past_meta = True
                continue
            if not past_meta:
                k, _, v = line.strip().partition(":")
                meta[k.strip()] = v.strip()
            else:
                lines.append(line)
    import pandas as pd
    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t") if lines else pd.DataFrame()
    return meta, df
