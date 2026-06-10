import argparse
import logging
import os
import time
import joblib
import numpy as np
from config import RANDOM_SEED, RESULTS_DIR, RAW_CSV, N_JOBS_FEATURES, N_JOBS_MODELS
from data_utils import prepare_data
from metrics import evaluate, save_results, timed
from gpu_utils import to_numpy, seed as gpu_seed
from features import ALL_EXTRACTORS
from models import ALL_SKLEARN_MODELS, ALL_LLM_MODELS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("parallel_pipeline")


def _extract_one(feat_name, X_train, X_val, X_test):
    gpu_seed(RANDOM_SEED)
    ext = ALL_EXTRACTORS[feat_name]()
    return feat_name, ext.fit_transform(X_train), ext.transform(X_val), ext.transform(X_test)


def extract_all_parallel(X_train, X_val, X_test, feature_names, n_jobs):
    t0 = time.perf_counter()
    results = joblib.Parallel(
        n_jobs=min(n_jobs if n_jobs > 0 else len(feature_names), len(feature_names)),
        backend="multiprocessing", verbose=5)(
        joblib.delayed(_extract_one)(fn, X_train, X_val, X_test)
        for fn in feature_names)
    elapsed = time.perf_counter() - t0
    return {fn: (Xtr, Xv, Xt) for fn, Xtr, Xv, Xt in results}, elapsed


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


def train_models_parallel(feat_name, Xtr, Xv, Xt,
                           y_train, y_val, y_test, label_classes, model_names, n_jobs):
    Xtr_np = to_numpy(Xtr).astype("float32")
    Xv_np  = to_numpy(Xv).astype("float32")
    Xt_np  = to_numpy(Xt).astype("float32")
    t0 = time.perf_counter()
    results = joblib.Parallel(
        n_jobs=min(n_jobs if n_jobs > 0 else len(model_names), len(model_names)),
        backend="loky", verbose=5)(
        joblib.delayed(_train_one)(
            mn, feat_name, Xtr_np, Xv_np, Xt_np,
            y_train, y_val, y_test, label_classes)
        for mn in model_names)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default=RAW_CSV)
    parser.add_argument("--out",        default=RESULTS_DIR)
    parser.add_argument("--feat-jobs",  type=int, default=N_JOBS_FEATURES)
    parser.add_argument("--model-jobs", type=int, default=N_JOBS_MODELS)
    parser.add_argument("--llm",        action="store_true")
    parser.add_argument("--features",   nargs="+", default=list(ALL_EXTRACTORS.keys()))
    parser.add_argument("--models",     nargs="+", default=list(ALL_SKLEARN_MODELS.keys()))
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    (X_train, y_train, X_val, y_val, X_test, y_test, le) = prepare_data(args.data)

    t0 = time.perf_counter(); all_results = []; timing = {}

    features_dict, feat_elapsed = extract_all_parallel(
        X_train, X_val, X_test, args.features, args.feat_jobs)
    timing["feature_extraction_total"] = feat_elapsed

    ms_t0 = time.perf_counter()
    for feat_name, (Xtr, Xv, Xt) in features_dict.items():
        res_list, elapsed = train_models_parallel(
            feat_name, Xtr, Xv, Xt, y_train, y_val, y_test,
            le.classes_, args.models, args.model_jobs)
        timing[f"model_training_{feat_name}"] = elapsed
        all_results.extend(res_list)
    timing["model_training_total"] = time.perf_counter() - ms_t0

    if args.llm:
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

    timing["total_wall_time"] = time.perf_counter() - t0
    logger.info("Total: %.1f s", timing["total_wall_time"])
    os.makedirs(args.out, exist_ok=True)
    save_results({"timing": timing, "runs": all_results},
                 os.path.join(args.out, "results_parallel.json"))


if __name__ == "__main__":
    main()
