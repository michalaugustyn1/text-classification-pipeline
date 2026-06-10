import argparse
import logging
import os
import time
import numpy as np
from config import RANDOM_SEED, RESULTS_DIR, RAW_CSV
from data_utils import prepare_data
from metrics import evaluate, save_results, timed
from features import ALL_EXTRACTORS
from models import ALL_SKLEARN_MODELS, ALL_LLM_MODELS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("serial_pipeline")


def run_combination(feat_name, model_name,
                    X_train, y_train, X_val, y_val, X_test, y_test, le):
    result = {"feature": feat_name, "model": model_name}
    ext = ALL_EXTRACTORS[feat_name]()
    with timed("feature_fit_transform", result): Xtr = ext.fit_transform(X_train)
    with timed("feature_transform_val", result):  Xv  = ext.transform(X_val)
    with timed("feature_transform_test", result): Xt  = ext.transform(X_test)
    model = ALL_SKLEARN_MODELS[model_name]()
    with timed("train", result):          model.fit(Xtr, y_train)
    with timed("inference_val", result):  y_pred_v = model.predict(Xv)
    with timed("inference_test", result): y_pred_t = model.predict(Xt)
    vm = evaluate(y_val,  y_pred_v, le.classes_)
    tm = evaluate(y_test, y_pred_t, le.classes_)
    result.update({
        "val_accuracy": vm["accuracy"], "val_f1_macro": vm["f1_macro"],
        "val_f1_weighted": vm["f1_weighted"],
        "test_accuracy": tm["accuracy"], "test_f1_macro": tm["f1_macro"],
        "test_f1_weighted": tm["f1_weighted"],
    })
    logger.info("[%s × %s]  acc=%.4f  f1=%.4f  feat=%.2fs  train=%.2fs  inf=%.4fs",
                feat_name, model_name, result["test_accuracy"], result["test_f1_macro"],
                result["feature_fit_transform_time"], result["train_time"],
                result["inference_test_time"])
    return result


def run_llm_combination(model_name, X_train, y_train, X_test, y_test, le):
    result = {"feature": "raw_text", "model": model_name}
    llm = ALL_LLM_MODELS[model_name]()
    with timed("train", result):          llm.fit(X_train, y_train, label_encoder=le)
    with timed("inference_test", result): y_pred = llm.predict(X_test)
    n = min(len(X_test), llm.sample_size)
    m = evaluate(y_test[:n], y_pred[:n], le.classes_)
    result.update({"test_accuracy": m["accuracy"], "test_f1_macro": m["f1_macro"],
                   "test_f1_weighted": m["f1_weighted"],
                   "val_accuracy": None, "val_f1_macro": None})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default=RAW_CSV)
    parser.add_argument("--out",      default=RESULTS_DIR)
    parser.add_argument("--llm",      action="store_true")
    parser.add_argument("--features", nargs="+", default=list(ALL_EXTRACTORS.keys()))
    parser.add_argument("--models",   nargs="+", default=list(ALL_SKLEARN_MODELS.keys()))
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    (X_train, y_train, X_val, y_val, X_test, y_test, le) = prepare_data(args.data)

    all_results = []
    t0 = time.perf_counter()

    for feat_name in args.features:
        for model_name in args.models:
            logger.info("=== %s × %s ===", feat_name, model_name)
            try:
                all_results.append(run_combination(
                    feat_name, model_name,
                    X_train, y_train, X_val, y_val, X_test, y_test, le))
            except Exception as exc:
                logger.error("FAILED [%s × %s]: %s", feat_name, model_name, exc)
                all_results.append({"feature": feat_name, "model": model_name, "error": str(exc)})

    if args.llm:
        for llm_name in ("llama", "mistral"):
            logger.info("=== raw_text × %s ===", llm_name)
            try:
                all_results.append(run_llm_combination(
                    llm_name, X_train, y_train, X_test, y_test, le))
            except Exception as exc:
                logger.error("FAILED [raw_text × %s]: %s", llm_name, exc)
                all_results.append({"feature": "raw_text", "model": llm_name, "error": str(exc)})

    total = time.perf_counter() - t0
    logger.info("Total: %.1f s", total)
    os.makedirs(args.out, exist_ok=True)
    save_results({"total_time": total, "runs": all_results},
                 os.path.join(args.out, "results_serial.json"))


if __name__ == "__main__":
    main()
