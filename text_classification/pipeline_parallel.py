import argparse
import logging
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import RANDOM_SEED, RESULTS_DIR, N_JOBS_FEATURES, N_JOBS_MODELS
from utils.data_utils import prepare_data
from utils.metrics import save_results
from parallel.parallel_features import extract_all_features_parallel
from parallel.parallel_models import train_all_models_parallel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("parallel_pipeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default=None)
    parser.add_argument("--out",        default=RESULTS_DIR)
    parser.add_argument("--feat-jobs",  type=int, default=N_JOBS_FEATURES)
    parser.add_argument("--model-jobs", type=int, default=N_JOBS_MODELS)
    parser.add_argument("--llm",        action="store_true")
    parser.add_argument("--features",   nargs="+",
                        default=["bow", "tfidf", "ngram", "word2vec", "doc2vec"])
    parser.add_argument("--models",     nargs="+",
                        default=["logistic_regression", "random_forest",
                                 "xgboost", "svm", "naive_bayes", "knn", "mlp"])
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    from configs.config import RAW_CSV
    (X_train, y_train, X_val, y_val, X_test, y_test, le) = prepare_data(args.data or RAW_CSV)

    t0          = time.perf_counter()
    all_results = []
    timing      = {}

    features_dict, feat_elapsed = extract_all_features_parallel(
        X_train, X_val, X_test, feature_names=args.features, n_jobs=args.feat_jobs)
    timing["feature_extraction_total"] = feat_elapsed

    ms_t0 = time.perf_counter()
    for feat_name, (Xtr, Xv, Xt) in features_dict.items():
        res_list, elapsed = train_all_models_parallel(
            feat_name, Xtr, Xv, Xt, y_train, y_val, y_test,
            le.classes_, args.models, n_jobs=args.model_jobs)
        timing[f"model_training_{feat_name}"] = elapsed
        all_results.extend(res_list)
    timing["model_training_total"] = time.perf_counter() - ms_t0

    if args.llm:
        from models.llm_models import ALL_LLM_MODELS
        from utils.metrics import timed, evaluate
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
