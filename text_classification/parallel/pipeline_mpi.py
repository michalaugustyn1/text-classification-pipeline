import argparse
import logging
import os
import sys
import time
import numpy as np
from utils.gpu_utils import to_numpy, seed as gpu_seed
from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import RANDOM_SEED, RESULTS_DIR
from utils.data_utils import prepare_data
from utils.metrics import evaluate, save_results
from features.feature_extractors import ALL_EXTRACTORS
from models.sklearn_models import ALL_SKLEARN_MODELS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s RANK%(rank)s %(message)s")

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def worker_process(rank):
    logger = logging.LoggerAdapter(logging.getLogger("mpi_worker"), {"rank": rank})
    while True:
        task = COMM.recv(source=0, tag=MPI.ANY_TAG)
        if task is None:
            break
        feat_name, model_name, X_train, y_train, X_test, y_test, label_classes = task
        gpu_seed(RANDOM_SEED + rank)
        result = {"feature": feat_name, "model": model_name}
        try:
            ext = ALL_EXTRACTORS[feat_name]()
            t0 = time.perf_counter()
            Xtr = ext.fit_transform(X_train)
            Xt  = ext.transform(X_test)
            result["feature_time"] = time.perf_counter() - t0
            model = ALL_SKLEARN_MODELS[model_name]()
            t1 = time.perf_counter()
            model.fit(to_numpy(Xtr), y_train)
            result["train_time"] = time.perf_counter() - t1
            t2 = time.perf_counter()
            y_pred = model.predict(to_numpy(Xt))
            result["inference_time"] = time.perf_counter() - t2
            m = evaluate(y_test, y_pred, label_classes)
            result.update({"test_accuracy": m["accuracy"],
                            "test_f1_macro": m["f1_macro"],
                            "test_f1_weighted": m["f1_weighted"]})
        except Exception as exc:
            result["error"] = str(exc)
            logger.error("[%s × %s] ERROR: %s", feat_name, model_name, exc)
        COMM.send(result, dest=0)


def master_process(args):
    logger = logging.LoggerAdapter(logging.getLogger("mpi_master"), {"rank": 0})
    from configs.config import RAW_CSV
    (X_train, y_train, _, _, X_test, y_test, le) = prepare_data(args.data or RAW_CSV)
    label_classes = le.classes_.tolist()
    tasks = [(fn, mn, X_train, y_train, X_test, y_test, label_classes)
             for fn in ALL_EXTRACTORS for mn in ALL_SKLEARN_MODELS]
    n_tasks, n_workers = len(tasks), SIZE - 1
    logger.info("Tasks:%d  Workers:%d", n_tasks, n_workers)
    total_start = time.perf_counter()
    all_results = []
    task_idx = 0
    for w in range(1, min(n_workers + 1, n_tasks + 1)):
        COMM.send(tasks[task_idx], dest=w); task_idx += 1
    received = 0
    while received < n_tasks:
        status = MPI.Status()
        result = COMM.recv(source=MPI.ANY_SOURCE, status=status)
        src = status.Get_source()
        all_results.append(result); received += 1
        if task_idx < n_tasks:
            COMM.send(tasks[task_idx], dest=src); task_idx += 1
        else:
            COMM.send(None, dest=src)
    for w in range(received + 1, n_workers + 1):
        COMM.send(None, dest=w)
    if args.llm:
        from models.llm_models import ALL_LLM_MODELS
        from utils.metrics import timed
        for llm_name in ("llama", "mistral"):
            result = {"feature": "raw_text", "model": llm_name}
            try:
                llm = ALL_LLM_MODELS[llm_name]()
                with timed("train", result):
                    llm.fit(X_train, y_train, label_encoder=le)
                with timed("inference_test", result):
                    y_pred = llm.predict(X_test)
                n = min(len(X_test), llm.sample_size)
                m = evaluate(y_test[:n], y_pred[:n], le.classes_)
                result.update({"test_accuracy": m["accuracy"],
                                "test_f1_macro": m["f1_macro"],
                                "test_f1_weighted": m["f1_weighted"]})
            except Exception as exc:
                result["error"] = str(exc)
            all_results.append(result)
    total_elapsed = time.perf_counter() - total_start
    os.makedirs(args.out, exist_ok=True)
    save_results({"total_time": total_elapsed, "n_ranks": SIZE, "runs": all_results},
                 os.path.join(args.out, "results_mpi.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--out",  default=RESULTS_DIR)
    parser.add_argument("--llm",  action="store_true")
    args = parser.parse_args()
    gpu_seed(RANDOM_SEED)
    if RANK == 0: master_process(args)
    else:         worker_process(RANK)


if __name__ == "__main__":
    main()
