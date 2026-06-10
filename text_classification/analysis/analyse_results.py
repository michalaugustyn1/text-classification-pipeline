import argparse
import json
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import NumpyEncoder


def load_runs(path):
    with open(path) as f: d = json.load(f)
    runs = d.get("runs", d) if isinstance(d, dict) else d
    df = pd.DataFrame(runs)
    df.dropna(subset=["test_accuracy"], inplace=True)
    return df


def load_scaling_results(scaling_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(scaling_dir, "njobs_*"))):
        n = int(os.path.basename(d).replace("njobs_", ""))
        rf = os.path.join(d, "results_parallel.json")
        if not os.path.exists(rf): continue
        with open(rf) as f: data = json.load(f)
        rows.append({"n_jobs": n,
                     "total_time": data.get("timing", {}).get("total_wall_time", np.nan)})
    return pd.DataFrame(rows).sort_values("n_jobs")


def plot_heatmap(df, metric, title, out_path):
    pivot = df.pivot_table(index="feature", columns="model", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, cbar_kws={"label": metric})
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Model", fontsize=11); ax.set_ylabel("Feature Extractor", fontsize=11)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_time_comparison(result_files, out_path):
    labels, times = [], []
    for label, path in result_files.items():
        if not os.path.exists(path): continue
        with open(path) as f: d = json.load(f)
        t = d.get("total_time") or d.get("timing", {}).get("total_wall_time", 0)
        labels.append(label); times.append(t / 60)
    if not times: return
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, times, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.bar_label(bars, fmt="%.1f min", padding=3, fontsize=10)
    ax.set_ylabel("Wall-clock time (minutes)")
    ax.set_title("Total Pipeline Runtime by Variant")
    ax.set_ylim(0, max(times) * 1.2)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_speedup(df_scaling, out_path):
    if df_scaling.empty: return
    base = df_scaling.loc[df_scaling["n_jobs"].idxmin(), "total_time"]
    df_scaling = df_scaling.copy()
    df_scaling["speedup"] = base / df_scaling["total_time"]
    df_scaling["ideal"]   = df_scaling["n_jobs"] / df_scaling["n_jobs"].min()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_scaling["n_jobs"], df_scaling["speedup"], "o-", label="Actual", lw=2)
    ax.plot(df_scaling["n_jobs"], df_scaling["ideal"],   "k--", label="Ideal", lw=1.5)
    ax.set_xlabel("n_jobs"); ax.set_ylabel("Speedup")
    ax.set_title("Strong-Scaling Speedup"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_train_time(df, title, out_path):
    if "train_time" not in df.columns: return
    df.pivot_table(index="model", columns="feature", values="train_time", aggfunc="mean")\
      .plot(kind="bar", figsize=(12, 5), width=0.7)
    plt.title(title); plt.ylabel("Training time (s)"); plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    args   = parser.parse_args()
    rdir   = args.results_dir
    pdir   = os.path.join(rdir, "plots")
    os.makedirs(pdir, exist_ok=True)

    result_files = {
        "Serial":    os.path.join(rdir, "results_serial.json"),
        "Optimized": os.path.join(rdir, "results_optimized.json"),
        "Parallel":  os.path.join(rdir, "results_parallel.json"),
        "MPI":       os.path.join(rdir, "results_mpi.json"),
    }

    all_dfs = []
    for label, path in result_files.items():
        if not os.path.exists(path): continue
        df = load_runs(path)
        for metric, tag in (("test_accuracy", "accuracy"), ("test_f1_macro", "f1")):
            plot_heatmap(df, metric, f"{label} — Test {tag.upper()} (feature × model)",
                         os.path.join(pdir, f"heatmap_{tag}_{label.lower()}.png"))
        plot_train_time(df, f"{label} — Training Time",
                        os.path.join(pdir, f"train_time_{label.lower()}.png"))
        df["pipeline"] = label
        all_dfs.append(df)

    plot_time_comparison(result_files, os.path.join(pdir, "walltime_comparison.png"))
    plot_speedup(load_scaling_results(os.path.join(rdir, "scaling")),
                 os.path.join(pdir, "speedup.png"))

    if all_dfs:
        summary = pd.concat(all_dfs, ignore_index=True)
        summary.to_csv(os.path.join(rdir, "summary.csv"), index=False)
        best = (summary.sort_values("test_f1_macro", ascending=False)
                       .groupby("pipeline").first()
                       [["feature", "model", "test_accuracy", "test_f1_macro"]])
        print(best.to_string())


if __name__ == "__main__":
    main()
