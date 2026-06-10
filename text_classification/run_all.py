import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger  = logging.getLogger("run_all")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script, extra_args):
    cmd = [sys.executable, os.path.join(BASE_DIR, script)] + extra_args
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True)
    parser.add_argument("--out",   default="results/")
    parser.add_argument("--mode",  choices=["serial", "optimized", "parallel", "all"],
                        default="all")
    parser.add_argument("--llm",   action="store_true")
    args   = parser.parse_args()
    common = ["--data", args.data, "--out", args.out] + (["--llm"] if args.llm else [])
    modes  = (["serial", "optimized", "parallel"] if args.mode == "all" else [args.mode])
    scripts = {"serial": "pipeline_serial.py",
               "optimized": "pipeline_optimized.py",
               "parallel":  "pipeline_parallel.py"}
    codes  = {m: run_script(scripts[m], common) for m in modes}
    run_script("analysis/analyse_results.py", ["--results-dir", args.out])
    return max(codes.values(), default=0)


if __name__ == "__main__":
    sys.exit(main())
