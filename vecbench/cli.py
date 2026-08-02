from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from vecbench.benchmark.experiment_matrix import (
    ADAPTER_REGISTRY,
    QUICK_CONFIG,
    build_matrix,
)
from vecbench.benchmark.runner import run_experiment
from vecbench.data.generator import compute_ground_truth, generate_vectors

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(config: dict = QUICK_CONFIG, checkpoint_path: Path | None = None) -> pd.DataFrame:
    checkpoint_path = checkpoint_path or (RESULTS_DIR / "experiments.parquet")
    RESULTS_DIR.mkdir(exist_ok=True)

    done = pd.read_parquet(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    done_keys = {(r["db"], r["size"], r["dim"], r["k"]) for r in rows}

    matrix = build_matrix(config)
    for exp in matrix:
        key = (exp["db"], exp["size"], exp["dim"], exp["k"])
        if key in done_keys:
            print(f"skipping (checkpointed): {exp}")
            continue

        print(f"running: {exp}")
        vectors = generate_vectors(exp["size"], exp["dim"], seed=0)
        n_queries = min(config["n_queries"], exp["size"])
        queries = generate_vectors(n_queries, exp["dim"], seed=1)
        ground_truth = compute_ground_truth(vectors, queries, exp["k"])

        adapter = ADAPTER_REGISTRY[exp["db"]]()
        result = run_experiment(
            adapter, vectors, ground_truth, queries, exp["k"], n_threads=config["concurrency"],
        )
        rows.append(result.to_dict())
        pd.DataFrame(rows).to_parquet(checkpoint_path)  # checkpoint after every experiment

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} experiments -> {checkpoint_path}")
    return df


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        print("usage: vecbench run | vecbench report")
        sys.exit(1)

    if sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "report":
        from vecbench.report.html_generator import generate_report

        df = pd.read_parquet(RESULTS_DIR / "experiments.parquet")
        generate_report(df, RESULTS_DIR / "report.html")
        print(f"wrote {RESULTS_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
