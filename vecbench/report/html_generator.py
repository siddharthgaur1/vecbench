from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio


def generate_report(df: pd.DataFrame, output_path: Path) -> None:
    figs = []

    recall = px.bar(df, x="db", y="recall_at_k", color="size", barmode="group",
                     title="Recall@k by database and dataset size")
    figs.append(recall)

    latency = px.bar(df, x="db", y="latency_p95", color="size", barmode="group",
                      title="p95 query latency (ms) by database and dataset size")
    figs.append(latency)

    if "qps_c1" in df.columns:
        qps_cols = [c for c in df.columns if c.startswith("qps_c")]
        qps_long = df.melt(id_vars=["db", "size"], value_vars=qps_cols, var_name="concurrency", value_name="qps")
        qps_long["concurrency"] = qps_long["concurrency"].str.replace("qps_c", "").astype(int)
        qps_fig = px.line(qps_long, x="concurrency", y="qps", color="db", title="QPS vs concurrency")
        figs.append(qps_fig)

    tradeoff = px.scatter(df, x="latency_p95", y="recall_at_k", color="db", size="size",
                           title="Recall-latency tradeoff (top-left is best)")
    figs.append(tradeoff)

    memory = px.bar(df, x="db", y="peak_memory_mb", color="size", barmode="group",
                     title="Peak memory (MB) — Python-heap only via tracemalloc, see README caveat")
    figs.append(memory)

    winner_by_size = df.loc[df.groupby("size")["recall_at_k"].idxmax()][["size", "db", "recall_at_k"]]
    summary_rows = "".join(
        f"<li>Best recall at {int(r['size'])} vectors: <b>{r['db']}</b> ({r['recall_at_k']:.3f})</li>"
        for _, r in winner_by_size.iterrows()
    )

    body_parts = [f"<h1>vecbench report</h1><h2>Executive summary</h2><ul>{summary_rows}</ul>"]
    for i, fig in enumerate(figs):
        include_js = "inline" if i == 0 else False
        body_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=include_js))

    output_path.write_text(
        f"<!doctype html><html><head><title>vecbench report</title></head>"
        f"<body>{''.join(body_parts)}</body></html>",
        encoding="utf-8",
    )
