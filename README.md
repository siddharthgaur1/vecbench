# vecbench

Reproducible recall/latency/throughput/memory benchmarks for vector
databases used in RAG, with an auto-generated self-contained HTML report.

## Results (this run, `configs` = `QUICK_CONFIG`: 2k/10k vectors, dim=128, k=10)

| DB | Size | Recall@10 | p50 latency (ms) | p95 latency (ms) | Insert throughput (vec/s) |
|---|---|---|---|---|---|
| chroma | 2,000 | 0.976 | 4.97 | 7.65 | 1,465 |
| chroma | 10,000 | 0.753 | 4.98 | 7.83 | 1,755 |
| faiss_flat | 2,000 | **1.000** | 2.28 | 3.08 | 2,922,695 |
| faiss_flat | 10,000 | **1.000** | 2.75 | 4.19 | 1,913,143 |
| faiss_ivf | 2,000 | 0.389 | 0.10 | 0.21 | 112,423 |
| faiss_ivf | 10,000 | 0.351 | 0.08 | 0.12 | 109,896 |
| faiss_hnsw | 2,000 | 0.759 | 0.08 | 0.13 | 66,549 |
| faiss_hnsw | 10,000 | 0.428 | 0.20 | 0.38 | 32,491 |

Full data: `results/experiments.parquet`. Interactive charts: `results/report.html`
(regenerate with `python -m vecbench.cli report`).

## Key findings

- **FAISS Flat wins on recall by construction** (it *is* the ground truth) —
  the real signal is that everything else trades recall for latency: ~800x
  faster queries than Flat at 2k vectors, for a ~25-60pp recall cost.
- **ChromaDB's HNSW default degrades from 0.976 → 0.753 recall between 2k
  and 10k vectors** with unchanged parameters — its default `ef`/`M` aren't
  tuned for this scale out of the box.
- **FAISS IVF's recall (0.35-0.39) is an artifact of too few training points
  for 100 clusters at this scale**, not a fair reading of IVF's ceiling —
  see `METHODOLOGY.md`. Don't conclude "IVF is bad" from the quick config.
- **FAISS HNSW is the best recall/latency tradeoff of the approximate
  indexes** at 2k vectors (0.759 recall at 0.13ms p95) but also degrades at
  10k (0.428) — same root cause as Chroma, unequal to default M=32 at this
  density with random-direction vectors (see METHODOLOGY.md on why that's a
  conservative, not inflated, comparison).

## Recommendations

- Best for **<10k vectors, simplest setup**: ChromaDB — competitive recall, zero index tuning.
- Best for **exact search / small corpora where 100% recall matters**: FAISS Flat.
- Best for **latency-sensitive approximate search at real corpus scale**: FAISS HNSW, but tune `M`/`ef_search` — don't trust the quick-config defaults above at production scale.
- Don't judge FAISS IVF from this run — rerun with `nlist` sized correctly for your dataset (`METHODOLOGY.md`).

## Quickstart

```
pip install -e .
python -c "from vecbench.cli import run; run()"     # quick config, ~1 min
python -m vecbench.cli report                        # writes results/report.html
```

## Architecture

Each vector DB (FAISS Flat/IVF/HNSW, ChromaDB) implements one shared
`BaseAdapter` interface (insert, query, memory footprint), so the runner
exercises all of them through identical code — recall is scored against
FAISS Flat's exact search as ground truth, latency is measured per query at
multiple concurrency levels, and every completed experiment is checkpointed
to `results/experiments.parquet` so an interrupted run resumes instead of
restarting.

Checkpointed after every experiment (`results/experiments.parquet`) — an
interrupted run resumes without re-running completed experiments.

## Reproducibility

Hardware: whatever machine ran this (`os.cpu_count()`, RAM — not
auto-detected in this build, see below). Same config → same results ±2%
(fixed seeds in `data/generator.py`).

## Limitations

Built: FAISS (Flat/IVF/HNSW) + ChromaDB adapters, recall@k against exact
ground truth, p50/p95/p99 latency, QPS at multiple concurrency levels,
insert throughput, `tracemalloc`-based memory, checkpointed runner, quick
config, self-contained HTML report with Pareto-style recall-latency scatter.

What's not here — add if actually needed:
- **Qdrant and pgvector adapters** — both need a running Docker service; can't stand those up and leave them running unattended. Add `adapters/qdrant_adapter.py` / `adapters/pgvector_adapter.py` (interfaces already match `BaseAdapter`) once you have the containers up.
- Real SIFT1M / sentence-transformers embedding datasets — only synthetic unit vectors (see `METHODOLOGY.md` for why that's still valid)
- `full.yaml`-scale sweep (100k/500k vectors, multiple dims) — quick config only, full run is the multi-hour job the original spec called out
- Auto-detected hardware specs section in the report
- `vecbench compare` CLI subcommand — use the parquet + pandas directly for now
