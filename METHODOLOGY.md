# Methodology

## Recall@k

`recall_at_k(predicted, ground_truth)` — for each query, intersect the
adapter's returned ids with the true top-k ids (from `FAISS IndexFlatL2`
exact brute-force search over the same vectors) and divide total hits by
total ground-truth items across all queries. 1.0 means every returned
neighbor across every query was actually in the true top-k.

## Why synthetic (random unit) vectors are valid

Vector database performance for nearest-neighbor search is a function of
dimensionality, index structure, and distance-computation cost — not of
what the vectors semantically represent. Uniformly distributed unit vectors
exercise the same L2/cosine distance machinery and index traversal as real
embeddings, without requiring a specific embedding model or corpus. They are
a **worse-than-real-world case for clustering-based indexes** (IVF, HNSW)
because embeddings from real corpora cluster in semantically meaningful
regions of the space, which these indexes exploit — recall numbers on
uniform random data are a conservative lower bound, not an inflated one.

## Known limitations

- **Memory measurement**: `peak_memory_mb` uses `tracemalloc`, which only
  tracks Python-heap allocations. FAISS and ChromaDB's HNSW graph live
  largely in native (C++) memory, so these numbers understate true peak RSS.
  A `psutil`-based RSS delta would be more representative — not implemented
  here to avoid an extra dependency for the quick-config scope.
- **IVF recall is deliberately unflattering** at the dataset sizes in
  `QUICK_CONFIG` (2k/10k vectors, 100 clusters): FAISS warns that 100
  centroids need ~3900 training points to cluster well below that. This
  reflects nlist being tuned for the "IVF100" spec from the original design,
  not a real-world 100k+ vector deployment — run the full config (larger
  sizes) before drawing conclusions about IVF's actual recall ceiling.
- **Docker-based engines (Qdrant, pgvector) are not benchmarked** in this
  build — see README "what's here vs. spec".
- **Concurrency (QPS) uses `ThreadPoolExecutor`**, which measures Python-level
  dispatch overhead + each adapter's actual concurrency handling (both FAISS
  and ChromaDB release the GIL during native search calls, so this is a
  reasonable proxy, not a network-level load test).
