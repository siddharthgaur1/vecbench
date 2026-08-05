"""`cli.run()` opens the committed results checkpoint through pandas before it
does anything else:

    done = pd.read_parquet(checkpoint_path) if checkpoint_path.exists() else ...

pandas needs pyarrow (or fastparquet) for that, and neither is pulled in by
numpy/pandas alone. Until pyarrow was declared, a fresh `pip install -e .`
followed by the README's quickstart died with:

    ImportError: Unable to find a usable engine; tried using: 'pyarrow',
    'fastparquet'.

Which only ever showed up on a clean environment - never on a dev machine that
had pyarrow from something else.
"""

import pandas as pd


def test_a_parquet_engine_is_installed():
    """Fails on any environment where the quickstart would fail."""
    pd.io.parquet.get_engine("auto")


def test_roundtrip_through_the_checkpoint_format(tmp_path):
    p = tmp_path / "experiments.parquet"
    df = pd.DataFrame([{"db": "faiss_flat", "size": 2000, "dim": 128, "k": 10}])
    df.to_parquet(p)
    assert pd.read_parquet(p).equals(df)
