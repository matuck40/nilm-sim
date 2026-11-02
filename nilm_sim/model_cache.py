"""Train once, reuse from disk.

The app's models come from fixed seeds, so a model trained yesterday is identical to one trained
now as long as the code and the libraries have not changed. The cache file name carries a
fingerprint of both; any change gives a new name and the model is trained again.
Only files this function wrote itself are ever loaded.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

PACKAGE_DIR = Path(__file__).parent
DEFAULT_DIR = PACKAGE_DIR.parent / "models"


def code_fingerprint() -> str:
    """12 hex characters from every nilm_sim source file and the library versions."""
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        h.update(str(path.relative_to(PACKAGE_DIR)).encode())
        h.update(path.read_bytes())
    for version in (np.__version__, pd.__version__, sklearn.__version__, joblib.__version__):
        h.update(version.encode())
    return h.hexdigest()[:12]


def cached_fit(name: str, train: Callable[[], object], cache_dir: Path | None = None) -> object:
    """Return train()'s result, loading it from cache_dir when this code already produced it."""
    if cache_dir is None:
        cache_dir = Path(os.environ.get("NILM_MODEL_DIR", DEFAULT_DIR))
    path = Path(cache_dir) / f"{name}-{code_fingerprint()}.joblib"
    if path.exists():
        return joblib.load(path)
    obj = train()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(obj, tmp)
    tmp.replace(path)                       # never leave a half-written model behind
    return obj
