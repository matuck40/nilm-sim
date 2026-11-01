"""Synthetic dataset: several houses and the split by house.

Why split by house: splitting in time inside the same house lets the model
see that family's routine and appliances during training, so the test score
is optimistic. Holding out whole houses measures what matters in production:
being right on a house the model has never seen.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .appliances import CATALOG
from .house import simulate_house

APPLIANCES = list(CATALOG)


def make_dataset(n_houses: int = 10, n_days: int = 1, seed0: int = 0) -> list[pd.DataFrame]:
    return [simulate_house(seed0 + i, n_days=n_days) for i in range(n_houses)]


def split_by_house(houses: list[pd.DataFrame], n_test: int = 3
                   ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """The last `n_test` houses are kept for testing only."""
    return houses[:-n_test], houses[-n_test:]


def save_dataset(houses: list[pd.DataFrame], folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for df in houses:
        df.to_csv(folder / f"house_{df.attrs['house']:02d}.csv.gz")
