"""ML model to compare against Hart: Random Forest over sliding windows.

The sliding window turns the series into a table: each second gets features
from the recent past (rolling means and deviations, steps, power factor;
optionally time of day), and
the target is each appliance's power at that second. A single model predicts
all appliances at once (multi-output regression).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

WINDOWS = (10, 60, 600)


def make_features(p: np.ndarray, q: np.ndarray, t_seconds: np.ndarray | None = None) -> pd.DataFrame:
    """Features use only past and present samples (nothing from the future: no leakage)."""
    P, Q = pd.Series(p, dtype=float), pd.Series(q, dtype=float)
    f = {"P": P, "Q": Q, "dP": P.diff().fillna(0), "dQ": Q.diff().fillna(0)}
    for w in WINDOWS:
        f[f"P_mean_{w}"] = P.rolling(w, min_periods=1).mean()
        f[f"P_std_{w}"] = P.rolling(w, min_periods=1).std().fillna(0)
        f[f"Q_mean_{w}"] = Q.rolling(w, min_periods=1).mean()
    base = P.rolling(1800, min_periods=1).min()          # base-load estimate
    f["P_above_base"] = P - base
    f["PF"] = P / np.sqrt(P**2 + Q**2).clip(lower=1)     # power factor
    if t_seconds is not None:
        h = (np.asarray(t_seconds) % 86_400) / 3600
        f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    return pd.DataFrame(f)


def _t_seconds(df: pd.DataFrame) -> np.ndarray:
    if isinstance(df.index, pd.DatetimeIndex):
        return ((df.index - df.index[0]).total_seconds()).to_numpy()
    return np.asarray(df.index)


class ForestDisaggregator:
    def __init__(self, use_hour: bool = False, stride: int = 5, **rf_kwargs):
        self.use_hour = use_hour
        self.stride = stride        # train on 1 of every `stride` seconds: neighbours are near-duplicates
        self.rf = RandomForestRegressor(**({"n_estimators": 60, "max_depth": 16,
                                            "min_samples_leaf": 5, "n_jobs": -1,
                                            "random_state": 0} | rf_kwargs))
        self.appliances: list[str] = []

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        t = _t_seconds(df) if self.use_hour else None
        return make_features(df["P"].to_numpy(), df["Q"].to_numpy(), t)

    def fit(self, houses: list[pd.DataFrame], appliances: list[str]) -> ForestDisaggregator:
        self.appliances = appliances
        X = pd.concat([self._X(df).iloc[::self.stride] for df in houses])
        y = pd.concat([df[[f"P_{a}" for a in appliances]].iloc[::self.stride] for df in houses])
        self.rf.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        y = self.rf.predict(self._X(df).to_numpy()).clip(min=0)
        return pd.DataFrame(y, columns=[f"P_{a}" for a in self.appliances], index=df.index)
