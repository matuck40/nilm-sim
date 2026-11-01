"""Per-appliance NILM metrics.

- MAE (W): mean absolute error, sample by sample.
- SAE: relative error on total energy over the period (what the bill sees).
- F1: on/off state accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y_hat)))


def sae(y: np.ndarray, y_hat: np.ndarray) -> float:
    total = np.sum(y)
    return float(abs(np.sum(y_hat) - total) / total) if total > 0 else float("nan")


def f1_on(on_true: np.ndarray, on_pred: np.ndarray) -> float:
    tp = np.sum(on_true & on_pred)
    fp = np.sum(~on_true & on_pred)
    fn = np.sum(on_true & ~on_pred)
    return float(2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else float("nan")


def evaluate(df: pd.DataFrame, pred: pd.DataFrame, appliances: list[str],
             on_threshold: float = 15.0) -> pd.DataFrame:
    rows = []
    for a in appliances:
        y = df[f"P_{a}"].to_numpy()
        y_hat = pred[f"P_{a}"].to_numpy()
        on_true = df[f"on_{a}"].to_numpy()
        rows.append({"appliance": a, "MAE_W": mae(y, y_hat), "SAE": sae(y, y_hat),
                     "F1": f1_on(on_true, y_hat > on_threshold),
                     "energy_true_kWh": y.sum() / 3.6e6,
                     "energy_pred_kWh": y_hat.sum() / 3.6e6})
    return pd.DataFrame(rows).set_index("appliance")
