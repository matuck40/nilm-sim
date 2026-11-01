"""Hart's (1992) event detector, minimal version.

1. Find the steps in the aggregate (something switched on or off).
2. Measure each step in the P-Q plane: (ΔP, ΔQ).
3. Assign each step to the appliance with the nearest signature.
4. Pair each appliance's "on" with its "off".

Signatures are learned from the sub-metering of the training houses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_events(p: np.ndarray, q: np.ndarray, threshold: float = 30.0,
                  median_w: int = 5, ctx: int = 6, merge_s: int = 4) -> pd.DataFrame:
    """Steps in P larger than `threshold` watts.

    The median filter removes the fridge start-up spike (2 s) and the noise
    while keeping the steps. ΔP and ΔQ are the median after minus the median
    before the event, so they measure the new level, not the spike.
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    m = pd.Series(p).rolling(median_w, center=True, min_periods=1).median().to_numpy()
    jumps = np.flatnonzero(np.abs(np.diff(m)) > threshold / 2) + 1

    # merge neighbouring jumps into a single event
    groups: list[list[int]] = []
    for i in jumps:
        if groups and i - groups[-1][-1] <= merge_s:
            groups[-1].append(i)
        else:
            groups.append([i])

    rows = []
    n = len(p)
    for g in groups:
        a, b = g[0], g[-1]
        before = slice(max(0, a - ctx - 1), max(1, a - 1))
        after = slice(min(n - 1, b + median_w), min(n, b + median_w + ctx))
        dp = np.median(p[after]) - np.median(p[before])
        dq = np.median(q[after]) - np.median(q[before])
        if abs(dp) >= threshold:
            rows.append((a, dp, dq))
    return pd.DataFrame(rows, columns=["i", "dP", "dQ"])


def learn_signatures(houses: list[pd.DataFrame], appliances: list[str]) -> pd.DataFrame:
    """Each appliance's signature = median (ΔP, ΔQ) over the times it switched
    on, measured on the sub-metering of the training houses."""
    rows = []
    for name in appliances:
        dps, dqs = [], []
        for df in houses:
            on = df[f"on_{name}"].to_numpy()
            starts = np.flatnonzero(~on[:-1] & on[1:]) + 1
            p, q = df[f"P_{name}"].to_numpy(), df[f"Q_{name}"].to_numpy()
            for s in starts:
                if s - 3 < 0 or s + 12 > len(p):
                    continue
                dps.append(np.median(p[s + 4:s + 12]) - np.median(p[s - 3:s]))
                dqs.append(np.median(q[s + 4:s + 12]) - np.median(q[s - 3:s]))
        rows.append((name, float(np.median(dps)), float(np.median(dqs)), len(dps)))
    return pd.DataFrame(rows, columns=["appliance", "dP", "dQ", "n_events"]).set_index("appliance")


class HartDisaggregator:
    """Hart baseline: events + signatures + on/off pairing."""

    def __init__(self, threshold: float = 30.0, max_dist: float = 0.45):
        self.threshold = threshold
        self.max_dist = max_dist
        self.signatures: pd.DataFrame | None = None

    def fit(self, houses: list[pd.DataFrame], appliances: list[str]) -> HartDisaggregator:
        self.signatures = learn_signatures(houses, appliances)
        return self

    def _nearest(self, dp: float, dq: float) -> tuple[str | None, float]:
        """Appliance whose signature is closest to the step, in relative distance."""
        sig = self.signatures
        scale = np.maximum(sig["dP"].abs(), 30.0)
        dist = (np.abs(abs(dp) - sig["dP"]) + np.abs(abs(dq) - np.abs(sig["dQ"]))) / scale
        best = dist.idxmin()
        return (best, float(dist[best])) if dist[best] <= self.max_dist else (None, float(dist[best]))

    def predict(self, p: np.ndarray, q: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (estimated P per appliance, table of labelled events)."""
        if self.signatures is None:
            raise RuntimeError("call fit() first")
        n = len(p)
        events = detect_events(p, q, self.threshold)
        names = list(self.signatures.index)
        on_since: dict[str, int | None] = {a: None for a in names}
        est = {a: np.zeros(n) for a in names}
        labels = []

        for _, ev in events.iterrows():
            i = int(ev.i)
            app, _dist = self._nearest(ev.dP, ev.dQ)
            labels.append(app)
            if app is None:
                continue
            if ev.dP > 0 and on_since[app] is None:            # switch on
                on_since[app] = i
            elif ev.dP < 0 and on_since[app] is not None:      # switch off: close the pair
                est[app][on_since[app]:i] = self.signatures.loc[app, "dP"]
                on_since[app] = None

        for app, start in on_since.items():                     # still on at the end
            if start is not None:
                est[app][start:] = self.signatures.loc[app, "dP"]

        events = events.assign(appliance=labels)
        return pd.DataFrame({f"P_{a}": v for a, v in est.items()}), events
