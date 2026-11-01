"""A house: each appliance's routine, the sum at the meter, and noise.

Two modes:
- `simulate_house`: a whole day at once (to build the dataset).
- `LiveHouse`: second by second, with states coming from outside
  (the app's switches), to test the models "live".

Output is always 1 Hz, like a low-frequency smart meter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .appliances import CATALOG, Appliance

DAY = 86_400


def _sessions(n: int, rng: np.random.Generator, windows: list[tuple[float, float]],
              n_range: tuple[int, int], dur_min: tuple[float, float],
              level: int | tuple[int, ...] = 1) -> np.ndarray:
    """Random usage sessions inside time-of-day windows (in hours)."""
    state = np.zeros(n, dtype=int)
    for day0 in range(0, n, DAY):
        for _ in range(rng.integers(n_range[0], n_range[1] + 1)):
            h0, h1 = windows[rng.integers(len(windows))]
            start = day0 + int(rng.uniform(h0, h1) * 3600)
            dur = int(rng.uniform(*dur_min) * 60)
            lv = level if isinstance(level, int) else int(rng.choice(level))
            state[start:min(start + dur, n)] = lv
    return state


def _fridge_cycles(n: int, rng: np.random.Generator) -> np.ndarray:
    """Switches itself on and off: about 15 min on, 30 min off."""
    state = np.zeros(n, dtype=int)
    t = int(rng.uniform(0, 30 * 60))
    while t < n:
        on = int(rng.normal(15, 2) * 60)
        state[t:t + on] = 1
        t += on + int(rng.normal(30, 4) * 60)
    return state


def default_schedule(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name == "fridge":
        return _fridge_cycles(n, rng)
    if name == "tv":
        return _sessions(n, rng, [(12, 14), (19, 23)], (1, 3), (30, 120))
    if name == "light":
        return _sessions(n, rng, [(6, 7.5), (18, 23)], (2, 4), (20, 150))
    if name == "fan":
        return _sessions(n, rng, [(13, 17), (21, 23.5)], (1, 3), (40, 180), level=(1, 2))
    if name == "shower":
        return _sessions(n, rng, [(6, 8), (18, 22)], (1, 3), (8, 15))
    raise KeyError(name)


def house_appliances(rng: np.random.Generator) -> dict[str, Appliance]:
    """Each house has different brands: power ±15% (shower 4.5 to 6.5 kW)."""
    out = {}
    for name, app in CATALOG.items():
        factor = rng.uniform(4500, 6500) / 5500 if name == "shower" else rng.uniform(0.85, 1.15)
        out[name] = app.scaled(factor)
    return out


def simulate_house(seed: int, n_days: int = 1,
                   noise_p: float = 5.0, noise_q: float = 3.0) -> pd.DataFrame:
    """Simulate one house. Columns: aggregate P and Q (what the meter sees) and,
    per appliance, P, Q and on/off state (the sub-metering, i.e. ground truth)."""
    rng = np.random.default_rng(seed)
    n = n_days * DAY
    apps = house_appliances(rng)
    base = rng.uniform(40, 100)                       # base load: what never switches off

    data: dict[str, np.ndarray] = {}
    p_agg = np.full(n, base) + rng.normal(0, noise_p, n)
    q_agg = np.full(n, base * 0.2) + rng.normal(0, noise_q, n)
    for name, app in apps.items():
        state = default_schedule(name, n, rng)
        p, q = app.render(state, rng)
        data[f"P_{name}"], data[f"Q_{name}"], data[f"on_{name}"] = p, q, state > 0
        p_agg += p
        q_agg += q

    df = pd.DataFrame({"P": p_agg, "Q": q_agg, **data})
    df.index = pd.date_range("2026-01-01", periods=n, freq="s", name="t")
    df.attrs["house"] = seed
    return df


class LiveHouse:
    """House generated second by second. TV, light, fan and shower states come
    from the switches; the fridge cycles on its own."""

    def __init__(self, seed: int = 999, buffer_s: int = 6 * 3600, noise_p: float = 5.0,
                 noise_q: float = 3.0):
        self.rng = np.random.default_rng(seed)
        self.apps = house_appliances(self.rng)
        self.base = self.rng.uniform(40, 100)
        self.noise_p, self.noise_q = noise_p, noise_q
        self.buffer_s = buffer_s
        self.prev = {name: 0 for name in self.apps}
        self._fridge_left = int(self.rng.uniform(0, 30 * 60))
        self._fridge_on = False
        self.df = pd.DataFrame()
        self.t = 0

    def _fridge_step(self) -> int:
        if self._fridge_left <= 0:
            self._fridge_on = not self._fridge_on
            mean, sd = (15, 2) if self._fridge_on else (30, 4)
            self._fridge_left = int(self.rng.normal(mean, sd) * 60)
        self._fridge_left -= 1
        return int(self._fridge_on)

    def step(self, seconds: int, states: dict[str, int]) -> pd.DataFrame:
        n = seconds
        rows: dict[str, np.ndarray] = {}
        p_agg = np.full(n, self.base) + self.rng.normal(0, self.noise_p, n)
        q_agg = np.full(n, self.base * 0.2) + self.rng.normal(0, self.noise_q, n)
        for name, app in self.apps.items():
            if name == "fridge":
                state = np.array([self._fridge_step() for _ in range(n)])
            else:
                state = np.full(n, int(states.get(name, 0)))
            p, q = app.render(state, self.rng, prev_state=self.prev[name])
            self.prev[name] = int(state[-1])
            rows[f"P_{name}"], rows[f"Q_{name}"], rows[f"on_{name}"] = p, q, state > 0
            p_agg += p
            q_agg += q
        new = pd.DataFrame({"P": p_agg, "Q": q_agg, **rows},
                           index=pd.RangeIndex(self.t, self.t + n, name="t"))
        self.t += n
        self.df = pd.concat([self.df, new]).iloc[-self.buffer_s:]
        return self.df
