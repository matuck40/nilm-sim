"""Appliances as state machines with an electrical signature (P, Q).

Each appliance has a list of states. State 0 is off (or standby) and the
others are operating levels. Each state has an active power P (W) and a
reactive power Q (var). Hart's event detector recognises appliances by this
signature in the P-Q plane.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Appliance:
    name: str
    states: tuple[tuple[float, float], ...]   # (P, Q) per state; index 0 = off
    spike_p: float = 0.0     # start-up spike added to P, in W
    spike_q: float = 0.0     # start-up spike added to Q, in var
    spike_s: int = 0         # spike duration, in seconds
    noise_rel: float = 0.02  # multiplicative noise of the load itself

    @property
    def n_states(self) -> int:
        return len(self.states)

    def scaled(self, factor: float) -> Appliance:
        """Same appliance, different brand: all powers multiplied by `factor`."""
        states = tuple((p * factor, q * factor) for p, q in self.states)
        return replace(self, states=states,
                       spike_p=self.spike_p * factor, spike_q=self.spike_q * factor)

    def render(self, state: np.ndarray, rng: np.random.Generator,
               prev_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Turn a state sequence (one per second) into P(t) and Q(t).

        `prev_state` is the state one second before the first sample, so the
        start-up spike is right when the signal is generated in chunks (live mode).
        """
        state = np.asarray(state, dtype=int)
        table = np.asarray(self.states, dtype=float)
        p = table[state, 0].copy()
        q = table[state, 1].copy()

        if self.spike_s > 0:
            prev = np.concatenate([[prev_state], state[:-1]])
            starts = np.flatnonzero((prev == 0) & (state > 0))
            for s in starts:
                end = min(s + self.spike_s, len(state))
                p[s:end] += self.spike_p
                q[s:end] += self.spike_q

        on = p > 0
        p[on] *= 1 + rng.normal(0, self.noise_rel, on.sum())
        q[on] *= 1 + rng.normal(0, self.noise_rel, on.sum())
        return p, q


# Base catalogue. Each house draws a power variation around these values.
FRIDGE = Appliance("fridge", ((0, 0), (150, 90)), spike_p=450, spike_q=300, spike_s=2)
TV = Appliance("tv", ((1, 0), (100, 15)))
LIGHT = Appliance("light", ((0, 0), (60, 0)))
FAN = Appliance("fan", ((0, 0), (55, 40), (80, 60)))
SHOWER = Appliance("shower", ((0, 0), (5500, 0)), noise_rel=0.01)

CATALOG: dict[str, Appliance] = {a.name: a for a in (FRIDGE, TV, LIGHT, FAN, SHOWER)}
