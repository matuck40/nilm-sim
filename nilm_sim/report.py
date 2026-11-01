"""Train both models on the training houses and evaluate on the test houses.

Usage: python -m nilm_sim.report
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .dataset import APPLIANCES, make_dataset, split_by_house
from .events import HartDisaggregator
from .metrics import evaluate
from .models import ForestDisaggregator


def run(n_houses: int = 10, n_test: int = 3, out: str | Path = "reports/metrics.md") -> pd.DataFrame:
    t0 = time.time()
    houses = make_dataset(n_houses)
    train, test = split_by_house(houses, n_test)

    hart = HartDisaggregator().fit(train, APPLIANCES)
    forest = ForestDisaggregator().fit(train, APPLIANCES)

    results = []
    for df in test:
        h_pred, _ = hart.predict(df["P"].to_numpy(), df["Q"].to_numpy())
        h_pred.index = df.index
        for model, pred in (("Hart", h_pred), ("RandomForest", forest.predict(df))):
            r = evaluate(df, pred, APPLIANCES).assign(model=model, house=df.attrs["house"])
            results.append(r.reset_index())

    res = pd.concat(results)
    summary = (res.groupby(["model", "appliance"])[["MAE_W", "SAE", "F1"]]
               .mean().round(3).unstack("model"))
    summary.columns = [f"{metric} {model}" for metric, model in summary.columns]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    test_ids = [df.attrs["house"] for df in test]
    out.write_text(
        "# Metrics on unseen houses\n\n"
        f"Training: {len(train)} houses; test: houses {test_ids} (one day each, 1 Hz). "
        "Mean over the test houses.\n\n"
        "## Learned signatures (Hart)\n\n" + hart.signatures.round(1).to_markdown() +
        "\n\n## Results\n\n" + summary.to_markdown() + "\n",
        encoding="utf-8")
    print(hart.signatures.round(1), "\n")
    print(summary)
    print(f"\n{time.time() - t0:.0f} s; report written to {out}")
    return summary


if __name__ == "__main__":
    run()
