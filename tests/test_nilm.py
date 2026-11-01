import numpy as np
import pytest

from nilm_sim.appliances import CATALOG, FRIDGE
from nilm_sim.dataset import APPLIANCES, make_dataset, split_by_house
from nilm_sim.events import HartDisaggregator, detect_events
from nilm_sim.house import LiveHouse, simulate_house
from nilm_sim.metrics import f1_on, mae, sae


@pytest.fixture(scope="module")
def house():
    return simulate_house(42)


def test_house_shape_and_columns(house):
    assert len(house) == 86_400
    for a in APPLIANCES:
        assert {f"P_{a}", f"Q_{a}", f"on_{a}"} <= set(house.columns)


def test_aggregate_is_sum_plus_base_and_noise(house):
    """Physical invariant: the aggregate is the sum of appliances plus base load and noise."""
    parts = house[[f"P_{a}" for a in APPLIANCES]].sum(axis=1)
    resid = house["P"] - parts
    assert resid.std() < 10              # only 5 W noise around the base load
    assert 30 < resid.mean() < 110       # base load between 40 and 100 W


def test_same_seed_same_house():
    a, b = simulate_house(7), simulate_house(7)
    assert np.allclose(a["P"], b["P"])


def test_fridge_spike_only_at_start():
    state = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    p, _ = FRIDGE.render(state, np.random.default_rng(0))
    assert p[2] > 400 and p[3] > 400     # 2 s spike
    assert p[4] < 200                    # then the steady level


def test_detect_single_step():
    p = np.r_[np.full(100, 80.0), np.full(100, 140.0)]
    ev = detect_events(p, np.zeros_like(p), threshold=30)
    assert len(ev) == 1
    assert ev.dP.iloc[0] == pytest.approx(60, abs=1)


def test_split_is_by_house():
    houses = make_dataset(4)
    train, test = split_by_house(houses, 1)
    ids_train = {d.attrs["house"] for d in train}
    ids_test = {d.attrs["house"] for d in test}
    assert ids_train.isdisjoint(ids_test)


def test_hart_recovers_shower():
    houses = make_dataset(3)
    hart = HartDisaggregator().fit(houses[:2], APPLIANCES)
    df = houses[2]
    pred, _ = hart.predict(df["P"].to_numpy(), df["Q"].to_numpy())
    assert f1_on(df["on_shower"].to_numpy(), pred["P_shower"].to_numpy() > 15) > 0.9


def test_metrics():
    y = np.array([0, 100, 100, 0.0])
    assert mae(y, y) == 0
    assert sae(y, y * 1.1) == pytest.approx(0.1)
    assert f1_on(np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 0], bool)) == pytest.approx(2 / 3)


def test_live_house_buffer():
    live = LiveHouse(seed=1, buffer_s=50)
    live.step(30, {"shower": 1})
    df = live.step(30, {"shower": 0})
    assert len(df) == 50
    assert df["P_shower"].iloc[0] > 4000 and df["P_shower"].iloc[-1] == 0
    assert set(CATALOG) == set(APPLIANCES)
