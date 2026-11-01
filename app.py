"""nilm-sim interface. Run with: streamlit run app.py

Tab 1, Live: switch appliances on and off, the meter shows the sum, and both
models say what they think is on.
Tab 2, Simulated day: a whole house over one day, the events in the P-Q plane
and the models' metrics against ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nilm_sim.dataset import APPLIANCES, make_dataset, split_by_house
from nilm_sim.events import HartDisaggregator
from nilm_sim.house import LiveHouse, simulate_house
from nilm_sim.metrics import evaluate
from nilm_sim.models import ForestDisaggregator

N_HOUSES, N_TEST = 10, 3
COLORS = dict(zip(APPLIANCES, px.colors.qualitative.Set2, strict=False))

st.set_page_config(page_title="nilm-sim", layout="wide")


@st.cache_resource(show_spinner="Training Hart and Random Forest on the training houses (about 1 min)...")
def train_models():
    houses = make_dataset(N_HOUSES)
    train, _ = split_by_house(houses, N_TEST)
    hart = HartDisaggregator().fit(train, APPLIANCES)
    forest = ForestDisaggregator().fit(train, APPLIANCES)
    return hart, forest, [df.attrs["house"] for df in train]


hart, forest, train_ids = train_models()

st.title("nilm-sim")
st.caption("Non-intrusive load monitoring on a synthetic house at 1 Hz. "
           f"Models trained on houses {train_ids}; any other house is unseen.")

tab_live, tab_day = st.tabs(["Live", "Simulated day"])

# ------------------------------------------------------------------- Live
with tab_live:
    if "live" not in st.session_state:
        st.session_state.live = LiveHouse(seed=999)
        st.session_state.live.step(120, {})          # 2 min of history for the windows

    c1, c2, c3, c4, c5 = st.columns(5)
    states = {
        "tv": int(c1.toggle("TV")),
        "light": int(c2.toggle("Light")),
        "fan": {"off": 0, "speed 1": 1, "speed 2": 2}[
            c3.selectbox("Fan", ["off", "speed 1", "speed 2"])],
        "shower": int(c4.toggle("Shower")),
    }
    c5.markdown("**Fridge:** cycles on its own")
    s1, s2 = st.columns([1, 3])
    running = s1.toggle("Run", value=True)
    speed = s2.slider("Simulated seconds per refresh", 1, 30, 5)

    @st.fragment(run_every=1.0 if running else None)
    def live_panel():
        house: LiveHouse = st.session_state.live
        df = house.step(speed, states) if running else house.df
        h_pred, _ = hart.predict(df["P"].to_numpy(), df["Q"].to_numpy())
        # Hart replays the whole buffer (6 h) so it still sees each switch-on event;
        # the forest only needs its longest window (600 s) before the plotted 15 min.
        f_pred = forest.predict(df.iloc[-1800:].reset_index(drop=True))

        rows = []
        for a in APPLIANCES:
            rows.append({"appliance": a, "true (W)": round(df[f"P_{a}"].iloc[-1]),
                         "Hart (W)": round(h_pred[f"P_{a}"].iloc[-1]),
                         "Random Forest (W)": round(f_pred[f"P_{a}"].iloc[-1])})
        left, right = st.columns([3, 2])
        tail = df.iloc[-900:]
        minutes = (tail.index - tail.index[-1]) / 60
        fig = go.Figure(go.Scatter(x=minutes, y=tail["P"], mode="lines", name="aggregate P"))
        fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                          title="What the meter sees (last 15 min)",
                          xaxis_title="minutes", yaxis_title="W")
        left.plotly_chart(fig, width="stretch")
        right.markdown(f"**Now:** P = {df['P'].iloc[-1]:.0f} W, Q = {df['Q'].iloc[-1]:.0f} var")
        right.dataframe(pd.DataFrame(rows).set_index("appliance"), width="stretch")

        pick = st.selectbox("Compare one appliance over time", APPLIANCES, key="live_pick")
        comp = pd.DataFrame({"true": tail[f"P_{pick}"].to_numpy(),
                             "Hart": h_pred[f"P_{pick}"].iloc[-900:].to_numpy(),
                             "Random Forest": f_pred[f"P_{pick}"].iloc[-900:].to_numpy()},
                            index=minutes)
        st.line_chart(comp, height=220)

    live_panel()

# ---------------------------------------------------------- Simulated day
with tab_day:
    seed = st.number_input("House seed", 0, 10_000, value=N_HOUSES - 1, step=1)
    seen = seed in train_ids
    st.markdown("House **seen in training**: the result is optimistic." if seen
                else "House **unseen** by the models: this is the honest test.")

    @st.cache_data(show_spinner="Simulating and disaggregating the day...")
    def run_day(seed: int):
        df = simulate_house(int(seed))
        h_pred, events = hart.predict(df["P"].to_numpy(), df["Q"].to_numpy())
        h_pred.index = df.index
        f_pred = forest.predict(df)
        m = pd.concat({"Hart": evaluate(df, h_pred, APPLIANCES),
                       "Random Forest": evaluate(df, f_pred, APPLIANCES)}, axis=1)
        return df, h_pred, f_pred, events, m

    df, h_pred, f_pred, events, m = run_day(seed)

    hours = (df.index - df.index[0]).total_seconds() / 3600
    step = 10                                            # plot 1 of every 10 s
    fig = go.Figure()
    for a in APPLIANCES:
        fig.add_trace(go.Scatter(x=hours[::step], y=df[f"P_{a}"].iloc[::step], name=a,
                                 stackgroup="true", line={"width": 0, "color": COLORS[a]}))
    fig.add_trace(go.Scatter(x=hours[::step], y=df["P"].iloc[::step], name="aggregate",
                             line={"color": "black", "width": 1}))
    fig.update_layout(height=380, title="One day: aggregate and sub-metering (ground truth)",
                      xaxis_title="hour", yaxis_title="W", margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    ev = events.assign(label=events["appliance"].fillna("unidentified"))
    sc = px.scatter(ev, x="dP", y="dQ", color="label",
                    title="Events in the P-Q plane (each point is a step)",
                    labels={"dP": "ΔP (W)", "dQ": "ΔQ (var)"})
    sig = hart.signatures
    sc.add_trace(go.Scatter(x=sig["dP"], y=sig["dQ"], mode="markers+text", text=sig.index,
                            textposition="top center", name="signatures",
                            marker={"symbol": "x", "size": 12, "color": "black"}))
    sc.update_xaxes(range=[-250, 250])
    sc.update_yaxes(range=[-150, 150])
    a.plotly_chart(sc, width="stretch")
    a.caption("Zoomed to ±250 W; the shower sits outside, at ±5 kW.")

    b.markdown("**Metrics for this day**")
    b.dataframe(m.xs("MAE_W", axis=1, level=1).round(1).rename(columns=lambda c: f"MAE {c} (W)")
                .join(m.xs("F1", axis=1, level=1).round(3).rename(columns=lambda c: f"F1 {c}")),
                width="stretch")
    pick = b.selectbox("Compare one appliance over the day", APPLIANCES, key="day_pick")
    comp = pd.DataFrame({"true": df[f"P_{pick}"].iloc[::step].to_numpy(),
                         "Hart": h_pred[f"P_{pick}"].iloc[::step].to_numpy(),
                         "Random Forest": f_pred[f"P_{pick}"].iloc[::step].to_numpy()},
                        index=np.asarray(hours[::step]))
    st.line_chart(comp, height=260)
