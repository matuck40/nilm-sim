# nilm-sim

A small sandbox for **non-intrusive load monitoring (NILM)**: a synthetic house generator at 1 Hz, Hart's event-based baseline, a Random Forest disaggregator, and a Streamlit app to switch appliances on and off and watch both models react in real time.

> Study project. A Portuguese walkthrough of the code is in `docs/TOUR.pt-BR.md`.
> The data is synthetic; real-data experiments (UK-DALE) are on the roadmap.

## What it does

- **Simulator** (`nilm_sim/house.py`, `appliances.py`): each appliance is a state machine with an active/reactive power signature (P, Q). Fridge cycles on its own with a motor start-up spike, the fan is inductive with two speeds, the shower is a 4.5 to 6.5 kW resistive step. Each house draws its own appliance variants, base load and routine. The output is the aggregate (what a smart meter sees) plus sub-metered ground truth.
- **Hart (1992) baseline** (`events.py`): median-filtered step detection, (ΔP, ΔQ) signatures learned from sub-metered training homes, nearest-signature labelling and on/off pairing.
- **Random Forest** (`models.py`): sliding-window features using only past and present samples (rolling means and standard deviations, steps, power factor, power above base load), multi-output regression of each appliance's power.
- **CI** (GitHub Actions): ruff, pytest on Python 3.12 and 3.13, and gitleaks on pushes to main and on pull requests.
- **Evaluation** (`metrics.py`, `report.py`): **split by house**, so the test homes were never seen in training. MAE in watts, SAE (error on total energy) and F1 on the on/off state.

## Results on unseen homes

7 training homes, 3 test homes, one day each at 1 Hz (`python -m nilm_sim.report`):

| Appliance | MAE Hart (W) | MAE RF (W) | F1 Hart | F1 RF |
|---|---|---|---|---|
| Shower | 10.8 | 1.6 | 0.99 | 1.00 |
| Fridge | 3.5 | 3.5 | 1.00 | 0.98 |
| Light | 1.5 | 4.6 | 0.92 | 0.80 |
| TV | 7.5 | 5.3 | 0.82 | 0.84 |
| Fan | 2.7 | 1.1 | 0.67 | 0.98 |

Reading: Hart does well on clean resistive steps (light) and worse on the two-speed fan. A likely cause is that Hart learns a single signature per appliance, so one value (about 72 W, between the 55 W and 80 W speeds) has to cover both speeds, and switch-ons far from it can go unidentified. That is what happens in one of the four test fan sessions (house 7, a 51 W speed-1 switch-on falls just outside the matching distance), so Hart's fan F1 is the mean of 0, 1 and 1. Hart also assigns every shower the same learned wattage, while the forest estimates each house's actual shower power. Part of Hart's TV and light error comes from the shower itself: its simulated per-second noise (standard deviation 1 % of its 4.5 to 6.5 kW, so 45 to 65 W) produces spurious steps (median about 50 W, close to the 60 W light) that this minimal detector, which has no steady-state check, labels mostly as the light. The forest's lower F1 on the light suggests it confuses the small 60 W step with other loads. These are three test houses of one day each, with few events per appliance (four fan sessions in total), so treat the differences as indicative.

## Run it

Requires Python 3.12 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # tests, including the physical invariant on the aggregate
python -m nilm_sim.report      # metrics on unseen homes -> reports/metrics.md
streamlit run app.py           # opens http://localhost:8501
```

The app trains its models once and saves them under `models/` (git-ignored). Later starts load the file; a change to the `nilm_sim` code, to `app.py` (where the training settings live) or to the Python or library versions changes the file name, so the models are trained again; a file that fails to load is trained again too.

## Roadmap

- Real data: UK-DALE / REFIT
- Seq2point CNN against both baselines
- Experiment tracking with MLflow and a Docker image
- Drift monitoring on the input distribution
