# Non-Intrusive Load Monitoring (NILM) on UK-DALE

Predicting per-appliance on/off states (fridge, kettle) from whole-house
aggregate power readings, using the UK-DALE smart meter dataset.

## Why this matters

Non-Intrusive Load Monitoring lets you infer which appliances are running
in a home using only a single whole-house power meter — no per-appliance
hardware required. This has real applications in smart grid demand
forecasting, energy efficiency programs, and utility-side analytics.

## Dataset

- **Source:** [UK-DALE](https://jack-kelly.com/data/) (house_1)
- **Scale:** 22.6M+ aligned readings after preprocessing
- **Channels used:** whole-house aggregate, fridge, kettle, microwave, washing machine

## Pipeline

1. **Data loading & alignment** — Each appliance channel is logged
   independently on its own ~6-second timer, so raw timestamps don't
   line up across channels. Resampled every channel onto a shared 6-second
   grid, forward-filled gaps up to 30 seconds, and merged into a single
   aligned table (<3% missing values across all channels).

   - *Debugging note:* an early sanity-check plot showed the fridge
     channel as completely blank. Root cause: the fridge meter started
     recording over a month after the aggregate meter (Dec 15, 2012 vs.
     Nov 9, 2012), so the initial exploration window predated any fridge
     data. Fixed by computing the common overlapping period across all
     channels before selecting a plotting/analysis window.

2. **Feature engineering** — Built multiple time-scale features from the
   aggregate signal:
   - Rolling mean (30s, 1min, 5min windows)
   - Rolling standard deviation (5min)
   - First-order difference (rate of change)

3. **Labeling** — Converted raw appliance wattage into binary on/off
   labels using thresholds chosen from each appliance's actual wattage
   distribution (kettle: >500W; fridge: >10W), rather than arbitrary
   guesses.

4. **Train/test split** — Chronological split (not random), training on
   data through mid-2016 and testing on the most recent ~9 months, to
   avoid leaking future information into training.

5. **Modeling** — Trained and compared Random Forest and XGBoost
   classifiers for each appliance.

## Results

| Appliance | Model | Accuracy | F1 Score |
|---|---|---|---|
| Kettle | Random Forest | 99.64% | 0.754 |
| Fridge | Random Forest (max_depth=20) | 73.8% | 0.721 |
| Fridge | XGBoost | 73.0% | 0.718 |

**Key finding — feature iteration on kettle:** the first version of the
kettle model scored 99.3% accuracy but only 0.174 F1 — a classic accuracy
trap caused by class imbalance (kettle is "on" only ~0.6% of the time).
Diagnosed the issue as the 5-minute rolling window smoothing out kettle's
short (~2-3 minute) boiling spikes. Adding shorter 1-minute and 30-second
rolling windows raised recall from ~10% to ~76% of true kettle-on events
caught, taking F1 from 0.174 to 0.754.

**Key finding — model comparison:** XGBoost did not outperform Random
Forest on this feature set (0.718 vs. 0.721 F1 on fridge) — a useful,
honest result showing that a more sophisticated model isn't automatically
better without matching feature/hyperparameter work.

## Tech stack

Python, pandas, scikit-learn, XGBoost, matplotlib

## Repo structure

```
load_data.py            # Data loading, alignment, resampling, exploration plot
feature_engineering.py  # Rolling window features + on/off labels
train_test_split.py     # Chronological train/test split
train_model.py           # Random Forest + XGBoost training and evaluation
day1_exploration.png     # Sanity-check plot (aggregate/fridge/kettle)
```

## Possible extensions

- Multi-house generalization (train on one home, test on another)
- Sequence models (LSTM/1D-CNN) for windowed appliance signatures
- NILM-specific evaluation metrics (NDE, SAE)