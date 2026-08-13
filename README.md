# Evil Twin Attack Detection in Non-Urban Terrain

Machine learning pipeline for detecting Evil Twin Wi-Fi attacks from RSSI readings collected by four ESP32 sensors, and for localizing an ongoing attack (which point and which rogue antenna). Companion code for the accompanying thesis/article (see `docs/`).

## Repository structure

```
data/
  rssi_raw.csv                    Raw labeled RSSI readings (long format)
docs/
  materials_and_methods.md        Methodology write-up (Ukrainian)
  materials_and_methods.docx      Same, as Word document
notebooks/
  01_eda_and_prototyping.ipynb    Data exploration + first pass at every model
  02_final_comparison.ipynb       Consolidated GridSearchCV benchmark (54 configs) + final results
  tuning/                         Per-algorithm hyperparameter tuning + diagnostics (9 algorithms x Base/FE)
```

### The three notebook layers

The notebooks form three layers that build on each other rather than duplicate one another:

1. **`01_eda_and_prototyping.ipynb`** — loads the raw CSV, visualizes the RSSI signal (baseline vs. attack windows), builds the synchronized `[s1, s2, s3, s4]` feature vector, and trains every model once at default/example hyperparameters. This is where the cascade task setup (Attack Detection → Point Prediction → Antenna Prediction) and the feature engineering idea are first introduced and explained.
2. **`tuning/*.ipynb`** — one notebook per algorithm (Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Decision Tree, Logistic Regression, SVM, KNN), each doing a full `GridSearchCV` sweep for that algorithm plus diagnostics unique to it (accuracy-vs-hyperparameter curves, confusion matrices, feature importance, and extras like the Decision Tree plot or the SVM C×γ heatmap). The `*_fe.ipynb` variant of each repeats the same tuning on the 13 feature-engineered columns instead of the original 4.
3. **`02_final_comparison.ipynb`** — re-runs the same 54 `GridSearchCV` configurations (9 algorithms × 2 feature sets × 3 tasks) in a single consolidated loop and produces the final cross-model comparison table, the Feature-Engineering-effect table, and the winner summary used in the paper's results section.

You don't strictly need to run all three layers to get the paper's headline numbers — `02_final_comparison.ipynb` alone reproduces the final comparison table. The EDA notebook and the per-algorithm tuning notebooks are where the detail and reasoning behind those numbers live.

## Setup

Tested with Python 3.9. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost jupyter
```

Approximate versions used during development: `pandas 1.4`, `numpy 1.21`, `scikit-learn 1.0`, `xgboost 2.1`, `lightgbm 4.6`, `catboost 1.2`.

## Data

`data/rssi_raw.csv` is long-format: one row per sensor reading.

| column     | meaning                                                        |
|------------|-----------------------------------------------------------------|
| `time`     | reading timestamp                                                |
| `deviceID` | which sensor took the reading (`sensor_1`…`sensor_4`)            |
| `value`    | RSSI in dBm                                                      |
| `point`    | ground-truth location label (0 = no attack, 1–8 = attack point)  |
| `attack`   | binary attack flag (0/1) — target for Attack Detection           |
| `antenna`  | which rogue antenna was active (0 = none, 1–6) — target for Antenna Prediction |

All notebooks resample and inner-join the four sensors onto a common 5-second grid to build the synchronized `[s1, s2, s3, s4]` vector used as model input (see Section 4 of `01_eda_and_prototyping.ipynb`).

## Running the notebooks

Run everything from inside `notebooks/` (or `notebooks/tuning/`) so the relative paths to `data/rssi_raw.csv` resolve correctly — every notebook reads it via `../data/rssi_raw.csv` or `../../data/rssi_raw.csv`.

Recommended order:

1. **`notebooks/01_eda_and_prototyping.ipynb`** — run this first. It explains the dataset, the resampling pipeline, the cascade task setup, and gives an untuned baseline for every model. Every other notebook refers back to it.
2. **`notebooks/tuning/*.ipynb`** — run in any order; each is self-contained (reloads the data, rebuilds the cascade splits, then tunes one algorithm). Runtime per notebook is a few minutes, dominated by the `GridSearchCV` cell. Run both the base (`catboost.ipynb`) and `_fe` (`catboost_fe.ipynb`) variant of a model if you want the Base-vs-FE comparison for that algorithm.
3. **`notebooks/02_final_comparison.ipynb`** — run last. Its "Main Search Loop" cell repeats all 54 tuning runs from step 2 in one place (several minutes) and produces the final comparison table, the FE-effect table, and the winner summary.

To run a notebook non-interactively from the command line, e.g.:

```bash
cd notebooks/tuning
jupyter nbconvert --to notebook --execute --inplace knn.ipynb
```
