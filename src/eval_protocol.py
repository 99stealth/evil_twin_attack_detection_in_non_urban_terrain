"""Session-aware evaluation protocol for the Evil Twin RSSI dataset.

Single source of truth for dataset construction, grouping, cross-validation
protocols, model configs, and the metrics used across the manuscript notebooks.

Background: the experiment consists of 48 attack *sessions* (8 points x 6
antennas), each ~55s long, separated by ~2min no-attack gaps. A row-shuffled
train/test split or a shuffled K-fold puts near-identical rows of the same
session into both train and test, so a model can recognise the session
instead of generalising (see README / manuscript response letter). All CV
protocols here are defined over groups derived from the raw, time-ordered
labels so that no session straddles a train/test boundary.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None

RANDOM_STATE = 42

ANTENNA_MAP = {
    "none": 0,
    "Alfa ARS-N05": 1,
    "Alfa ARS-N19": 2,
    "Alfa APA-M25": 3,
    "AOSIYANT Yagi 16dBi": 4,
    "Logperiodic": 5,
    "Alfa APA-M04": 6,
}
ANTENNA_NAMES = {v: k for k, v in ANTENNA_MAP.items()}
ANTENNA_SHORT = ["ARS-N05", "ARS-N19", "APA-M25", "Yagi", "Logper", "APA-M04"]

# Point layout: 8 points on an 11m-radius circle, 45 deg apart, numbered
# counter-clockwise from the east (see manuscript Fig. 1).
POINT_RADIUS_M = 11.0
N_POINTS = 8


def point_distance_m(i: int, j: int) -> float:
    """Chord distance in metres between points i and j (1..8) on the circle."""
    d = min(abs(i - j), N_POINTS - abs(i - j))
    theta = np.deg2rad(45.0 * d)
    return 2 * POINT_RADIUS_M * np.sin(theta / 2)


def point_xy_m(i: int) -> tuple[float, float]:
    """Cartesian (x, y) in metres of point i (1..8), AP at the origin, east=+x,
    north=+y, counter-clockwise from east."""
    theta = np.deg2rad(45.0 * (i - 1))
    return POINT_RADIUS_M * np.cos(theta), POINT_RADIUS_M * np.sin(theta)


# Task A2 (R2.1/R2.2) geometry. AP at the origin; sensors on the axes at
# SENSOR_DISTANCE_M from the AP (E, N, W, S).
#
# SENSOR_POSITIONS is inferred from data, not from the author's recollection
# (which could not be confirmed — see TASK_A2_fix / sensor_mapping_scores()).
# All 24 permutations of sensor_1..4 -> {E,N,W,S} were scored by the mean
# Pearson correlation between median attack RSSI and -log10(distance from the
# attack point to the sensor), averaged over the 48 (point, antenna)
# sessions. Best: s1=N, s2=S, s3=E, s4=W, score 0.479 (0.450 restricted to
# the two omnidirectional antennas). The author's recollection
# (s1=E, s2=W, s3=N, s4=S) scored -0.201; the opposite pairs (1-2, 3-4) match
# that recollection, so the axes appear rotated 90 deg relative to it.
SENSOR_DISTANCE_M = 6.0
SENSOR_POSITIONS = {
    "s1": (0.0, SENSOR_DISTANCE_M),    # North
    "s2": (0.0, -SENSOR_DISTANCE_M),   # South
    "s3": (SENSOR_DISTANCE_M, 0.0),    # East
    "s4": (-SENSOR_DISTANCE_M, 0.0),   # West
}
# Inferred from data (best of 24 permutations, mean corr(RSSI, -log10 d) = 0.479);
# the author's recollection of the mapping could not be confirmed. See sensor_mapping_scores().
SENSOR_POSITIONS_CONFIRMED = False
SENSOR_POSITIONS_SOURCE = "inferred_from_data"

# Expected chord error of a uniformly random point on the ring (R2.1 reference line).
RANDOM_GUESS_ERROR_M = 4 * POINT_RADIUS_M / np.pi


def diagnostic_sensor_direction_rssi(dataset: pd.DataFrame) -> pd.DataFrame:
    """Median attack-RSSI per point per sensor (5s-resampled dataset) — use
    this table to confirm/correct SENSOR_POSITIONS. At an axis point (1=E,
    3=N, 5=W, 7=S) the co-located sensor (5m away) should read clearly
    strongest, the opposite sensor (17m away) clearly weakest, and the two
    perpendicular sensors (12.53m away) roughly tied in between."""
    return dataset[dataset["attack"] == 1].groupby("point")[FEATURES_BASE].median()


_DIRECTION_XY = {
    "N": (0.0, SENSOR_DISTANCE_M),
    "S": (0.0, -SENSOR_DISTANCE_M),
    "E": (SENSOR_DISTANCE_M, 0.0),
    "W": (-SENSOR_DISTANCE_M, 0.0),
}


def sensor_mapping_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Score all 24 permutations of sensor_1..4 -> {E, N, W, S} by how well
    they explain the observed attack RSSI, to infer SENSOR_POSITIONS from
    data instead of relying on an unconfirmed recollection.

    For each (point, antenna) attack session (48 total) and each candidate
    permutation: take the median RSSI per sensor over that session's *raw*
    scans, and compute the Pearson correlation, across the 4 sensors,
    between that median RSSI and -log10(distance from the attack point to
    the candidate sensor position) — the sign log-distance path loss
    predicts (closer -> stronger RSSI -> larger -log10(d)). `score_all` is
    the mean of that per-session correlation over all 48 sessions;
    `score_omni` restricts it to the two omnidirectional antennas (1, 2:
    ARS-N05, ARS-N19), where a directional antenna's radiation pattern can't
    distort the distance relationship.

    Returns a DataFrame with columns `mapping` (the directions of s1..s4,
    e.g. "NSEW"), `score_all`, `score_omni`, sorted by `score_all` descending.
    """
    grouped = add_groups(raw)
    attack = grouped[grouped["attack"] == 1]
    device_names = sorted(raw["deviceID"].unique())
    sensor_cols = [f"s{i}" for i in range(1, len(device_names) + 1)]
    dev_to_col = {dev: col for dev, col in zip(device_names, sensor_cols)}

    med = attack.groupby(["point", "antenna", "deviceID"])["value"].median().reset_index()
    med["sensor"] = med["deviceID"].map(dev_to_col)
    wide = med.pivot(index=["point", "antenna"], columns="sensor", values="value").reset_index()
    wide = wide.dropna(subset=sensor_cols)

    rows = []
    for perm in itertools.permutations("NSEW"):
        mapping_str = "".join(perm)
        sensor_xy = {col: np.array(_DIRECTION_XY[d]) for col, d in zip(sensor_cols, perm)}

        corrs_all, corrs_omni = [], []
        for _, session_row in wide.iterrows():
            point_xy = np.array(point_xy_m(int(session_row["point"])))
            dists = np.array([np.linalg.norm(point_xy - sensor_xy[c]) for c in sensor_cols])
            rssi = session_row[sensor_cols].to_numpy(dtype=float)
            neg_log_d = -np.log10(dists)
            if np.std(rssi) == 0 or np.std(neg_log_d) == 0:
                continue
            r = float(np.corrcoef(rssi, neg_log_d)[0, 1])
            corrs_all.append(r)
            if int(session_row["antenna"]) in (1, 2):
                corrs_omni.append(r)

        rows.append({
            "mapping": mapping_str,
            "score_all": float(np.mean(corrs_all)) if corrs_all else float("nan"),
            "score_omni": float(np.mean(corrs_omni)) if corrs_omni else float("nan"),
        })

    return pd.DataFrame(rows).sort_values("score_all", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 1. Data loading & grouping
# --------------------------------------------------------------------------

def _find_repo_root(marker: str = "data/rssi_raw.csv") -> Path:
    p = Path.cwd().resolve()
    for candidate in [p, *p.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate {marker} from {p}")


def default_dataset_path() -> Path:
    return _find_repo_root() / "data" / "rssi_raw.csv"


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw long-format RSSI CSV.

    Uses format="ISO8601" explicitly: with pandas >= 2, read_csv(parse_dates=...)
    silently leaves the column as strings when fractional-second precision is
    mixed across rows (some timestamps have milliseconds, some don't).
    """
    if path is None:
        path = default_dataset_path()
    raw = pd.read_csv(path)
    raw["time"] = pd.to_datetime(raw["time"], format="ISO8601")
    return raw.sort_values("time").reset_index(drop=True)


def add_groups(raw: pd.DataFrame) -> pd.DataFrame:
    """Attach `session` and `block` group ids to a time-ordered raw dataframe.

    session: run-length id over (attack, point, antenna) in time order.
      48 attack sessions + 49 no-attack gaps = 97 segments.
    block: the antenna "block" a row belongs to. For an attack session this
      is just its antenna. Each no-attack gap is assigned to the *next*
      attack session's antenna (bfill), since the experiment ran one
      antenna block at a time; the trailing gap after the very last
      session (no next session) is assigned to the last block (6).
    """
    raw = raw.sort_values("time").reset_index(drop=True).copy()
    label_cols = ["attack", "point", "antenna"]
    changed = (raw[label_cols] != raw[label_cols].shift()).any(axis=1)
    raw["session"] = changed.cumsum()

    session_antenna = (
        raw.loc[raw["attack"] == 1]
        .groupby("session")["antenna"]
        .first()
    )
    all_sessions = pd.Series(index=sorted(raw["session"].unique()), dtype="float64")
    all_sessions.loc[session_antenna.index] = session_antenna.values
    all_sessions = all_sessions.bfill()
    all_sessions = all_sessions.ffill()  # trailing gap (no next session) -> last block
    raw["block"] = raw["session"].map(all_sessions).astype(int)
    return raw


# --------------------------------------------------------------------------
# 2. Dataset construction
# --------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    freq: str
    n_rows: int
    n_attack_rows: int
    n_sessions_attack: int
    imputed_share: dict
    rows_per_session: pd.Series
    impute: str = "nearest"


def build_dataset(
    raw: pd.DataFrame, freq: str = "5s", impute: str = "nearest"
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Resample each sensor to a common grid, inner-join, attach labels+groups.

    Pipeline (unchanged from the original notebooks): resample(freq).median()
    -> interpolate(nearest) -> inner join across sensors -> merge_asof the
    per-reading labels and groups onto the resulting grid (direction="nearest").

    impute="nearest" (default): fill grid points with no direct reading via
    nearest-value interpolation (the main-task pipeline).
    impute="none": leave gaps as NaN; the inner join's dropna() then keeps
    only grid points where all sensors had a direct reading in that bucket
    (V7 sensitivity check — no imputed rows).
    """
    if impute not in ("nearest", "none"):
        raise ValueError(f"Unknown impute mode: {impute!r}")
    grouped = add_groups(raw)

    sensors = {}
    imputed_share = {}
    for dev in sorted(raw["deviceID"].unique()):
        s = raw.loc[raw["deviceID"] == dev, ["time", "value"]].copy()
        s = s.set_index("time").sort_index()
        resampled = s.resample(freq).median()
        was_missing = resampled["value"].isna()
        if impute == "nearest":
            resampled = resampled.interpolate(method="nearest")
        sensors[dev] = resampled.rename(columns={"value": dev})
        # share of grid points that had no direct reading and were imputed,
        # among the grid points that end up with a value at all
        has_value = resampled["value"].notna()
        imputed_share[dev] = float((was_missing & has_value).sum() / max(has_value.sum(), 1))

    device_names = sorted(raw["deviceID"].unique())
    sensor_cols = [f"s{i}" for i in range(1, len(device_names) + 1)]
    rename_map = {dev: col for dev, col in zip(device_names, sensor_cols)}

    df_pivot = None
    for dev in device_names:
        s = sensors[dev].rename(columns=rename_map)
        df_pivot = s if df_pivot is None else df_pivot.join(s, how="inner")
    df_pivot = df_pivot.dropna().reset_index()

    labels = (
        grouped[["time", "point", "attack", "antenna", "session", "block"]]
        .drop_duplicates("time")
        .sort_values("time")
    )
    dataset = pd.merge_asof(
        df_pivot.sort_values("time"), labels, on="time", direction="nearest"
    )
    dataset = dataset[["time"] + sensor_cols + ["point", "attack", "antenna", "session", "block"]]

    rows_per_session = dataset.groupby("session").size()
    info = DatasetInfo(
        freq=freq,
        n_rows=len(dataset),
        n_attack_rows=int((dataset["attack"] == 1).sum()),
        n_sessions_attack=int(dataset.loc[dataset["attack"] == 1, "session"].nunique()),
        imputed_share={rename_map[k]: v for k, v in imputed_share.items()},
        rows_per_session=rows_per_session,
        impute=impute,
    )
    return dataset, info


def build_dataset_scan_cycle(
    raw: pd.DataFrame, tolerance_seconds: float = 3.6
) -> tuple[pd.DataFrame, DatasetInfo]:
    """V7 option (a): align on the anchor sensor's own scan cycle instead of
    a resampling grid, with no imputation.

    For each scan of the first sensor (by device name), take the nearest scan
    of every other sensor within +/- tolerance_seconds; the row is dropped if
    any sensor has no scan in that window (`merge_asof(..., tolerance=...)`
    leaves those as NaN, then `dropna()`).
    """
    grouped = add_groups(raw)
    device_names = sorted(raw["deviceID"].unique())
    sensor_cols = [f"s{i}" for i in range(1, len(device_names) + 1)]
    tol = pd.Timedelta(seconds=tolerance_seconds)

    anchor_dev = device_names[0]
    merged = (
        raw.loc[raw["deviceID"] == anchor_dev, ["time", "value"]]
        .sort_values("time")
        .rename(columns={"value": sensor_cols[0]})
    )
    for dev, col in zip(device_names[1:], sensor_cols[1:]):
        other = (
            raw.loc[raw["deviceID"] == dev, ["time", "value"]]
            .sort_values("time")
            .rename(columns={"value": col})
        )
        merged = pd.merge_asof(merged.sort_values("time"), other, on="time", direction="nearest", tolerance=tol)
    merged = merged.dropna().reset_index(drop=True)

    labels = (
        grouped[["time", "point", "attack", "antenna", "session", "block"]]
        .drop_duplicates("time")
        .sort_values("time")
    )
    dataset = pd.merge_asof(merged.sort_values("time"), labels, on="time", direction="nearest")
    dataset = dataset[["time"] + sensor_cols + ["point", "attack", "antenna", "session", "block"]]

    rows_per_session = dataset.groupby("session").size()
    info = DatasetInfo(
        freq=f"scan_cycle(anchor={anchor_dev}, tol={tolerance_seconds}s)",
        n_rows=len(dataset),
        n_attack_rows=int((dataset["attack"] == 1).sum()),
        n_sessions_attack=int(dataset.loc[dataset["attack"] == 1, "session"].nunique()),
        imputed_share={c: 0.0 for c in sensor_cols},
        rows_per_session=rows_per_session,
        impute="scan_cycle",
    )
    return dataset, info


def add_fe(df: pd.DataFrame, sensor_cols: list[str] | None = None) -> pd.DataFrame:
    """Add pairwise deltas + row-wise aggregates computed from a single snapshot."""
    df = df.copy()
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
        sensor_cols = sorted(sensor_cols, key=lambda c: int(c[1:]))
    for idx_a in range(len(sensor_cols)):
        for idx_b in range(idx_a + 1, len(sensor_cols)):
            a, b = sensor_cols[idx_a], sensor_cols[idx_b]
            df[f"d{a[1:]}{b[1:]}"] = df[a] - df[b]
    df["mean_rssi"] = df[sensor_cols].mean(axis=1)
    df["std_rssi"] = df[sensor_cols].std(axis=1)
    df["range_rssi"] = df[sensor_cols].max(axis=1) - df[sensor_cols].min(axis=1)
    return df


FEATURES_BASE = ["s1", "s2", "s3", "s4"]
FEATURES_FE = FEATURES_BASE + [
    "d12", "d13", "d14", "d23", "d24", "d34", "mean_rssi", "std_rssi", "range_rssi",
]


# --------------------------------------------------------------------------
# 3. Tasks & protocols
# --------------------------------------------------------------------------

TASKS = {
    "attack": {
        "label": "Attack Detection",
        "rows": lambda df: pd.Series(True, index=df.index),
        "y_col": "attack",
        "default_group": "block",
    },
    "point": {
        "label": "Point Prediction",
        "rows": lambda df: df["attack"] == 1,
        "y_col": "point",
        "default_group": "antenna",
    },
    "antenna": {
        "label": "Antenna Prediction",
        "rows": lambda df: df["attack"] == 1,
        "y_col": "antenna",
        "default_group": "point",
    },
}

TASKS_ORDER = ["attack", "point", "antenna"]


def _make_outer_cv(name: str, groups: pd.Series):
    """Build an outer-CV splitter object for a named protocol, given the
    group series that will be used with it (only its cardinality matters
    for GroupKFold's n_splits clamp)."""
    if name == "shuffled":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if name == "legacy_unshuffled":
        return StratifiedKFold(n_splits=5, shuffle=False)
    if name in ("primary", "loso_groupwise"):
        return LeaveOneGroupOut()
    if name == "loso_session_gkfold":
        n_splits = min(10, groups.nunique())
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unknown protocol: {name}")


@dataclass
class Protocol:
    name: str
    group_col_by_task: dict  # task -> column name to use as groups (or None -> no grouping)
    needs_groups: bool


PROTOCOLS = {
    # primary: LeaveOneGroupOut with the task's default group column
    "primary": Protocol(
        name="primary",
        group_col_by_task={"attack": "block", "point": "antenna", "antenna": "point"},
        needs_groups=True,
    ),
    # supplementary: leave-one-session-out (point/antenna) / GroupKFold(10, session) (attack)
    "loso": Protocol(
        name="loso",
        group_col_by_task={"attack": "session", "point": "session", "antenna": "session"},
        needs_groups=True,
    ),
    # legacy Table 1 protocol: StratifiedKFold(5, shuffle=False) on time-ordered rows
    "legacy_unshuffled": Protocol(
        name="legacy_unshuffled",
        group_col_by_task={"attack": None, "point": None, "antenna": None},
        needs_groups=False,
    ),
    # explicit-leak baseline: StratifiedKFold(5, shuffle=True)
    "shuffled": Protocol(
        name="shuffled",
        group_col_by_task={"attack": None, "point": None, "antenna": None},
        needs_groups=False,
    ),
}


def get_outer_cv_and_groups(protocol_name: str, task: str, df_task: pd.DataFrame):
    """Return (splitter, groups_array_or_None) for a given protocol+task,
    reading the group column off df_task (already row-filtered for the task)."""
    proto = PROTOCOLS[protocol_name]
    group_col = proto.group_col_by_task[task]
    groups = df_task[group_col].to_numpy() if group_col is not None else None

    if protocol_name == "shuffled":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    elif protocol_name == "legacy_unshuffled":
        splitter = StratifiedKFold(n_splits=5, shuffle=False)
    elif protocol_name == "primary":
        splitter = LeaveOneGroupOut()
    elif protocol_name == "loso":
        if task == "attack":
            n_splits = min(10, df_task[group_col].nunique())
            splitter = GroupKFold(n_splits=n_splits)
        else:
            splitter = LeaveOneGroupOut()
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")
    return splitter, groups


# --------------------------------------------------------------------------
# 4. Model configs
# --------------------------------------------------------------------------

class _CatBoostFlat(CatBoostClassifier if CatBoostClassifier is not None else object):
    """CatBoostClassifier.predict() returns shape (n, 1); flatten it so it's
    consistent with every other sklearn classifier's predict()."""

    def predict(self, X, **kwargs):
        return np.asarray(super().predict(X, **kwargs)).ravel()


def _make_model_configs() -> dict:
    cfgs = {
        "Logistic Regression": {
            "make": lambda: LogisticRegression(solver="saga", max_iter=5000, random_state=RANDOM_STATE),
            "grid": {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty": ["l1", "l2"], "class_weight": [None, "balanced"]},
            "scale": True,
        },
        "SVM (RBF)": {
            "make": lambda: SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
            "grid": {"C": [0.01, 0.1, 1, 10, 100, 1000], "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10]},
            "scale": True,
        },
        "KNN": {
            "make": lambda: KNeighborsClassifier(),
            "grid": {"n_neighbors": list(range(1, 26)), "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan", "chebyshev"]},
            "scale": True,
        },
        "Decision Tree": {
            "make": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
            "grid": {"max_depth": [3, 5, 8, 10, 15, None], "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 2, 5, 10], "criterion": ["gini", "entropy"]},
            "scale": False,
        },
        "Random Forest": {
            "make": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"]},
            "scale": False,
        },
        "Extra Trees": {
            "make": lambda: ExtraTreesClassifier(random_state=RANDOM_STATE),
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"]},
            "scale": False,
        },
        "XGBoost": {
            "make": "xgboost",  # handled specially: needs 0-based labels for multiclass
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3], "subsample": [0.8, 1.0]},
            "scale": False,
        },
        "LightGBM": {
            # n_jobs=1: GridSearchCV(n_jobs=-1) already parallelizes across CV folds/
            # hyperparameter combos at the process level: LightGBM's own default
            # (use all cores) would oversubscribe 10 worker processes x 10 threads
            # each on a 10-core machine.
            "make": lambda: LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1),
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [-1, 3, 5, 7], "learning_rate": [0.01, 0.1, 0.3], "num_leaves": [15, 31, 63]},
            "scale": False,
        },
        "CatBoost": {
            "make": lambda: _CatBoostFlat(random_state=RANDOM_STATE, verbose=False, allow_writing_files=False, thread_count=1),
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7, 9], "learning_rate": [0.01, 0.1, 0.3], "l2_leaf_reg": [1, 3, 10]},
            "scale": False,
        },
        "MLP": {
            "make": lambda: MLPClassifier(random_state=RANDOM_STATE, early_stopping=True, max_iter=2000),
            "grid": {
                "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-3, 1e-2],
            },
            "scale": True,
        },
    }
    return cfgs


MODEL_CONFIGS = _make_model_configs()


class _XGBWrapper(BaseEstimator, ClassifierMixin):
    """0-based-label wrapper for XGBoost multiclass, transparent for binary.

    Shifting labels by a constant offset before fit and shifting predictions
    back after predict is a bijective relabeling; it does not change accuracy
    or any label-permutation-invariant metric.

    Follows the sklearn estimator convention (params stored unmodified in
    __init__, matching XGBoost's grid keys exactly) so GridSearchCV/clone
    work without a custom get_params/set_params.
    """

    def __init__(self, is_multiclass=False, n_estimators=100, max_depth=3,
                 learning_rate=0.1, subsample=1.0):
        self.is_multiclass = is_multiclass
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample

    def fit(self, X, y):
        y = np.asarray(y)
        self._offset = y.min() if self.is_multiclass else 0
        eval_metric = "mlogloss" if self.is_multiclass else "logloss"
        self.model_ = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric=eval_metric, n_jobs=1,
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
        )
        self.model_.fit(X, y - self._offset)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model_.predict(X) + self._offset

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        return self.model_.predict_proba(X)


def make_estimator(model_name: str, is_multiclass: bool, params: dict | None = None):
    """Build a fresh (unfitted) estimator/pipeline for a model config.

    `params` is expected in the same key format as `best_params_` from a grid
    search over `_search_space(model_name)` — i.e. "model__"-prefixed for
    scaled models, flat for everything else — so a fold's winning
    hyperparameters can be replayed directly.
    """
    cfg = MODEL_CONFIGS[model_name]
    params = dict(params or {})
    if model_name == "XGBoost":
        return _XGBWrapper(is_multiclass=is_multiclass, **params)

    base = cfg["make"]()
    if cfg["scale"]:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", base)])
        if params:
            pipe.set_params(**params)
        return pipe
    if params:
        base.set_params(**params)
    return base


def _search_space(model_name: str) -> dict:
    cfg = MODEL_CONFIGS[model_name]
    if cfg["scale"]:
        return {f"model__{k}": v for k, v in cfg["grid"].items()}
    return cfg["grid"]


# --------------------------------------------------------------------------
# 5. Nested CV
# --------------------------------------------------------------------------

# Grids expensive enough that nested CV with full GridSearchCV is impractical;
# use RandomizedSearchCV instead (documented per section 2.3 of the task).
RANDOMIZED_SEARCH_MODELS = {"XGBoost", "LightGBM", "CatBoost"}
RANDOMIZED_SEARCH_N_ITER = 30


def _assert_no_group_overlap(groups, train_idx, test_idx, context: str):
    g = np.asarray(groups)
    train_groups = set(g[train_idx])
    test_groups = set(g[test_idx])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"Group overlap between train/test in {context}: {overlap}")


def nested_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    outer_cv,
    inner_cv_factory: Callable[[np.ndarray], object],
    is_multiclass: bool,
) -> dict:
    """Nested CV: GridSearchCV/RandomizedSearchCV inside each outer fold.

    Returns a dict with pooled out-of-fold predictions (built by concatenating
    each fold's held-out predictions, so summing per-fold confusion matrices
    reproduces the pooled accuracy exactly), per-fold scores, and the winning
    hyperparameters of every outer fold.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    groups = np.asarray(groups)

    n = len(y)
    oof_pred = np.full(n, np.nan, dtype=object)
    oof_proba = [None] * n
    fold_scores = []
    best_params_per_fold = []
    needs_proba = hasattr(make_estimator(model_name, is_multiclass), "predict_proba") or model_name == "SVM (RBF)"

    space = _search_space(model_name)
    use_randomized = model_name in RANDOMIZED_SEARCH_MODELS

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        _assert_no_group_overlap(groups, train_idx, test_idx, f"{model_name} outer fold {fold_i}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train = groups[train_idx]

        inner_cv = inner_cv_factory(g_train)
        estimator = make_estimator(model_name, is_multiclass)

        # Validate inner splits too (guards inner_cv_factory implementations).
        for in_tr, in_te in inner_cv.split(X_train, y_train, g_train):
            _assert_no_group_overlap(g_train, in_tr, in_te, f"{model_name} outer fold {fold_i} inner split")

        search_cls = RandomizedSearchCV if use_randomized else GridSearchCV
        search_kwargs = dict(cv=inner_cv, scoring="accuracy", n_jobs=-1)
        if use_randomized:
            search_kwargs["n_iter"] = RANDOMIZED_SEARCH_N_ITER
            search_kwargs["random_state"] = RANDOM_STATE
        search = search_cls(estimator, space, **search_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train, groups=g_train)

        best = search.best_estimator_
        # CatBoost predict() returns shape (n, 1) instead of (n,)
        pred = np.asarray(best.predict(X_test)).ravel()
        oof_pred[test_idx] = pred
        fold_scores.append(float((pred == y_test.to_numpy()).mean()))
        best_params_per_fold.append(search.best_params_)

        if needs_proba:
            try:
                proba = best.predict_proba(X_test)
            except AttributeError:
                proba = best.decision_function(X_test)
            for local_i, global_i in enumerate(test_idx):
                oof_proba[global_i] = proba[local_i]

    oof_pred = oof_pred.astype(y.dtype if y.dtype != object else int)
    return {
        "oof_true": y.to_numpy(),
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "fold_scores": fold_scores,
        "best_params_per_fold": best_params_per_fold,
        "groups": groups,
    }


def group_kfold_inner_factory(n_splits_cap: int = 5):
    def factory(g_train: np.ndarray):
        n_inner_groups = len(np.unique(g_train))
        return GroupKFold(n_splits=min(n_splits_cap, n_inner_groups))
    return factory


# --------------------------------------------------------------------------
# 6. Metrics & plotting
# --------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, task: str, y_score=None) -> dict:
    """Core metric set for one true/pred array pair (no fold structure).

    Shared by `summarize` (OOF predictions from nested_cv) and any single
    train/test split (e.g. the V1 locked hold-out), so both report the same
    metric definitions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pooled_acc = float((y_true == y_pred).mean())

    out = {
        "pooled_acc": pooled_acc,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    classes = sorted(np.unique(y_true))
    recalls = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    out["per_class_recall"] = dict(zip(classes, recalls.tolist()))

    if task == "attack":
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
        if y_score is not None:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            except ValueError:
                out["roc_auc"] = float("nan")

    if task == "point":
        errors = [point_distance_m(int(t), int(p)) for t, p in zip(y_true, y_pred)]
        out["mean_localization_error_m"] = float(np.mean(errors))

    return out


def summarize(result: dict, task: str) -> dict:
    y_true = result["oof_true"]
    y_pred = result["oof_pred"]
    fold_scores = np.array(result["fold_scores"])

    y_score = None
    if task == "attack":
        proba = result.get("oof_proba")
        if proba is not None and all(p is not None for p in proba):
            proba_arr = np.array(proba)
            y_score = proba_arr[:, 1] if proba_arr.ndim == 2 and proba_arr.shape[1] == 2 else proba_arr.ravel()

    out = classification_metrics(y_true, y_pred, task, y_score=y_score)
    out["acc_mean"] = float(fold_scores.mean())
    out["acc_std"] = float(fold_scores.std())
    return out


def confusion_matrix_from_oof(result: dict, labels=None):
    return confusion_matrix(result["oof_true"], result["oof_pred"], labels=labels)


def most_frequent_params(best_params_per_fold: list[dict]) -> dict:
    """Mode of the per-outer-fold best_params_, ties broken by first occurrence."""
    keyed = [tuple(sorted(p.items())) for p in best_params_per_fold]
    counts: dict = {}
    order: list = []
    for k in keyed:
        if k not in counts:
            order.append(k)
        counts[k] = counts.get(k, 0) + 1
    best_key = max(order, key=lambda k: counts[k])
    return dict(best_key)


def permutation_importance_oof(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    outer_cv,
    best_params_per_fold: list[dict],
    is_multiclass: bool,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Permutation importance averaged over outer test folds.

    Re-splits with the same outer_cv/groups used by nested_cv (deterministic,
    so fold order matches best_params_per_fold), refits each fold's winning
    config on that fold's train data, and averages sklearn's
    permutation_importance() on the held-out test fold across folds.
    """
    from sklearn.inspection import permutation_importance

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    groups = np.asarray(groups)

    importances = []
    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        params = best_params_per_fold[fold_i]
        est = make_estimator(model_name, is_multiclass, params=params)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            est, X.iloc[test_idx], y.iloc[test_idx],
            n_repeats=n_repeats, random_state=random_state, scoring="accuracy",
        )
        importances.append(result.importances_mean)

    avg = np.mean(importances, axis=0)
    return pd.Series(avg, index=X.columns)


# --------------------------------------------------------------------------
# 7. Validation & robustness (task A1: locked hold-out, forward-chaining,
#    dependence structure, cluster bootstrap, session/window aggregation,
#    session-level permutation tests)
# --------------------------------------------------------------------------

def attack_session_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per attack session: point, antenna, block, start/end time, n raw rows.

    `gap_session` is the id of the attack-free session immediately preceding
    it (session id - 1): the experiment alternates gap/attack segments, so
    this is always the gap that was recorded right before that attack run.
    """
    grouped = add_groups(raw)
    tbl = (
        grouped[grouped["attack"] == 1]
        .groupby("session")
        .agg(point=("point", "first"), antenna=("antenna", "first"), block=("block", "first"),
             start=("time", "min"), end=("time", "max"), n_raw_rows=("time", "size"))
        .reset_index()
    )
    tbl["gap_session"] = tbl["session"] - 1
    return tbl


def select_balanced_holdout_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    """V1b: one attack session per point, antenna assigned by a fixed Latin
    rule (antenna = ((point-1) mod 6) + 1), so every point appears once and
    every antenna at least once among the 8 held-out sessions."""
    session_tbl = attack_session_table(raw)
    rows = []
    for point in range(1, N_POINTS + 1):
        antenna = ((point - 1) % 6) + 1
        match = session_tbl[(session_tbl["point"] == point) & (session_tbl["antenna"] == antenna)]
        if len(match) != 1:
            raise AssertionError(f"Expected exactly one session for point={point}, antenna={antenna}, got {len(match)}")
        row = match.iloc[0]
        rows.append({
            "point": point, "antenna": antenna,
            "session": int(row["session"]), "gap_session": int(row["gap_session"]),
        })
    return pd.DataFrame(rows)


def assert_no_session_overlap(sessions_a, sessions_b, context: str) -> None:
    overlap = set(np.asarray(sessions_a).tolist()) & set(np.asarray(sessions_b).tolist())
    if overlap:
        raise AssertionError(f"Session overlap between dev/test in {context}: {sorted(overlap)}")


def locked_holdout_eval(
    model_name: str,
    X_dev, y_dev, groups_dev, sessions_dev,
    X_test, y_test, sessions_test,
    is_multiclass: bool,
    task: str,
    inner_cv_factory=None,
) -> dict:
    """V1: select hyperparameters with nested CV on the dev set only, refit
    on all of dev, evaluate exactly once on the locked test set.

    Asserts that no session appears on both sides (acceptance check #1) and
    that every inner-CV split used for model selection is itself group-safe.
    """
    assert_no_session_overlap(sessions_dev, sessions_test, f"locked holdout ({model_name})")

    inner_cv_factory = inner_cv_factory or group_kfold_inner_factory(5)
    inner_cv = inner_cv_factory(np.asarray(groups_dev))
    for train_idx, val_idx in inner_cv.split(X_dev, y_dev, groups_dev):
        _assert_no_group_overlap(groups_dev, train_idx, val_idx, f"locked holdout inner CV ({model_name})")

    space = _search_space(model_name)
    use_randomized = model_name in RANDOMIZED_SEARCH_MODELS
    estimator = make_estimator(model_name, is_multiclass)
    search_cls = RandomizedSearchCV if use_randomized else GridSearchCV
    search_kwargs = dict(cv=inner_cv, scoring="accuracy", n_jobs=-1)
    if use_randomized:
        search_kwargs.update(n_iter=RANDOMIZED_SEARCH_N_ITER, random_state=RANDOM_STATE)
    search = search_cls(estimator, space, **search_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_dev, y_dev, groups=groups_dev)

    best = search.best_estimator_  # refit=True (default) -> already refit on all of dev
    y_pred = np.asarray(best.predict(X_test)).ravel()

    y_score = None
    if task == "attack":
        try:
            y_score = best.predict_proba(X_test)[:, 1]
        except AttributeError:
            try:
                y_score = best.decision_function(X_test)
            except AttributeError:
                y_score = None

    metrics = classification_metrics(y_test, y_pred, task, y_score=y_score)
    metrics.update({
        "best_params": search.best_params_,
        "n_test_rows": int(len(np.asarray(y_test))),
        "n_test_sessions": int(len(np.unique(sessions_test))),
        "y_pred": y_pred,
    })
    return metrics


def select_best_model_on_dev(
    task: str,
    X_dev, y_dev, groups_dev,
    is_multiclass: bool,
    model_names: list[str] | None = None,
    inner_cv_factory=None,
) -> tuple[str, dict]:
    """V1 (F1 fix): rank models by nested-CV pooled accuracy computed
    *entirely* on the dev split, so the model choice for a locked hold-out
    design cannot see the held-out test data.

    This function's signature has no X_test/y_test/sessions_test parameter at
    all, so there is nothing to leak through by construction. The runtime
    guarantee that the held-out sessions are never touched lives in
    `locked_holdout_eval` (called separately, per model, to actually score on
    the locked test set), which asserts `assert_no_session_overlap` before
    doing anything else — that is the "assertion added to locked_holdout_eval"
    referenced in the F1 acceptance check.

    Returns (best_model_name, {model_name: dev_pooled_acc}).
    """
    model_names = model_names or list(MODEL_CONFIGS.keys())
    inner_cv_factory = inner_cv_factory or group_kfold_inner_factory(5)
    outer_cv = LeaveOneGroupOut()

    dev_scores = {}
    for model_name in model_names:
        res = nested_cv(model_name, X_dev, y_dev, groups_dev, outer_cv, inner_cv_factory, is_multiclass=is_multiclass)
        dev_scores[model_name] = summarize(res, task)["pooled_acc"]

    best_model = max(dev_scores, key=dev_scores.get)
    return best_model, dev_scores


def mode_best_params(best_params_df: pd.DataFrame, model: str, task_label: str, feature_set: str) -> dict:
    """Most frequent best_params_ for (model, task, feature set) from a
    `results/best_params.csv`-shaped DataFrame produced by 02_final_comparison
    (its "Best Params" column holds a stringified dict per outer fold)."""
    import ast

    sub = best_params_df[
        (best_params_df["Model"] == model)
        & (best_params_df["Task"] == task_label)
        & (best_params_df["Feature Set"] == feature_set)
    ]
    parsed = [ast.literal_eval(s) if isinstance(s, str) else s for s in sub["Best Params"]]
    if not parsed:
        raise ValueError(f"No best_params rows found for {model}/{task_label}/{feature_set}")
    return most_frequent_params(parsed)


# --- V3: within-session dependence -----------------------------------------

def lag1_autocorr_by_session(df: pd.DataFrame, value_col: str, session_col: str = "session",
                              time_col: str = "time") -> float:
    """Median (over sessions) lag-1 autocorrelation of `value_col` within each
    session, time-ordered."""
    corrs = []
    for _, g in df.sort_values(time_col).groupby(session_col):
        v = g[value_col].to_numpy(dtype=float)
        if len(v) > 2 and np.std(v[:-1]) > 0 and np.std(v[1:]) > 0:
            c = np.corrcoef(v[:-1], v[1:])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
    return float(np.median(corrs)) if corrs else float("nan")


def icc1(df: pd.DataFrame, value_col: str, group_col: str = "session") -> float:
    """One-way random-effects ICC(1): share of variance in `value_col`
    explained by `group_col` (session) membership, with the standard
    unbalanced-design k0 adjustment (Shrout & Fleiss 1979 / McGraw & Wong 1996)."""
    g = df.groupby(group_col)[value_col]
    k = g.size()
    n_groups = len(k)
    N = int(k.sum())
    if n_groups < 2 or N <= n_groups:
        return float("nan")
    grand_mean = df[value_col].mean()
    group_means = g.mean()
    ssb = float((k * (group_means - grand_mean) ** 2).sum())
    ssw = float(g.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum())
    dfb = n_groups - 1
    dfw = N - n_groups
    msb = ssb / dfb
    msw = ssw / dfw if dfw > 0 else float("nan")
    k0 = (N - (k ** 2).sum() / N) / dfb
    denom = msb + (k0 - 1) * msw
    return float((msb - msw) / denom) if denom else float("nan")


# --- V4: cluster (session-level) bootstrap ----------------------------------

def cluster_bootstrap_ci(y_true, y_pred, cluster_ids, metric_fn=None,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI for a metric by resampling clusters (sessions/segments) with
    replacement, never rows — acceptance check #2."""
    if metric_fn is None:
        metric_fn = lambda yt, yp: float((yt == yp).mean())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cluster_ids = np.asarray(cluster_ids)
    clusters = np.unique(cluster_ids)
    n = len(clusters)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in clusters}

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(clusters, size=n, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        stats[i] = metric_fn(y_true[idx], y_pred[idx])

    return {
        "ci_low": float(np.nanpercentile(stats, 2.5)),
        "ci_high": float(np.nanpercentile(stats, 97.5)),
        "boot_mean": float(np.nanmean(stats)),
        "boot_std": float(np.nanstd(stats)),
    }


# --- V5: session-level results & sliding-window aggregation ----------------

def majority_vote(labels: np.ndarray):
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)]


def session_level_accuracy(y_true, y_pred, sessions) -> tuple[float, pd.DataFrame]:
    """Majority-vote-per-session accuracy (one prediction per attack session)."""
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "session": np.asarray(sessions)})
    rows = []
    for sid, g in df.groupby("session"):
        rows.append({
            "session": sid,
            "y_true": g["y_true"].iloc[0],
            "y_pred": majority_vote(g["y_pred"].to_numpy()),
            "n_rows": len(g),
        })
    out = pd.DataFrame(rows)
    acc = float((out["y_true"] == out["y_pred"]).mean())
    return acc, out


def localization_error_summary(y_true, y_pred, sessions=None) -> dict:
    """F5: mean/median Point Prediction localization error in metres
    (`point_distance_m`, chord on the 11m circle), the share of predictions
    that are exact / adjacent (8.42m) / farther, and — if `sessions` is
    given — the same after a session-level majority vote (`session_level_accuracy`)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = np.array([point_distance_m(int(t), int(p)) for t, p in zip(y_true, y_pred)])
    adjacent_m = point_distance_m(1, 2)  # 8.42 m — the chord between any adjacent pair of points

    def _bucket_shares(errs):
        exact = float(np.mean(errs == 0))
        adjacent = float(np.mean(np.isclose(errs, adjacent_m)))
        farther = float(np.mean(errs > adjacent_m + 1e-6))
        return exact, adjacent, farther

    exact, adjacent, farther = _bucket_shares(errors)
    out = {
        "mean_error_m": float(errors.mean()),
        "median_error_m": float(np.median(errors)),
        "share_exact": exact,
        "share_adjacent": adjacent,
        "share_farther": farther,
        "n": int(len(errors)),
    }

    if sessions is not None:
        _, session_tbl = session_level_accuracy(y_true, y_pred, sessions)
        session_errors = np.array([
            point_distance_m(int(t), int(p))
            for t, p in zip(session_tbl["y_true"], session_tbl["y_pred"])
        ])
        s_exact, s_adjacent, s_farther = _bucket_shares(session_errors)
        out["session"] = {
            "mean_error_m": float(session_errors.mean()),
            "median_error_m": float(np.median(session_errors)),
            "share_exact": s_exact,
            "share_adjacent": s_adjacent,
            "share_farther": s_farther,
            "n": int(len(session_errors)),
        }
    return out


def sliding_window_accuracy(y_true, y_pred, sessions, window_sizes=(1, 2, 3, 6, 12)) -> dict:
    """Accuracy of a causal majority vote over the trailing w rows of the same
    session (partial windows at a session's start use whatever rows exist)."""
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "session": np.asarray(sessions)})
    out = {}
    for w in window_sizes:
        correct, total = 0, 0
        for _, g in df.groupby("session"):
            preds = g["y_pred"].to_numpy()
            trues = g["y_true"].to_numpy()
            for t in range(len(preds)):
                voted = majority_vote(preds[max(0, t - w + 1): t + 1])
                correct += int(voted == trues[t])
                total += 1
        out[w] = correct / total if total else float("nan")
    return out


def _full_window_alarm_flags(preds: np.ndarray, w: int) -> np.ndarray:
    """Boolean array, one per row of `preds` (a single session's predictions,
    in time order): True at row t iff the *full* trailing window of w rows
    (t-w+1..t) is entirely predicted attack. The first w-1 rows of a session
    can never fire (there isn't yet a full same-session window)."""
    n = len(preds)
    flags = np.zeros(n, dtype=bool)
    for t in range(n):
        window = preds[max(0, t - w + 1): t + 1]
        if len(window) == w and np.all(window == 1):
            flags[t] = True
    return flags


def false_alarm_rate_per_hour(y_true, y_pred, sessions, freq_seconds: float,
                               window_sizes=(1, 2, 3)) -> tuple[dict, float]:
    """Attack-only false-alarm rate on attack-free (y_true==0) rows: an alarm
    fires when a *full* causal window of w consecutive rows is all predicted
    attack. Rate is alarms / total attack-free hours (row count x grid step)."""
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "session": np.asarray(sessions)})
    no_attack = df[df["y_true"] == 0]
    total_hours = len(no_attack) * freq_seconds / 3600.0

    rates = {}
    for w in window_sizes:
        alarms = 0
        for _, g in no_attack.groupby("session"):
            alarms += int(_full_window_alarm_flags(g["y_pred"].to_numpy(), w).sum())
        rates[w] = alarms / total_hours if total_hours > 0 else float("nan")
    return rates, total_hours


def attack_detection_sensitivity(y_true, y_pred, sessions, freq_seconds: float,
                                  window_sizes=(1, 2, 3)) -> dict:
    """F3: the cost side of the false-alarm rate, under the same full-window
    alarm rule (`_full_window_alarm_flags`) — for each w:

    - detection_rate: share of attack sessions with >=1 alarm during the session;
    - median_delay_s: median, over *detected* sessions, of the time from the
      session's first row to its first alarm (assumes rows are on a
      `freq_seconds` grid within a session, i.e. delay = alarm_row_index * freq_seconds);
    - row_recall: fraction of true-attack rows (y_true==1, anywhere, not just
      within attack sessions' own window) whose windowed alarm flag is True.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "session": np.asarray(sessions)})

    out = {}
    for w in window_sizes:
        alarm_flags = np.zeros(len(df), dtype=bool)
        delays = []
        n_attack_sessions = 0
        n_detected = 0
        for sid, g in df.groupby("session"):
            is_attack_session = bool((g["y_true"] == 1).any())
            local_flags = _full_window_alarm_flags(g["y_pred"].to_numpy(), w)
            alarm_flags[g.index.to_numpy()] = local_flags
            if is_attack_session:
                n_attack_sessions += 1
                fired = np.where(local_flags)[0]
                if len(fired) > 0:
                    n_detected += 1
                    delays.append(float(fired[0]) * freq_seconds)

        is_true_attack = df["y_true"].to_numpy() == 1
        out[w] = {
            "detection_rate": n_detected / n_attack_sessions if n_attack_sessions else float("nan"),
            "median_delay_s": float(np.median(delays)) if delays else float("nan"),
            "row_recall": float(alarm_flags[is_true_attack].mean()) if is_true_attack.any() else float("nan"),
            "n_attack_sessions": n_attack_sessions,
            "n_detected": n_detected,
        }
    return out


# --- V6: session-level permutation test -------------------------------------

def session_label_permutation(y, sessions, rng: np.random.Generator, within=None) -> np.ndarray:
    """Permute session-level labels (each session's whole label reassigned to
    all of its rows), optionally restricted to permuting only within groups
    of `within` (e.g. keep each session's point fixed while shuffling antenna
    among the sessions that share that point) — acceptance check #3."""
    y = np.asarray(y)
    sessions = np.asarray(sessions)
    within = np.asarray(within) if within is not None else np.zeros(len(y), dtype=int)

    df = pd.DataFrame({"session": sessions, "y": y, "within": within})
    session_tbl = df.groupby("session").agg(y=("y", "first"), within=("within", "first"))
    permuted = session_tbl["y"].copy()
    for _, grp in session_tbl.groupby("within"):
        idx = grp.index
        permuted.loc[idx] = rng.permutation(grp["y"].to_numpy())

    mapping = permuted.to_dict()
    return df["session"].map(mapping).to_numpy()


def session_permutation_test(
    X, y, sessions, groups, outer_cv, within=None,
    n_perm: int = 1000, seed: int = 0, model_factory=None, n_jobs: int = -1,
) -> dict:
    """Null hypothesis: labels are unrelated to the RSSI profile. Permutes
    labels at the session level (see `session_label_permutation`), rescoring
    a fixed model under the primary grouped-CV protocol each time."""
    from sklearn.model_selection import cross_val_predict

    if model_factory is None:
        model_factory = lambda: ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE)

    y = np.asarray(y)
    sessions = np.asarray(sessions)
    groups = np.asarray(groups)

    def score(y_arr):
        pred = cross_val_predict(model_factory(), X, y_arr, cv=outer_cv, groups=groups, n_jobs=n_jobs)
        return float((pred == y_arr).mean())

    observed = score(y)
    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = session_label_permutation(y, sessions, rng, within=within)
        null_scores[i] = score(y_perm)

    p_value = float((1 + np.sum(null_scores >= observed)) / (n_perm + 1))
    return {
        "observed": observed,
        "null_mean": float(null_scores.mean()),
        "null_p95": float(np.percentile(null_scores, 95)),
        "p_value": p_value,
        "null_scores": null_scores,
    }


def plot_confusion(oof_true, oof_pred, labels, title, path=None, cmap="Blues",
                    tick_labels=None, paper: bool = False):
    """paper=True (F4): no title at all (no task name, no model name, no
    suptitle) — axis labels only ("Predicted"/"Actual"), for camera-ready
    figures. paper=False (default) keeps the full internal title."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(oof_true, oof_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    display_labels = tick_labels if tick_labels is not None else labels
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=display_labels, yticklabels=display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if not paper:
        ax.set_title(title)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig, cm


# --------------------------------------------------------------------------
# 8. Localization regression & baselines (Task A2: R2.1, R2.2)
# --------------------------------------------------------------------------

def _make_regressor_configs() -> dict:
    cfgs = {
        "Extra Trees": {
            "make": lambda: ExtraTreesRegressor(random_state=RANDOM_STATE),
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", 1.0]},
            "scale": False,
        },
        "Random Forest": {
            "make": lambda: RandomForestRegressor(random_state=RANDOM_STATE),
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", 1.0]},
            "scale": False,
        },
        "KNN": {
            "make": lambda: KNeighborsRegressor(),
            "grid": {"n_neighbors": list(range(1, 16)), "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
            "scale": True,
        },
        "MLP": {
            "make": lambda: MLPRegressor(random_state=RANDOM_STATE, early_stopping=True, max_iter=2000),
            "grid": {
                "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-3, 1e-2],
            },
            "scale": True,
        },
        "CatBoost": {
            # loss_function="MultiRMSE": CatBoost's native multi-output regression.
            "make": lambda: CatBoostRegressor(
                random_state=RANDOM_STATE, verbose=False, allow_writing_files=False,
                thread_count=1, loss_function="MultiRMSE",
            ),
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3]},
            "scale": False,
        },
    }
    return cfgs


REGRESSOR_CONFIGS = _make_regressor_configs()
REGRESSOR_RANDOMIZED = {"MLP", "CatBoost"}


def make_localization_scorer(variant: str):
    """A scorer(estimator, X, y) callable for GridSearchCV/RandomizedSearchCV
    that scores by *physical* localization error (metres), not RMSE —
    negated, since sklearn always maximizes the score. variant="xy": y is
    already (x, y). variant="angle": y is (cos theta, sin theta); the
    prediction is projected onto the R=POINT_RADIUS_M ring before scoring
    (the ring-prior specific to this dataset, see nested_cv_regression)."""
    def scorer(estimator, X, y):
        y_pred = np.asarray(estimator.predict(X))
        y_true = np.asarray(y)
        if variant == "angle":
            theta_p = np.arctan2(y_pred[:, 1], y_pred[:, 0])
            xp, yp = POINT_RADIUS_M * np.cos(theta_p), POINT_RADIUS_M * np.sin(theta_p)
            theta_t = np.arctan2(y_true[:, 1], y_true[:, 0])
            xt, yt = POINT_RADIUS_M * np.cos(theta_t), POINT_RADIUS_M * np.sin(theta_t)
            err = np.sqrt((xp - xt) ** 2 + (yp - yt) ** 2)
        else:
            err = np.sqrt(((y_true - y_pred) ** 2).sum(axis=1))
        return -float(np.mean(err))
    return scorer


def make_regressor(model_name: str, params: dict | None = None):
    """Analogous to make_estimator(), for REGRESSOR_CONFIGS."""
    cfg = REGRESSOR_CONFIGS[model_name]
    params = dict(params or {})
    base = cfg["make"]()
    if cfg["scale"]:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", base)])
        if params:
            pipe.set_params(**params)
        return pipe
    if params:
        base.set_params(**params)
    return base


def _regressor_search_space(model_name: str) -> dict:
    cfg = REGRESSOR_CONFIGS[model_name]
    if cfg["scale"]:
        return {f"model__{k}": v for k, v in cfg["grid"].items()}
    return cfg["grid"]


def get_regression_outer_cv_and_groups(protocol: str, df_attack: pd.DataFrame):
    """LOPO (groups=point, 8 folds): the held-out position is never seen —
    an interpolation test on the ring (its two ±45 deg neighbours are always
    in training). LOAO (groups=antenna, 6 folds): positions are all seen,
    only the antenna hardware is held out."""
    if protocol == "LOPO":
        return LeaveOneGroupOut(), df_attack["point"].to_numpy()
    if protocol == "LOAO":
        return LeaveOneGroupOut(), df_attack["antenna"].to_numpy()
    raise ValueError(f"Unknown regression protocol: {protocol}")


def nested_cv_regression(
    model_name: str,
    X: pd.DataFrame,
    y_xy: np.ndarray,
    groups: np.ndarray,
    outer_cv,
    inner_cv_factory: Callable[[np.ndarray], object],
    variant: str = "xy",
) -> dict:
    """Nested CV for (x, y) localization regression — the regression analogue
    of nested_cv(). variant="xy": regress (x, y) directly. variant="angle":
    regress (cos theta, sin theta) and project every prediction onto the
    R=POINT_RADIUS_M ring — a prior specific to this dataset (all attack
    positions lie on one ring), not a general deployment assumption; state
    this wherever variant="angle" results are reported.

    Asserts no group overlap in every outer AND inner split, exactly like
    nested_cv() (acceptance check: LOPO's held-out point never touches
    training, outer or inner).
    """
    X = X.reset_index(drop=True)
    y_xy = np.asarray(y_xy, dtype=float)
    groups = np.asarray(groups)
    n = len(y_xy)

    if variant == "angle":
        theta = np.arctan2(y_xy[:, 1], y_xy[:, 0])
        y_target = np.column_stack([np.cos(theta), np.sin(theta)])
    elif variant == "xy":
        y_target = y_xy
    else:
        raise ValueError(f"Unknown variant: {variant}")

    oof_pred_xy = np.full((n, 2), np.nan)
    fold_mean_errors = []
    best_params_per_fold = []

    space = _regressor_search_space(model_name)
    use_randomized = model_name in REGRESSOR_RANDOMIZED
    scorer = make_localization_scorer(variant)

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_target, groups)):
        _assert_no_group_overlap(groups, train_idx, test_idx, f"{model_name} regression outer fold {fold_i}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_target[train_idx]
        g_train = groups[train_idx]

        inner_cv = inner_cv_factory(g_train)
        for in_tr, in_te in inner_cv.split(X_train, y_train, g_train):
            _assert_no_group_overlap(g_train, in_tr, in_te, f"{model_name} regression outer fold {fold_i} inner split")

        estimator = make_regressor(model_name)
        search_cls = RandomizedSearchCV if use_randomized else GridSearchCV
        search_kwargs = dict(cv=inner_cv, scoring=scorer, n_jobs=-1)
        if use_randomized:
            search_kwargs.update(n_iter=RANDOMIZED_SEARCH_N_ITER, random_state=RANDOM_STATE)
        search = search_cls(estimator, space, **search_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train, groups=g_train)

        best = search.best_estimator_
        pred = np.asarray(best.predict(X_test))
        if variant == "angle":
            theta_p = np.arctan2(pred[:, 1], pred[:, 0])
            pred_xy = np.column_stack([POINT_RADIUS_M * np.cos(theta_p), POINT_RADIUS_M * np.sin(theta_p)])
        else:
            pred_xy = pred
        oof_pred_xy[test_idx] = pred_xy

        fold_err = np.sqrt(((y_xy[test_idx] - pred_xy) ** 2).sum(axis=1))
        fold_mean_errors.append(float(fold_err.mean()))
        best_params_per_fold.append(search.best_params_)

    oof_errors_m = np.sqrt(((y_xy - oof_pred_xy) ** 2).sum(axis=1))
    return {
        "oof_true_xy": y_xy,
        "oof_pred_xy": oof_pred_xy,
        "oof_errors_m": oof_errors_m,
        "fold_mean_errors_m": fold_mean_errors,
        "best_params_per_fold": best_params_per_fold,
        "groups": groups,
    }


def regression_error_summary(errors_m) -> dict:
    e = np.asarray(errors_m, dtype=float)
    return {
        "mean_error_m": float(e.mean()),
        "median_error_m": float(np.median(e)),
        "p90_error_m": float(np.percentile(e, 90)),
        "n": int(len(e)),
    }


def session_level_localization_error(oof_true_xy, oof_pred_xy, sessions) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean predicted (x, y) per session vs. that session's (constant) true
    position — the regression analogue of session_level_accuracy()."""
    oof_true_xy = np.asarray(oof_true_xy)
    oof_pred_xy = np.asarray(oof_pred_xy)
    df = pd.DataFrame({
        "session": np.asarray(sessions),
        "tx": oof_true_xy[:, 0], "ty": oof_true_xy[:, 1],
        "px": oof_pred_xy[:, 0], "py": oof_pred_xy[:, 1],
    })
    agg = df.groupby("session").agg(tx=("tx", "first"), ty=("ty", "first"), px=("px", "mean"), py=("py", "mean"))
    errors = np.sqrt((agg["tx"] - agg["px"]) ** 2 + (agg["ty"] - agg["py"]) ** 2).to_numpy()
    return errors, agg.reset_index()


def xy_cluster_bootstrap_ci(true_xy, pred_xy, cluster_ids, n_boot: int = 2000, seed: int = 0,
                             metric: str = "mean") -> dict:
    """cluster_bootstrap_ci, specialised for (x, y) localization error arrays
    instead of classification labels (same session-resampling guarantee —
    resamples clusters, never rows)."""
    metric_fn = {
        "mean": lambda te, pe: float(np.sqrt(((te - pe) ** 2).sum(axis=1)).mean()),
        "median": lambda te, pe: float(np.median(np.sqrt(((te - pe) ** 2).sum(axis=1)))),
        "p90": lambda te, pe: float(np.percentile(np.sqrt(((te - pe) ** 2).sum(axis=1)), 90)),
    }[metric]

    true_xy = np.asarray(true_xy)
    pred_xy = np.asarray(pred_xy)
    cluster_ids = np.asarray(cluster_ids)
    clusters = np.unique(cluster_ids)
    n = len(clusters)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in clusters}

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(clusters, size=n, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        stats[i] = metric_fn(true_xy[idx], pred_xy[idx])

    return {
        "ci_low": float(np.nanpercentile(stats, 2.5)),
        "ci_high": float(np.nanpercentile(stats, 97.5)),
        "boot_mean": float(np.nanmean(stats)),
        "boot_std": float(np.nanstd(stats)),
    }


# --- R2.2(a): nearest centroid --------------------------------------------

def nearest_centroid_oof(X: pd.DataFrame, y_point: np.ndarray, groups: np.ndarray, outer_cv) -> dict:
    """R2.2(a): standardize on train, predict the nearest class centroid
    (Euclidean, in standardized feature space). No hyperparameters to tune,
    so this is a plain OOF loop (still asserts no outer-fold group overlap).
    Under LOPO the held-out point's centroid never exists, so it can never be
    predicted — its error is always >= the adjacent-point chord (8.42 m)."""
    X = X.reset_index(drop=True)
    y_point = np.asarray(y_point)
    groups = np.asarray(groups)
    n = len(y_point)
    oof_pred_point = np.full(n, -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_point, groups)):
        _assert_no_group_overlap(groups, train_idx, test_idx, f"nearest-centroid outer fold {fold_i}")

        scaler = StandardScaler().fit(X.iloc[train_idx])
        Xs_train = scaler.transform(X.iloc[train_idx])
        Xs_test = scaler.transform(X.iloc[test_idx])

        y_train = y_point[train_idx]
        classes = np.unique(y_train)
        centroids = np.array([Xs_train[y_train == c].mean(axis=0) for c in classes])

        dists = np.linalg.norm(Xs_test[:, None, :] - centroids[None, :, :], axis=2)  # (n_test, n_classes)
        oof_pred_point[test_idx] = classes[np.argmin(dists, axis=1)]

    oof_pred_xy = np.array([point_xy_m(int(p)) for p in oof_pred_point])
    return {"oof_true_point": y_point, "oof_pred_point": oof_pred_point, "oof_pred_xy": oof_pred_xy, "groups": groups}


# --- R2.2(b): weighted centroid localization (WCL), no training -----------

def wcl_predict_xy(df: pd.DataFrame, sensor_cols: list[str] = FEATURES_BASE,
                    sensor_positions: dict | None = None, project_to_ring: bool = False) -> np.ndarray:
    """R2.2(b): position estimate = sum_k w_k p_k / sum_k w_k, weights
    w_k = 10^(RSSI_k/10) (linear power). Training-free. project_to_ring=True
    additionally projects onto the R=POINT_RADIUS_M ring along the estimated
    direction (WCL's raw estimate is always inside the sensor polygon, i.e.
    biased toward the origin relative to the true R=11m ring)."""
    sensor_positions = sensor_positions or SENSOR_POSITIONS
    rssi = df[sensor_cols].to_numpy(dtype=float)
    w = np.power(10.0, rssi / 10.0)
    pos = np.array([sensor_positions[c] for c in sensor_cols])  # (k, 2)
    xy = (w @ pos) / w.sum(axis=1, keepdims=True)
    if project_to_ring:
        theta = np.arctan2(xy[:, 1], xy[:, 0])
        xy = np.column_stack([POINT_RADIUS_M * np.cos(theta), POINT_RADIUS_M * np.sin(theta)])
    return xy


# --- R2.2(c): log-distance trilateration with unknown transmit power ------

def _trilateration_grid(bound: float = 15.0, step: float = 0.25) -> np.ndarray:
    coords = np.arange(-bound, bound + step / 2, step)
    gx, gy = np.meshgrid(coords, coords)
    return np.column_stack([gx.ravel(), gy.ravel()])


def trilaterate_xy(rssi: np.ndarray, n_exp: float, sensor_cols: list[str] = FEATURES_BASE,
                    sensor_positions: dict | None = None, grid: np.ndarray | None = None) -> np.ndarray:
    """R2.2(c): RSSI_k = P0 - 10*n*log10(||p - p_k||); for a candidate p, P0
    has the closed-form least-squares estimate mean_k(RSSI_k + 10*n*log10(d_k)).
    Grid-searches p over `grid` (default: 0.25m grid on [-15, 15]^2) to
    minimize the residual sum of squares, for every row of `rssi` at once —
    vectorized via the algebraic identity
    ||c + b||^2 = ||c||^2 + 2*b.c + ||b||^2 where c depends only on
    (grid, n_exp) and b only on the row's RSSI, avoiding an O(rows x grid x
    sensors) tensor.
    """
    sensor_positions = sensor_positions or SENSOR_POSITIONS
    grid = _trilateration_grid() if grid is None else grid
    pos = np.array([sensor_positions[c] for c in sensor_cols])  # (k, 2)

    d = np.linalg.norm(grid[:, None, :] - pos[None, :, :], axis=2)  # (M, k)
    d = np.maximum(d, 0.1)
    logd = np.log10(d)  # (M, k)
    mean_logd = logd.mean(axis=1)  # (M,)
    c = 10.0 * n_exp * (mean_logd[:, None] - logd)  # (M, k)
    C = (c ** 2).sum(axis=1)  # (M,)

    rssi = np.atleast_2d(np.asarray(rssi, dtype=float))  # (rows, k)
    mean_rssi = rssi.mean(axis=1)  # (rows,)
    b = mean_rssi[:, None] - rssi  # (rows, k)
    B = (b ** 2).sum(axis=1)  # (rows,)

    residuals = C[None, :] + 2.0 * (b @ c.T) + B[:, None]  # (rows, M)
    best_idx = np.argmin(residuals, axis=1)
    return grid[best_idx]


def fit_trilateration_n(rssi: np.ndarray, true_xy: np.ndarray, sensor_cols: list[str] = FEATURES_BASE,
                         sensor_positions: dict | None = None, grid: np.ndarray | None = None,
                         n_range=np.arange(1.5, 4.01, 0.1)) -> tuple[float, float]:
    """R2.2(c): fit the path-loss exponent n on training rows by grid search,
    minimizing mean localization error. Returns (best_n, best_mean_error_m)."""
    true_xy = np.asarray(true_xy)
    best_n, best_err = None, np.inf
    for n_exp in n_range:
        pred = trilaterate_xy(rssi, float(n_exp), sensor_cols=sensor_cols, sensor_positions=sensor_positions, grid=grid)
        err = float(np.sqrt(((true_xy - pred) ** 2).sum(axis=1)).mean())
        if err < best_err:
            best_n, best_err = float(n_exp), err
    return best_n, best_err


def trilateration_oof(df: pd.DataFrame, y_xy: np.ndarray, groups: np.ndarray, outer_cv,
                       sensor_cols: list[str] = FEATURES_BASE, sensor_positions: dict | None = None,
                       fit_n: bool = True, fixed_n: float = 2.0) -> dict:
    """R2.2(c) as an OOF loop matching the rest of the framework: for each
    outer fold, optionally fit n on the training rows (`fit_trilateration_n`),
    then trilaterate the held-out rows with that n (or with `fixed_n` if
    fit_n=False). No inner CV — n-fitting *is* the only "training" here."""
    rssi_all = df[sensor_cols].to_numpy(dtype=float)
    y_xy = np.asarray(y_xy, dtype=float)
    groups = np.asarray(groups)
    n = len(y_xy)
    grid = _trilateration_grid()

    oof_pred_xy = np.full((n, 2), np.nan)
    n_per_fold = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(df, y_xy, groups)):
        _assert_no_group_overlap(groups, train_idx, test_idx, f"trilateration outer fold {fold_i}")
        if fit_n:
            best_n, _ = fit_trilateration_n(
                rssi_all[train_idx], y_xy[train_idx], sensor_cols=sensor_cols,
                sensor_positions=sensor_positions, grid=grid,
            )
        else:
            best_n = fixed_n
        n_per_fold.append(best_n)
        oof_pred_xy[test_idx] = trilaterate_xy(
            rssi_all[test_idx], best_n, sensor_cols=sensor_cols, sensor_positions=sensor_positions, grid=grid,
        )

    return {"oof_true_xy": y_xy, "oof_pred_xy": oof_pred_xy, "n_per_fold": n_per_fold, "groups": groups}
