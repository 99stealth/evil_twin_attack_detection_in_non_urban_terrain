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

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
from sklearn.neighbors import KNeighborsClassifier
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
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

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
            preds = g["y_pred"].to_numpy()
            for t in range(len(preds)):
                window = preds[max(0, t - w + 1): t + 1]
                if len(window) == w and np.all(window == 1):
                    alarms += 1
        rates[w] = alarms / total_hours if total_hours > 0 else float("nan")
    return rates, total_hours


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


def plot_confusion(oof_true, oof_pred, labels, title, path=None, cmap="Blues", tick_labels=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(oof_true, oof_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    display_labels = tick_labels if tick_labels is not None else labels
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=display_labels, yticklabels=display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig, cm
