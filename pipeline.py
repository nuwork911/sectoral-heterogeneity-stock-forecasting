"""
====================================================================================================================
Sectoral Heterogeneity in Stock Price Direction Forecasting: Implications for AI-Driven Investment Decision Support
====================================================================================================================

Yuvraj Sachan - KF7029 MSc Artificial Intelligence, Northumbria University

Run from the folder that holds merged_data.csv and ticker_sectors.xlsx:

    python pipeline.py

Outputs (all under results/):
    sci_by_sector.csv               five SCI metrics + index + band, per sector
    sci_hurst_sensitivity.csv       SCI under both Hurst sign conventions
    predictions.parquet             EVERY out-of-sample prediction (see note 1)
    performance_by_model_sector.csv pooled metrics per (model, sector)
    performance_by_fold.csv         per-fold metrics  (powers the RQ1 test)
    lstm_seed_stability.csv         per-seed LSTM metrics, mean and SD
    auc_bootstrap_ci.csv            date-block bootstrap CIs for every cell
    mcnemar_pairs.csv               all model pairs x all sectors, Holm-adjusted
    rq_summary.json                 every test statistic in one place
    robustness_*.csv                per-ticker split, Hurst min_n, lag-0 feature
    sci_decomposition.csv           each SCI component vs performance (5 tests)
    sci_pca_weighted.csv            PC1-weighted SCI, alternative to equal weights
    auc_bootstrap_sectors.csv       bootstrap CI for each SECTOR's mean AUC
    shap_importance.csv             per-sector, per-feature mean |SHAP|
    shap_groups.csv                 the same, collapsed to six feature groups
    shap_vs_sci.csv                 does the attribution profile track the index?
    global_vs_sector.csv            one panel-wide model vs the per-sector models
    numbers.tex                     numbers procuded in a clean file
    tab_*.tex                       LaTeX table bodies for the thesis


=============================================================================
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# FutureWarning spam from pandas/xgboost is noise we don't need in the log.
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional deps - degrade gracefully if not installed.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# =========================================================================== #
# CONFIG                                                                      #
# =========================================================================== #
PRICE_CSV = "merged_data.csv"
SECTOR_MAP = "ticker_sectors.xlsx"
OUT_DIR = "results"

# Column names - kept here so a schema change is a one-line edit.
COL_DATE, COL_TICKER, COL_SECTOR = "date", "ticker", "Sector"
COL_OPEN, COL_HIGH, COL_LOW = "Open", "High", "Low"
COL_CLOSE, COL_VOLUME = "Close", "Volume"

# Column `sentiment` is out of scope for this study; the
# assertion in build_features() enforces that.
FORBIDDEN_FEATURES = ("sentiment",)

RANDOM_STATE = 42
N_SPLITS = 5                    # expanding-window walk-forward folds
ALPHA = 0.05
METRIC = "auc"                  # primary metric

# --- Sectoral Complexity Index -------------------------------------------- #
# +1 if a HIGHER value means MORE complex.  Hurst is -1:
# stronger persistence aids forecasting and therefore lowers complexity
# (Eom et al., 2008).  HURST_SENSITIVITY re-runs the index with +1.
SCI_DIRECTION = {"volatility": +1, "sampen": +1, "hurst": -1,
                 "amihud": +1, "vol_var": +1}
SCI_ZSCORE_DDOF = 0             # population SD across the eleven sectors
SAMPEN_M, SAMPEN_R = 2, 0.2
HURST_MIN_N = 8                 # smallest R/S block
HURST_MIN_N_ROBUST = 16         # sensitivity check (R/S is biased at small n)
N_BANDS = 3
MIN_HISTORY_SCI = 250

# --- Features -------------------------------------------------------------- #
LAGS = [1, 2, 3, 5, 10]         # r_{t-1} ... r_{t-10}
INCLUDE_LAG0 = False            # r_t itself.  Main spec = False to match the
                                # documented feature set; the robustness block
                                # re-runs with True.
MA_WINDOWS = [5, 10, 20]
VOL_WINDOW = 10
RSI_WINDOW = 14
MOM_WINDOW = 10

# --- LSTM ------------------------------------------------------------------ #
SEQ_LEN = 10
LSTM_HIDDEN = 32
LSTM_EPOCHS = 10
LSTM_BATCH = 512
LSTM_LR = 1e-3
LSTM_VAL_FRAC = 0.10            # tail of the training partition, for epoch choice
N_SEEDS_LSTM = 5

# --- Inference ------------------------------------------------------------- #
BOOTSTRAP_B = 2000              # date-block bootstrap replicates
BOOTSTRAP_SEED = 7
REUSE_PREDICTIONS = True        # load results/predictions.parquet if present
                                # instead of refitting every model
RUN_ROBUSTNESS = True
QUICK_MODE = False              # True -> 3 sectors, 1 seed, 50 bootstraps


# The headline results are
# unchanged when predictions are reused from an earlier run.
RUN_SHAP = True                 # per-sector feature attribution 
SHAP_MAX_ROWS = 20000           # cap on test rows explained per sector
SHAP_MODEL = "XGBoost"          # TreeExplainer is exact and fast for GBMs
RUN_GLOBAL_BASELINE = True      # all-sectors-pooled fit vs the per-sector fits
GLOBAL_BASELINE_MODELS = None   # None = every tabular model.  RandomForest on
                                # the full ~580k-row panel is the slow one; set
                                
RUN_SCI_PCA = True              # PCA-weighted SCI as an alternative weighting



RUN_PANEL = True                # sector x fold panel with SCI x model
RUN_REGIMES = True              # volatility and trend market states
RUN_ABSTENTION = True           # selective prediction and reliability gating
RUN_SELECTION_BASELINES = True  # SCI policy against the baselines it must beat
SCI_FOLD_WINDOW = 250           # trailing days used for the per-fold SCI
SCI_FOLD_MIN_OBS = 120          # minimum ticker observations inside that window
REGIME_VOL_WINDOW = 21          # trailing window for the volatility state
REGIME_TREND_WINDOW = 63        # trailing window for the bull/bear state
ABSTAIN_COVERAGES = (1.0, 0.75, 0.50, 0.25, 0.10)
SELECTION_RANDOM_DRAWS = 2000   # draws behind the random-selection baseline

# Nemenyi q_alpha, alpha=0.05, Demsar (2006) Table 5(a), indexed by #models
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

NUMBERS: dict[str, str] = {}    # macro name -> formatted value, for numbers.tex


def _save_frame(frame: pd.DataFrame, stem: str) -> str:
    """Write a frame, preferring parquet and falling back to gzipped CSV."""
    try:
        frame.to_parquet(f"{stem}.parquet", index=False)
        return f"{stem}.parquet"
    except Exception:
        # No pyarrow/fastparquet installed - fall back to gzip CSV.
        frame.to_csv(f"{stem}.csv.gz", index=False, compression="gzip")
        return f"{stem}.csv.gz"


def _load_frame(stem: str):
    """Read back whichever format _save_frame produced, or None."""
    for ext, reader in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        path = f"{stem}{ext}"
        if os.path.exists(path):
            try:
                frame = reader(path)
                frame[COL_DATE] = pd.to_datetime(frame[COL_DATE])
                return frame, path
            except Exception:
                return None, None
    return None, None


def note(msg: str) -> None:
    """Tagged console line so logs are easy to grep."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def macro(name: str, value, fmt: str = "{:.4f}") -> None:
    """Record a value for emission as a LaTeX \\renewcommand."""
    NUMBERS[name] = value if isinstance(value, str) else fmt.format(value)


def set_all_seeds(seed: int) -> None:
    """One place to seed everything that uses RNG."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  
        torch.backends.cudnn.benchmark = False


# =========================================================================== #
# DATA                                                                        #
# =========================================================================== #
def load_data(price_csv: str = PRICE_CSV, sector_map: str = SECTOR_MAP) -> pd.DataFrame:
    """Load and merge price + sector files.  Drops anything unphysical."""
    df = pd.read_csv(price_csv, parse_dates=[COL_DATE])
    n_raw = len(df)

    # Read the sector lookup 
    smap = (pd.read_excel(sector_map) if str(sector_map).lower().endswith((".xlsx", ".xls"))
            else pd.read_csv(sector_map))
    smap = smap.iloc[:, :2]   # first two columns are ticker, sector
    smap.columns = [COL_TICKER, COL_SECTOR]

    df = df.merge(smap, on=COL_TICKER, how="inner")
    keep = [COL_DATE, COL_TICKER, COL_SECTOR, COL_OPEN, COL_HIGH,
            COL_LOW, COL_CLOSE, COL_VOLUME]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=[COL_CLOSE])
    
    df = df[(df[COL_CLOSE] > 0) & (df[COL_VOLUME] > 0)]
    df = df.drop_duplicates([COL_TICKER, COL_DATE])
    df = df.sort_values([COL_TICKER, COL_DATE]).reset_index(drop=True)

    lengths = df.groupby(COL_TICKER).size()
    note(f"    rows {n_raw} raw -> {len(df)} clean | "
         f"{df[COL_TICKER].nunique()} tickers | {df[COL_SECTOR].nunique()} sectors")

    # Cache headline numbers for the thesis tables.
    macro("NRowsRaw", f"{n_raw:,}")
    macro("NRowsClean", f"{len(df):,}")
    macro("NTickers", str(df[COL_TICKER].nunique()))
    macro("NSectors", str(df[COL_SECTOR].nunique()))
    # Windows strftime doesn't support %-d so guard against that.
    macro("DateStart", df[COL_DATE].min().strftime("%-d %B %Y")
          if os.name != "nt" else df[COL_DATE].min().strftime("%d %B %Y"))
    macro("DateEnd", df[COL_DATE].max().strftime("%-d %B %Y")
          if os.name != "nt" else df[COL_DATE].max().strftime("%d %B %Y"))
    macro("MinTickerDays", str(int(lengths.min())))
    macro("MaxTickerDays", str(int(lengths.max())))
    macro("NTickersSCI", str(int((lengths >= MIN_HISTORY_SCI).sum())))
    macro("NTickersDropped", str(int((lengths < MIN_HISTORY_SCI).sum())))
    return df


# =========================================================================== #
# SECTORAL COMPLEXITY INDEX                                                   #
# =========================================================================== #
def log_returns(close) -> np.ndarray:
    """Log returns, finite-only - drops the first NaN."""
    r = np.diff(np.log(np.asarray(close, dtype=float)))
    return r[np.isfinite(r)]


def annualised_vol(r: np.ndarray, periods: int = 252) -> float:
    """Annualised volatility of a return series (252 trading days)."""
    return float(np.std(r, ddof=1) * np.sqrt(periods)) if r.size >= 2 else np.nan


def sample_entropy(x, m: int = SAMPEN_M, r_frac: float = SAMPEN_R) -> float:
    """Richman and Moorman (2000) sample entropy, -ln(A/B)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    if N < m + 2:
        return np.nan
    r = r_frac * np.std(x, ddof=0)
    if r <= 0:
        return np.nan

    def _count(mm: int) -> int:
        # Count pairs of length-mm templates that match within tolerance r.
        M = N - m                      # same template count for m and m+1
        templates = np.lib.stride_tricks.sliding_window_view(x, mm)[:M]
        total = 0
        for i in range(M - 1):
            d = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            total += int(np.count_nonzero(d <= r))
        return total

    B, A = _count(m), _count(m + 1)
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan


def hurst_rs(x, min_n: int = HURST_MIN_N) -> float:
    """Hurst exponent by rescaled-range analysis on non-overlapping blocks.

    classical R/S is upward biased at small block sizes. The robustness
    block re-estimates the index with min_n = HURST_MIN_N_ROBUST.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = x.size
    if N < 4 * min_n:
        return np.nan
    # logarithmically spaced block sizes between min_n and N/2
    ns = np.unique(np.floor(np.logspace(np.log10(min_n),
                                        np.log10(N // 2), 20)).astype(int))
    ns = ns[ns >= min_n]
    rs, valid = [], []
    for n in ns:
        n = int(n)
        blocks = N // n
        if blocks < 1:
            continue
        vals = []
        for b in range(blocks):
            seg = x[b * n:(b + 1) * n]
            y = np.cumsum(seg - seg.mean())
            R = y.max() - y.min()
            S = seg.std(ddof=1)
            if S > 0 and R > 0:
                vals.append(R / S)
        if vals:
            rs.append(float(np.mean(vals)))
            valid.append(n)
    if len(valid) < 4:
        return np.nan
    # Hurst = slope of log(R/S) vs log(n)
    return float(np.polyfit(np.log(valid), np.log(rs), 1)[0])


def amihud_illiquidity(r: np.ndarray, close, volume, scale: float = 1e6) -> float:
    """Amihud (2002): mean |return| per unit of dollar volume, times 1e6."""
    close = np.asarray(close, dtype=float)[1:]
    volume = np.asarray(volume, dtype=float)[1:]
    if close.size != r.size:                       # guard against ragged input
        k = min(close.size, r.size)
        close, volume, r = close[:k], volume[:k], r[:k]
    dollar = close * volume
    mask = (dollar > 0) & np.isfinite(r) & np.isfinite(dollar)
    if mask.sum() < 2:
        return np.nan
    return float(np.mean(np.abs(r[mask]) / dollar[mask]) * scale)


def volume_variability(volume) -> float:
    """Coefficient of variation of volume - dimensionless dispersion measure."""
    v = np.asarray(volume, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1) / v.mean()) if v.size >= 2 and v.mean() > 0 else np.nan


def _ticker_metrics(g: pd.DataFrame, hurst_min_n: int = HURST_MIN_N) -> pd.Series:
    """Per-ticker bundle of the five raw SCI metrics."""
    close = g[COL_CLOSE].to_numpy()
    vol = g[COL_VOLUME].to_numpy()
    r = log_returns(close)
    return pd.Series({
        "volatility": annualised_vol(r),
        "sampen": sample_entropy(r),
        "hurst": hurst_rs(r, min_n=hurst_min_n),
        "amihud": amihud_illiquidity(r, close, vol),
        "vol_var": volume_variability(vol),
    })


def _index_from_raw(per_sector: pd.DataFrame, direction: dict) -> pd.Series:
    """Z-score each metric, sign it per SCI_DIRECTION, and average."""
    metrics = list(direction)
    z = ((per_sector[metrics] - per_sector[metrics].mean())
         / per_sector[metrics].std(ddof=SCI_ZSCORE_DDOF))
    return z.mul(pd.Series(direction)).mean(axis=1)


def compute_sci(df: pd.DataFrame, hurst_min_n: int = HURST_MIN_N,
                record: bool = True) -> pd.DataFrame:
    """Per-sector table: five raw metrics, n_tickers, SCI and complexity band."""
    counts = df.groupby(COL_TICKER)[COL_DATE].transform("size")
    long = df[counts >= MIN_HISTORY_SCI]

    # groupby.apply with include_groups=False avoids the deprecation warning
    per_ticker = (long.groupby([COL_SECTOR, COL_TICKER], sort=False)
                      .apply(lambda g: _ticker_metrics(g, hurst_min_n),
                             include_groups=False)
                      .reset_index())

    metrics = list(SCI_DIRECTION)
    per_sector = per_ticker.groupby(COL_SECTOR)[metrics].mean(numeric_only=True)
    per_sector["n_tickers"] = per_ticker.groupby(COL_SECTOR).size()
    per_sector["SCI"] = _index_from_raw(per_sector, SCI_DIRECTION)
    # Three equal-frequency bands via qcut.
    per_sector["band"] = pd.qcut(per_sector["SCI"], N_BANDS,
                                 labels=["low", "medium", "high"])
    per_sector = per_sector.sort_values("SCI")

    # sensitivity: flip the Hurst orientation
    flipped = dict(SCI_DIRECTION, hurst=+1)
    per_sector["SCI_hurst_plus"] = _index_from_raw(per_sector, flipped)
    tau, tau_p = stats.kendalltau(per_sector["SCI"], per_sector["SCI_hurst_plus"])
    if record: macro("HurstFlipTau", tau)
    if record: macro("HurstFlipTauP", tau_p, "{:.3f}")

    
    corr = per_sector[metrics].corr()
    if record: macro("CorrVolHurst", corr.loc["volatility", "hurst"], "{:.2f}")
    if record: macro("CorrHurstVolVar", corr.loc["hurst", "vol_var"], "{:.2f}")
    if record: macro("CorrAmihudMin", corr.loc["amihud", [c for c in metrics if c != "amihud"]].min(), "{:.2f}")
    if record: macro("CorrAmihudMax", corr.loc["amihud", [c for c in metrics if c != "amihud"]].max(), "{:.2f}")
    if record: macro("HurstMin", per_sector["hurst"].min(), "{:.3f}")
    if record: macro("HurstMax", per_sector["hurst"].max(), "{:.3f}")
    if record: macro("SCIMin", per_sector["SCI"].min(), "{:+.3f}")
    if record: macro("SCIMax", per_sector["SCI"].max(), "{:+.3f}")
    if record: macro("NLowBand", str(int((per_sector["band"] == "low").sum())))
    if record: macro("NMediumBand", str(int((per_sector["band"] == "medium").sum())))
    if record: macro("NHighBand", str(int((per_sector["band"] == "high").sum())))
    return per_sector


# =========================================================================== #
# FEATURES                                                                    #
# =========================================================================== #
def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI on close prices.  Vectorised with rolling means."""
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _features_one_ticker(g: pd.DataFrame, include_lag0: bool) -> pd.DataFrame:
    """Build the feature row for a single ticker, sorted by date.

    Target is the sign of r_{t+1}, so the model is asked to forecast next-day
    direction."""
    g = g.sort_values(COL_DATE)
    close, vol = g[COL_CLOSE], g[COL_VOLUME]
    ret = np.log(close).diff()

    feat = pd.DataFrame(index=g.index)
    if include_lag0:
        feat["ret_lag0"] = ret                       # r_t, known at close of t
    for k in LAGS:
        feat[f"ret_lag{k}"] = ret.shift(k)
    for w in MA_WINDOWS:
        # methodology eq. (3): P_t / MA_w(P) - 1
        feat[f"close_ma{w}"] = close / close.rolling(w).mean() - 1
    feat[f"vol_{VOL_WINDOW}"] = ret.rolling(VOL_WINDOW).std()
    feat["rsi"] = _rsi(close, RSI_WINDOW) / 100   # rescaled to [0,1]
    feat[f"mom{MOM_WINDOW}"] = close / close.shift(MOM_WINDOW) - 1
    feat["vol_chg"] = vol.pct_change()
    feat["vol_ma_ratio"] = vol / vol.rolling(MA_WINDOWS[-1]).mean()

    feat[COL_DATE] = g[COL_DATE].to_numpy()
    feat[COL_TICKER] = g[COL_TICKER].to_numpy()
    feat["target"] = (ret.shift(-1) > 0).astype(int).to_numpy()   # sign of r_{t+1}
    return feat


def build_features(df: pd.DataFrame, include_lag0: bool = INCLUDE_LAG0,
                   record: bool = True) -> pd.DataFrame:
    """Concatenate per-ticker feature frames, drop NaNs, attach sector."""
    parts = [_features_one_ticker(g, include_lag0)
             for _, g in df.groupby(COL_TICKER, sort=False)]
    out = pd.concat(parts).replace([np.inf, -np.inf], np.nan).dropna()
    tmap = df[[COL_TICKER, COL_SECTOR]].drop_duplicates()
    out = out.merge(tmap, on=COL_TICKER, how="left")
    out = out.sort_values([COL_TICKER, COL_DATE]).reset_index(drop=True)

    # Safety net: nothing forbidden should ever reach the model.  Fail loud
    # if it does, because leaking sentiment would invalidate every result.
    cols = feature_columns(out)
    bad = [c for c in cols if any(f in c.lower() for f in FORBIDDEN_FEATURES)]
    assert not bad, f"out-of-scope column reached the feature matrix: {bad}"
    assert "target" not in cols and COL_DATE not in cols
    if record:
        macro("NFeatures", str(len(cols)))
        macro("NModelRows", f"{len(out):,}")
    note(f"    feature panel {len(out):,} rows x {len(cols)} features")
    return out


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """Everything that's not an ID / target is a model input."""
    return [c for c in feat.columns
            if c not in (COL_DATE, COL_TICKER, COL_SECTOR, "target")]


# =========================================================================== #
# MODELS                                                                      #
# =========================================================================== #
def make_sklearn(name: str, seed: int = RANDOM_STATE):
    """Factory for the tabular classifiers.  Hyperparameters chosen to be
    'sensible defaults', not tuned."""
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=seed,
                             n_jobs=-1, tree_method="hist")
    raise ValueError(name)


def available_models() -> list[str]:
    """Return the list of models we can actually build on this machine."""
    models = ["LogisticRegression", "RandomForest"]
    if _HAS_XGB:
        models.append("XGBoost")
    else:
        note("    [warn] xgboost not installed -> XGBoost skipped")
    if _HAS_TORCH:
        models.append("LSTM")
    else:
        note("    [warn] torch not installed -> LSTM skipped")
    return models


def _windows_for_ticker(X: np.ndarray, y: np.ndarray, keep: np.ndarray,
                        seq_len: int):
    """Windows ENDING AT t inclusive: rows t-seq_len+1 .. t predict y_t.

    used rows t-seq_len .. t-1, so the LSTM never saw day t while the
    tabular models did.

    `keep` marks which end-rows to emit. The LSTM is scored on the same sample as the tabular models.
    """
    n = X.shape[0]
    if n < seq_len:
        return None, None, None
    # sliding_window_view gives shape (m, f, seq_len); we want (m, seq_len, f)
    wins = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=seq_len, axis=0)                 # (m, f, seq_len)
    wins = np.ascontiguousarray(wins.transpose(0, 2, 1))  # (m, seq_len, f)
    ends = np.arange(seq_len - 1, n)
    sel = keep[ends]
    if not sel.any():
        return None, None, None
    return wins[sel], y[ends][sel], ends[sel]


def build_sequences(frame: pd.DataFrame, cols: list[str], keep_mask: np.ndarray,
                    seq_len: int = SEQ_LEN):
    """frame must be sorted by (ticker, date) with a positional RangeIndex."""
    Xs, ys, pos = [], [], []
    arr_all = frame[cols].to_numpy(np.float32)
    y_all = frame["target"].to_numpy(np.float32)
    for _, idx in frame.groupby(COL_TICKER, sort=False).indices.items():
        idx = np.sort(idx)
        Xw, yw, ends = _windows_for_ticker(arr_all[idx], y_all[idx],
                                          keep_mask[idx], seq_len)
        if Xw is None:
            continue
        Xs.append(Xw)
        ys.append(yw)
        pos.append(idx[ends])
    if not Xs:
        return None, None, None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(pos)


def make_fold_sequences(frame: pd.DataFrame, cols: list[str],
                        train_mask: np.ndarray, test_mask: np.ndarray,
                        scaler: StandardScaler):
    """Build the train/test sequence tensors ONCE per fold.

    rebuilt every LSTM call; 
    """
    scaled = frame.copy()
    scaled[cols] = scaler.transform(frame[cols])
    Xtr, ytr, pos_tr = build_sequences(scaled, cols, train_mask)
    Xte, yte, pos_te = build_sequences(scaled, cols, test_mask)
    if Xtr is None or Xte is None or len(np.unique(ytr)) < 2:
        return None
    # inner validation split = latest LSTM_VAL_FRAC of training windows in time
    order = np.argsort(frame[COL_DATE].to_numpy()[pos_tr], kind="stable")
    n_val = max(1, int(len(order) * LSTM_VAL_FRAC))
    val_idx, tr_idx = order[-n_val:], order[:-n_val]
    # fall back to no validation split if the training tail is too small
    if len(tr_idx) < LSTM_BATCH or len(np.unique(ytr[tr_idx])) < 2:
        tr_idx, val_idx = np.arange(len(ytr)), None
    return {"Xtr": Xtr, "ytr": ytr, "tr_idx": tr_idx, "val_idx": val_idx,
            "Xte": Xte, "yte": yte, "pos_te": pos_te}


if _HAS_TORCH:

    class SeqLSTM(nn.Module):
        """Minimal single-layer LSTM with a linear readout on the last state."""
        def __init__(self, n_feat: int, hidden: int = LSTM_HIDDEN):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    def train_lstm_once(seq: dict, n_feat: int, seed: int) -> np.ndarray:
        """One seeded training run; returns test-set probabilities.

        ran LSTM_EPOCHS *full-batch* gradient steps (eight Adam updates from
        random initialisation), so it measured a near-random projection of the
        features.  This is minibatch training with the epoch chosen on a
        held-out tail of the training partition.
        """
        set_all_seeds(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = SeqLSTM(n_feat).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=LSTM_LR)
        loss_fn = nn.BCEWithLogitsLoss()

        Xtr, ytr = seq["Xtr"], seq["ytr"]
        tr_idx, val_idx = seq["tr_idx"], seq["val_idx"]
        ds = TensorDataset(torch.from_numpy(Xtr[tr_idx]), torch.from_numpy(ytr[tr_idx]))
        gen = torch.Generator().manual_seed(seed)   # makes the DataLoader deterministic too
        dl = DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True, generator=gen)

        def _probs(X: np.ndarray) -> np.ndarray:
            """Batched inference - keep large test sets off the GPU at once."""
            net.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 4096):
                    xb = torch.from_numpy(X[i:i + 4096]).to(dev)
                    out.append(torch.sigmoid(net(xb)).cpu().numpy())
            return np.concatenate(out)

        # Early stopping by AUC on the validation tail; remember the best state.
        best_state, best_score = None, -np.inf
        for _ in range(LSTM_EPOCHS):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()
            if val_idx is None:
                # no validation -> keep the final epoch
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                continue
            pv = _probs(Xtr[val_idx])
            score = (roc_auc_score(ytr[val_idx], pv)
                     if len(np.unique(ytr[val_idx])) > 1 else 0.5)
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
        return _probs(seq["Xte"])


# =========================================================================== #
# WALK-FORWARD EVALUATION -> long prediction table                            #
# =========================================================================== #
def run_sector(fs: pd.DataFrame, models: list[str],
               n_seeds_lstm: int = N_SEEDS_LSTM) -> pd.DataFrame:
    """Every out-of-sample prediction for one sector, as a long table."""
    fs = fs.sort_values([COL_TICKER, COL_DATE]).reset_index(drop=True)
    cols = feature_columns(fs)
    dates = np.sort(fs[COL_DATE].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(dates), start=1):
        train_mask = fs[COL_DATE].isin(dates[tr_idx]).to_numpy()
        test_mask = fs[COL_DATE].isin(dates[te_idx]).to_numpy()
        if not train_mask.any() or not test_mask.any():
            continue
        train, test = fs[train_mask], fs[test_mask]
        if train["target"].nunique() < 2:
            # can't fit anything if training is single-class
            continue

        scaler = StandardScaler().fit(train[cols])
        train_major = float(max(train["target"].mean(), 1 - train["target"].mean()))
        # the constant predictor a decision maker could actually deploy
        train_const = int(train["target"].mean() >= 0.5)

        for name in models:
            if name == "LSTM":
                if not _HAS_TORCH or n_seeds_lstm < 1:
                    continue
                seq = make_fold_sequences(fs, cols, train_mask, test_mask, scaler)
                if seq is None:
                    continue
                pos, y_true = seq["pos_te"], seq["yte"].astype(int)
                # Multiple seeds so we can report a mean and SD for the LSTM.
                for sd in range(n_seeds_lstm):
                    prob = train_lstm_once(seq, len(cols), seed=RANDOM_STATE + sd)
                    rows.append(pd.DataFrame({
                        COL_DATE: fs[COL_DATE].to_numpy()[pos],
                        COL_TICKER: fs[COL_TICKER].to_numpy()[pos],
                        COL_SECTOR: fs[COL_SECTOR].to_numpy()[pos],
                        "model": "LSTM", "seed": sd, "fold": fold,
                        "y_true": y_true, "y_prob": prob,
                        "train_major": train_major, "train_const": train_const}))
            else:
                clf = make_sklearn(name, seed=RANDOM_STATE)
                clf.fit(scaler.transform(train[cols]), train["target"])
                prob = clf.predict_proba(scaler.transform(test[cols]))[:, 1]
                rows.append(pd.DataFrame({
                    COL_DATE: test[COL_DATE].to_numpy(),
                    COL_TICKER: test[COL_TICKER].to_numpy(),
                    COL_SECTOR: test[COL_SECTOR].to_numpy(),
                    "model": name, "seed": 0, "fold": fold,
                    "y_true": test["target"].to_numpy(),
                    "y_prob": prob,
                    "train_major": train_major, "train_const": train_const}))

    if not rows:
        return pd.DataFrame()
    # Sector is carried per row rather than stamped from fs.iloc[0]: identical
    # for the per-sector calls, but it lets the same function fit ONE model on
    # the whole panel and still report performance by sector (global_vs_sector).
    return pd.concat(rows, ignore_index=True)


def restrict_to_common_support(preds: pd.DataFrame) -> pd.DataFrame:
    """Keep only (sector, date, ticker) keys scored by EVERY model.

    Without this, McNemar and the paired band comparisons are computed on
    slightly different samples for the LSTM.  The dropped share is reported.
    """
    n_models = preds["model"].nunique()
    key = [COL_SECTOR, COL_DATE, COL_TICKER]
    cover = preds.groupby(key, sort=False, observed=True)["model"].transform("nunique")
    before = len(preds)
    out = preds[cover == n_models].reset_index(drop=True)
    macro("CommonSupportPct", 100 * len(out) / before, "{:.2f}")
    note(f"    common support keeps {len(out):,}/{before:,} prediction rows "
         f"({100 * len(out) / before:.2f}%)")
    return out


# =========================================================================== #
# METRICS                                                                     #
# =========================================================================== #
def _metric_block(y: np.ndarray, p: np.ndarray, train_major: float,
                  train_const: np.ndarray) -> dict:
    """Compute all metrics for one (y, prob) pair, plus the two baselines."""
    yhat = (p >= 0.5).astype(int)
    two = len(np.unique(y)) > 1   # AUC/MCC need both classes present
    return {
        "accuracy": accuracy_score(y, yhat),
        "auc": roc_auc_score(y, p) if two else np.nan,
        "f1": f1_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat) if two else np.nan,
        "baseline_test": float(max(y.mean(), 1 - y.mean())),
        "baseline_train": float(np.mean(y == np.asarray(train_const))),
        "train_major": train_major,
        "n": int(len(y)),
    }


def metrics_by_foldblocked(preds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-fold metrics averaged over folds the primary specification.

    AUC is a ranking statistic, so computing it over the pooled predictions of
    several separately fitted folds estimates a quantity belonging to no fitted
    model, and is biased downwards when the target base rate drifts between
    folds.Folds are weighted equally because the test windows are near-identical in
    length.
    """
    per_fold = metrics_by(preds, list(keys) + ["fold"])
    metric_cols = [c for c in per_fold.columns
                   if c not in set(keys) | {"fold", "n"}]
    agg = {c: (c, "mean") for c in metric_cols}
    agg["auc_fold_sd"] = ("auc", "std")
    agg["n_folds"] = ("fold", "nunique")
    agg["n"] = ("n", "sum")
    return per_fold.groupby(list(keys), as_index=False).agg(**agg)


def metrics_by(preds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """ Use metrics_by_foldblocked for
    cell-level reporting; this pools across any folds present in `preds`."""
    out = []
    for k, g in preds.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        m = _metric_block(g["y_true"].to_numpy(), g["y_prob"].to_numpy(),
                          float(g["train_major"].mean()),
                          g["train_const"].to_numpy())
        out.append(dict(zip(keys, k), **m))
    return pd.DataFrame(out)


def collapse_lstm_seeds(per_seed: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and SD over LSTM seeds; other models pass through untouched.

    Non-LSTM rows have seed=0 only, so their SD is NaN - which is fine."""
    num = ["accuracy", "auc", "f1", "precision", "recall", "mcc",
           "baseline_test", "baseline_train", "n"]
    grp = per_seed.groupby(keys, sort=False, observed=True)
    mean = grp[num].mean()
    sd = grp[num].std(ddof=1).add_suffix("_sd")
    n_seeds = grp.size().rename("n_seeds")
    return pd.concat([mean, sd, n_seeds], axis=1).reset_index()


# =========================================================================== #
# INFERENCE                                                                   #
# =========================================================================== #
def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def nemenyi_cd(k: int, n: int) -> float:
    """Critical difference for the Nemenyi post-hoc test."""
    q = _Q05.get(k)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n))) if q else np.nan


def kendalls_w(chi2: float, n_blocks: int, k_treatments: int) -> float:
    """Kendall's coefficient of concordance, W = chi2 / (N (k - 1)).

    The Friedman statistic answers if the sector differences systematically;
    W answers how strongly do the blocks agree on the ordering 
    Reporting both separates statistical detectability from magnitude
    because the sector spread is around one AUC point.

    W is descriptive rather than inferential in this design as the expanding windows share training data and all models see
    identical fold rows. 
    """
    if n_blocks <= 0 or k_treatments <= 1:
        return float("nan")
    return float(chi2 / (n_blocks * (k_treatments - 1)))


def rq1_sector_heterogeneity(fold_perf: pd.DataFrame, pooled: pd.DataFrame,
                             metric: str = METRIC) -> dict:
    """Primary test uses per-fold cells so the block count is adequate."""
    # Wide format: rows are (model, fold) blocks, columns are sector treatments.
    wide = (fold_perf.pivot_table(index=["model", "fold"], columns=COL_SECTOR,
                                  values=metric, observed=True)
                     .dropna(axis=0, how="any").dropna(axis=1, how="any"))
    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])

    # v1's specification, kept for comparability
    wide_v1 = pooled.pivot(index="model", columns=COL_SECTOR,
                           values=metric).dropna(axis=1)
    stat_v1, p_v1 = stats.friedmanchisquare(*[wide_v1[c].to_numpy()
                                              for c in wide_v1.columns])
    means = pooled.groupby(COL_SECTOR, observed=True)[metric].mean().sort_values()

    # --- effect size, not just significance -------------------------------- #
    # A p-value says the sector differences are systematic;   
    # complementary magnitudes are reported alongside it.
    w = kendalls_w(float(stat), int(wide.shape[0]), int(wide.shape[1]))
    w_v1 = kendalls_w(float(stat_v1), int(wide_v1.shape[0]), int(wide_v1.shape[1]))
    sd_between = float(means.std(ddof=1))     # dispersion around the grand mean
    return {"test": "Friedman, sectors as treatments, model x fold as blocks",
            "chi2": float(stat), "p": float(p),
            "k_treatments": int(wide.shape[1]), "n_blocks": int(wide.shape[0]),
            "chi2_pooled_blocks": float(stat_v1), "p_pooled_blocks": float(p_v1),
            "n_blocks_pooled": int(wide_v1.shape[0]),
            "significant": bool(p < ALPHA),
            "sector_mean_spread": float(means.max() - means.min()),
            "sector_mean_sd": sd_between,
            "kendall_w": w, "kendall_w_pooled_blocks": w_v1,
            "sector_means": means.to_dict()}


def rq2_complexity(pooled: pd.DataFrame, sci: pd.DataFrame,
                   metric: str = METRIC) -> dict:
    """RQ2: does SCI predict forecasting performance?"""
    perf = pooled.groupby(COL_SECTOR, observed=True)[metric].mean()
    d = sci[["SCI", "band"]].join(perf.rename("perf")).dropna(subset=["perf"])
    rho, p = stats.spearmanr(d["SCI"], d["perf"])

    # Band comparison, does low-complexity beat high-complexity?
    lo = d.loc[d["band"] == "low", "perf"]
    hi = d.loc[d["band"] == "high", "perf"]
    u_h2 = stats.mannwhitneyu(lo, hi, alternative="greater")   # H2's direction
    u_two = stats.mannwhitneyu(lo, hi, alternative="two-sided")

    # Per-model Spearman, Holm-corrected across the family of four.
    per_model, names = [], []
    for m, g in pooled.groupby("model", observed=True):
        dd = sci[["SCI"]].join(g.set_index(COL_SECTOR)[metric]).dropna()
        r, pp = stats.spearmanr(dd["SCI"], dd[metric])
        per_model.append((float(r), float(pp)))
        names.append(m)
    praw = np.array([pp for _, pp in per_model])
    padj = holm(praw)
    table = pd.DataFrame({"model": names,
                          "rho": [r for r, _ in per_model],
                          "p": praw, "p_holm": padj}).sort_values("rho", ascending=False)

    return {"test": "Spearman(SCI, mean AUC) across sectors",
            "rho": float(rho), "p": float(p),
            "negative_as_H2_predicts": bool(rho < 0),
            "significant": bool(p < ALPHA),
            "band_means": d.groupby("band", observed=True)["perf"].mean().to_dict(),
            "mannwhitney_U_H2": float(u_h2.statistic), "mannwhitney_p_H2": float(u_h2.pvalue),
            "mannwhitney_p_two_sided": float(u_two.pvalue),
            "per_model": table.to_dict("records"),
            "table": d.sort_values("SCI")}


def rq2_metric_decomposition(pooled: pd.DataFrame, sci: pd.DataFrame,
                             metric: str = METRIC) -> pd.DataFrame:
    """Relate each SCI component to performance separately.

    The composite could in principle conceal components that push in opposite
    directions like entropy in predictability. Correlating each of the five metrics with per-sector mean AUC
    on its own distinguishes those two readings.

    Metrics are signed as they enter Hurst is negated, so a coefficient here has the same interpretation.  Holm adjustment runs across the five tests.
    """
    perf = pooled.groupby(COL_SECTOR, observed=True)[metric].mean()
    rows = []
    for m, d in SCI_DIRECTION.items():
        dd = sci[[m]].join(perf.rename("perf")).dropna()
        if len(dd) < 3:
            continue
        signed = d * dd[m]
        rho, p = stats.spearmanr(signed, dd["perf"])
        rho_raw, _ = stats.spearmanr(dd[m], dd["perf"])
        rows.append({"metric": m, "direction": d,
                     "rho_signed": float(rho), "rho_raw": float(rho_raw),
                     "p": float(p), "n": int(len(dd))})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
        out["supports_H2"] = out["rho_signed"] < 0
        out = out.sort_values("rho_signed")
    return out


def sci_pca_weighted(sci: pd.DataFrame) -> pd.DataFrame:
    """
    Equal weighting is the primary specification as the
    five metrics are treated as reflective indicators of one construct rather
    than as competing predictors.  
    """
    metrics = list(SCI_DIRECTION)
    z = ((sci[metrics] - sci[metrics].mean())
         / sci[metrics].std(ddof=SCI_ZSCORE_DDOF))
    z = z.mul(pd.Series(SCI_DIRECTION))          # orient before extracting
    X = z.to_numpy(dtype=float)
    # PC1 by SVD on the (already centred, already scaled) matrix.
    U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    load = Vt[0]
    # Sign convention: make PC1 point the same way as the equal-weight index,
    # otherwise the correlation's sign is an arbitrary artefact of the SVD.
    score = X @ load
    if np.corrcoef(score, sci["SCI"].to_numpy())[0, 1] < 0:
        load, score = -load, -score
    out = pd.DataFrame({"SCI_pca": score}, index=sci.index)
    out["band_pca"] = pd.qcut(out["SCI_pca"], N_BANDS,
                              labels=["low", "medium", "high"])
    var_explained = float(S[0] ** 2 / np.sum(S ** 2))
    tau, tau_p = stats.kendalltau(sci["SCI"], out["SCI_pca"])
    out.attrs["var_explained"] = var_explained
    out.attrs["kendall_tau_vs_equal"] = float(tau)
    out.attrs["kendall_tau_p"] = float(tau_p)
    out.attrs["loadings"] = dict(zip(metrics, load.round(4)))
    return out


def rq3_models(pooled: pd.DataFrame, metric: str = METRIC) -> dict:
    """RQ3: do the models differ?  Friedman + Nemenyi if k >= 3, else Wilcoxon."""
    wide = pooled.pivot(index=COL_SECTOR, columns="model", values=metric).dropna()
    models = list(wide.columns)
    k, n = len(models), len(wide)
    if k < 3:
        stat, p = stats.wilcoxon(wide[models[0]], wide[models[1]])
        test = "Wilcoxon signed-rank"
        cd = np.nan
    else:
        stat, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in models])
        test = "Friedman + Nemenyi"
        cd = nemenyi_cd(k, n)
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    return {"test": test, "chi2": float(stat), "p": float(p),
            "significant": bool(p < ALPHA),
            "avg_ranks": ranks.to_dict(),
            "max_rank_gap": float(ranks.max() - ranks.min()),
            "critical_difference": float(cd)}


def mcnemar_all_pairs(preds: pd.DataFrame) -> pd.DataFrame:
    """Every model pair in every sector, on identical held-out observations."""
    p = preds.copy()
    p["correct"] = ((p["y_prob"] >= 0.5).astype(int) == p["y_true"]).astype(int)
    # collapse LSTM seeds to the seed-averaged probability for a single decision
    key = [COL_SECTOR, COL_DATE, COL_TICKER, "model"]
    p = (p.groupby(key, sort=False)
          .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
          .reset_index())
    p["correct"] = ((p["y_prob"] >= 0.5).astype(int) == p["y_true"]).astype(int)

    rows = []
    for sector, g in p.groupby(COL_SECTOR, sort=False):
        wide = g.pivot_table(index=[COL_DATE, COL_TICKER], columns="model",
                             values="correct")
        wide = wide.dropna()
        models = list(wide.columns)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b_ = wide[models[i]].to_numpy(), wide[models[j]].to_numpy()
                # McNemar's b and c - the discordant cells
                b = int(np.sum((a == 1) & (b_ == 0)))
                c = int(np.sum((a == 0) & (b_ == 1)))
                if b + c == 0:
                    pv = 1.0
                elif b + c < 25:
                    # exact binomial for small samples
                    pv = float(stats.binomtest(min(b, c), b + c, 0.5,
                                               alternative="two-sided").pvalue)
                else:
                    # Edwards' continuity-corrected chi-square
                    chi = (abs(b - c) - 1) ** 2 / (b + c)
                    pv = float(stats.chi2.sf(chi, 1))
                rows.append({"sector": sector, "model_a": models[i],
                             "model_b": models[j], "n_pairs": int(len(wide)),
                             "b": b, "c": c, "p": pv})
    out = pd.DataFrame(rows)
    # Holm correction across the whole family of pairwise tests.
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
        out["significant_holm"] = out["p_holm"] < ALPHA
    return out.sort_values("p")


def date_block_bootstrap(preds: pd.DataFrame, sci: pd.DataFrame,
                         B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict:
    """Resample whole trading DATES within fold, not rows.

    Dates are the resampling unit because several hundred
    tickers share each date and their errors are driven by a common market
    factor; folds are respected because each fold is a distinct fitted model and
    a pooled ranking across folds is biased (Section 3.7.3).

    Roughly five hundred tickers share each date, so their prediction errors
    are driven by a common market factor. Resampling dates preserves the cross-sectional dependence.
    """
    rng = np.random.default_rng(seed)
    # collapse LSTM seeds per (sector, date, ticker, model, fold)
    p = (preds.groupby([COL_SECTOR, COL_DATE, COL_TICKER, "model", "fold"],
                       sort=False)
              .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
              .reset_index())

    # Dates are factorised WITHIN FOLD, because resampling is within fold: a
    # replicate must never rank one fold's predictions against another's.
    
    blocks: dict[tuple, list] = {}
    fold_ndates: dict[int, int] = {}
    for f in np.sort(p["fold"].unique()):
        pf = p[p["fold"] == f]
        codes_all, dates_all = pd.factorize(pf[COL_DATE], sort=True)
        fold_ndates[f] = len(dates_all)
        pf = pf.assign(_dcode=codes_all)
        for (sector, model), g in pf.groupby([COL_SECTOR, "model"], sort=False):
            g = g.sort_values("_dcode", kind="stable")
            codes = g["_dcode"].to_numpy()
            uniq = np.unique(codes)
            # pre-compute the slice boundaries for every date - hot path
            blocks.setdefault((sector, model), []).append({
                "fold": f,
                "y": g["y_true"].to_numpy(),
                "p": g["y_prob"].to_numpy(),
                "starts": np.searchsorted(codes, uniq, side="left"),
                "ends": np.searchsorted(codes, uniq, side="right"),
                "local": uniq})

    draws: dict[tuple, list] = {k: [] for k in blocks}
    band = sci["band"].astype(str).to_dict()
    rho_draws, band_draws, margin_draws = [], [], []
    
    sector_draws, spread_draws = [], []

    for _ in range(B):
        # One date resample per fold, shared by every cell in the replicate, so
        # the cross-sectional dependence among stocks sharing a date survives.
        picks = {f: np.bincount(rng.integers(0, n, n), minlength=n)
                 for f, n in fold_ndates.items()}
        cell_auc = {}
        for k, fold_list in blocks.items():
            per = []
            for c in fold_list:
                reps = picks[c["fold"]][c["local"]]
                take = np.nonzero(reps)[0]
                if take.size == 0:
                    continue
                # expand the date repeats into row indices
                idx = np.concatenate([
                    np.repeat(np.arange(c["starts"][i], c["ends"][i]), reps[i])
                    for i in take])
                y, pr = c["y"][idx], c["p"][idx]
                if len(np.unique(y)) > 1:
                    per.append(roc_auc_score(y, pr))
            a = float(np.mean(per)) if per else np.nan
            cell_auc[k] = a
            draws[k].append(a)

        # Per-replicate RQ2 statistic.
        df = (pd.Series(cell_auc).rename_axis([COL_SECTOR, "model"])
                .reset_index(name="auc"))
        sec_mean = df.groupby(COL_SECTOR)["auc"].mean()
        sector_draws.append(sec_mean)
        if sec_mean.notna().sum() > 1:
            
            spread_draws.append(float(sec_mean.max() - sec_mean.min()))
        common = [s for s in sec_mean.index if s in sci.index]
        if len(common) > 2:
            r, _ = stats.spearmanr(sci.loc[common, "SCI"], sec_mean.loc[common])
            rho_draws.append(r)

        # Band-level means + the LSTM-minus-Logistic margin per replicate.
        df["band"] = df[COL_SECTOR].map(band)
        bm = df.groupby(["band", "model"])["auc"].mean()
        band_draws.append(bm)
        if {"LSTM", "LogisticRegression"} <= set(df["model"]):
            margin_draws.append(bm.xs("LSTM", level="model")
                                - bm.xs("LogisticRegression", level="model"))

    def ci(v):
        """95% percentile interval from a vector of bootstrap draws."""
        v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
        if v.size < 10:
            return (np.nan, np.nan)
        return tuple(np.percentile(v, [2.5, 97.5]))

    # Assemble the cell-level CI table.
    cell_rows = []
    for (sector, model), v in draws.items():
        lo, hi = ci(v)
        v = np.asarray([x for x in v if np.isfinite(x)])
        cell_rows.append({"sector": sector, "model": model,
                          "auc_boot_mean": float(np.mean(v)) if v.size else np.nan,
                          "ci_lo": lo, "ci_hi": hi,
                          # two-sided p = 2 * min(P(X<=0.5), P(X>=0.5))
                          "p_two_sided_vs_half": float(2 * min((v <= 0.5).mean(),
                                                               (v >= 0.5).mean()))
                          if v.size else np.nan,
                          "excludes_half": bool(np.isfinite(lo) and (lo > 0.5 or hi < 0.5))})
    cell_df = pd.DataFrame(cell_rows)
    if len(cell_df):
        cell_df["p_holm"] = holm(cell_df["p_two_sided_vs_half"].fillna(1).to_numpy())

    # Band-level CIs.
    band_df = pd.DataFrame()
    if band_draws:
        stacked = pd.concat(band_draws, axis=1)
        band_df = pd.DataFrame({"mean": stacked.mean(axis=1),
                                "ci_lo": stacked.quantile(.025, axis=1),
                                "ci_hi": stacked.quantile(.975, axis=1)}).reset_index()

    # LSTM-minus-Logistic margin CIs (per band).
    margin_df = pd.DataFrame()
    if margin_draws:
        st = pd.concat(margin_draws, axis=1)
        margin_df = pd.DataFrame({"mean": st.mean(axis=1),
                                  "ci_lo": st.quantile(.025, axis=1),
                                  "ci_hi": st.quantile(.975, axis=1)}).reset_index()

    # Sector-mean CIs, for the heterogeneity table.
    sector_df = pd.DataFrame()
    if sector_draws:
        st = pd.concat(sector_draws, axis=1)
        sector_df = pd.DataFrame({
            "auc_boot_mean": st.mean(axis=1),
            "ci_lo": st.quantile(.025, axis=1),
            "ci_hi": st.quantile(.975, axis=1)}).rename_axis("sector").reset_index()
        sector_df["excludes_half"] = ((sector_df["ci_lo"] > 0.5)
                                      | (sector_df["ci_hi"] < 0.5))

    rho_ci = ci(rho_draws)
    return {"cells": cell_df, "bands": band_df, "margins": margin_df,
            "sectors": sector_df, "spread_ci": ci(spread_draws),
            "rho_ci": rho_ci, "B": B}


def adaptive_selection(fold_perf: pd.DataFrame, pooled: pd.DataFrame,
                       sci: pd.DataFrame, metric: str = METRIC) -> dict:
    """Band-to-model policy, fitted on early folds and scored on late folds."""
    fp = fold_perf.merge(sci["band"].rename_axis(COL_SECTOR).reset_index(),
                         on=COL_SECTOR)
    folds = sorted(fp["fold"].unique())
    # Hold out the last two folds for out-of-sample scoring.
    cut = max(1, len(folds) - 2)
    fit_folds, score_folds = folds[:cut], folds[cut:]

    # Pick the best model per band on the fit folds.
    fit = fp[fp["fold"].isin(fit_folds)]
    choice = (fit.groupby(["band", "model"], observed=True)[metric].mean()
                 .reset_index().sort_values(metric, ascending=False)
                 .drop_duplicates("band").set_index("band"))

    # Score that policy on the held-out folds.
    score = fp[fp["fold"].isin(score_folds)]
    sel = []
    for sector, g in score.groupby(COL_SECTOR, observed=True):
        b = str(sci.loc[sector, "band"])
        if b not in choice.index:
            continue
        m = choice.loc[b, "model"]
        v = g.loc[g["model"] == m, metric]
        if len(v):
            sel.append(v.mean())
    oos_adaptive = float(np.nanmean(sel)) if sel else np.nan
    fixed = score.groupby("model", observed=True)[metric].mean()
    oos_best_fixed_model = fixed.idxmax()
    oos_best_fixed = float(fixed.max())

    # in-sample version reported explicitly as an upper bound
    wide = (fit.groupby([COL_SECTOR, "model"], observed=True)[metric]
               .mean()                      # fold-blocked: mean over fit folds
               .unstack("model"))
    in_sel = [wide.loc[s, choice.loc[str(sci.loc[s, "band"]), "model"]]
              for s in wide.index if str(sci.loc[s, "band"]) in choice.index]
    return {"fit_folds": fit_folds, "score_folds": score_folds,
            "band_to_model": choice[["model", metric]].to_dict("index"),
            "oos_adaptive_mean": oos_adaptive,
            "oos_best_fixed_model": str(oos_best_fixed_model),
            "oos_best_fixed_mean": oos_best_fixed,
            "oos_uplift": oos_adaptive - oos_best_fixed,
            "in_sample_adaptive_mean": float(np.nanmean(in_sel)),
            "in_sample_best_fixed_mean": float(np.nanmax(wide.mean().values)),
            "in_sample_best_fixed_model": str(wide.mean().idxmax()),
            "per_sector_oracle_mean": float(wide.max(axis=1).mean())}


# =========================================================================== #
# FEATURE ATTRIBUTION (SHAP)                                                  #
# =========================================================================== #
# Grouping used to summarise the thirteen features into interpretable blocks.

FEATURE_GROUPS = {
    "lagged returns": lambda c: c.startswith("ret_lag"),
    "trend (MA ratios)": lambda c: c.startswith("close_ma"),
    "momentum": lambda c: c.startswith("mom"),
    "volatility": lambda c: c.startswith("vol_") and "chg" not in c
                            and "ma_ratio" not in c,
    "RSI": lambda c: c == "rsi",
    "volume": lambda c: c in ("vol_chg", "vol_ma_ratio"),
}


def _feature_group(col: str) -> str:
    for name, test in FEATURE_GROUPS.items():
        if test(col):
            return name
    return "other"


def shap_analysis(feat: pd.DataFrame, sci: pd.DataFrame,
                  model_name: str = SHAP_MODEL,
                  max_rows: int = SHAP_MAX_ROWS, seed: int = RANDOM_STATE):
    """Per-sector SHAP attribution for the tree ensemble.

    
    The model is refitted on the final fold's training window, which is the
    state a deployed system would be in, and explained on that fold's test
    rows.  TreeExplainer is exact for gradient-boosted trees, so no sampling
    approximation enters the attribution itself; only the row subsample does.

    Returns (per-feature table, per-group table).  Falls back to the model's
    own split-gain importances if the shap package is unavailable, so the
    artefact is always produced -- the fallback is flagged in the output.
    """
    cols = feature_columns(feat)
    rng = np.random.default_rng(seed)
    rows, method = [], "shap-tree"

    for sector, fs in feat.groupby(COL_SECTOR, sort=False):
        fs = fs.sort_values([COL_TICKER, COL_DATE])
        dates = np.sort(fs[COL_DATE].unique())
        splits = list(TimeSeriesSplit(n_splits=N_SPLITS).split(dates))
        if not splits:
            continue
        tr_idx, te_idx = splits[-1]                  # final fold = latest state
        train = fs[fs[COL_DATE].isin(dates[tr_idx])]
        test = fs[fs[COL_DATE].isin(dates[te_idx])]
        if train["target"].nunique() < 2 or not len(test):
            continue

        scaler = StandardScaler().fit(train[cols])
        Xtr = scaler.transform(train[cols])
        clf = make_sklearn(model_name, seed=RANDOM_STATE)
        clf.fit(Xtr, train["target"])

        # Subsample test rows TreeExplainer is exact but linear in rows, and
        # 20k rows already pins a mean |SHAP| to three decimal places.
        Xte = scaler.transform(test[cols])
        if len(Xte) > max_rows:
            Xte = Xte[rng.choice(len(Xte), max_rows, replace=False)]

        if _HAS_SHAP:
            try:
                expl = shap.TreeExplainer(clf)
                sv = expl.shap_values(Xte)
                if isinstance(sv, list):          # older API: one array per class
                    sv = sv[-1]
                sv = np.asarray(sv)
                if sv.ndim == 3:                  # (rows, features, classes)
                    sv = sv[:, :, -1]
                imp = np.abs(sv).mean(axis=0)
            except Exception as e:                # noqa: BLE001 - degrade, don't die
                note(f"    [warn] SHAP failed for {sector} ({e}); using gain")
                imp = np.asarray(getattr(clf, "feature_importances_",
                                         np.full(len(cols), np.nan)), dtype=float)
                method = "gain-fallback"
        else:
            imp = np.asarray(getattr(clf, "feature_importances_",
                                     np.full(len(cols), np.nan)), dtype=float)
            method = "gain-fallback"

        total = float(np.nansum(imp)) or np.nan
        for c, v in zip(cols, imp):
            rows.append({"sector": sector, "feature": c,
                         "group": _feature_group(c),
                         "mean_abs_shap": float(v),
                         "share": float(v) / total,
                         "n_explained": int(len(Xte))})

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail
    detail.attrs["method"] = method
    detail.attrs["model"] = model_name

    groups = (detail.groupby(["sector", "group"], as_index=False)["share"].sum()
                    .pivot(index="sector", columns="group", values="share")
                    .fillna(0.0))
    groups = groups.reindex(sci.index.intersection(groups.index))
    # sci.index is named 'Sector' and groups' is 'sector', so the intersection
    # drops the name.  
    groups.index.name = "sector"
    groups.attrs["method"] = method
    groups.attrs["model"] = model_name
    return detail, groups


def shap_vs_complexity(groups: pd.DataFrame, sci: pd.DataFrame) -> pd.DataFrame:
    """

    H3 rests on the premise that sectors differ in the structure they afford.
    If that premise holds, the share of attribution a model places on, say,
    volume or volatility inputs should track the index even where the
    resulting AUC does not. Holm-adjusted across the feature groups.
    """
    common = sci.index.intersection(groups.index)
    if len(common) < 3:
        return pd.DataFrame()
    rows = []
    for g in groups.columns:
        rho, p = stats.spearmanr(sci.loc[common, "SCI"], groups.loc[common, g])
        rows.append({"group": g, "rho": float(rho), "p": float(p)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    return out.sort_values("rho", ascending=False)


# =========================================================================== #
# GLOBAL (ALL-SECTOR) FIT vs THE PER-SECTOR FITS                              #
# =========================================================================== #
def global_vs_sector(feat: pd.DataFrame, pooled: pd.DataFrame,
                     models: list[str], metric: str = METRIC):
    """Fit one model on the whole panel and score it by sector, for comparison
    with the per-sector models used throughout.

    The main analysis trains a separate model inside every sector, so any
    sector difference it finds is a difference in what each sector's own data
    affords rather than a difference in how one shared model happens to behave
    there.  This check quantifies what that separation buys.  If sector-specific
    training adds nothing over a single pooled model, the sectors are not
    offering materially different learnable structure. The comparison is on identical folds, features and
    evaluation rows, so the only thing that varies is the training partition.

    
    """
    tab = [m for m in models if m != "LSTM"]
    if GLOBAL_BASELINE_MODELS:
        tab = [m for m in tab if m in GLOBAL_BASELINE_MODELS]
    if not tab:
        return pd.DataFrame(), pd.DataFrame()
    note(f"    fitting global (all-sector) models: {', '.join(tab)}")
    gp = run_sector(feat, tab, n_seeds_lstm=0)
    if not len(gp):
        return pd.DataFrame(), pd.DataFrame()
    gp["seed"] = 0
    glob = metrics_by_foldblocked(gp, ["model", COL_SECTOR])
    glob = glob.rename(columns={metric: "auc_global"})

    per = pooled[pooled["model"].isin(tab)][["model", COL_SECTOR, metric]]
    cmp_ = per.rename(columns={metric: "auc_sector"}).merge(
        glob[["model", COL_SECTOR, "auc_global"]], on=["model", COL_SECTOR])
    cmp_["gain_from_sector_fit"] = cmp_["auc_sector"] - cmp_["auc_global"]
    by_sector = (cmp_.groupby(COL_SECTOR, as_index=False)
                     .agg(auc_sector=("auc_sector", "mean"),
                          auc_global=("auc_global", "mean"),
                          gain=("gain_from_sector_fit", "mean"))
                     .sort_values("gain", ascending=False))
    return cmp_, by_sector


# =========================================================================== #
# ROBUSTNESS                                                                  #
# =========================================================================== #
def robustness_per_ticker(feat: pd.DataFrame, sci: pd.DataFrame, split: float = 0.7):
    """One model per ticker on a single 70/30 time split, aggregated to sector.

    MIN_HISTORY_SCI is applied here to the POST-dropna feature frame, which is about twenty-five rows
    shorter per ticker, so a ticker with 250-275 raw days qualifies for the
    index but not for this check.  That is why n can be one lower than the SCI
    column. Both counts are reported.
    """
    cols = feature_columns(feat)
    rows = []
    for (sec, tk), g in feat.groupby([COL_SECTOR, COL_TICKER], sort=False):
        g = g.sort_values(COL_DATE)
        if len(g) < MIN_HISTORY_SCI:
            continue
        cut = int(len(g) * split)
        tr, te = g.iloc[:cut], g.iloc[cut:]
        if tr["target"].nunique() < 2 or te["target"].nunique() < 2:
            # can't fit / score if either split is single-class
            continue
        sc = StandardScaler().fit(tr[cols])
        Xtr, Xte = sc.transform(tr[cols]), sc.transform(te[cols])
        for name, clf in [("LR", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                          ("RF", RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                        random_state=RANDOM_STATE))]:
            clf.fit(Xtr, tr["target"])
            pr = clf.predict_proba(Xte)[:, 1]
            rows.append({"sector": sec, "ticker": tk, "model": name,
                         "acc": accuracy_score(te["target"], (pr >= 0.5).astype(int)),
                         "auc": roc_auc_score(te["target"], pr)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail
    summary = (detail.groupby("sector")
                     .agg(n_tickers=("ticker", "nunique"),
                          perticker_auc=("auc", "mean"),
                          perticker_acc=("acc", "mean"),
                          pct_auc_gt_50=("auc", lambda s: (s > 0.5).mean() * 100))
                     .join(sci["band"]).sort_values("perticker_auc"))
    return detail, summary


# =========================================================================== #
# LATEX EMISSION                                                              #
# =========================================================================== #
def _fmt(v, nd=4, signed=False):
    """Format a number for LaTeX.  Returns '--' for missing/inf."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return f"${s}$" if signed else s


def write_tables(out: str, sci: pd.DataFrame, pooled: pd.DataFrame,
                 boot: dict, mcn: pd.DataFrame, seeds: pd.DataFrame,
                 rob_summary: pd.DataFrame, sci_flip: pd.DataFrame) -> None:
    """
    Each table is a standalone fragment"""
    os.makedirs(out, exist_ok=True)

    # --- Table: SCI and components ----------------------------------------
    lines = [r"\begin{tabular}{lrrrrrrrl}", r"\toprule",
             r"Sector & Vol. & SampEn & Hurst & Amihud & Vol.\ var. & $n$ & SCI & Band \\",
             r"\midrule"]
    for s, r in sci.iterrows():
        # Amihud scaled by 1e4 so the columns line up nicely
        lines.append(f"{s} & {r.volatility:.3f} & {r.sampen:.3f} & {r.hurst:.3f} & "
                     f"{r.amihud * 1e4:.3f} & {r.vol_var:.3f} & {int(r.n_tickers)} & "
                     f"${r.SCI:+.3f}$ & {r.band} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_sci.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: mean performance by model ---------------------------------
    agg = (pooled.groupby("model", observed=True)[["accuracy", "auc", "f1", "mcc"]]
                 .mean().sort_values("auc", ascending=False))
    ranks = (pooled.pivot(index=COL_SECTOR, columns="model", values="auc")
                   .rank(axis=1, ascending=False).mean())
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & Accuracy & AUC & F1 & MCC & Mean rank \\", r"\midrule"]
    for m, r in agg.iterrows():
        lines.append(f"{m} & {r.accuracy:.4f} & {r.auc:.4f} & {r.f1:.4f} & "
                     f"${r.mcc:+.4f}$ & {ranks[m]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_perf_model.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: bootstrap CIs (longtable) --------------------------------
    cells = boot["cells"].merge(pooled[[COL_SECTOR, "model", "auc"]]
                               .rename(columns={COL_SECTOR: "sector"}),
                               on=["sector", "model"], how="left")
    cells = cells.sort_values("p_two_sided_vs_half")
    head = (r"Sector & Model & AUC & 95\% CI & $p_{\text{Holm}}$ & Verdict \\")
    lines = [r"\begin{longtable}{llrrrl}",
             r"\caption{Date-block bootstrap 95\% confidence intervals for the "
             r"out-of-sample AUC of every model--sector combination, with "
             r"Holm-adjusted two-sided $p$-values against an AUC of 0.5. Whole "
             r"trading dates are resampled, so cross-sectional dependence between "
             r"tickers is preserved.}\label{tab:ci}\\",
             r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\toprule", head, r"\midrule", r"\endhead",
             r"\midrule \multicolumn{6}{r}{\emph{continued on next page}}\\",
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for _, r in cells.iterrows():
        # verdict string for the reader
        verdict = ("above chance" if r.ci_lo > 0.5 else
                   "below chance" if r.ci_hi < 0.5 else "indistinguishable")
        lines.append(f"{r.sector} & {r.model} & {r.auc:.4f} & "
                     f"$[{r.ci_lo:.4f},\\,{r.ci_hi:.4f}]$ & {r.p_holm:.3f} & {verdict} \\\\")
    lines.append(r"\end{longtable}")
    open(f"{out}/tab_ci.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: full per-model per-sector results (longtable) ------------
    band_order = {"low": 0, "medium": 1, "high": 2}
    full = pooled.copy()
    full["band"] = full[COL_SECTOR].map(sci["band"].astype(str).to_dict())
    full["_b"] = full["band"].map(band_order)
    full = full.sort_values(["_b", COL_SECTOR, "model"])
    head = (r"Sector & Model & Acc. & AUC & F1 & MCC & Base (test) & Base (train) \\")
    lines = [r"\begin{longtable}{llrrrrrr}",
             r"\caption{Out-of-sample performance for all model--sector "
             r"combinations. Baseline (test) is the hindsight majority class; "
             r"baseline (train) applies the training-partition majority to the "
             r"test partition. LSTM rows are means across seeds.}\label{tab:full}\\",
             r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\toprule", head, r"\midrule", r"\endhead",
             r"\midrule \multicolumn{8}{r}{\emph{continued on next page}}\\",
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for _, r in full.iterrows():
        lines.append(f"{r[COL_SECTOR]} & {r['model']} & {r.accuracy:.4f} & {r.auc:.4f} & "
                     f"{r.f1:.4f} & ${r.mcc:+.4f}$ & {r.baseline_test:.4f} & "
                     f"{r.baseline_train:.4f} \\\\")
    lines.append(r"\end{longtable}")
    open(f"{out}/tab_perf_full.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: mean AUC by model x band, with bootstrap margin row ------
    pb = pooled.merge(sci["band"].rename_axis(COL_SECTOR).reset_index(), on=COL_SECTOR)
    bt = pb.pivot_table(index="model", columns="band", values="auc",
                        aggfunc="mean", observed=True)
    bands = [b for b in ["low", "medium", "high"] if b in bt.columns]
    bt = bt[bands]
    lines = [r"\begin{tabular}{l" + "r" * len(bands) + "}", r"\toprule",
             "Model & " + " & ".join(b.capitalize() for b in bands) + r" \\",
             r"\midrule"]
    for m in bt.index:
        lines.append(m + " & " + " & ".join(f"{bt.loc[m, b]:.4f}" for b in bands) + r" \\")
    # Add the LSTM-minus-Logistic margin row plus its bootstrap CI.
    if {"LSTM", "LogisticRegression"}.issubset(set(bt.index)):
        lines.append(r"\midrule")
        marg = bt.loc["LSTM"] - bt.loc["LogisticRegression"]
        lines.append(r"LSTM $-$ Logistic & " +
                     " & ".join(f"${marg[b]:+.4f}$" for b in bands) + r" \\")
        if len(boot.get("margins", pd.DataFrame())):
            mi = boot["margins"].set_index("band")
            cells_ci = []
            for b in bands:
                if b in mi.index:
                    cells_ci.append(f"$[{mi.loc[b, 'ci_lo']:+.4f},\\,{mi.loc[b, 'ci_hi']:+.4f}]$")
                else:
                    cells_ci.append("--")
            lines.append(r"\quad 95\% CI & " + " & ".join(cells_ci) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_band.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: LSTM seed stability --------------------------------------
    if len(seeds):
        lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
                 r"Sector & Mean AUC & SD & Min & Max \\", r"\midrule"]
        for s, g in seeds.groupby(COL_SECTOR, observed=True):
            lines.append(f"{s} & {g.auc.mean():.4f} & {g.auc.std(ddof=1):.4f} & "
                         f"{g.auc.min():.4f} & {g.auc.max():.4f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{out}/tab_lstm_seeds.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: McNemar, every pair in every sector (longtable) ---------
    if len(mcn):
        head = (r"Sector & Model A & Model B & $b$ & $c$ & $p$ & "
                r"$p_{\text{Holm}}$ \\")
        lines = [r"\begin{longtable}{lllrrrr}",
                 r"\caption{McNemar tests of paired predictions for every model "
                 r"pair in every sector, on the common evaluation support. "
                 r"$b$ and $c$ are the discordant counts; $p$-values are "
                 r"Holm-adjusted across the whole family.}\label{tab:mcnemar}\\",
                 r"\toprule", head, r"\midrule", r"\endfirsthead",
                 r"\toprule", head, r"\midrule", r"\endhead",
                 r"\midrule \multicolumn{7}{r}{\emph{continued on next page}}\\",
                 r"\endfoot", r"\bottomrule", r"\endlastfoot"]
        for _, r in mcn.iterrows():
            lines.append(f"{r.sector} & {r.model_a} & {r.model_b} & {int(r.b)} & "
                         f"{int(r.c)} & {r.p:.4f} & {r.p_holm:.4f} \\\\")
        lines.append(r"\end{longtable}")
        open(f"{out}/tab_mcnemar.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: per-ticker robustness -----------------------------------
    if len(rob_summary):
        lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
                 r"Sector & Band & $n$ & Per-ticker AUC & Per-ticker acc. & "
                 r"\% AUC $>0.5$ & Pooled AUC \\", r"\midrule"]
        pooled_auc = pooled.groupby(COL_SECTOR, observed=True)["auc"].mean()
        for s, r in rob_summary.iterrows():
            lines.append(f"{s} & {r.band} & {int(r.n_tickers)} & {r.perticker_auc:.4f} & "
                         f"{r.perticker_acc:.4f} & {r.pct_auc_gt_50:.1f} & "
                         f"{pooled_auc.get(s, float('nan')):.4f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{out}/tab_perticker.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: Hurst sign sensitivity ---------------------------------
    lines = [r"\begin{tabular}{lrrl}", r"\toprule",
             r"Sector & SCI ($d_{\text{Hurst}}=-1$) & SCI ($d_{\text{Hurst}}=+1$) & Band \\",
             r"\midrule"]
    for s, r in sci_flip.iterrows():
        lines.append(f"{s} & ${r.SCI:+.3f}$ & ${r.SCI_hurst_plus:+.3f}$ & {r.band} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_hurst_sensitivity.tex", "w").write("\n".join(lines) + "\n")


def write_extra_tables(out: str, sci: pd.DataFrame, rq1: dict, boot: dict,
                       decomp: pd.DataFrame, shap_groups: pd.DataFrame,
                       shap_corr: pd.DataFrame, gvs: pd.DataFrame,
                       sci_pca: pd.DataFrame) -> None:
    """
    Kept in a separate function from write_tables so the original artefacts
    are identical when predictions are reused.
    """
    # --- Table: sectoral heterogeneity, magnitude alongside significance ---
    sec = boot.get("sectors", pd.DataFrame())
    means = pd.Series(rq1["sector_means"]).sort_values(ascending=False)
    lines = [r"\begin{tabular}{lrrl}", r"\toprule",
             r"Sector & Mean AUC & 95\% CI & Band \\", r"\midrule"]
    band = sci["band"].astype(str).to_dict()
    ci_map = (sec.set_index("sector")[["ci_lo", "ci_hi"]].to_dict("index")
              if len(sec) else {})
    for s, v in means.items():
        c = ci_map.get(s)
        ci_s = (f"$[{c['ci_lo']:.4f},\\,{c['ci_hi']:.4f}]$" if c else "--")
        lines.append(f"{s} & {v:.4f} & {ci_s} & {band.get(s, '--')} \\\\")
    lines += [r"\midrule",
              f"Range (max $-$ min) & {rq1['sector_mean_spread']:.4f} & & \\\\",
              f"SD between sectors & {rq1['sector_mean_sd']:.4f} & & \\\\",
              f"Kendall's $W$ & {rq1['kendall_w']:.3f} & & \\\\",
              r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tab_heterogeneity.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: SCI component decomposition -------------------------------
    if len(decomp):
        label = {"volatility": "Volatility", "sampen": "Sample entropy",
                 "hurst": "Hurst exponent", "amihud": "Amihud illiquidity",
                 "vol_var": "Volume variability"}
        lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
                 r"Component & $d_k$ & Spearman $\rho$ & $p$ & $p_{\text{Holm}}$ \\",
                 r"\midrule"]
        for _, r in decomp.iterrows():
            lines.append(f"{label.get(r['metric'], r['metric'])} & "
                         f"${r['direction']:+d}$ & ${r['rho_signed']:+.3f}$ & "
                         f"{r['p']:.3f} & {r['p_holm']:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{out}/tab_sci_decomposition.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: SHAP attribution shares by sector and feature group -------
    if len(shap_groups):
        gcols = list(shap_groups.columns)
        lines = [r"\begin{tabular}{l" + "r" * len(gcols) + "}", r"\toprule",
                 "Sector & " + " & ".join(c.replace("%", r"\%") for c in gcols)
                 + r" \\", r"\midrule"]
        for s, r in shap_groups.iterrows():
            lines.append(f"{s} & " + " & ".join(f"{r[c]:.3f}" for c in gcols)
                         + r" \\")
        lines += [r"\midrule",
                  "Mean & " + " & ".join(f"{shap_groups[c].mean():.3f}"
                                         for c in gcols) + r" \\"]
        if len(shap_corr):
            cm = shap_corr.set_index("group")
            lines.append(r"Spearman vs SCI & " + " & ".join(
                (f"${cm.loc[c, 'rho']:+.3f}$" if c in cm.index else "--")
                for c in gcols) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{out}/tab_shap.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: per-sector fit vs single global fit -----------------------
    if len(gvs):
        lines = [r"\begin{tabular}{lrrr}", r"\toprule",
                 r"Sector & Per-sector AUC & Global AUC & Gain \\", r"\midrule"]
        for _, r in gvs.iterrows():
            lines.append(f"{r[COL_SECTOR]} & {r['auc_sector']:.4f} & "
                         f"{r['auc_global']:.4f} & ${r['gain']:+.4f}$ \\\\")
        lines += [r"\midrule",
                  f"Mean & {gvs['auc_sector'].mean():.4f} & "
                  f"{gvs['auc_global'].mean():.4f} & "
                  f"${gvs['gain'].mean():+.4f}$ \\\\",
                  r"\bottomrule", r"\end{tabular}"]
        open(f"{out}/tab_global_vs_sector.tex", "w").write("\n".join(lines) + "\n")

    # --- Table: PCA-weighted SCI ------------------------------------------
    if len(sci_pca):
        lines = [r"\begin{tabular}{lrrll}", r"\toprule",
                 r"Sector & SCI (equal) & SCI (PC1) & Band (equal) & Band (PC1) \\",
                 r"\midrule"]
        j = sci[["SCI", "band"]].join(sci_pca).sort_values("SCI")
        for s, r in j.iterrows():
            lines.append(f"{s} & ${r['SCI']:+.3f}$ & ${r['SCI_pca']:+.3f}$ & "
                         f"{r['band']} & {r['band_pca']} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(f"{out}/tab_sci_pca.tex", "w").write("\n".join(lines) + "\n")


def write_numbers(out: str) -> None:
    """
    A companion declarations file is written alongside numbers.tex that uses
    \\renewcommand. 
    """
    with open(f"{out}/numbers.tex", "w") as f:
        f.write("% Auto-generated by pipeline.py -- do not edit by hand.\n")
        f.write(f"% {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(r"\renewcommand{\NumbersAreStale}{false}" + "\n")
        for k, v in sorted(NUMBERS.items()):
            f.write("\\renewcommand{\\%s}{%s}\n" % (k, v))
    with open(f"{out}/numbers_declarations.tex", "w") as f:
        f.write("% Auto-generated by pipeline.py -- \\input this BEFORE\n"
                "% numbers.tex so every macro exists to be renewed.\n")
        f.write(r"\providecommand{\NumbersAreStale}{true}" + "\n")
        for k in sorted(NUMBERS):
            f.write("\\providecommand{\\%s}{??}\n" % k)
    note(f"    wrote {out}/numbers.tex with {len(NUMBERS)} macros "
         f"(+ numbers_declarations.tex)")


_ROB_SUMMARY = pd.DataFrame()   # module-level so the robustness helper can set it


def _run_robustness_block(df, feat, sci, pooled, models, n_seeds):
    """Isolated so a failure here cannot cost the main results."""
    global _ROB_SUMMARY
    if not RUN_ROBUSTNESS:
        return
    rob_detail, _ROB_SUMMARY = robustness_per_ticker(feat, sci)
    rob_summary = _ROB_SUMMARY
    if len(rob_detail):
        rob_detail.to_csv(f"{OUT_DIR}/robustness_perticker.csv", index=False)
        rob_summary.to_csv(f"{OUT_DIR}/robustness_by_sector.csv")
        macro("PerTickerAuc", rob_detail["auc"].mean())
        macro("PooledAuc", pooled["auc"].mean())
        macro("PerTickerPctAbove",
              100 * (rob_detail.groupby("ticker")["auc"].mean() > 0.5).mean(), "{:.0f}")
        macro("PerTickerBest", rob_detail["auc"].max(), "{:.3f}")
        macro("NTickersRobust", str(rob_detail["ticker"].nunique()))

    # Hurst min_n sensitivity
    sci_h = compute_sci(df, hurst_min_n=HURST_MIN_N_ROBUST, record=False)
    tau, _ = stats.kendalltau(sci["SCI"], sci_h.reindex(sci.index)["SCI"])
    sci_h.to_csv(f"{OUT_DIR}/robustness_sci_hurst_minn.csv")
    macro("HurstMinNTau", tau)
    macro("HurstMinNRobust", str(HURST_MIN_N_ROBUST))

    # lag-0 feature sensitivity, tabular models only (cheap)
    feat0 = build_features(df, include_lag0=True, record=False)
    tab = [m for m in models if m != "LSTM"]
    rows = []
    for sector, fs in feat0.groupby(COL_SECTOR, sort=False):
        p = run_sector(fs, tab, n_seeds_lstm=0)
        if len(p):
            rows.append(p)
    if rows:
        p0 = pd.concat(rows, ignore_index=True)
        m0 = metrics_by(p0, ["model", COL_SECTOR])
        m0.to_csv(f"{OUT_DIR}/robustness_lag0.csv", index=False)
        base = pooled[pooled["model"].isin(tab)]["auc"].mean()
        macro("LagZeroAuc", m0["auc"].mean())
        macro("LagZeroBaseAuc", base)
        macro("LagZeroDelta", m0["auc"].mean() - base, "{:+.5f}")



# =========================================================================== #
#                                                                             #
# =========================================================================== #
# =========================================================================== #
# EXTENSIONS REQUESTED AT SUPERVISION                                         #
# =========================================================================== #
# Every function below is additive.  None of them refits a forecasting model,
# none of them touches a quantity already reported, and each writes its own
# artefacts, so a re-run with cached predictions reproduces the headline
# results exactly and simply gains new files.
#
# =========================================================================== #


def fold_windows(preds: pd.DataFrame) -> pd.DataFrame:
    """First and last test date of each walk-forward fold."""
    g = preds.groupby("fold")[COL_DATE]
    return pd.DataFrame({"start": g.min(), "end": g.max()}).reset_index()


def compute_sci_by_fold(df: pd.DataFrame, preds: pd.DataFrame,
                        window: int = SCI_FOLD_WINDOW,
                        min_obs: int = SCI_FOLD_MIN_OBS) -> pd.DataFrame:
    """Time-varying SCI, one index per (sector, fold).

    The headline SCI is a single static number per sector, computed once over
    the whole sample.  That makes it collinear with a sector fixed effect.

    Each fold's index is computed on the `window` trading days immediately
    before that fold's test window opens, so it remains an ex-ante quantity:

    Standardisation pools all (sector, fold) cells, so one z-scale carries both
    the cross-sectional and the temporal variation and the two are comparable.
    """
    fw = fold_windows(preds)
    metrics = list(SCI_DIRECTION)
    frames = []
    for _, row in fw.iterrows():
        hist = df[df[COL_DATE] < row["start"]]
        if not len(hist):
            continue
        # last `window` trading days available before this fold opens
        dates = np.sort(hist[COL_DATE].unique())[-window:]
        hist = hist[hist[COL_DATE].isin(dates)]
        counts = hist.groupby(COL_TICKER)[COL_DATE].transform("size")
        hist = hist[counts >= min_obs]
        if not len(hist):
            continue
        per_ticker = (hist.groupby([COL_SECTOR, COL_TICKER], sort=False)
                          .apply(_ticker_metrics, include_groups=False)
                          .reset_index())
        per_sector = per_ticker.groupby(COL_SECTOR)[metrics].mean(numeric_only=True)
        per_sector["fold"] = int(row["fold"])
        per_sector["n_tickers"] = per_ticker.groupby(COL_SECTOR).size()
        per_sector["window_start"] = dates.min()
        per_sector["window_end"] = dates.max()
        frames.append(per_sector.reset_index())

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)

    # One standardisation across every (sector, fold) cell.
    z = ((panel[metrics] - panel[metrics].mean())
         / panel[metrics].std(ddof=SCI_ZSCORE_DDOF))
    panel["SCI_ft"] = z.mul(pd.Series(SCI_DIRECTION)).mean(axis=1)
    panel["band_ft"] = pd.qcut(panel["SCI_ft"], N_BANDS,
                               labels=["low", "medium", "high"])

    
    within = panel.groupby(COL_SECTOR)["SCI_ft"].std(ddof=1).mean()
    between = panel.groupby(COL_SECTOR)["SCI_ft"].mean().std(ddof=1)
    panel.attrs["within_sector_sd"] = float(within)
    panel.attrs["between_sector_sd"] = float(between)
    return panel


# --------------------------------------------------------------------------- #
# Least squares with cluster-robust standard errors                            #
# --------------------------------------------------------------------------- #
def _ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    """cluster-robust covariance matrix.

    Clustering is by sector, the eleven sectors are the independent units, and
    the five folds within a sector share both a fitted model lineage and the
    same market history.  With G = 11 the correction is generous rather than
    exact, so inference here is reported as indicative and the point estimates
    carry the argument.  Degrees of freedom are G - 1.
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    uniq = np.unique(groups)
    G, n, k = len(uniq), X.shape[0], X.shape[1]
    meat = np.zeros((k, k))
    for g in uniq:
        m = groups == g
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    scale = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = scale * (XtX_inv @ meat @ XtX_inv)

    se = np.sqrt(np.clip(np.diag(V), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    dfree = max(G - 1, 1)
    p = 2 * stats.t.sf(np.abs(t), dfree)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": t, "p": p, "V": V, "df": dfree,
            "n": n, "k": k, "G": G,
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan}


def _wald(fit: dict, idx: list[int]) -> tuple[float, float, int]:
    """Joint test that the coefficients at `idx` are all zero."""
    if not idx:
        return np.nan, np.nan, 0
    b = fit["beta"][idx]
    V = fit["V"][np.ix_(idx, idx)]
    stat = float(b @ np.linalg.pinv(V) @ b)
    q = len(idx)
    # F form, appropriate with few clusters
    f = stat / q
    p = float(stats.f.sf(f, q, fit["df"]))
    return f, p, q


def _dummies(values: pd.Series, drop_first: bool = True) -> tuple[np.ndarray, list[str]]:
    levels = sorted(pd.unique(values.astype(str)))
    if drop_first:
        levels = levels[1:]
    cols = [(values.astype(str) == lv).astype(float).to_numpy() for lv in levels]
    return (np.column_stack(cols) if cols else np.zeros((len(values), 0))), levels


def panel_forecastability(fold_perf: pd.DataFrame, sci_fold: pd.DataFrame,
                          metric: str = METRIC) -> tuple[pd.DataFrame, dict]:
    """Sector x fold panel: does complexity explain forecastability, and does
    its effect differ by model family?

    It uses every (model, sector, fold) cell, it lets the
    complexity effect differ by model family, and it can separate the between-sector effect from the
    within-sector movement over time.

    Two specifications are estimated because they answer different questions:
      A  fold FE only      - b is identified from BOTH cross-sector and
                              temporal variation, and is comparable to the
                              headline Spearman correlation.
      B  fold + sector FE  - b is identified from within sector movement
                              only, 
    """
    if not len(sci_fold):
        return pd.DataFrame(), {}
    keep = [COL_SECTOR, "fold", "SCI_ft"]
    d = fold_perf.merge(sci_fold[keep], on=[COL_SECTOR, "fold"], how="inner")
    d = d.dropna(subset=[metric, "SCI_ft"])
    if len(d) < 20 or d["model"].nunique() < 2:
        return pd.DataFrame(), {}

    # z-score the index within the estimation panel so b reads as "AUC change
    # per one standard deviation of complexity".
    d = d.copy()
    d["sci_z"] = (d["SCI_ft"] - d["SCI_ft"].mean()) / d["SCI_ft"].std(ddof=0)

    y = d[metric].to_numpy(dtype=float)
    groups = d[COL_SECTOR].astype(str).to_numpy()
    mod_d, mod_levels = _dummies(d["model"])
    fold_d, fold_levels = _dummies(d["fold"])
    sec_d, sec_levels = _dummies(d[COL_SECTOR])
    sci = d["sci_z"].to_numpy(dtype=float).reshape(-1, 1)
    inter = mod_d * sci if mod_d.shape[1] else np.zeros((len(d), 0))

    rows, summary = [], {}
    specs = {
        "A: fold FE": (fold_d, fold_levels, False),
        "B: fold + sector FE": (np.column_stack([fold_d, sec_d]) if sec_d.shape[1]
                                else fold_d, fold_levels, True),
    }
    for spec, (fe, _lv, has_sector) in specs.items():
        X = np.column_stack([np.ones(len(d)), sci, mod_d, inter, fe])
        names = (["intercept", "SCI (z)"]
                 + [f"model={m}" for m in mod_levels]
                 + [f"SCI x model={m}" for m in mod_levels]
                 + [f"FE{i}" for i in range(fe.shape[1])])
        fit = _ols_cluster(y, X, groups)
        i_int = [i for i, nm in enumerate(names) if nm.startswith("SCI x")]
        f_int, p_int, q_int = _wald(fit, i_int)
        for i, nm in enumerate(names):
            if nm.startswith("FE"):
                continue        # nuisance parameters, not reported
            rows.append({"spec": spec, "term": nm, "coef": fit["beta"][i],
                         "se": fit["se"][i], "t": fit["t"][i], "p": fit["p"][i]})
        summary[spec] = {
            "sci_coef": float(fit["beta"][1]), "sci_se": float(fit["se"][1]),
            "sci_p": float(fit["p"][1]),
            "interaction_F": float(f_int), "interaction_p": float(p_int),
            "interaction_df": int(q_int),
            "n": int(fit["n"]), "clusters": int(fit["G"]),
            "r2": float(fit["r2"]), "sector_fe": has_sector}
    return pd.DataFrame(rows), summary


def temporal_heterogeneity(fold_perf: pd.DataFrame,
                           metric: str = METRIC) -> tuple[pd.DataFrame, dict]:
    """
    RQ1 asks whether sectors differ. The walk-forward design supplies a second axis for free, and if a sector is unpredictable on average but
    predictable in particular periods. A two-way decomposition of the (sector x fold) table of mean AUC gives the relative size of the two sources directly.
    """
    d = fold_perf.dropna(subset=[metric])
    if not len(d) or d["fold"].nunique() < 2:
        return pd.DataFrame(), {}

    cell = (d.groupby([COL_SECTOR, "fold"], observed=True)[metric]
             .mean().unstack("fold"))
    cell = cell.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if cell.shape[0] < 2 or cell.shape[1] < 2:
        return pd.DataFrame(), {}

    grand = float(cell.to_numpy().mean())
    row_m = cell.mean(axis=1)          # sector effects
    col_m = cell.mean(axis=0)          # fold (period) effects
    S, T = cell.shape
    ss_sector = float(T * ((row_m - grand) ** 2).sum())
    ss_fold = float(S * ((col_m - grand) ** 2).sum())
    resid = cell.to_numpy() - row_m.to_numpy()[:, None] - col_m.to_numpy()[None, :] + grand
    ss_resid = float((resid ** 2).sum())
    ss_total = ss_sector + ss_fold + ss_resid

    # Friedman with folds as treatments, (model, sector) as blocks.
    wide = (d.pivot_table(index=["model", COL_SECTOR], columns="fold",
                          values=metric, observed=True)
             .dropna(axis=0, how="any"))
    chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    w = kendalls_w(float(chi2), int(wide.shape[0]), int(wide.shape[1]))

    # Note on the third row: with a single AUC per (sector, fold) cell there
    # is no replication within a cell, so the interaction cannot be separated
    # from residual error.  The row is labelled accordingly rather than being
    # reported as a pure interaction.
    out = pd.DataFrame({
        "source": ["sector", "period (fold)",
                   "sector x period and residual"],
        "ss": [ss_sector, ss_fold, ss_resid],
        "share": [ss_sector / ss_total, ss_fold / ss_total, ss_resid / ss_total],
        "df": [S - 1, T - 1, (S - 1) * (T - 1)]})

    info = {
        "friedman_folds_chi2": float(chi2), "friedman_folds_p": float(p),
        "kendall_w_folds": float(w),
        "n_blocks": int(wide.shape[0]), "n_folds": int(wide.shape[1]),
        "share_sector": ss_sector / ss_total,
        "share_period": ss_fold / ss_total,
        "share_interaction": ss_resid / ss_total,
        "cross_sector_range": float(row_m.max() - row_m.min()),
        "period_range": float(col_m.max() - col_m.min()),
        "mean_within_sector_range": float((cell.max(axis=1) - cell.min(axis=1)).mean()),
        "best_fold_sector_pairs": int(S)}
    return out, info


def market_regimes(df: pd.DataFrame, vol_window: int = REGIME_VOL_WINDOW,
                   trend_window: int = REGIME_TREND_WINDOW) -> pd.DataFrame:
    """Label each trading date by market state, using trailing data only.

    Two states are built.  Volatility: the realised standard deviation of the
    equal-weighted market return over the previous `vol_window` days, split at
    its median.  Trend: the sign of the cumulative market return over the
    previous `trend_window` days, Both use a window that closes the day before.
    """
    px = (df.sort_values(COL_DATE)
            .assign(_lr=lambda x: np.log(x[COL_CLOSE]))
            .groupby([COL_TICKER])["_lr"].diff())
    tmp = df[[COL_DATE]].copy()
    tmp["ret"] = px.to_numpy()
    mkt = (tmp.dropna().groupby(COL_DATE)["ret"].mean()
              .sort_index().rename("mkt_ret").to_frame())
    if len(mkt) < max(vol_window, trend_window) + 5:
        return pd.DataFrame()

    mkt["vol"] = mkt["mkt_ret"].rolling(vol_window).std().shift(1)
    mkt["trend"] = mkt["mkt_ret"].rolling(trend_window).sum().shift(1)
    mkt = mkt.dropna()
    cut = float(mkt["vol"].median())
    mkt["vol_regime"] = np.where(mkt["vol"] > cut, "high vol", "low vol")
    mkt["trend_regime"] = np.where(mkt["trend"] > 0, "bull", "bear")
    mkt.attrs["vol_cut"] = cut
    return mkt.reset_index()


def regime_analysis(preds: pd.DataFrame,
                    regimes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Performance and sector spread within each market state.

    The question is if the SECTOR differences widen in a particular state.  If they do,
    the conditioning variable a decision support system needs is sector x
    state rather than sector alone.
    """
    if not len(regimes):
        return pd.DataFrame(), pd.DataFrame(), {}

    p = preds.merge(regimes[[COL_DATE, "vol_regime", "trend_regime"]],
                    on=COL_DATE, how="inner")
    if not len(p):
        return pd.DataFrame(), pd.DataFrame(), {}

    frames, summary = [], []
    for kind in ("vol_regime", "trend_regime"):
        # fold-blocked within regime, then seed-averaged, as everywhere else
        per_seed = metrics_by_foldblocked(p, ["model", COL_SECTOR, "seed", kind])
        cells = collapse_lstm_seeds(per_seed, ["model", COL_SECTOR, kind])
        cells = cells.rename(columns={kind: "regime"})
        cells["regime_kind"] = kind
        frames.append(cells)

        for reg, g in cells.groupby("regime", observed=True):
            sect = g.groupby(COL_SECTOR, observed=True)["auc"].mean()
            wide = g.pivot_table(index="model", columns=COL_SECTOR, values="auc",
                                 observed=True).dropna(axis=1, how="any")
            if wide.shape[1] >= 3 and wide.shape[0] >= 2:
                chi2, pv = stats.friedmanchisquare(
                    *[wide[c].to_numpy() for c in wide.columns])
            else:
                chi2, pv = np.nan, np.nan
            summary.append({
                "regime_kind": kind, "regime": str(reg),
                "mean_auc": float(g["auc"].mean()),
                "sector_spread": float(sect.max() - sect.min()),
                "sector_sd": float(sect.std(ddof=1)),
                "best_model": str(g.groupby("model", observed=True)["auc"]
                                   .mean().idxmax()),
                "friedman_chi2": float(chi2) if chi2 == chi2 else np.nan,
                "friedman_p": float(pv) if pv == pv else np.nan,
                "n_rows": int(g["n"].sum())})

    cells_all = pd.concat(frames, ignore_index=True)
    summ = pd.DataFrame(summary)

    info = {}
    for kind, lo, hi in (("vol_regime", "low vol", "high vol"),
                         ("trend_regime", "bear", "bull")):
        sub = cells_all[cells_all["regime_kind"] == kind]
        a = sub[sub["regime"] == hi].groupby(COL_SECTOR, observed=True)["auc"].mean()
        b = sub[sub["regime"] == lo].groupby(COL_SECTOR, observed=True)["auc"].mean()
        common = a.index.intersection(b.index)
        if len(common) >= 5:
            try:
                w = stats.wilcoxon(a[common], b[common])
                info[f"{kind}_wilcoxon_p"] = float(w.pvalue)
            except ValueError:
                info[f"{kind}_wilcoxon_p"] = np.nan
            info[f"{kind}_mean_gap"] = float((a[common] - b[common]).mean())
            info[f"{kind}_spread_{hi.replace(' ', '_')}"] = float(a.max() - a.min())
            info[f"{kind}_spread_{lo.replace(' ', '_')}"] = float(b.max() - b.min())
    return cells_all, summ, info


def _seed_averaged(preds: pd.DataFrame) -> pd.DataFrame:
    """One probability per (sector, date, ticker, model, fold)."""
    return (preds.groupby([COL_SECTOR, COL_DATE, COL_TICKER, "model", "fold"],
                          sort=False, observed=True)
                 .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"),
                      train_const=("train_const", "first"))
                 .reset_index())


def abstention_curve(preds: pd.DataFrame,
                     coverages: tuple = ABSTAIN_COVERAGES) -> pd.DataFrame:
    """Selective prediction: accuracy when the system is allowed to abstain.
    
    Two baselines are therefore computed on exactly the same covered rows: the hindsight
    majority class of that subset, and the constant a deployable system would
    have emitted, fixed from the training partition.  Skill is the margin over
    each, and it is the margin rather than the raw accuracy that carries the
    argument.

    Ranking is within fold, since each fold is a separately fitted model whose
    probability scale is its own.
    """
    p = _seed_averaged(preds)
    p["conf"] = (p["y_prob"] - 0.5).abs()
    p["correct"] = ((p["y_prob"] >= 0.5).astype(int) == p["y_true"]).astype(int)

    rows = []
    for (model, fold), g in p.groupby(["model", "fold"], sort=False, observed=True):
        g = g.sort_values("conf", ascending=False)
        n = len(g)
        for cov in coverages:
            k = max(int(round(cov * n)), 50)
            k = min(k, n)
            sub = g.iloc[:k]
            y = sub["y_true"].to_numpy()
            base_major = float(max(y.mean(), 1 - y.mean()))
            base_const = float(np.mean(y == sub["train_const"].to_numpy()))
            acc = float(sub["correct"].mean())
            rows.append({"model": model, "fold": fold, "coverage": cov,
                         "accuracy": acc,
                         "baseline_majority": base_major,
                         "baseline_const": base_const,
                         "skill_vs_majority": acc - base_major,
                         "skill_vs_const": acc - base_const,
                         "up_share": float(y.mean()),
                         "threshold": float(sub["conf"].min()),
                         "n": int(k)})
    per_fold = pd.DataFrame(rows)
    if not len(per_fold):
        return per_fold
    out = (per_fold.groupby(["model", "coverage"], as_index=False)
                   .agg(accuracy=("accuracy", "mean"),
                        accuracy_sd=("accuracy", "std"),
                        baseline_majority=("baseline_majority", "mean"),
                        baseline_const=("baseline_const", "mean"),
                        skill_vs_majority=("skill_vs_majority", "mean"),
                        skill_vs_majority_sd=("skill_vs_majority", "std"),
                        skill_vs_const=("skill_vs_const", "mean"),
                        up_share=("up_share", "mean"),
                        threshold=("threshold", "mean"),
                        n=("n", "sum")))
    # gain relative to answering everything, per model
    full = out[out["coverage"] == max(coverages)].set_index("model")["accuracy"]
    out["gain_vs_full"] = out["accuracy"] - out["model"].map(full)
    return out.sort_values(["model", "coverage"], ascending=[True, False])


def abstention_gate(fold_perf: pd.DataFrame, preds: pd.DataFrame,
                    metric: str = METRIC) -> tuple[pd.DataFrame, dict]:
    """Reliability gating, deploy a sector, model only after it has passed.

    The rule is the one an operator runs entering fold t, look
    at the cell's validated performance on fold t-1; emit signals in fold t
    only if that came out above chance.  
    """
    fp = fold_perf.dropna(subset=[metric]).copy()
    folds = sorted(fp["fold"].unique())
    if len(folds) < 2:
        return pd.DataFrame(), {}

    prev = fp.copy()
    prev["fold"] = prev["fold"] + 1
    prev = prev.rename(columns={metric: "prev_auc"})[[COL_SECTOR, "model",
                                                      "fold", "prev_auc"]]
    d = fp.merge(prev, on=[COL_SECTOR, "model", "fold"], how="inner")
    if not len(d):
        return pd.DataFrame(), {}
    d["deployed"] = d["prev_auc"] > 0.5

    dep = d[d["deployed"]]
    info = {
        "n_cells": int(len(d)),
        "n_deployed": int(d["deployed"].sum()),
        "coverage": float(d["deployed"].mean()),
        "auc_deployed": float(dep[metric].mean()) if len(dep) else np.nan,
        "auc_all": float(d[metric].mean()),
        "acc_deployed": float(dep["accuracy"].mean()) if len(dep) else np.nan,
        "acc_all": float(d["accuracy"].mean()),
        "share_deployed_above_half": (float((dep[metric] > 0.5).mean())
                                      if len(dep) else np.nan),
        "share_all_above_half": float((d[metric] > 0.5).mean())}
    info["auc_gain"] = info["auc_deployed"] - info["auc_all"]
    return d, info


def selection_policies(fold_perf: pd.DataFrame, sci: pd.DataFrame,
                       metric: str = METRIC, n_draws: int = SELECTION_RANDOM_DRAWS,
                       seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, dict]:
    """Score the SCI selection rule against the baselines.

    Six policies are scored on identical held-out folds:

      sci_band            band -> model map fitted on the early folds
      universal_fit       one model for everything, chosen on the early folds
      universal_hindsight the best single model chosen on the scored folds
                   
      sector_hindsight    the best model per sector on the scored folds
                       
      random              a model drawn uniformly per sector, averaged over many draws
                          
      validated_recent    per sector, the model that led on the previous fold

    """
    rng = np.random.default_rng(seed)
    band = sci["band"].astype(str)
    fp = fold_perf.dropna(subset=[metric]).copy()
    fp["band"] = fp[COL_SECTOR].map(band)
    folds = sorted(fp["fold"].unique())
    if len(folds) < 3:
        return pd.DataFrame(), {}
    cut = max(1, len(folds) - 2)
    fit_folds, score_folds = folds[:cut], folds[cut:]
    fit, score = fp[fp["fold"].isin(fit_folds)], fp[fp["fold"].isin(score_folds)]
    models = sorted(fp["model"].unique())

    # cell means on the scored folds: rows = sector, cols = model
    wide = (score.groupby([COL_SECTOR, "model"], observed=True)[metric]
                 .mean().unstack("model"))
    wide = wide.dropna(axis=0, how="any")
    if not len(wide):
        return pd.DataFrame(), {}

    per_sector = {}

    # 1. SCI band policy, fitted on early folds
    choice = (fit.groupby(["band", "model"], observed=True)[metric].mean()
                 .reset_index().sort_values(metric, ascending=False)
                 .drop_duplicates("band").set_index("band")["model"])
    per_sector["sci_band"] = pd.Series(
        {s: wide.loc[s, choice[band[s]]] for s in wide.index
         if band.get(s) in choice.index and choice[band[s]] in wide.columns})

    # 2. one model everywhere, chosen on the early folds
    best_fit_model = fit.groupby("model", observed=True)[metric].mean().idxmax()
    per_sector["universal_fit"] = wide[best_fit_model]

    # 3. one model everywhere, chosen with hindsight
    best_hind = wide.mean().idxmax()
    per_sector["universal_hindsight"] = wide[best_hind]

    # 4. per-sector hindsight (the ceiling)
    per_sector["sector_hindsight"] = wide.max(axis=1)

    # 5. random choice per sector
    draws = np.empty(n_draws)
    arr = wide.to_numpy()
    for i in range(n_draws):
        pick = rng.integers(0, arr.shape[1], size=arr.shape[0])
        draws[i] = arr[np.arange(arr.shape[0]), pick].mean()
    random_mean, random_sd = float(draws.mean()), float(draws.std(ddof=1))

    # 6. previous-fold winner, rolled forward one fold at a time
    rec = []
    for f in score_folds:
        prev = fp[fp["fold"] == f - 1]
        cur = fp[fp["fold"] == f]
        if not len(prev) or not len(cur):
            continue
        winner = (prev.groupby([COL_SECTOR, "model"], observed=True)[metric]
                      .mean().groupby(COL_SECTOR, observed=True).idxmax())
        for s, key in winner.items():
            m = key[1]
            v = cur[(cur[COL_SECTOR] == s) & (cur["model"] == m)][metric]
            if len(v):
                rec.append({COL_SECTOR: s, "fold": f, metric: float(v.mean())})
    rec_df = pd.DataFrame(rec)
    if len(rec_df):
        per_sector["validated_recent"] = (rec_df.groupby(COL_SECTOR)[metric].mean())

    rows = []
    ref = per_sector["universal_fit"]
    for name, series in per_sector.items():
        series = series.dropna()
        common = series.index.intersection(ref.index)
        if len(common) >= 5 and name != "universal_fit":
            try:
                pv = float(stats.wilcoxon(series[common], ref[common]).pvalue)
            except ValueError:
                pv = np.nan
        else:
            pv = np.nan
        rows.append({"policy": name, "deployable": name in
                     ("sci_band", "universal_fit", "validated_recent", "random"),
                     "mean_auc": float(series.mean()),
                     "sd_across_sectors": float(series.std(ddof=1)),
                     "gain_vs_universal": float(series.mean() - ref.mean()),
                     "wilcoxon_p_vs_universal": pv,
                     "n_sectors": int(len(series))})
    rows.append({"policy": "random", "deployable": True,
                 "mean_auc": random_mean, "sd_across_sectors": random_sd,
                 "gain_vs_universal": random_mean - float(ref.mean()),
                 "wilcoxon_p_vs_universal": np.nan,
                 "n_sectors": int(len(wide))})
    out = (pd.DataFrame(rows).drop_duplicates("policy", keep="last")
             .sort_values("mean_auc", ascending=False).reset_index(drop=True))

    info = {"fit_folds": fit_folds, "score_folds": score_folds,
            "band_to_model": choice.to_dict(),
            "universal_fit_model": str(best_fit_model),
            "universal_hindsight_model": str(best_hind),
            "random_draws": n_draws}
    return out, info


def decision_relevant_heterogeneity(rq1: dict, lstm_seeds: pd.DataFrame,
                                    sector_ci: pd.DataFrame,
                                    fold_perf: pd.DataFrame,
                                    metric: str = METRIC) -> dict:
    """Is the sector effect large enough to condition a decision on?

    A significant Friedman test says the sector ordering is systematic.  It
    does not say the gap is worth acting on.  The observed spread is compared
    against three noise floors, each of which is a quantity that moves the
    same AUC without any sector effect at all:

      seed floor      refitting the LSTM with a different random seed
      sampling floor  the half-width of a sector's bootstrap interval
      period floor    the same sector's movement from one fold to the next

    Heterogeneity is called decision-relevant only if it clears the largest of
    them.
    """
    spread = float(rq1.get("sector_mean_spread", np.nan))
    sd = float(rq1.get("sector_mean_sd", np.nan))

    seed_floor = np.nan
    if lstm_seeds is not None and len(lstm_seeds):
        seed_floor = float(lstm_seeds.groupby(COL_SECTOR, observed=True)[metric]
                                     .std(ddof=1).mean())
    sampling_floor, overlap = np.nan, np.nan
    if sector_ci is not None and len(sector_ci):
        half = (sector_ci["ci_hi"] - sector_ci["ci_lo"]) / 2
        sampling_floor = float(half.mean())
        lo = sector_ci["ci_lo"].to_numpy()
        hi = sector_ci["ci_hi"].to_numpy()
        n = len(lo)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if pairs:
            overlap = float(np.mean([not (hi[i] < lo[j] or hi[j] < lo[i])
                                     for i, j in pairs]))
    period_floor = np.nan
    if fold_perf is not None and len(fold_perf):
        cell = (fold_perf.groupby([COL_SECTOR, "fold"], observed=True)[metric]
                         .mean().unstack("fold"))
        if cell.shape[1] > 1:
            period_floor = float(cell.std(axis=1, ddof=1).mean())

    floors = {"seed": seed_floor, "sampling": sampling_floor,
              "period": period_floor}
    finite = [v for v in floors.values() if v == v]
    binding = max(finite) if finite else np.nan
    binding_name = (max(floors, key=lambda k: (floors[k] if floors[k] == floors[k]
                                               else -np.inf)) if finite else "none")
    return {"spread": spread, "sd_between": sd,
            "floor_seed": seed_floor, "floor_sampling": sampling_floor,
            "floor_period": period_floor,
            "binding_floor": binding, "binding_floor_name": binding_name,
            "ratio_to_binding": spread / binding if binding and binding == binding else np.nan,
            "decision_relevant": bool(spread > binding) if binding == binding else False,
            "share_sector_pairs_overlapping": overlap}


def heterogeneity_summary(pooled: pd.DataFrame, rq1: dict,
                          sector_ci: pd.DataFrame, drh: dict,
                          metric: str = METRIC) -> pd.DataFrame:
    """Sector heterogeneity is what the study sets out to explain; the SCI is one
    candidate explanation for it.  Keeping them in separate artefacts stops
    the two being read as the same construct.  This table holds only how predictable
    each sector is, how uncertain that estimate is, and how big the spread is
    against significance and against the noise floors.
    """
    means = (pooled.groupby(COL_SECTOR, observed=True)[metric]
                   .mean().sort_values(ascending=False).rename("mean_auc"))
    out = means.to_frame()
    out["rank"] = np.arange(1, len(out) + 1)
    if sector_ci is not None and len(sector_ci):
        ci = sector_ci.set_index("sector")[["ci_lo", "ci_hi", "excludes_half"]]
        out = out.join(ci, how="left")
    out = out.rename_axis("sector").reset_index()
    out.attrs["footer"] = {
        "range": rq1.get("sector_mean_spread"),
        "sd": rq1.get("sector_mean_sd"),
        "friedman_chi2": rq1.get("chi2"), "friedman_p": rq1.get("p"),
        "kendall_w": rq1.get("kendall_w"),
        "binding_noise_floor": drh.get("binding_floor"),
        "decision_relevant": drh.get("decision_relevant")}
    return out


def write_extension_tables(out: str, het: pd.DataFrame, panel: pd.DataFrame,
                           tvar: pd.DataFrame, regimes: pd.DataFrame,
                           abst: pd.DataFrame, policies: pd.DataFrame) -> None:
    """LaTeX bodies for the tables added by this section."""
    if len(het):
        f = het.attrs.get("footer", {})
        with open(f"{out}/tab_heterogeneity_summary.tex", "w") as fh:
            has_ci = "ci_lo" in het.columns
            fh.write("\\begin{tabular}{lr" + ("r" if has_ci else "") + "c}\n\\toprule\n")
            fh.write("Sector & Mean AUC & " + ("95\\% CI & " if has_ci else "")
                     + "Excludes 0.5 \\\\\n\\midrule\n")
            for _, r in het.iterrows():
                ci = (f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] & " if has_ci
                      and r.get("ci_lo") == r.get("ci_lo") else ("-- & " if has_ci else ""))
                mark = "yes" if r.get("excludes_half") is True else "no"
                fh.write(f"{r['sector']} & {r['mean_auc']:.4f} & {ci}{mark} \\\\\n")
            fh.write("\\midrule\n")
            fh.write(f"\\multicolumn{{{3 + int(has_ci)}}}{{l}}{{Range "
                     f"{_fmt(f.get('range'))}, SD {_fmt(f.get('sd'))}, "
                     f"Friedman $\\chi^2$ {_fmt(f.get('friedman_chi2'), 2)} "
                     f"($p$ {_fmt(f.get('friedman_p'), 3)}), "
                     f"Kendall's $W$ {_fmt(f.get('kendall_w'), 3)}}} \\\\\n")
            fh.write(f"\\multicolumn{{{3 + int(has_ci)}}}{{l}}{{Binding noise "
                     f"floor {_fmt(f.get('binding_noise_floor'))}; "
                     f"decision-relevant: "
                     f"{'yes' if f.get('decision_relevant') else 'no'}}} \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")

    if len(panel):
        with open(f"{out}/tab_panel_regression.tex", "w") as fh:
            fh.write("\\begin{tabular}{llrrr}\n\\toprule\n")
            fh.write("Specification & Term & Coef. & SE & $p$ \\\\\n\\midrule\n")
            for spec, g in panel.groupby("spec", sort=False):
                first = True
                for _, r in g.iterrows():
                    if r["term"] == "intercept":
                        continue
                    lab = spec if first else ""
                    first = False
                    fh.write(f"{lab} & {r['term']} & ${r['coef']:+.4f}$ & "
                             f"{r['se']:.4f} & {r['p']:.3f} \\\\\n")
                fh.write("\\midrule\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")

    if len(tvar):
        with open(f"{out}/tab_temporal_variance.tex", "w") as fh:
            fh.write("\\begin{tabular}{lrrr}\n\\toprule\n")
            fh.write("Source & d.f. & Sum of squares & Share \\\\\n\\midrule\n")
            for _, r in tvar.iterrows():
                fh.write(f"{r['source']} & {int(r['df'])} & {r['ss']:.5f} & "
                         f"{100 * r['share']:.1f}\\% \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")

    if len(regimes):
        with open(f"{out}/tab_regimes.tex", "w") as fh:
            fh.write("\\begin{tabular}{llrrrl}\n\\toprule\n")
            fh.write("State & Regime & Mean AUC & Sector range & Sector SD & "
                     "Leading model \\\\\n\\midrule\n")
            for _, r in regimes.iterrows():
                kind = ("volatility" if r["regime_kind"] == "vol_regime"
                        else "trend")
                fh.write(f"{kind} & {r['regime']} & {r['mean_auc']:.4f} & "
                         f"{r['sector_spread']:.4f} & {r['sector_sd']:.4f} & "
                         f"{r['best_model']} \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")

    if len(abst):
        with open(f"{out}/tab_abstention.tex", "w") as fh:
            models = list(dict.fromkeys(abst["model"]))
            covs = sorted(abst["coverage"].unique(), reverse=True)
            fh.write("\\begin{tabular}{l" + "r" * len(covs) + "}\n\\toprule\n")
            fh.write("Model & " + " & ".join(f"{100 * c:.0f}\\%" for c in covs)
                     + " \\\\\n\\midrule\n")
            for m in models:
                gm = abst[abst["model"] == m].set_index("coverage")
                acc = " & ".join(f"{gm['accuracy'].get(c, float('nan')):.4f}"
                                 for c in covs)
                fh.write(f"{m} & {acc} \\\\\n")
            fh.write("\\midrule\n")
            for m in models:
                gm = abst[abst["model"] == m].set_index("coverage")
                sk = " & ".join(
                    f"${gm['skill_vs_majority'].get(c, float('nan')):+.4f}$"
                    for c in covs)
                fh.write(f"{m}, over majority & {sk} \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")

    if len(policies):
        with open(f"{out}/tab_selection_policies.tex", "w") as fh:
            fh.write("\\begin{tabular}{llrrr}\n\\toprule\n")
            fh.write("Policy & Deployable & Mean AUC & Gain & $p$ \\\\\n\\midrule\n")
            for _, r in policies.iterrows():
                pv = (f"{r['wilcoxon_p_vs_universal']:.3f}"
                      if r["wilcoxon_p_vs_universal"] == r["wilcoxon_p_vs_universal"]
                      else "--")
                fh.write(f"{r['policy'].replace('_', ' ')} & "
                         f"{'yes' if r['deployable'] else 'no'} & "
                         f"{r['mean_auc']:.4f} & ${r['gain_vs_universal']:+.4f}$ & "
                         f"{pv} \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")


def run_extensions(df: pd.DataFrame, preds: pd.DataFrame, feat: pd.DataFrame,
                   fold_perf: pd.DataFrame, pooled: pd.DataFrame,
                   lstm_seeds: pd.DataFrame, sci: pd.DataFrame,
                   rq1: dict, boot: dict) -> dict:
    """Run the whole extension block, writing artefacts and macros.
    Wrapped individually in try/except.
    """
    res: dict = {}

    # ---- point 4: time-varying index and the sector x fold panel ----------
    sci_fold = pd.DataFrame()
    if RUN_PANEL:
        try:
            note("    per-fold SCI (trailing window, ex ante)")
            sci_fold = compute_sci_by_fold(df, preds)
            if len(sci_fold):
                sci_fold.to_csv(f"{OUT_DIR}/sci_by_sector_fold.csv", index=False)
                macro("SCIFoldWindow", str(SCI_FOLD_WINDOW))
                macro("SCIFoldWithinSD", sci_fold.attrs["within_sector_sd"], "{:.3f}")
                macro("SCIFoldBetweenSD", sci_fold.attrs["between_sector_sd"], "{:.3f}")
                # does the static index survive as a description of the panel?
                static = sci["SCI"]
                mean_ft = sci_fold.groupby(COL_SECTOR)["SCI_ft"].mean()
                common = static.index.intersection(mean_ft.index)
                if len(common) >= 5:
                    tau, tau_p = stats.kendalltau(static[common], mean_ft[common])
                    macro("SCIFoldTauVsStatic", float(tau), "{:+.3f}")
                    macro("SCIFoldTauVsStaticP", float(tau_p), "{:.3f}")
                # band churn: how often would a sector change band over time?
                churn = (sci_fold.groupby(COL_SECTOR)["band_ft"]
                                 .nunique().gt(1).mean())
                macro("SCIFoldBandChurn", 100 * float(churn), "{:.0f}")
                res["sci_fold"] = sci_fold

            panel, psum = panel_forecastability(fold_perf, sci_fold)
            if len(panel):
                panel.to_csv(f"{OUT_DIR}/panel_regression.csv", index=False)
                res["panel"] = panel
                res["panel_summary"] = psum
                for key, tag in (("A: fold FE", "Between"),
                                 ("B: fold + sector FE", "Within")):
                    if key in psum:
                        s = psum[key]
                        macro(f"Panel{tag}Beta", s["sci_coef"], "{:+.4f}")
                        macro(f"Panel{tag}SE", s["sci_se"], "{:.4f}")
                        macro(f"Panel{tag}P", s["sci_p"], "{:.3f}")
                        macro(f"Panel{tag}InterF", s["interaction_F"], "{:.2f}")
                        macro(f"Panel{tag}InterP", s["interaction_p"], "{:.3f}")
                        macro(f"Panel{tag}Rsq", s["r2"], "{:.3f}")
                        macro(f"Panel{tag}N", str(s["n"]))
                note(f"    panel: between-sector beta "
                     f"{psum['A: fold FE']['sci_coef']:+.4f} "
                     f"(p={psum['A: fold FE']['sci_p']:.3f}), interaction p="
                     f"{psum['A: fold FE']['interaction_p']:.3f}")
        except Exception as e:
            note(f"    [WARN] panel block failed ({type(e).__name__}: {e})")

    # ---- point 5: temporal heterogeneity ----------------------------------
    try:
        tvar, tinfo = temporal_heterogeneity(fold_perf)
        if len(tvar):
            tvar.to_csv(f"{OUT_DIR}/temporal_variance.csv", index=False)
            res["temporal_variance"], res["temporal"] = tvar, tinfo
            macro("TempFriedChi", tinfo["friedman_folds_chi2"], "{:.2f}")
            macro("TempFriedP", tinfo["friedman_folds_p"], "{:.4f}")
            macro("TempKendallW", tinfo["kendall_w_folds"], "{:.3f}")
            macro("ShareVarSector", 100 * tinfo["share_sector"], "{:.1f}")
            macro("ShareVarPeriod", 100 * tinfo["share_period"], "{:.1f}")
            macro("ShareVarInteraction", 100 * tinfo["share_interaction"], "{:.1f}")
            macro("PeriodRange", tinfo["period_range"])
            macro("WithinSectorRange", tinfo["mean_within_sector_range"])
            note(f"    temporal: period explains "
                 f"{100 * tinfo['share_period']:.1f}% of cell variance vs "
                 f"{100 * tinfo['share_sector']:.1f}% for sector")
    except Exception as e:
        note(f"    [WARN] temporal heterogeneity failed ({type(e).__name__}: {e})")

    # ---- point 6: market regimes ------------------------------------------
    if RUN_REGIMES:
        try:
            reg = market_regimes(df)
            if len(reg):
                reg[[COL_DATE, "mkt_ret", "vol", "trend", "vol_regime",
                     "trend_regime"]].to_csv(f"{OUT_DIR}/market_regimes.csv",
                                             index=False)
                rcells, rsumm, rinfo = regime_analysis(preds, reg)
                if len(rsumm):
                    rcells.to_csv(f"{OUT_DIR}/regime_cells.csv", index=False)
                    rsumm.to_csv(f"{OUT_DIR}/regime_summary.csv", index=False)
                    res["regime_cells"], res["regime_summary"] = rcells, rsumm
                    res["regime_info"] = rinfo
                    for _, r in rsumm.iterrows():
                        tag = {"high vol": "HighVol", "low vol": "LowVol",
                               "bull": "Bull", "bear": "Bear"}.get(r["regime"])
                        if not tag:
                            continue
                        macro(f"Regime{tag}Auc", r["mean_auc"])
                        macro(f"Regime{tag}Spread", r["sector_spread"])
                        macro(f"Regime{tag}Model", str(r["best_model"]))
                    if "vol_regime_mean_gap" in rinfo:
                        macro("RegimeVolGap", rinfo["vol_regime_mean_gap"], "{:+.4f}")
                        macro("RegimeVolGapP", rinfo.get("vol_regime_wilcoxon_p",
                                                         np.nan), "{:.3f}")
                    if "trend_regime_mean_gap" in rinfo:
                        macro("RegimeTrendGap", rinfo["trend_regime_mean_gap"], "{:+.4f}")
                        macro("RegimeTrendGapP", rinfo.get("trend_regime_wilcoxon_p",
                                                           np.nan), "{:.3f}")
                    note("    regimes: " + ", ".join(
                        f"{r['regime']} AUC {r['mean_auc']:.4f} "
                        f"(spread {r['sector_spread']:.4f})"
                        for _, r in rsumm.iterrows()))
        except Exception as e:
            note(f"    [WARN] regime analysis failed ({type(e).__name__}: {e})")

    # ---- point 8: abstention ----------------------------------------------
    if RUN_ABSTENTION:
        try:
            curve = abstention_curve(preds)
            if len(curve):
                curve.to_csv(f"{OUT_DIR}/abstention_curve.csv", index=False)
                res["abstention_curve"] = curve
                best = curve.sort_values("accuracy", ascending=False).iloc[0]
                macro("AbstainBestModel", str(best["model"]))
                macro("AbstainBestCoverage", 100 * float(best["coverage"]), "{:.0f}")
                macro("AbstainBestAcc", float(best["accuracy"]))
                macro("AbstainBestGain", float(best["gain_vs_full"]), "{:+.4f}")
                tight = curve[curve["coverage"] == min(curve["coverage"])]
                macro("AbstainBestSkill", float(best["skill_vs_majority"]), "{:+.4f}")
                macro("AbstainTightCoverage", 100 * float(tight["coverage"].iloc[0]), "{:.0f}")
                macro("AbstainTightAccMean", float(tight["accuracy"].mean()))
                macro("AbstainTightGainMean", float(tight["gain_vs_full"].mean()), "{:+.4f}")
                macro("AbstainTightBaseMean", float(tight["baseline_majority"].mean()))
                macro("AbstainTightSkillMean", float(tight["skill_vs_majority"].mean()), "{:+.4f}")
                macro("AbstainTightSkillConst", float(tight["skill_vs_const"].mean()), "{:+.4f}")
                macro("AbstainTightUpShare", 100 * float(tight["up_share"].mean()), "{:.1f}")
                macro("NAbstainModelsImprove",
                      str(int((tight["gain_vs_full"] > 0).sum())))
                macro("NAbstainModelsSkilled",
                      str(int((tight["skill_vs_majority"] > 0).sum())))
                full_cov = curve[curve["coverage"] == max(ABSTAIN_COVERAGES)]
                macro("AbstainFullSkillMean",
                      float(full_cov["skill_vs_majority"].mean()), "{:+.4f}")
            gate_cells, ginfo = abstention_gate(fold_perf, preds)
            if ginfo:
                gate_cells.to_csv(f"{OUT_DIR}/abstention_gate.csv", index=False)
                res["abstention_gate"] = ginfo
                macro("GateCoverage", 100 * ginfo["coverage"], "{:.1f}")
                macro("GateAucDeployed", ginfo["auc_deployed"])
                macro("GateAucAll", ginfo["auc_all"])
                macro("GateAucGain", ginfo["auc_gain"], "{:+.4f}")
                macro("GateShareAboveHalf", 100 * ginfo["share_deployed_above_half"], "{:.1f}")
                macro("GateShareAboveHalfAll", 100 * ginfo["share_all_above_half"], "{:.1f}")
                note(f"    abstention gate: deploys "
                     f"{100 * ginfo['coverage']:.1f}% of cells, AUC "
                     f"{ginfo['auc_deployed']:.4f} vs {ginfo['auc_all']:.4f} "
                     f"({ginfo['auc_gain']:+.4f})")
        except Exception as e:
            note(f"    [WARN] abstention block failed ({type(e).__name__}: {e})")

    # ---- point 9: selection baselines --------------------------------------
    if RUN_SELECTION_BASELINES:
        try:
            pol, pinfo = selection_policies(fold_perf, sci)
            if len(pol):
                pol.to_csv(f"{OUT_DIR}/selection_policies.csv", index=False)
                res["policies"], res["policy_info"] = pol, pinfo
                for _, r in pol.iterrows():
                    tag = "".join(w.capitalize() for w in r["policy"].split("_"))
                    macro(f"Policy{tag}Auc", r["mean_auc"])
                    macro(f"Policy{tag}Gain", r["gain_vs_universal"], "{:+.4f}")
                macro("PolicyBest", str(pol.iloc[0]["policy"].replace("_", " ")))
                dep = pol[pol["deployable"]]
                macro("PolicyBestDeployable",
                      str(dep.iloc[0]["policy"].replace("_", " ")) if len(dep) else "none")
                if {"sci_band", "validated_recent"} <= set(pol["policy"]):
                    a = float(pol.loc[pol["policy"] == "validated_recent", "mean_auc"].iloc[0])
                    b = float(pol.loc[pol["policy"] == "sci_band", "mean_auc"].iloc[0])
                    macro("PolicyValidatedMinusSCI", a - b, "{:+.4f}")
                note("    selection policies: " + ", ".join(
                    f"{r['policy']} {r['mean_auc']:.4f}" for _, r in pol.iterrows()))
        except Exception as e:
            note(f"    [WARN] selection policies failed ({type(e).__name__}: {e})")

    # ---- points 1 and 10: the phenomenon, and whether it is actionable -----
    try:
        sector_ci = boot.get("sectors", pd.DataFrame())
        drh = decision_relevant_heterogeneity(rq1, lstm_seeds, sector_ci, fold_perf)
        res["decision_relevant"] = drh
        macro("HetSpread", drh["spread"])
        macro("HetSD", drh["sd_between"])
        macro("HetFloorSeed", drh["floor_seed"])
        macro("HetFloorSampling", drh["floor_sampling"])
        macro("HetFloorPeriod", drh["floor_period"])
        macro("HetFloorBinding", drh["binding_floor"])
        macro("HetFloorBindingName", str(drh["binding_floor_name"]))
        macro("HetRatioToFloor", drh["ratio_to_binding"], "{:.2f}")
        macro("HetDecisionRelevant", "yes" if drh["decision_relevant"] else "no")
        if drh["share_sector_pairs_overlapping"] == drh["share_sector_pairs_overlapping"]:
            macro("HetPairsOverlapping",
                  100 * drh["share_sector_pairs_overlapping"], "{:.0f}")

        het = heterogeneity_summary(pooled, rq1, sector_ci, drh)
        het.to_csv(f"{OUT_DIR}/heterogeneity_summary.csv", index=False)
        res["heterogeneity"] = het
        note(f"    heterogeneity: spread {drh['spread']:.4f} against a binding "
             f"{drh['binding_floor_name']} floor of {drh['binding_floor']:.4f} "
             f"-> decision-relevant: "
             f"{'yes' if drh['decision_relevant'] else 'no'}")
    except Exception as e:
        note(f"    [WARN] heterogeneity summary failed ({type(e).__name__}: {e})")

    # ---- LaTeX bodies ------------------------------------------------------
    try:
        write_extension_tables(
            OUT_DIR,
            res.get("heterogeneity", pd.DataFrame()),
            res.get("panel", pd.DataFrame()),
            res.get("temporal_variance", pd.DataFrame()),
            res.get("regime_summary", pd.DataFrame()),
            res.get("abstention_curve", pd.DataFrame()),
            res.get("policies", pd.DataFrame()))
    except Exception as e:
        note(f"    [WARN] extension tables failed ({type(e).__name__}: {e})")

    return res


def main() -> None:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(RANDOM_STATE)

    # QUICK_MODE trims everything for a sanity-check run
    n_seeds = 1 if QUICK_MODE else N_SEEDS_LSTM
    n_boot = 50 if QUICK_MODE else BOOTSTRAP_B

    note("environment")
    note(f"    python {platform.python_version()} | numpy {np.__version__} | "
         f"pandas {pd.__version__} | xgboost {'yes' if _HAS_XGB else 'NO'} | "
         f"torch {'yes' if _HAS_TORCH else 'NO'}")
    macro("SeedValue", str(RANDOM_STATE))
    macro("NFolds", str(N_SPLITS))
    macro("NSeedsLSTM", str(n_seeds))
    macro("BootstrapB", str(n_boot))
    macro("SeqLen", str(SEQ_LEN))
    macro("LSTMEpochs", str(LSTM_EPOCHS))
    macro("LSTMBatch", str(LSTM_BATCH))
    macro("LSTMHidden", str(LSTM_HIDDEN))
    macro("MinHistorySCI", str(MIN_HISTORY_SCI))

    # ---- 1/8  load raw data ------------------------------------------------
    note("1/8  loading data")
    df = load_data()
    if QUICK_MODE:
        keep = df[COL_SECTOR].drop_duplicates().head(3)
        df = df[df[COL_SECTOR].isin(keep)]
        note(f"    QUICK_MODE: {list(keep)}")

    # ---- 2/8  build the SCI ------------------------------------------------
    note("2/8  Sectoral Complexity Index")
    sci = compute_sci(df)
    sci.to_csv(f"{OUT_DIR}/sci_by_sector.csv")
    sci[["SCI", "SCI_hurst_plus", "band"]].to_csv(f"{OUT_DIR}/sci_hurst_sensitivity.csv")
    print(sci[["SCI", "band"]].round(4).to_string())

    # ---- 3/8  build the feature panel -------------------------------------
    note("3/8  features")
    feat = build_features(df)

    # ---- 4/8  walk-forward forecasting ------------------------------------
    note("4/8  walk-forward forecasting")
    models = available_models()
    if len(models) < 4:
        note("    [WARN] fewer than four models available. Macros for the "
             "missing model will NOT be written to numbers.tex, so the thesis "
             "would silently keep its previous values. Fix the install first.")
    macro("NModels", str(len(models)))
    macro("ModelList", ", ".join(models))
    parts_dir = f"{OUT_DIR}/pred_parts"
    os.makedirs(parts_dir, exist_ok=True)
    preds, cache_path = (_load_frame(f"{OUT_DIR}/predictions")
                         if REUSE_PREDICTIONS else (None, None))
    if preds is not None:
        note(f"    reusing {len(preds):,} cached predictions from {cache_path}")
        note("    (delete results/predictions.* or set REUSE_PREDICTIONS=False "
             "to refit)")
    else:
        # One checkpoint per sector.  
        all_preds = []
        sectors = list(feat.groupby(COL_SECTOR, sort=False))
        for i, (sector, fs) in enumerate(sectors, 1):
            stem = f"{parts_dir}/" + "".join(
                ch if ch.isalnum() else "_" for ch in str(sector))
            part, part_path = (_load_frame(stem) if REUSE_PREDICTIONS
                               else (None, None))
            if part is not None:
                note(f"    [{i}/{len(sectors)}] {sector}  -- checkpoint found, "
                     f"skipping ({len(part):,} rows)")
                all_preds.append(part)
                continue
            note(f"    [{i}/{len(sectors)}] {sector}  ({len(fs):,} rows)")
            part = run_sector(fs, models, n_seeds_lstm=n_seeds)
            if not len(part):
                note(f"    [warn] {sector} produced no predictions")
                continue
            written = _save_frame(part, stem)
            note(f"        checkpoint -> {written}")
            all_preds.append(part)

        if not all_preds:
            raise SystemExit("no predictions produced; nothing to analyse")
        preds = pd.concat(all_preds, ignore_index=True)
        preds = restrict_to_common_support(preds)
        written = _save_frame(preds, f"{OUT_DIR}/predictions")
        note(f"    saved {len(preds):,} predictions -> {written}")

    # ---- 5/8  metrics -----------------------------------------------------
    note("5/8  metrics")
    # PRIMARY SPECIFICATION: fold-blocked. AUC is a
    # ranking statistic, so computing it over the pooled predictions of several
    # separately fitted folds estimates are biased downwards when the target base rate drifts between folds.
    
    per_seed = metrics_by_foldblocked(preds, ["model", COL_SECTOR, "seed"])
    per_seed.to_csv(f"{OUT_DIR}/performance_by_model_sector_seed.csv", index=False)
    lstm_seeds = per_seed[per_seed["model"] == "LSTM"]
    lstm_seeds.to_csv(f"{OUT_DIR}/lstm_seed_stability.csv", index=False)

    pooled = collapse_lstm_seeds(per_seed, ["model", COL_SECTOR])
    pooled.to_csv(f"{OUT_DIR}/performance_by_model_sector.csv", index=False)
    pooled.to_csv(f"{OUT_DIR}/performance_foldblocked.csv", index=False)

    
    across = collapse_lstm_seeds(
        metrics_by(preds, ["model", COL_SECTOR, "seed"]), ["model", COL_SECTOR])
    across.to_csv(f"{OUT_DIR}/performance_pooled.csv", index=False)
    cmp_df = (across[[COL_SECTOR, "model", "auc"]].rename(columns={"auc": "pooled"})
              .merge(pooled[[COL_SECTOR, "model", "auc"]]
                     .rename(columns={"auc": "fold_blocked"}),
                     on=[COL_SECTOR, "model"]))
    cmp_df["gap"] = cmp_df["fold_blocked"] - cmp_df["pooled"]
    cmp_df["side_flip"] = (cmp_df["pooled"] < .5) != (cmp_df["fold_blocked"] < .5)
    cmp_df.to_csv(f"{OUT_DIR}/foldblock_comparison.csv", index=False)
    macro("AggMeanAbsGap", cmp_df["gap"].abs().mean())
    macro("AggSideFlips", str(int(cmp_df["side_flip"].sum())))
    macro("AggWorstGap", cmp_df["gap"].max(), "{:+.4f}")
    macro("PooledSpread", across.groupby(COL_SECTOR)["auc"].mean()
                                .pipe(lambda s: s.max() - s.min()))
    # Cache Real Estate specifically because the appendix picks it out.
    for _lbl, _fr in (("RealEstatePooled", across), ("RealEstateFoldBlocked", pooled)):
        _v = _fr.loc[_fr[COL_SECTOR] == "Real Estate", "auc"]
        if len(_v):
            macro(_lbl, _v.mean())
    # The 8 worst aggregation gaps go into a small LaTeX table.
    _tab = (cmp_df.reindex(cmp_df["gap"].abs().sort_values(ascending=False).index)
                  .head(8))
    with open(f"{OUT_DIR}/tab_aggregation.tex", "w") as _f:
        _f.write("\\begin{tabular}{llrrrc}\n\\toprule\n")
        _f.write("Sector & Model & Pooled & Fold-blocked & Gap & Side change "
                 "\\\\\n\\midrule\n")
        for _, _r in _tab.iterrows():
            _f.write(f"{_r[COL_SECTOR]} & {_r['model']} & {_r['pooled']:.4f} & "
                     f"{_r['fold_blocked']:.4f} & ${_r['gap']:+.4f}$ & "
                     f"{'yes' if _r['side_flip'] else 'no'} \\\\\n")
        _f.write("\\bottomrule\n\\end{tabular}\n")

    fold_seed = metrics_by(preds, ["model", COL_SECTOR, "fold", "seed"])
    fold_perf = collapse_lstm_seeds(fold_seed, ["model", COL_SECTOR, "fold"])
    fold_perf.to_csv(f"{OUT_DIR}/performance_by_fold.csv", index=False)

    print(pooled.groupby("model", observed=True)[["accuracy", "auc", "f1", "mcc"]]
                .mean().round(4).to_string())

    # Headline win/lose counts vs baselines.
    beat_test = int((pooled["accuracy"] > pooled["baseline_test"]).sum())
    beat_train = int((pooled["accuracy"] > pooled["baseline_train"]).sum())
    macro("NCells", str(len(pooled)))
    macro("NBeatBaselineTest", str(beat_test))
    macro("NBeatBaselineTrain", str(beat_train))
    macro("NAucAboveHalf", str(int((pooled["auc"] > 0.5).sum())))
    macro("NMccPositive", str(int((pooled["mcc"] > 0).sum())))
    macro("AucMin", pooled["auc"].min(), "{:.4f}")
    macro("AucMax", pooled["auc"].max(), "{:.4f}")
    # Per-model means 
    for m, g in pooled.groupby("model", observed=True):
        tag = m.replace("Regression", "Reg")
        macro(f"Mean{tag}Auc", g["auc"].mean())
        macro(f"Mean{tag}Acc", g["accuracy"].mean())
        macro(f"Mean{tag}FOne", g["f1"].mean())
        macro(f"Mean{tag}Mcc", g["mcc"].mean(), "{:+.4f}")
    if len(lstm_seeds):
        sd = lstm_seeds.groupby(COL_SECTOR, observed=True)["auc"].std(ddof=1)
        macro("LstmSeedSdMean", sd.mean())
        macro("LstmSeedSdMax", sd.max())

    # ---- 6/8  hypothesis tests --------------------------------------------
    note("6/8  hypothesis tests")
    rq1 = rq1_sector_heterogeneity(fold_perf, pooled)
    rq2 = rq2_complexity(pooled, sci)
    rq3 = rq3_models(pooled)
    mcn = mcnemar_all_pairs(preds)
    mcn.to_csv(f"{OUT_DIR}/mcnemar_pairs.csv", index=False)

    # Component-wise decomposition of the index
    
    decomp = rq2_metric_decomposition(pooled, sci)
    if len(decomp):
        decomp.to_csv(f"{OUT_DIR}/sci_decomposition.csv", index=False)
        for _, r in decomp.iterrows():
            tag = {"volatility": "Vol", "sampen": "SampEn", "hurst": "Hurst",
                   "amihud": "Amihud", "vol_var": "VolVar"}[r["metric"]]
            macro(f"Decomp{tag}Rho", r["rho_signed"], "{:+.3f}")
            macro(f"Decomp{tag}P", r["p"], "{:.3f}")
            macro(f"Decomp{tag}PHolm", r["p_holm"], "{:.3f}")
        macro("NDecompNegative", str(int((decomp["rho_signed"] < 0).sum())))
        macro("NDecompSigHolm", str(int((decomp["p_holm"] < ALPHA).sum())))
        macro("DecompRhoMin", decomp["rho_signed"].min(), "{:+.3f}")
        macro("DecompRhoMax", decomp["rho_signed"].max(), "{:+.3f}")
        print("\nSCI component decomposition (signed as in Eq. 13):")
        print(decomp[["metric", "rho_signed", "p", "p_holm"]]
              .round(4).to_string(index=False))

    # PCA-weighted SCI as an alternative weighting.
    sci_pca = pd.DataFrame()
    if RUN_SCI_PCA:
        try:
            sci_pca = sci_pca_weighted(sci)
            sci_pca.to_csv(f"{OUT_DIR}/sci_pca_weighted.csv")
            perf_s = pooled.groupby(COL_SECTOR, observed=True)["auc"].mean()
            dd = sci_pca.join(perf_s.rename("perf")).dropna()
            rho_pca, p_pca = stats.spearmanr(dd["SCI_pca"], dd["perf"])
            macro("SCIPCAVarExplained", 100 * sci_pca.attrs["var_explained"], "{:.1f}")
            macro("SCIPCATau", sci_pca.attrs["kendall_tau_vs_equal"], "{:.3f}")
            macro("SCIPCATauP", sci_pca.attrs["kendall_tau_p"], "{:.3f}")
            macro("SCIPCARho", float(rho_pca), "{:+.3f}")
            macro("SCIPCARhoP", float(p_pca), "{:.3f}")
            macro("SCIPCABandChanges", str(int(
                (sci_pca["band_pca"].astype(str)
                 != sci["band"].astype(str)).sum())))
            note(f"    PCA-weighted SCI: PC1 explains "
                 f"{100 * sci_pca.attrs['var_explained']:.1f}% of variance, "
                 f"rho vs AUC = {rho_pca:+.3f} (p={p_pca:.3f})")
        except Exception as e:
            note(f"    [warn] PCA-weighted SCI failed ({type(e).__name__}: {e})")

    macro("RQoneKendallW", rq1["kendall_w"], "{:.3f}")
    macro("RQoneKendallWOld", rq1["kendall_w_pooled_blocks"], "{:.3f}")
    macro("RQoneSectorSD", rq1["sector_mean_sd"])

    macro("RQoneChiSq", rq1["chi2"], "{:.2f}")
    macro("RQonePval", rq1["p"], "{:.3f}")
    macro("RQoneBlocks", str(rq1["n_blocks"]))
    macro("RQoneTreatments", str(rq1["k_treatments"]))
    macro("RQoneChiSqOld", rq1["chi2_pooled_blocks"], "{:.2f}")
    macro("RQonePvalOld", rq1["p_pooled_blocks"], "{:.3f}")
    macro("RQoneSpread", rq1["sector_mean_spread"])
    macro("RQtwoRho", rq2["rho"], "{:+.3f}")
    macro("RQtwoPval", rq2["p"], "{:.3f}")
    macro("RQtwoMWU", rq2["mannwhitney_U_H2"], "{:.1f}")
    macro("RQtwoMWPval", rq2["mannwhitney_p_H2"], "{:.3f}")
    macro("RQtwoMWPvalTwo", rq2["mannwhitney_p_two_sided"], "{:.3f}")
    for b, v in rq2["band_means"].items():
        macro(f"BandMean{str(b).capitalize()}", v)
    for rec in rq2["per_model"]:
        tag = rec["model"].replace("Regression", "Reg")
        macro(f"Rho{tag}", rec["rho"], "{:+.3f}")
        macro(f"Rho{tag}P", rec["p"], "{:.3f}")
        macro(f"Rho{tag}PHolm", rec["p_holm"], "{:.3f}")
    macro("RQthreeChiSq", rq3["chi2"], "{:.2f}")
    macro("RQthreePval", rq3["p"], "{:.3f}")
    macro("RQthreeCD", rq3["critical_difference"], "{:.2f}")
    macro("RQthreeMaxGap", rq3["max_rank_gap"], "{:.2f}")
    for m, r in rq3["avg_ranks"].items():
        macro(f"Rank{m.replace('Regression', 'Reg')}", r, "{:.2f}")
    macro("NMcNemarTests", str(len(mcn)))
    macro("NMcNemarSig", str(int(mcn["significant_holm"].sum()) if len(mcn) else 0))

    # ---- 7/8  date-block bootstrap (the slow step) -----------------------
    note("7/8  date-block bootstrap  (this is the slow part)")
    boot = date_block_bootstrap(preds, sci, B=n_boot)
    boot["cells"].to_csv(f"{OUT_DIR}/auc_bootstrap_ci.csv", index=False)
    boot["bands"].to_csv(f"{OUT_DIR}/auc_bootstrap_bands.csv", index=False)
    boot["margins"].to_csv(f"{OUT_DIR}/auc_bootstrap_margins.csv", index=False)
    if len(boot.get("sectors", pd.DataFrame())):
        boot["sectors"].to_csv(f"{OUT_DIR}/auc_bootstrap_sectors.csv", index=False)
        macro("NSectorsCIExcludeHalf",
              str(int(boot["sectors"]["excludes_half"].sum())))
    if np.isfinite(boot.get("spread_ci", (np.nan, np.nan))[0]):
        macro("RQoneSpreadCILo", boot["spread_ci"][0])
        macro("RQoneSpreadCIHi", boot["spread_ci"][1])
    cells = boot["cells"]
    macro("NCellsAboveChance", str(int((cells["ci_lo"] > 0.5).sum())))
    macro("NCellsBelowChance", str(int((cells["ci_hi"] < 0.5).sum())))
    macro("NCellsCIExcludeHalf", str(int(cells["excludes_half"].sum())))
    macro("NCellsHolmSig", str(int((cells["p_holm"] < ALPHA).sum())))
    macro("RhoCILo", boot["rho_ci"][0], "{:+.3f}")
    macro("RhoCIHi", boot["rho_ci"][1], "{:+.3f}")
    if len(boot["margins"]):
        for b, r in boot["margins"].set_index("band").iterrows():
            macro(f"Margin{str(b).capitalize()}", r["mean"], "{:+.4f}")
            macro(f"Margin{str(b).capitalize()}CILo", r["ci_lo"], "{:+.4f}")
            macro(f"Margin{str(b).capitalize()}CIHi", r["ci_hi"], "{:+.4f}")

    # Adaptive selection policy (fit early, score late).
    ad = adaptive_selection(fold_perf, pooled, sci)
    macro("AdaptiveOOS", ad["oos_adaptive_mean"])
    macro("AdaptiveOOSFixed", ad["oos_best_fixed_mean"])
    macro("AdaptiveOOSFixedModel", ad["oos_best_fixed_model"])
    macro("AdaptiveOOSUplift", ad["oos_uplift"], "{:+.4f}")
    macro("AdaptiveInSample", ad["in_sample_adaptive_mean"])
    macro("AdaptiveInSampleFixed", ad["in_sample_best_fixed_mean"])
    macro("AdaptiveInSampleFixedModel", ad["in_sample_best_fixed_model"])
    macro("AdaptiveOracle", ad["per_sector_oracle_mean"])
    _lb = (pooled.assign(band=pooled[COL_SECTOR].map(sci["band"].astype(str)))
                 .query("model == 'LSTM'").groupby("band")["auc"].mean())
    if {"low", "medium", "high"} <= set(_lb.index):
        for b in ("low", "medium", "high"):
            macro(f"LstmBand{b.capitalize()}", _lb[b])
        macro("LstmBandRise", _lb["high"] - _lb["low"], "{:+.4f}")

    for b, v in ad["band_to_model"].items():
        macro(f"BestModel{str(b).capitalize()}", v["model"])

    # ---- 8/8  robustness --------------------------------------------------
    note("8/8  robustness, attribution and diagnostics")

    # --- SHAP feature attribution -----------------------------------------
    shap_detail, shap_groups, shap_corr = (pd.DataFrame(),) * 3
    if RUN_SHAP:
        try:
            note(f"    SHAP attribution ({SHAP_MODEL}, "
                 f"{'shap' if _HAS_SHAP else 'GAIN FALLBACK - pip install shap'})")
            shap_detail, shap_groups = shap_analysis(feat, sci)
            if len(shap_detail):
                shap_detail.to_csv(f"{OUT_DIR}/shap_importance.csv", index=False)
                shap_groups.to_csv(f"{OUT_DIR}/shap_groups.csv")
                shap_corr = shap_vs_complexity(shap_groups, sci)
                if len(shap_corr):
                    shap_corr.to_csv(f"{OUT_DIR}/shap_vs_sci.csv", index=False)
                macro("ShapModel", SHAP_MODEL)
                macro("ShapMethod", shap_detail.attrs.get("method", "shap-tree"))
                macro("ShapRowsPerSector", f"{int(shap_detail['n_explained'].max()):,}")
                # Which input does the model lean on, and how concentrated is it?
                overall = (shap_detail.groupby("feature")["share"].mean()
                                      .sort_values(ascending=False))
                macro("ShapTopFeature", str(overall.index[0]))
                macro("ShapTopShare", 100 * overall.iloc[0], "{:.1f}")
                gmean = shap_groups.mean().sort_values(ascending=False)
                macro("ShapTopGroup", str(gmean.index[0]))
                macro("ShapTopGroupShare", 100 * gmean.iloc[0], "{:.1f}")
                # How much does the profile move across sectors?  If the
                # premise behind H3 held, this would not be small.
                spread = (shap_groups.max() - shap_groups.min()).max()
                macro("ShapGroupSpread", 100 * float(spread), "{:.1f}")
                if len(shap_corr):
                    macro("NShapSigHolm",
                          str(int((shap_corr["p_holm"] < ALPHA).sum())))
                    macro("ShapRhoMax", shap_corr["rho"].abs().max(), "{:.3f}")
                print("\nSHAP attribution share by feature group:")
                print(shap_groups.round(3).to_string())
        except Exception as e:
            note(f"    [WARN] SHAP block failed ({type(e).__name__}: {e}); "
                 f"main results unaffected")

    # --- one global model vs the per-sector models -------------------------
    gvs = pd.DataFrame()
    if RUN_GLOBAL_BASELINE:
        try:
            gcmp, gvs = global_vs_sector(feat, pooled, models)
            if len(gvs):
                gcmp.to_csv(f"{OUT_DIR}/global_vs_sector_cells.csv", index=False)
                gvs.to_csv(f"{OUT_DIR}/global_vs_sector.csv", index=False)
                macro("GlobalFitAuc", gvs["auc_global"].mean())
                macro("SectorFitAuc", gvs["auc_sector"].mean())
                macro("SectorFitGain", gvs["gain"].mean(), "{:+.4f}")
                macro("SectorFitGainMax", gvs["gain"].max(), "{:+.4f}")
                macro("NSectorsFitHelps", str(int((gvs["gain"] > 0).sum())))
                macro("GlobalSpread",
                      gvs["auc_global"].max() - gvs["auc_global"].min())
                note(f"    per-sector {gvs['auc_sector'].mean():.4f} vs global "
                     f"{gvs['auc_global'].mean():.4f} "
                     f"({gvs['gain'].mean():+.4f})")
        except Exception as e:
            note(f"    [WARN] global-vs-sector check failed "
                 f"({type(e).__name__}: {e}); main results unaffected")

    rob_summary = pd.DataFrame()
    try:
        _run_robustness_block(df, feat, sci, pooled, models, n_seeds)
        rob_summary = _ROB_SUMMARY
    except Exception as e:
        # Robustness is optional - never let it sink the headline results.
        note(f"    [WARN] robustness block failed ({type(e).__name__}: {e}). "
             f"Main results and LaTeX artefacts are unaffected; the robustness "
             f"macros keep their fallback values.")

    # --- extensions requested at supervision  ---------------------
    note("    extensions: panel, regimes, abstention, selection baselines")
    ext = run_extensions(df, preds, feat, fold_perf, pooled, lstm_seeds,
                         sci, rq1, boot)

    note("writing LaTeX artefacts")
    write_tables(OUT_DIR, sci, pooled, boot, mcn, lstm_seeds, rob_summary,
                 sci[["SCI", "SCI_hurst_plus", "band"]])
    write_extra_tables(OUT_DIR, sci, rq1, boot, decomp, shap_groups,
                       shap_corr, gvs, sci_pca)
    write_numbers(OUT_DIR)

    # One JSON with everything for downstream inspection / archiving.
    summary = {"rq1": rq1, "rq2": {k: v for k, v in rq2.items() if k != "table"},
               "rq3": rq3, "adaptive": ad,
               "sci_decomposition": decomp.to_dict("records") if len(decomp) else [],
               "shap_vs_sci": shap_corr.to_dict("records") if len(shap_corr) else [],
               "global_vs_sector": gvs.to_dict("records") if len(gvs) else [],
               "bootstrap": {"B": n_boot, "rho_ci": boot["rho_ci"],
                             "spread_ci": boot.get("spread_ci"),
                             "cells_excluding_half": int(cells["excludes_half"].sum())},
               "mcnemar_significant": int(mcn["significant_holm"].sum()) if len(mcn) else 0,
               "panel": ext.get("panel_summary", {}),
               "temporal": ext.get("temporal", {}),
               "regimes": ext.get("regime_info", {}),
               "regime_summary": (ext["regime_summary"].to_dict("records")
                                  if len(ext.get("regime_summary", [])) else []),
               "abstention_gate": ext.get("abstention_gate", {}),
               "selection_policies": (ext["policies"].to_dict("records")
                                      if len(ext.get("policies", [])) else []),
               "decision_relevant_heterogeneity": ext.get("decision_relevant", {}),
               "runtime_minutes": round((time.time() - t0) / 60, 1),
               "config": {"n_splits": N_SPLITS, "n_seeds_lstm": n_seeds,
                          "include_lag0": INCLUDE_LAG0, "seq_len": SEQ_LEN,
                          "lstm_epochs": LSTM_EPOCHS, "bootstrap_B": n_boot,
                          "random_state": RANDOM_STATE, "quick_mode": QUICK_MODE}}
    with open(f"{OUT_DIR}/rq_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Final console summary - one screen of headline numbers.
    print("\n" + "=" * 70)
    print(f"RQ1  chi2={rq1['chi2']:.2f}  p={rq1['p']:.4f}  "
          f"({rq1['n_blocks']} blocks x {rq1['k_treatments']} sectors)")
    print(f"     effect size: Kendall W={rq1['kendall_w']:.3f}  "
          f"range={rq1['sector_mean_spread']:.4f}  "
          f"SD={rq1['sector_mean_sd']:.4f}")
    print(f"     v1 specification: chi2={rq1['chi2_pooled_blocks']:.2f}  "
          f"p={rq1['p_pooled_blocks']:.4f}  ({rq1['n_blocks_pooled']} blocks)")
    print(f"RQ2  rho={rq2['rho']:+.3f}  p={rq2['p']:.4f}  "
          f"bootstrap 95% CI [{boot['rho_ci'][0]:+.3f}, {boot['rho_ci'][1]:+.3f}]")
    print(f"     Mann-Whitney low>high p={rq2['mannwhitney_p_H2']:.4f}  "
          f"(two-sided {rq2['mannwhitney_p_two_sided']:.4f})")
    print(f"RQ3  chi2={rq3['chi2']:.2f}  p={rq3['p']:.4f}  CD={rq3['critical_difference']:.2f}  "
          f"max gap={rq3['max_rank_gap']:.2f}")
    print(f"     McNemar significant after Holm: "
          f"{int(mcn['significant_holm'].sum()) if len(mcn) else 0}/{len(mcn)}")
    print(f"CI   cells whose 95% CI excludes 0.5: {int(cells['excludes_half'].sum())}"
          f"/{len(cells)}  (above {int((cells['ci_lo'] > 0.5).sum())}, "
          f"below {int((cells['ci_hi'] < 0.5).sum())})")
    print(f"ADPT out-of-sample {ad['oos_adaptive_mean']:.4f} vs best fixed "
          f"{ad['oos_best_fixed_mean']:.4f} -> {ad['oos_uplift']:+.4f}   "
          f"(in-sample {ad['in_sample_adaptive_mean']:.4f})")
    _drh = ext.get("decision_relevant", {})
    if _drh:
        print(f"HET  spread={_drh['spread']:.4f} vs binding "
              f"{_drh['binding_floor_name']} floor {_drh['binding_floor']:.4f} "
              f"-> decision-relevant: "
              f"{'YES' if _drh['decision_relevant'] else 'NO'}")
    _ps = ext.get("panel_summary", {})
    for _k, _v in _ps.items():
        print(f"PNL  {_k}: SCI beta={_v['sci_coef']:+.4f} (p={_v['sci_p']:.3f})  "
              f"SCI x model F={_v['interaction_F']:.2f} "
              f"(p={_v['interaction_p']:.3f})  n={_v['n']}")
    _t = ext.get("temporal", {})
    if _t:
        print(f"TIME variance shares: sector {100 * _t['share_sector']:.1f}%  "
              f"period {100 * _t['share_period']:.1f}%  "
              f"interaction {100 * _t['share_interaction']:.1f}%")
    _pol = ext.get("policies")
    if _pol is not None and len(_pol):
        print("POL  " + "  ".join(f"{r['policy']}={r['mean_auc']:.4f}"
                                  for _, r in _pol.iterrows()))
    _g = ext.get("abstention_gate", {})
    if _g:
        print(f"ABST gate covers {100 * _g['coverage']:.1f}% of cells, AUC "
              f"{_g['auc_deployed']:.4f} vs {_g['auc_all']:.4f} "
              f"({_g['auc_gain']:+.4f})")
    print("=" * 70)
    note(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT_DIR}/")


if __name__ == "__main__":
    main()