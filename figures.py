"""
=============================================================================
Figures for the results chapter 
=============================================================================

Kept separate from pipeline.py so figures can be restyled without refitting.
Needs only matplotlib / pandas / numpy / scipy.

=============================================================================
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# --- Paths and global style ------------------------------------------------ #
OUT_DIR = "results"
FIG_DIR = f"{OUT_DIR}/figures"
METRIC = "auc"

# Keep band colour consistent across every figure 
BAND_ORDER = ["low", "medium", "high"]
BAND_COLOURS = {"low": "#4a7a4e", "medium": "#c79a2e", "high": "#b04545"}

# Metric labels used in axis legends / titles.
SCI_METRICS = ["volatility", "sampen", "hurst", "amihud", "vol_var"]
SCI_LABELS = {"volatility": "volatility", "sampen": "sample entropy",
              "hurst": "Hurst", "amihud": "Amihud", "vol_var": "volume var."}

# Nemenyi q_alpha, alpha = 0.05, Demsar (2006), indexed by number of models
_Q05 = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}

# A few rcParams so all figures share the same look.  
plt.rcParams.update({
    "figure.dpi": 120, "font.size": 10, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,          # embed real fonts, not Type 3
})


def _save(fig, name: str) -> None:
    """Wrapper so we always close the figure and print the same log line."""
    fig.savefig(f"{FIG_DIR}/{name}.pdf")
    plt.close(fig)
    print(f"  wrote {FIG_DIR}/{name}.pdf")


def _read(name: str, **kw):
    """Read a CSV from OUT_DIR, returning None if it's not there yet.

    Lets the optional figures degrade when their inputs are
    missing rather than crashing the whole script."""
    path = f"{OUT_DIR}/{name}"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kw)


def load():
    # Mandatory files.
    sci = _read("sci_by_sector.csv", index_col=0)
    if sci is None:
        raise SystemExit(f"{OUT_DIR}/sci_by_sector.csv not found -- run pipeline.py first")
    sci.index.name = "sector"

    perf = _read("performance_by_model_sector.csv")
    if perf is None:
        raise SystemExit(f"{OUT_DIR}/performance_by_model_sector.csv not found")
    
    if "Sector" in perf.columns:
        perf = perf.rename(columns={"Sector": "sector"})

    
    ci = _read("auc_bootstrap_ci.csv")
    seeds = _read("lstm_seed_stability.csv")
    if seeds is not None and "Sector" in seeds.columns:
        seeds = seeds.rename(columns={"Sector": "sector"})
    return sci, perf, ci, seeds


# --------------------------------------------------------------------------- #


def fig_metric_correlation(sci: pd.DataFrame) -> None:
    """5x5 correlation heatmap of the SCI components.
    """
    metrics = [m for m in SCI_METRICS if m in sci.columns]
    corr = sci[metrics].corr()
    labels = [SCI_LABELS.get(m, m) for m in metrics]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(labels)

    # Print the numeric correlation in each cell, white text on dark cells,
    
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Correlation among the five SCI metrics (across sectors)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)   # match matplotlib's magic fraction
    _save(fig, "fig_metric_correlation")


def fig_sci_vs_performance(sci: pd.DataFrame, perf: pd.DataFrame,
                           ci: pd.DataFrame | None) -> None:
    """The key RQ2 scatter: SCI vs mean AUC, coloured by complexity band."""
    mean_perf = perf.groupby("sector")[METRIC].mean()
    d = sci.join(mean_perf.rename("perf")).dropna(subset=["perf"])
    rho, p = stats.spearmanr(d["SCI"], d["perf"])

   
    err = None
    if ci is not None and {"sector", "ci_lo", "ci_hi"}.issubset(ci.columns):
        g = ci.groupby("sector")[["ci_lo", "ci_hi"]].mean()
        g = g.reindex(d.index)
        lo = (d["perf"] - g["ci_lo"]).clip(lower=0)
        hi = (g["ci_hi"] - d["perf"]).clip(lower=0)
        if lo.notna().all() and hi.notna().all():
            err = np.vstack([lo.to_numpy(), hi.to_numpy()])

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    
    if False:  
        ax.errorbar(d["SCI"], d["perf"], yerr=err, fmt="none",
                    ecolor="0.7", elinewidth=1, capsize=2, zorder=2)
    for b in BAND_ORDER:
        sub = d[d["band"] == b]
        ax.scatter(sub["SCI"], sub["perf"], s=70, color=BAND_COLOURS[b],
                   label=b, zorder=3)
    # Label each point with its sector so readers can see which is which.
    for s, row in d.iterrows():
        ax.annotate(s, (row["SCI"], row["perf"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.5, color="0.6", ls="--", lw=0.8, label="chance (AUC 0.5)")
    ax.set_xlabel("Sectoral Complexity Index")
    ax.set_ylabel(f"mean {METRIC.upper()} across models")
    ax.set_title(f"SCI vs forecasting performance "
                 f"(Spearman $\\rho$={rho:.2f}, p={p:.2f})")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_sci_vs_performance")


def _maximal_cliques(ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Maximal runs of models whose average ranks lie within CD of each other.
    """
    k = len(ranks)
    runs = []
    # Find every maximal run starting at i whose ranks stay within CD of i.
    for i in range(k):
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) <= cd:
            j += 1
        if j > i:
            runs.append((i, j))
    # Keep only maximal runs - throw away ones fully contained in another.
    maximal = [r for r in runs
               if not any(a <= r[0] and r[1] <= b and (a, b) != r for a, b in runs)]
    return maximal


def fig_critical_difference(perf: pd.DataFrame) -> None:
    """Nemenyi critical-difference diagram (Demsar 2006).
    """
    wide = perf.pivot_table(index="sector", columns="model", values=METRIC).dropna()
    models = list(wide.columns)
    k, n = len(models), len(wide)
    if k < 3:
        print("  [note] CD diagram needs >= 3 models; skipped")
        return
    # average ranks per model (lower = better)
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    q05 = _Q05.get(k, 2.569)
    cd = q05 * np.sqrt(k * (k + 1) / (6.0 * n))

    names = list(ranks.index)
    rvals = ranks.to_numpy()
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 2.4 + 0.3 * k))
    ax.set_xlim(lo - 0.6, hi + 0.6)
    ax.set_ylim(0, 1)
    ax.axis("off")   # CD diagrams don't use axes - everything is drawn manually

    # the rank axis itself
    ax.plot([lo, hi], [0.78, 0.78], "k-", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0.78, 0.81], "k-", lw=1)
        ax.text(r, 0.85, str(r), ha="center")

    # model labels: alternate left/right of the axis, stacked by depth
    for idx, (name, rv) in enumerate(zip(names, rvals)):
        left = idx < len(names) / 2
        depth = idx if left else (len(names) - 1 - idx)
        y = 0.58 - 0.10 * depth
        xtext = lo - 0.5 if left else hi + 0.5
        ax.plot([rv, rv], [0.78, y], "k-", lw=0.8)
        ax.plot([rv, xtext], [y, y], "k-", lw=0.8)
        ax.text(xtext + (-0.05 if left else 0.05), y, f"{name} ({rv:.2f})",
                ha="right" if left else "left", va="center", fontsize=9)

    # the CD reference bar in the top-left
    ax.plot([lo, lo + cd], [0.94, 0.94], "k-", lw=2.5)
    ax.text(lo + cd / 2, 0.965, f"CD = {cd:.2f}", ha="center", fontsize=9)

    # the actual clique bars
    yb = 0.72
    for i, j in _maximal_cliques(rvals, cd):
        ax.plot([rvals[i] - 0.04, rvals[j] + 0.04], [yb, yb], "k-", lw=3.5)
        yb -= 0.035
    ax.set_title("Critical-difference diagram (Nemenyi, lower rank = better)")
    _save(fig, "fig_critical_difference")


def fig_auc_confidence_intervals(sci: pd.DataFrame, ci: pd.DataFrame | None) -> None:
    """Caterpillar plot: every cell's AUC with its date-block bootstrap interval.
    """
    if ci is None or not {"sector", "model", "ci_lo", "ci_hi"}.issubset(ci.columns):
        print("  [note] auc_bootstrap_ci.csv not found; skipping CI figure")
        return
    d = ci.copy()

    # Prefer the fold-blocked point estimate to the bootstrap mean, but fall
    # back if the performance file is missing.
    perf_auc = _read("performance_by_model_sector.csv")
    if perf_auc is not None:
        if "Sector" in perf_auc.columns:
            perf_auc = perf_auc.rename(columns={"Sector": "sector"})
        d = d.merge(perf_auc[["sector", "model", "auc"]],
                    on=["sector", "model"], how="left")
        d["auc"] = d["auc"].fillna(d["auc_boot_mean"])
    else:
        d["auc"] = d["auc_boot_mean"]

    band = sci["band"].astype(str).to_dict()
    d["band"] = d["sector"].map(band)
    d = d.sort_values("auc").reset_index(drop=True)   # caterpillar = sorted

    fig, ax = plt.subplots(figsize=(7.6, 0.22 * len(d) + 2))   # height grows with rows
    for i, r in d.iterrows():
        excl = (r.ci_lo > 0.5) or (r.ci_hi < 0.5)   # does the CI exclude chance?
        col = BAND_COLOURS.get(r.band, "0.4")
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col,
                lw=2.4 if excl else 1.2, alpha=1.0 if excl else 0.55,
                solid_capstyle="round")
        ax.scatter(r.auc, i, s=26 if excl else 14, color=col,
                   edgecolor="black" if excl else "none", linewidth=0.6, zorder=3)
    ax.axvline(0.5, color="0.35", ls="--", lw=1)
    ax.set_yticks(range(len(d)))
    # Tiny font because there are 44 sector/model labels stacked vertically.
    ax.set_yticklabels([f"{r.sector} -- {r.model}" for _, r in d.iterrows()],
                       fontsize=7)
    ax.set_xlabel("out-of-sample AUC with 95% date-block bootstrap interval")
    n_excl = int(((d.ci_lo > 0.5) | (d.ci_hi < 0.5)).sum())
    ax.set_title(f"AUC and uncertainty for all {len(d)} model--sector cells\n"
                 f"({n_excl} intervals exclude 0.5; heavier lines)", fontsize=10)
    handles = [plt.Line2D([], [], color=BAND_COLOURS[b], lw=2.4, label=b)
               for b in BAND_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right",
              title="complexity band", title_fontsize=8)
    _save(fig, "fig_auc_confidence_intervals")


def fig_lstm_seed_stability(sci: pd.DataFrame, seeds: pd.DataFrame | None) -> None:
    """Seed-to-seed spread against the complexity ordering.
    """
    if seeds is None or "auc" not in seeds.columns or "sector" not in seeds.columns:
        print("  [note] lstm_seed_stability.csv not found; skipping seed figure")
        return
    order = [s for s in sci.sort_values("SCI").index if s in set(seeds["sector"])]
    if not order:
        print("  [note] no matching sectors in seed file; skipping")
        return
    groups = [seeds.loc[seeds["sector"] == s, "auc"].to_numpy() for s in order]
    band = sci["band"].astype(str).to_dict()

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(1)   # reproducible jitter, different from fig 4's
    for i, (s, vals) in enumerate(zip(order, groups)):
        col = BAND_COLOURS.get(band.get(s, "low"), "0.4")
        # jitter x to spread overlapping points
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)),
                   vals, s=26, color=col, alpha=0.75, edgecolor="white",
                   linewidth=0.4, zorder=3)
        if len(vals) > 1:
            # mean +/- 1 SD bar overlaid in black for emphasis
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="_",
                        color="black", markersize=16, capsize=4, lw=1.2, zorder=4)
    ax.axhline(0.5, color="0.6", ls="--", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("LSTM AUC")
    ax.set_title("LSTM AUC across random initialisations, sectors ordered by SCI\n"
                 "(black bars are mean $\\pm$ 1 SD across seeds)", fontsize=10)
    _save(fig, "fig_lstm_seed_stability")


# --------------------------------------------------------------------------- #
# Figures for the analyses added at supervision                              #
# --------------------------------------------------------------------------- #
def fig_sector_forest(sci: pd.DataFrame, perf: pd.DataFrame,
                      sec_ci: pd.DataFrame | None) -> None:
    """Sector mean AUC with its bootstrap interval, ordered by performance.

    Reading rank and precision off the same plot stops the ordering being read
    as more decisive than it is.
    """
    if sec_ci is None or not len(sec_ci):
        return
    d = sec_ci.set_index("sector").copy()
    d["band"] = sci["band"].reindex(d.index)
    d = d.sort_values("auc_boot_mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (s, r) in enumerate(d.iterrows()):
        col = BAND_COLOURS.get(str(r["band"]), "0.4")
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=col, lw=2, alpha=.85)
        ax.plot(r["auc_boot_mean"], i, "o", color=col, ms=5)
    ax.axvline(0.5, color="0.3", ls="--", lw=0.9)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index)
    ax.set_xlabel("mean AUC across models, with 95% date-block bootstrap interval")
    ax.set_title("Sector mean AUC and its uncertainty")
    handles = [plt.Line2D([], [], color=BAND_COLOURS[b], lw=2) for b in BAND_ORDER]
    ax.legend(handles, BAND_ORDER, title="complexity band", frameon=False,
              loc="lower right")
    _save(fig, "fig_sector_forest")


def fig_shap_heatmap(sci: pd.DataFrame, groups: pd.DataFrame | None) -> None:
    """Attribution share by feature group, sectors ordered by SCI.

    """
    if groups is None or not len(groups):
        return
        g = groups.copy()
    if g.shape[1] and not pd.api.types.is_numeric_dtype(g.iloc[:, 0]):       # 4 spaces
        g = g.set_index(g.columns[0])
    g = g.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    order = [s for s in sci.sort_values("SCI").index if s in g.index]
    if not order:
        print("  [note] SHAP groups share no sectors with the SCI; skipping")
        return
    g = g.loc[order]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(g.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(g.shape[1]))
    ax.set_xticklabels(g.columns, rotation=30, ha="right")
    ax.set_yticks(range(g.shape[0]))
    ax.set_yticklabels(g.index)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = float(g.iloc[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < g.to_numpy(dtype=float).mean() else "black")
    ax.set_title("SHAP attribution share by feature group\n"
                 "(sectors ordered by SCI)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of mean |SHAP|")
    _save(fig, "fig_shap_groups")


def fig_global_vs_sector(gvs: pd.DataFrame | None) -> None:
    """What sector-specific training buys over one model fitted on everything."""
    if gvs is None or not len(gvs):
        return
    sec_col = "Sector" if "Sector" in gvs.columns else "sector"
    d = gvs.sort_values("gain")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cols = ["#b04545" if v < 0 else "#4a7a4e" for v in d["gain"]]
    ax.barh(d[sec_col], d["gain"], color=cols)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("per-sector AUC $-$ global AUC (tabular models, identical folds)")
    ax.set_title("Gain from fitting within sector rather than across the panel")
    _save(fig, "fig_global_vs_sector")


# --------------------------------------------------------------------------- #
# Evidence figures added at supervision                                       #
# --------------------------------------------------------------------------- #
def fig_base_rate_drift() -> None:
    """Up-day share by fold, per sector.

    This is the mechanism behind the pooled-AUC bias. Pooling ranks predictions from five models
    that were each trained on a different base rate, so their probabilities
    sit on different scales; the wider the drift across folds, the more
    artificial structure pooling introduces.  Plotting the drift makes the
    argument visible rather than merely stated.
    """
    try:
        pr = pd.read_parquet(f"{OUT_DIR}/predictions.parquet",
                             columns=["fold", "Sector", "y_true"])
    except (FileNotFoundError, ImportError):
        # ImportError = no parquet engine installed; the gzip CSV is the
        # fallback the pipeline writes in exactly that case.
        try:
            pr = pd.read_csv(f"{OUT_DIR}/predictions.csv.gz",
                             usecols=["fold", "Sector", "y_true"])
        except (FileNotFoundError, ImportError):
            print("  [note] no predictions file; skipping base-rate figure")
            return
    rate = (pr.groupby(["Sector", "fold"])["y_true"].mean().unstack())
    if rate.empty:
        return
    overall = pr.groupby("fold")["y_true"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    for sector, row in rate.iterrows():
        ax.plot(row.index, row.values, marker="o", ms=3, lw=0.9,
                color="0.65", zorder=1)
    ax.plot(overall.index, overall.values, marker="o", ms=6, lw=2.2,
            color="#b04545", zorder=3, label="all sectors")
    ax.axhline(0.5, color="0.3", ls="--", lw=0.9, zorder=0)
    ax.set_xticks(list(overall.index))
    ax.set_xlabel("walk-forward fold")
    ax.set_ylabel("share of up-days in the fold's test window")
    ax.set_title("Target base rate drifts across folds\n"
                 "(grey lines are individual sectors)")
    ax.legend(frameon=False, loc="best")
    spread = float(overall.max() - overall.min())
    ax.annotate(f"overall drift {spread:.3f}",
                xy=(0.99, 0.02), xycoords="axes fraction",
                ha="right", fontsize=8, color="0.35")
    _save(fig, "fig_base_rate_drift")
    print(f"      overall up-day share by fold: "
          f"{', '.join(f'{v:.3f}' for v in overall)}")


# --------------------------------------------------------------------------- #
# Extended list of Figures requested at supervision                           #
# --------------------------------------------------------------------------- #
def fig_panel_interaction(sci_fold: pd.DataFrame | None,
                          fold_perf: pd.DataFrame | None) -> None:
    """AUC against the time-varying SCI, one fitted line per model family.

    Here every point is a (model, sector, fold) cell rather than one of eleven sector averages.
    """
    if sci_fold is None or fold_perf is None or not len(sci_fold) or not len(fold_perf):
        return
    key = "Sector" if "Sector" in fold_perf.columns else "sector"
    skey = "Sector" if "Sector" in sci_fold.columns else "sector"
    d = fold_perf.merge(sci_fold[[skey, "fold", "SCI_ft"]],
                        left_on=[key, "fold"], right_on=[skey, "fold"])
    d = d.dropna(subset=[METRIC, "SCI_ft"])
    if len(d) < 10:
        return
    z = (d["SCI_ft"] - d["SCI_ft"].mean()) / d["SCI_ft"].std(ddof=0)
    d = d.assign(sci_z=z)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    grid = np.linspace(d["sci_z"].min(), d["sci_z"].max(), 50)
    for model, g in d.groupby("model"):
        pts = ax.scatter(g["sci_z"], g[METRIC], s=14, alpha=.55, label=model)
        if len(g) > 2:
            b1, b0 = np.polyfit(g["sci_z"], g[METRIC], 1)
            ax.plot(grid, b0 + b1 * grid, lw=1.4,
                    color=pts.get_facecolor()[0])
    ax.axhline(0.5, color="0.3", ls="--", lw=0.9)
    ax.set_xlabel("sectoral complexity in that fold (z-score)")
    ax.set_ylabel("out-of-sample AUC")
    ax.set_title("Complexity and forecastability, sector $\\times$ fold panel")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, "fig_panel_interaction")


def fig_temporal_heterogeneity(tvar: pd.DataFrame | None,
                               fold_perf: pd.DataFrame | None) -> None:
    """
    Left: each sector's AUC path across the walk-forward folds.  Right: the
    two-way variance decomposition.  
    """
    if tvar is None or not len(tvar):
        return
    key = None
    if fold_perf is not None and len(fold_perf):
        key = "Sector" if "Sector" in fold_perf.columns else "sector"

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    if key:
        cell = fold_perf.groupby([key, "fold"])[METRIC].mean().unstack("fold")
        for sector, row in cell.iterrows():
            ax.plot(row.index, row.values, marker="o", ms=3, lw=1.0,
                    color="0.55", alpha=.8)
        ax.plot(cell.columns, cell.mean(axis=0).values, marker="s", ms=5,
                lw=2.0, color="#b04545", label="mean across sectors")
        ax.axhline(0.5, color="0.3", ls="--", lw=0.9)
        ax.set_xticks(list(cell.columns))
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("walk-forward fold")
    ax.set_ylabel("mean AUC")
    ax.set_title("Each sector across time")

    ax2 = axes[1]
    short = {"sector": "sector", "period (fold)": "period",
             "sector x period and residual": "interaction\n+ residual"}
    lab = [short.get(s, s.replace(" (fold)", "")) for s in tvar["source"]]
    ax2.bar(range(len(tvar)), 100 * tvar["share"], color=["#4a7a4e", "#c79a2e", "0.65"])
    ax2.set_xticks(range(len(tvar)))
    ax2.set_xticklabels(lab, fontsize=8)
    ax2.set_ylabel("share of cell variance (%)")
    ax2.set_title("Variance decomposition")
    for i, v in enumerate(100 * tvar["share"]):
        ax2.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    _save(fig, "fig_temporal_heterogeneity")


def fig_regime_spread(regimes: pd.DataFrame | None) -> None:
    """Mean AUC and the size of the sector gap, by market state."""
    if regimes is None or not len(regimes):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=False)
    order = ["low vol", "high vol", "bear", "bull"]
    d = regimes.set_index("regime").reindex([o for o in order
                                             if o in set(regimes["regime"])])
    x = np.arange(len(d))
    axes[0].bar(x, d["mean_auc"], color="0.65")
    axes[0].axhline(0.5, color="0.3", ls="--", lw=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(d.index, fontsize=8)
    axes[0].set_ylabel("mean AUC")
    axes[0].set_title("Performance by market state")
    lo = min(0.49, float(d["mean_auc"].min()) - 0.005)
    axes[0].set_ylim(lo, float(d["mean_auc"].max()) + 0.005)

    axes[1].bar(x, d["sector_spread"], color="#4a7a4e")
    axes[1].set_xticks(x); axes[1].set_xticklabels(d.index, fontsize=8)
    axes[1].set_ylabel("best minus worst sector (AUC)")
    axes[1].set_title("Sector spread by market state")
    _save(fig, "fig_regime_spread")


def fig_abstention(curve: pd.DataFrame | None) -> None:
    """Accuracy against coverage, a flat one means abstention buys nothing, which is
    the more useful finding for a decision support system.
    """
    if curve is None or not len(curve):
        return
    has_skill = "skill_vs_majority" in curve.columns
    ncol = 2 if has_skill else 1
    fig, axes = plt.subplots(1, ncol, figsize=(9.6 if has_skill else 6.8, 4.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for model, g in curve.groupby("model"):
        g = g.sort_values("coverage")
        line, = ax.plot(100 * g["coverage"], g["accuracy"], marker="o", ms=4,
                        lw=1.4, label=model)
        if "baseline_majority" in g:
            # the same rows, answered by always calling the majority class
            ax.plot(100 * g["coverage"], g["baseline_majority"], lw=1.0,
                    ls=":", color=line.get_color(), alpha=.9)
    ax.axhline(0.5, color="0.3", ls="--", lw=0.9)
    ax.invert_xaxis()
    ax.set_xlabel("coverage: share of days a signal is issued (%)")
    ax.set_ylabel("accuracy on the signals issued")
    ax.set_title("Accuracy (dotted: majority class,\nsame rows)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    if has_skill:
        ax2 = axes[1]
        for model, g in curve.groupby("model"):
            g = g.sort_values("coverage")
            ax2.plot(100 * g["coverage"], g["skill_vs_majority"], marker="o",
                     ms=4, lw=1.4, label=model)
            if "skill_vs_majority_sd" in g:
                ax2.fill_between(100 * g["coverage"],
                                 g["skill_vs_majority"] - g["skill_vs_majority_sd"],
                                 g["skill_vs_majority"] + g["skill_vs_majority_sd"],
                                 alpha=.12)
        ax2.axhline(0.0, color="0.3", ls="--", lw=0.9)
        ax2.invert_xaxis()
        ax2.set_xlabel("coverage: share of days a signal is issued (%)")
        ax2.set_ylabel("accuracy minus majority class")
        ax2.set_title("Skill: what abstention actually adds", fontsize=10)
    _save(fig, "fig_abstention")


def fig_selection_policies(pol: pd.DataFrame | None) -> None:
    """Model-selection policies on identical held-out folds.

    Hatched bars are not deployable: they use information from the scored
    folds and exist only to bound what any selection rule could achieve.
    """
    if pol is None or not len(pol):
        return
    d = pol.sort_values("mean_auc")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colours = ["#4a7a4e" if b else "0.75" for b in d["deployable"]]
    hatch = ["" if b else "//" for b in d["deployable"]]
    bars = ax.barh(range(len(d)), d["mean_auc"], color=colours)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([p.replace("_", " ") for p in d["policy"]], fontsize=9)
    ax.axvline(0.5, color="0.3", ls="--", lw=0.9)
    lo = min(0.49, float(d["mean_auc"].min()) - 0.004)
    span = float(d["mean_auc"].max()) - lo
    ax.set_xlim(lo, float(d["mean_auc"].max()) + 0.22 * span)   # room for labels
    ax.set_xlabel("mean out-of-sample AUC on the scored folds")
    ax.set_title("Selection policies (hatched = not deployable)")
    for i, v in enumerate(d["mean_auc"]):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    _save(fig, "fig_selection_policies")


def summary_tables(perf: pd.DataFrame) -> None:
    """Console-only summary - handy sanity check while iterating on figures."""
    print("\nMean performance by model (across sectors):")
    cols = [c for c in ["accuracy", "auc", "f1", "mcc"] if c in perf.columns]
    print(perf.groupby("model")[cols].mean().round(4).to_string())
    total = len(perf)
    for col, label in [("baseline_test", "hindsight test baseline"),
                       ("baseline_train", "deployable train baseline")]:
        if col in perf.columns:
            n = int((perf["accuracy"] > perf[col]).sum())
            print(f"Cells beating {label}: {n}/{total}")
    print(f"Cells with AUC > 0.5: {int((perf['auc'] > 0.5).sum())}/{total}")
    if "mcc" in perf.columns:
        print(f"Cells with MCC > 0: {int((perf['mcc'] > 0).sum())}/{total}")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    sci, perf, ci, seeds = load()
    fig_metric_correlation(sci)
    fig_sci_vs_performance(sci, perf, ci)
    fig_critical_difference(perf)
    fig_auc_confidence_intervals(sci, ci)   # optional, depends on bootstrap file
    fig_lstm_seed_stability(sci, seeds)     # optional, depends on seed-stability file

    # Figures for the analyses added at supervision.  Each input is optional,
    # so an older results/ directory still produces the original figure set.
    sec_ci = _read("auc_bootstrap_sectors.csv")
    shap_groups = _read("shap_groups.csv", index_col=0)
    gvs = _read("global_vs_sector.csv")
    fig_sector_forest(sci, perf, sec_ci)
    fig_base_rate_drift()
    fig_shap_heatmap(sci, shap_groups)
    fig_global_vs_sector(gvs)

    # Round-2 extensions.  Same rule: absent input -> figure quietly skipped.
    fold_perf = _read("performance_by_fold.csv")
    fig_panel_interaction(_read("sci_by_sector_fold.csv"), fold_perf)
    fig_temporal_heterogeneity(_read("temporal_variance.csv"), fold_perf)
    fig_regime_spread(_read("regime_summary.csv"))
    fig_abstention(_read("abstention_curve.csv"))
    fig_selection_policies(_read("selection_policies.csv"))

    summary_tables(perf)
    print(f"\nFigures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()