"""Export the two publication figures embedded in the README.

Usage (from the repo root):

    .venv/bin/python scripts/export_readme_figures.py

Writes:
    assets/target_drift.png   applications + default rate per week (the "why stability" hook)
    assets/weekly_gini.png    per-week validation Gini of the 02_base_model.ipynb baseline

The baseline reproduced here is exactly the one in notebooks/02_base_model.ipynb:
static A/P columns only, temporal 80/20 split by week, LightGBM with early stopping.
Training takes a few minutes.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "parquet_files" / "train"
ASSETS = REPO / "assets"

# Shared style: one hue for the data, one for reference lines, across both figures.
SERIES = "steelblue"
REFERENCE = "salmon"
DPI = 150

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#c8c8c8",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def thousands(x: float, _pos: int) -> str:
    return f"{x:,.0f}"


# --------------------------------------------------------------------------- #
# Figure 1 — target drift
# --------------------------------------------------------------------------- #
def make_target_drift() -> None:
    base = pl.read_parquet(TRAIN / "train_base.parquet", columns=["WEEK_NUM", "target"])

    weekly = (
        base.group_by("WEEK_NUM")
        .agg(
            pl.len().alias("applications"),
            pl.col("target").mean().alias("default_rate"),
        )
        .sort("WEEK_NUM")
    )
    weeks = weekly["WEEK_NUM"].to_numpy()
    applications = weekly["applications"].to_numpy()
    default_rate = weekly["default_rate"].to_numpy()
    overall_rate = base["target"].mean()

    print(f"[target_drift] weeks={len(weeks)}  overall default rate={overall_rate:.2%}")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="tight")

    ax_top.plot(weeks, applications, color=SERIES, linewidth=2)
    ax_top.fill_between(weeks, applications, color=SERIES, alpha=0.15)
    ax_top.set_ylabel("Applications")
    ax_top.set_title("Applications per week", loc="left")
    ax_top.set_ylim(bottom=0)
    ax_top.yaxis.set_major_formatter(FuncFormatter(thousands))

    ax_bot.plot(weeks, default_rate, color=SERIES, linewidth=2)
    ax_bot.axhline(
        overall_rate,
        color=REFERENCE,
        linestyle="--",
        linewidth=1.8,
        label=f"Overall default rate ({overall_rate:.2%})",
    )
    ax_bot.set_ylabel("Default rate")
    ax_bot.set_xlabel("Week number")
    ax_bot.set_title("Default rate per week", loc="left")
    ax_bot.set_ylim(bottom=0)
    ax_bot.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_bot.legend(loc="upper right")

    ax_bot.set_xlim(weeks.min(), weeks.max())

    out = ASSETS / "target_drift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[target_drift] wrote {out}")


# --------------------------------------------------------------------------- #
# Figure 2 — per-week validation Gini of the baseline
# --------------------------------------------------------------------------- #
def make_weekly_gini() -> None:
    static_paths = [TRAIN / "train_static_0_0.parquet", TRAIN / "train_static_0_1.parquet"]

    schema = pl.scan_parquet(static_paths[0]).collect_schema().names()
    feature_cols = [c for c in schema if c.endswith(("A", "P"))]
    print(f"[weekly_gini] feature columns: {len(feature_cols)}")

    static = pl.concat(
        [pl.read_parquet(p, columns=["case_id"] + feature_cols) for p in static_paths],
        how="vertical_relaxed",
    )
    base = pl.read_parquet(
        TRAIN / "train_base.parquet", columns=["case_id", "WEEK_NUM", "target"]
    )

    df = base.join(static, on="case_id", how="left")
    del static, base

    # Temporal split: first 80% of weeks train, last 20% validation.
    weeks = sorted(df["WEEK_NUM"].unique().to_list())
    n = len(weeks)
    train_weeks = weeks[: int(n * 0.8)]
    val_weeks = weeks[int(n * 0.8) :]
    print(
        f"[weekly_gini] train weeks {min(train_weeks)}-{max(train_weeks)}, "
        f"val weeks {min(val_weeks)}-{max(val_weeks)}"
    )

    train_df = df.filter(pl.col("WEEK_NUM").is_in(train_weeks))
    val_df = df.filter(pl.col("WEEK_NUM").is_in(val_weeks))
    del df

    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df["target"].to_pandas()
    X_val = val_df.select(feature_cols).to_pandas()
    y_val = val_df["target"].to_pandas()
    val_week = val_df["WEEK_NUM"].to_numpy()
    del train_df, val_df

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 1000,
        "is_unbalance": True,
        "verbosity": -1,
        "random_state": 42,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_preds)
    print(f"[weekly_gini] Val AUC:  {auc:.4f}")
    print(f"[weekly_gini] Val Gini: {2 * auc - 1:.4f}")

    y_val_np = y_val.to_numpy()
    plot_weeks = np.array(val_weeks)
    ginis = np.array(
        [
            2 * roc_auc_score(y_val_np[val_week == w], val_preds[val_week == w]) - 1
            for w in plot_weeks
        ]
    )
    mean_gini = float(ginis.mean())
    slope = float(np.polyfit(np.arange(len(ginis)), ginis, 1)[0])
    print(f"[weekly_gini] mean weekly Gini: {mean_gini:.4f}  slope: {slope:+.4f}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        plot_weeks,
        ginis,
        color=SERIES,
        linewidth=2,
        marker="o",
        markersize=6,
        markerfacecolor=SERIES,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Weekly Gini",
    )
    ax.axhline(
        mean_gini,
        color=REFERENCE,
        linestyle="--",
        linewidth=1.8,
        label=f"Mean Gini ({mean_gini:.3f})",
    )
    ax.set_xlabel("Week number")
    ax.set_ylabel("Gini")
    ax.set_title(
        f"Baseline Gini per validation week (weeks {plot_weeks.min()}–{plot_weeks.max()})",
        loc="left",
    )
    ax.set_xticks(plot_weeks)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = ASSETS / "weekly_gini.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[weekly_gini] wrote {out}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    make_target_drift()
    make_weekly_gini()


if __name__ == "__main__":
    main()
