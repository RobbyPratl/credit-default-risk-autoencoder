"""Competition metric: the stability-adjusted Gini."""

import numpy as np
from sklearn.metrics import roc_auc_score


def gini_stability(base_df, preds, w_fallingrate=88.0, w_resstd=-0.5):
    """Score predictions with the Home Credit stability-adjusted Gini.

    The competition does not reward raw discrimination alone: a model that scores
    well today but degrades over the observation window is worth less than a
    steady one. The metric therefore computes a Gini per calendar week, fits a
    straight line through those weekly Ginis, and applies two penalties:

    * ``w_fallingrate * min(0, slope)`` — a one-sided penalty on a *downward*
      trend. An improving model (positive slope) gets no bonus, only no penalty.
    * ``w_resstd * std(residuals)`` — a penalty on week-to-week volatility
      around the fitted trend (``w_resstd`` is negative by convention, so a
      noisier model scores lower).

    ``stability_score = mean_gini + w_fallingrate * min(0, slope) + w_resstd * residual_std``

    Weekly Gini is ``2 * AUC - 1``.

    Parameters
    ----------
    base_df : pandas.DataFrame
        Must contain ``WEEK_NUM`` and ``target`` columns, one row per case, in
        the same order as ``preds``.
    preds : array-like
        Predicted default probabilities (or any monotone score).
    w_fallingrate : float, default 88.0
        Weight on the falling-rate penalty.
    w_resstd : float, default -0.5
        Weight on the residual-standard-deviation penalty.

    Returns
    -------
    dict
        ``stability_score``, ``mean_gini``, ``slope`` and ``residual_std``, all
        as plain floats.

    Notes
    -----
    Every week must contain both classes, otherwise ``roc_auc_score`` raises.
    """
    base = base_df.copy()
    base["score"] = preds

    gini_in_time = (
        base[["WEEK_NUM", "target", "score"]]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", "score"]]
        .apply(lambda x: 2 * roc_auc_score(x["target"], x["score"]) - 1)
        .tolist()
    )

    x = np.arange(len(gini_in_time))
    y = np.array(gini_in_time)
    a, b = np.polyfit(x, y, 1)
    residuals = y - (a * x + b)

    return {
        "stability_score": float(
            np.mean(y) + w_fallingrate * min(0, a) + w_resstd * np.std(residuals)
        ),
        "mean_gini": float(np.mean(y)),
        "slope": float(a),
        "residual_std": float(np.std(residuals)),
    }
