import numpy as np
import pandas as pd
import pytest

from credit_risk import gini_stability

N_PER_WEEK = 200


def make_week(rng, week, gini, n=N_PER_WEEK):
    """Build one week whose Gini is approximately ``gini``.

    Half the rows are positives. Scores are the target plus Gaussian noise; the
    noise scale controls separability, so we calibrate it by bisection rather
    than by guessing a closed form.
    """
    target = np.tile([0, 1], n // 2)
    noise = rng.normal(size=n)

    lo, hi = 1e-3, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        score = target + mid * noise
        achieved = _gini(target, score)
        if achieved > gini:
            lo = mid
        else:
            hi = mid
    score = target + ((lo + hi) / 2) * noise

    return pd.DataFrame({"WEEK_NUM": week, "target": target, "score": score})


def _gini(target, score):
    from sklearn.metrics import roc_auc_score

    return 2 * roc_auc_score(target, score) - 1


def frame_from_ginis(ginis, seed=0):
    rng = np.random.default_rng(seed)
    parts = [make_week(rng, week, g) for week, g in enumerate(ginis)]
    df = pd.concat(parts, ignore_index=True)
    return df[["WEEK_NUM", "target"]], df["score"].to_numpy()


def test_result_keys_and_types():
    base, preds = frame_from_ginis([0.5] * 10)
    result = gini_stability(base, preds)

    assert set(result) == {"stability_score", "mean_gini", "slope", "residual_std"}
    assert all(isinstance(v, float) for v in result.values())
    assert result["mean_gini"] == pytest.approx(0.5, abs=0.02)
    assert result["residual_std"] >= 0.0


def test_declining_series_scores_lower_than_flat_series_with_same_mean():
    n = 11
    declining = np.linspace(0.7, 0.3, n)
    flat = np.full(n, declining.mean())

    declining_result = gini_stability(*frame_from_ginis(declining))
    flat_result = gini_stability(*frame_from_ginis(flat))

    # Same average discrimination...
    assert declining_result["mean_gini"] == pytest.approx(
        flat_result["mean_gini"], abs=0.03
    )
    # ...but only the declining one takes the falling-rate penalty.
    assert declining_result["slope"] < 0
    assert flat_result["slope"] == pytest.approx(0.0, abs=0.01)
    assert declining_result["stability_score"] < flat_result["stability_score"]


def test_noisy_residuals_score_lower_than_smooth_residuals():
    n = 12
    smooth = np.full(n, 0.5)
    noisy = np.full(n, 0.5)
    noisy[::2] += 0.15
    noisy[1::2] -= 0.15

    smooth_result = gini_stability(*frame_from_ginis(smooth))
    noisy_result = gini_stability(*frame_from_ginis(noisy))

    assert noisy_result["mean_gini"] == pytest.approx(
        smooth_result["mean_gini"], abs=0.03
    )
    assert noisy_result["residual_std"] > smooth_result["residual_std"]
    assert noisy_result["stability_score"] < smooth_result["stability_score"]


def test_weights_control_the_penalties():
    base, preds = frame_from_ginis(np.linspace(0.7, 0.3, 11))

    penalized = gini_stability(base, preds)
    unpenalized = gini_stability(base, preds, w_fallingrate=0.0, w_resstd=0.0)

    assert unpenalized["stability_score"] == pytest.approx(unpenalized["mean_gini"])
    assert penalized["stability_score"] < unpenalized["stability_score"]
    # Slope and mean are properties of the predictions, not of the weights.
    assert penalized["slope"] == pytest.approx(unpenalized["slope"])
