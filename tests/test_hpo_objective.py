"""`main` returns the scalar an HPO sweeper minimizes.

The sign convention is the easy thing to get backwards, and getting it backwards
would silently search for the *worst* hyperparameters while every run looks
healthy -- so it is pinned here rather than discovered after a night of compute.
"""

import math

import pytest

from rl_final.ppo.ppo_minigrid import WORST_OBJECTIVE, hpo_objective


def test_sign_is_flipped_for_minimizers():
    """Higher return must map to a LOWER objective."""
    good = hpo_objective([0.8, 0.9, 0.9], mode="auc")
    bad = hpo_objective([0.0, 0.1, 0.0], mode="auc")
    assert good < bad
    assert good < 0


def test_auc_is_the_mean_over_the_whole_run():
    assert hpo_objective([0.0, 0.5, 1.0], mode="auc") == pytest.approx(-0.5)


def test_auc_rewards_reaching_the_asymptote_earlier():
    """The property the whole objective choice rests on.

    Two runs that plateau at the same return, one of which got there sooner. Both
    are long enough that the last-10 window sees only the plateau, so final return
    cannot tell them apart; auc must.
    """
    fast = [0.0] + [0.8] * 12
    slow = [0.0] * 3 + [0.8] * 10
    assert hpo_objective(fast, mode="auc") < hpo_objective(slow, mode="auc")
    assert hpo_objective(fast, mode="final") == hpo_objective(slow, mode="final")


def test_final_averages_the_last_ten_evals():
    returns = [0.0] * 10 + [1.0] * 10
    assert hpo_objective(returns, mode="final") == pytest.approx(-1.0)
    # Fewer than ten evals is not an error, it just averages what exists.
    assert hpo_objective([0.4, 0.6], mode="final") == pytest.approx(-0.5)


def test_empty_and_nan_return_worst_case():
    """A crashed or diverged run must not look attractive, or poison the model."""
    assert hpo_objective([], mode="auc") == WORST_OBJECTIVE
    assert hpo_objective([0.5, math.nan], mode="auc") == WORST_OBJECTIVE


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown objective"):
        hpo_objective([0.5], mode="iqm")
