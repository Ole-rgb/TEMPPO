"""Time-varying schedules must not follow the fidelity.

Under multi-fidelity search `total_timesteps` IS the budget DEHB varies. If the LR
decay or the beta_llm anneal is derived from it, a low-fidelity run is not a
prefix of a full one -- it is a different schedule that fully decays early. Rungs
then rank schedules rather than hyperparameters, and nothing raises. These tests
pin the prefix property instead.
"""

import pytest
from omegaconf import OmegaConf

from rl_final.bonus.llm import anneal_factor, resolve_anneal_steps, schedule_horizon


def cfg_for(total, *, schedule=None, steps=None, frac=None):
    bonus = {"name": "rnd_llm_warmstart", "beta_rnd": 0.001, "beta_llm": 0.5}
    if steps is not None:
        bonus["anneal_llm_steps"] = steps
    if frac is not None:
        bonus["anneal_llm_frac"] = frac
    return OmegaConf.create(
        {"total_timesteps": total, "schedule_timesteps": schedule, "bonus": bonus}
    )


# --- schedule_horizon -------------------------------------------------------


def test_horizon_follows_total_timesteps_by_default():
    """Ordinary runs must be completely unaffected by any of this."""
    assert schedule_horizon(cfg_for(750_000)) == 750_000


def test_pinned_horizon_is_independent_of_the_budget():
    """The fix: every fidelity resolves against the same horizon."""
    horizons = {schedule_horizon(cfg_for(b, schedule=750_000)) for b in (83_000, 250_000, 750_000)}
    assert horizons == {750_000}


# --- the LR schedule --------------------------------------------------------


def lr_frac(iteration, schedule_iterations):
    """Mirrors the training loop's anneal_lr computation."""
    return max(0.0, 1.0 - (iteration - 1.0) / schedule_iterations)


def schedule_iterations(cfg, batch=512):
    """Mirrors main(): the horizon the LR decays over, in policy iterations."""
    return max(1, int(schedule_horizon(cfg) // batch))


def test_low_fidelity_lr_is_a_prefix_of_the_full_schedule():
    """Same iteration -> same learning rate, whatever the budget.

    This is the property that makes rung comparisons meaningful. Iteration 160 is
    ~82k env steps, i.e. inside even the lowest rung.
    """
    lrs = {
        lr_frac(160, schedule_iterations(cfg_for(b, schedule=750_000)))
        for b in (83_000, 250_000, 750_000)
    }
    assert len(lrs) == 1, f"LR at a fixed step differs across fidelities: {lrs}"
    assert lrs.pop() == pytest.approx(1 - 159 / (750_000 // 512))


def test_unpinned_lr_diverges_across_fidelities():
    """The bug itself, so the test above cannot pass vacuously."""
    batch = 512
    short, full = 83_000 // batch, 750_000 // batch
    # At the *end* of the short run the LR is spent; the long run is barely started.
    assert lr_frac(short, short) == pytest.approx(0.0, abs=0.01)
    assert lr_frac(short, full) > 0.85


# --- the beta_llm anneal ----------------------------------------------------


def test_absolute_steps_are_unchanged():
    """Every existing run and label must keep its meaning."""
    assert resolve_anneal_steps(cfg_for(750_000, steps=150_000)) == 150_000


def test_frac_resolves_against_the_pinned_horizon_not_the_budget():
    """0.2 means 20% of the schedule, identically at every fidelity."""
    resolved = {
        resolve_anneal_steps(cfg_for(b, schedule=750_000, frac=0.2))
        for b in (83_000, 250_000, 750_000)
    }
    assert resolved == {150_000}


def test_frac_and_steps_agree_on_the_same_horizon():
    """The two forms are interchangeable, so labels stay comparable."""
    assert resolve_anneal_steps(cfg_for(750_000, frac=0.2)) == resolve_anneal_steps(
        cfg_for(750_000, steps=150_000)
    )


def test_beta_llm_at_a_given_step_is_fidelity_invariant():
    """What the anneal actually has to guarantee, end to end."""
    at_50k = {
        anneal_factor(50_000, resolve_anneal_steps(cfg_for(b, schedule=750_000, frac=0.2)))
        for b in (83_000, 250_000, 750_000)
    }
    assert len(at_50k) == 1
    assert at_50k.pop() == pytest.approx(1 - 50_000 / 150_000)


def test_setting_both_forms_is_an_error():
    """Silently preferring one would be a config that lies about what it ran."""
    with pytest.raises(ValueError, match="not both"):
        resolve_anneal_steps(cfg_for(750_000, steps=150_000, frac=0.2))


def test_frac_outside_the_unit_interval_is_an_error():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resolve_anneal_steps(cfg_for(750_000, frac=1.5))


def test_no_anneal_keys_means_no_anneal():
    cfg = OmegaConf.create({"total_timesteps": 750_000, "bonus": {"name": "rnd"}})
    assert resolve_anneal_steps(cfg) == 0
    assert anneal_factor(500_000, 0) == 1.0
