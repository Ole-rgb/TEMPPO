"""The training entry point must stay picklable by reference.

Every parallel Hydra launcher (joblib/loky, submitit) -- and every sweeper that
dispatches through one, hypersweeper included -- cloudpickles the task function.
Hydra hands it the *undecorated* `ppo_minigrid.main`. If that gets pickled by
value, cloudpickle walks the module globals and dies on
`TypeError: cannot pickle 'CudnnModule' object`.

Two independent conditions have to hold, so there are two separate tests; either
one regressing alone is enough to break every parallel sweep.
"""

import subprocess
import sys
import types

import cloudpickle
import pytest

from rl_final.ppo import ppo_minigrid
from rl_final.ppo.ppo_minigrid import cli, main


def test_main_is_not_defined_in_dunder_main():
    """Condition 1: cloudpickle always pickles __main__ functions by value."""
    assert main.__module__ == "rl_final.ppo.ppo_minigrid"


def test_hydra_decorator_does_not_shadow_main():
    """Condition 2: the dotted name must resolve back to the same object.

    `@hydra.main` applied in place would rebind `main` to its wrapper, cloudpickle's
    by-reference lookup would find a *different* object, and it would silently fall
    back to by-value. That is the bug this whole module exists to prevent.
    """
    assert getattr(ppo_minigrid, main.__qualname__) is main
    assert cli is not main, "cli must be bound separately, not shadow main"


def test_main_pickles_by_reference():
    payload = cloudpickle.dumps(main)
    # By reference is a module path and a name; by value would drag in torch,
    # gymnasium and the rest of the module globals.
    assert len(payload) < 1024
    assert cloudpickle.loads(payload) is main


def test_by_value_pickling_still_fails():
    """Pin the failure mode itself, so the tests above cannot pass vacuously."""
    as_dunder_main = types.FunctionType(
        main.__code__, main.__globals__, main.__name__, main.__defaults__, main.__closure__
    )
    as_dunder_main.__module__ = "__main__"
    as_dunder_main.__qualname__ = main.__qualname__

    with pytest.raises(TypeError, match="cannot pickle"):
        cloudpickle.dumps(as_dunder_main)


def test_package_main_module_runs():
    """`python -m rl_final.ppo` resolves the Hydra config and exits cleanly."""
    proc = subprocess.run(
        [sys.executable, "-m", "rl_final.ppo", "--cfg", "job"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "total_timesteps" in proc.stdout


@pytest.mark.slow
def test_parallel_joblib_sweep(tmp_path):
    """End-to-end: the sweep that used to die on CudnnModule.

    Guards the whole chain rather than its parts -- the unit tests above can all
    pass while some newly imported global reintroduces an unpicklable object.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl_final.ppo",
            "--multirun",
            "hydra/launcher=joblib",
            "hydra.launcher.n_jobs=2",
            f"hydra.sweep.dir={tmp_path}",
            "env=empty_5x5",
            "bonus=none",
            "seed=0,1",
            "total_timesteps=5000",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert "cannot pickle" not in proc.stderr, proc.stderr[-3000:]
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_dir()) == ["0", "1"]
