"""Entry point for `python -m rl_final.ppo`.

Launch training through THIS module, not `python -m rl_final.ppo.ppo_minigrid`.

Every parallel Hydra launcher (joblib/loky, submitit) cloudpickles the task
function, and Hydra hands it the *undecorated* `ppo_minigrid.main`. Pickling that
by reference needs two things, both of which are easy to undo by accident:

1. `main.__module__` must not be `"__main__"` -- cloudpickle treats __main__ as
   non-importable on principle and always serializes it by value. Running
   `ppo_minigrid` directly as a script is exactly that case; importing it here is
   what avoids it.
2. `rl_final.ppo.ppo_minigrid.main` must resolve back to that same function
   object, which is why the Hydra decorator is bound to `cli` over there instead
   of shadowing `main`.
"""

from rl_final.ppo.ppo_minigrid import cli

if __name__ == "__main__":
    cli()
