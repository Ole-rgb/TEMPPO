import numpy as np


def hash_state(
    obs: np.ndarray,
    agent_state: tuple[int, int, int],
) -> tuple:
    """
    Turn a state into a hashable key for the count-based bonus.
    """
    ...
