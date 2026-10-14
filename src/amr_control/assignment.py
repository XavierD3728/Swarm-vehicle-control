"""Low-frequency central service; assignments are versioned and held between updates."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(position, roles, targets, target_roles, previous=None, switch_penalty=0.):
    result = np.empty(len(position), dtype=int)
    for role in np.unique(roles):
        robots = np.flatnonzero(roles == role)
        slots = np.flatnonzero(target_roles == role)
        if len(robots) != len(slots):
            raise ValueError("conditional masses must match")
        cost = np.sum((position[robots, None]-targets[slots][None])**2, axis=-1)
        if previous is not None:
            cost += switch_penalty*(slots[None] != previous[robots, None])
        row, column = linear_sum_assignment(cost)
        result[robots[row]] = slots[column]
    return result


def transport(position, roles, targets, target_roles):
    matching = assign(position, roles, targets, target_roles)
    return float(np.mean(np.sum((position-targets[matching])**2, axis=1))/2)

