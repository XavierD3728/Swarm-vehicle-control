from dataclasses import dataclass

import numpy as np

from .config import Config

COLORS = ("#dd8452", "#168f82", "#526acb")
CENTERS = np.array([[9.7, 6.35], [9.2, 3.65], [10.0, 1.25]])
REGION_RADII = np.array([1.7, 2.1, 1.4])


@dataclass(frozen=True)
class Scenario:
    position: np.ndarray
    heading: np.ndarray
    roles: np.ndarray
    capabilities: np.ndarray
    max_speed: np.ndarray
    max_acceleration: np.ndarray
    targets: np.ndarray
    target_roles: np.ndarray


def build_scenario(config: Config):
    """Same population, target anchors and seed order as the V5 warehouse."""
    rng = np.random.default_rng(config.seed)
    count = config.robots
    heavy, cold = max(2, round(.3 * count)), max(2, round(.2 * count))
    counts = (heavy, count-heavy-cold, cold)
    roles = np.repeat(np.arange(3), counts)
    rng.shuffle(roles)
    payload = np.where(roles == 0, 1., np.where(roles == 2, .55, .35))
    speed = np.where(roles == 1, 1.35, np.where(roles == 2, 1.05, .85))
    acceleration = np.where(roles == 0, .75, 1.)
    battery = rng.uniform(.55, 1., count)
    features = np.column_stack((payload, speed / 1.4, battery, roles == 2))
    position = np.array([[1.+.82*(i % 4), .8+.82*(i // 4)] for i in range(count)])
    position += rng.normal(0., .025, position.shape)
    heading = rng.normal(0., .08, count)
    targets = []
    for center, number, radius in zip(CENTERS, counts, (1.48, 1.90, 1.20)):
        index = np.arange(number)
        angle = index * np.pi * (3-np.sqrt(5))
        radial = radius * np.sqrt((index + .5) / number)
        targets.extend(center + np.column_stack((radial*np.cos(angle), radial*np.sin(angle))))
    # V6 includes workspace-wall safety; inset target anchors at construction only.
    inset = config.obstacle_clearance + .08
    targets = np.clip(np.asarray(targets), [inset, inset], [12.-inset, 8.-inset])
    # V5 has cross-role target anchors only ~0.049 m apart. Construct physically
    # compatible slots before simulation; never alter robot states or moving goals.
    spacing = max(config.pair_clearance+.12, .68)
    placed = []
    target_roles = np.repeat(np.arange(3), counts)
    for target, role in zip(targets, target_roles):
        selected = None
        for offset_radius in np.arange(0., 2.01, .04):
            angles = np.arange(72)*2*np.pi/72 if offset_radius else [0.]
            for angle in angles:
                candidate = target+offset_radius*np.array([np.cos(angle), np.sin(angle)])
                if np.any(candidate < inset) or np.any(candidate > [12.-inset, 8.-inset]):
                    continue
                if np.linalg.norm(candidate-CENTERS[role]) > REGION_RADII[role]-.08:
                    continue
                # Target region is on the right of the warehouse shelves.
                if candidate[0] < 7.95:
                    continue
                if all(np.linalg.norm(candidate-other) >= spacing for other in placed):
                    selected = candidate
                    break
            if selected is not None:
                break
        if selected is None:
            raise ValueError("cannot place safety-compatible target slots")
        placed.append(selected)
    targets = np.asarray(placed)
    return Scenario(position, heading, roles, features, speed, acceleration,
                    targets, np.repeat(np.arange(3), counts))
