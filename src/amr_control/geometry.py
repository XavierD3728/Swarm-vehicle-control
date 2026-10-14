"""Static convex geometry. No position projection or state repair."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Rectangle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("rectangle must have positive area")

    def distance_geometry(self, point):
        """Signed distance, outward normal, Hessian away from feature switches.

        Safety uses only the supporting normal: dropping nonnegative exterior
        distance curvature is conservative for this convex obstacle.
        """
        point = np.asarray(point, dtype=float)
        closest = np.clip(point, [self.xmin, self.ymin], [self.xmax, self.ymax])
        relative = point - closest
        distance = float(np.linalg.norm(relative))
        if distance > 1e-12:
            normal = relative / distance
            corner = abs(relative[0]) > 1e-12 and abs(relative[1]) > 1e-12
            hessian = (np.eye(2) - np.outer(normal, normal)) / distance if corner else np.zeros((2, 2))
            return distance, normal, hessian
        gaps = np.array([point[0] - self.xmin, self.xmax - point[0],
                         point[1] - self.ymin, self.ymax - point[1]])
        side = int(np.argmin(gaps))
        normals = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
        return -float(gaps[side]), normals[side], np.zeros((2, 2))


@dataclass(frozen=True)
class WorldMap:
    obstacles: tuple[Rectangle, ...]
    bounds: tuple[float, float, float, float] = (0., 12., 0., 8.)
    revision: int = 0

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(Rectangle(**item) for item in data["obstacles"]),
                   tuple(data.get("bounds", (0., 12., 0., 8.))))

    def clearance(self, points):
        return min(obstacle.distance_geometry(point)[0]
                   for point in points for obstacle in self.obstacles) if self.obstacles else float("inf")

    def boundary_clearance(self, points):
        lo_x, hi_x, lo_y, hi_y = self.bounds
        return float(np.min(np.column_stack((points[:, 0]-lo_x, hi_x-points[:, 0],
                                             points[:, 1]-lo_y, hi_y-points[:, 1]))))


def segment_rectangle_distance(start, goal, obstacle):
    """Exact Euclidean distance between a closed segment and an AABB."""
    delta = goal-start
    lo = np.array([obstacle.xmin, obstacle.ymin])
    hi = np.array([obstacle.xmax, obstacle.ymax])
    enter, leave = 0., 1.
    for axis in range(2):
        if abs(delta[axis]) < 1e-12:
            if start[axis] < lo[axis] or start[axis] > hi[axis]:
                enter, leave = 1., 0.
                break
        else:
            a, b = sorted(((lo[axis]-start[axis])/delta[axis], (hi[axis]-start[axis])/delta[axis]))
            enter, leave = max(enter, a), min(leave, b)
    if enter <= leave:
        return 0.
    best = min(obstacle.distance_geometry(start)[0], obstacle.distance_geometry(goal)[0])
    for corner in (lo, hi, np.array([lo[0], hi[1]]), np.array([hi[0], lo[1]])):
        fraction = np.clip(float((corner-start)@delta)/max(float(delta@delta), 1e-15), 0., 1.)
        best = min(best, float(np.linalg.norm(corner-start-fraction*delta)))
    return best


def default_map_path():
    repository_map = Path(__file__).resolve().parents[2]/"config/warehouse.json"
    return repository_map if repository_map.is_file() else Path(__file__).with_name("data")/"warehouse.json"
