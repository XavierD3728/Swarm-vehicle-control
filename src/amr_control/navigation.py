"""Role-conditioned Dijkstra field served as a read-only map product."""
import heapq
import numpy as np
from .geometry import segment_rectangle_distance


class NavigationField:
    def __init__(self, world, targets, target_roles, resolution=.16, inflation=.48, boundary_inflation=.30):
        self.world, self.resolution = world, resolution
        self.inflation = inflation
        xmin, xmax, ymin, ymax = world.bounds
        self.x = np.arange(xmin, xmax+resolution/2, resolution)
        self.y = np.arange(ymin, ymax+resolution/2, resolution)
        xx, yy = np.meshgrid(self.x, self.y)
        self.occupied = ((xx < xmin+boundary_inflation) | (xx > xmax-boundary_inflation)
                         | (yy < ymin+boundary_inflation) | (yy > ymax-boundary_inflation))
        for obstacle in world.obstacles:
            dx = np.maximum(np.maximum(obstacle.xmin-xx, xx-obstacle.xmax), 0.)
            dy = np.maximum(np.maximum(obstacle.ymin-yy, yy-obstacle.ymax), 0.)
            self.occupied |= np.hypot(dx, dy) <= inflation
        self.fields = []
        offsets = [(dy, dx, np.hypot(dx, dy)*resolution)
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
        for role in range(int(max(target_roles))+1):
            field = np.full(self.occupied.shape, np.inf)
            queue = []
            for target in targets[target_roles == role]:
                iy, ix = self.cell(target)
                if self.occupied[iy, ix]:
                    free = np.argwhere(~self.occupied)
                    coordinates = np.column_stack((self.x[free[:, 1]], self.y[free[:, 0]]))
                    iy, ix = free[np.argmin(np.sum((coordinates-target)**2, axis=1))]
                field[iy, ix] = 0.
                heapq.heappush(queue, (0., iy, ix))
            while queue:
                distance, iy, ix = heapq.heappop(queue)
                if distance != field[iy, ix]:
                    continue
                for dy, dx, length in offsets:
                    jy, jx = iy+dy, ix+dx
                    if not (0 <= jy < len(self.y) and 0 <= jx < len(self.x)):
                        continue
                    if self.occupied[jy, jx]:
                        continue
                    if dy and dx and (self.occupied[iy, jx] or self.occupied[jy, ix]):
                        continue
                    candidate = distance+length
                    if candidate < field[jy, jx]:
                        field[jy, jx] = candidate
                        heapq.heappush(queue, (candidate, jy, jx))
            self.fields.append(field)

    def cell(self, point):
        ix = int(np.clip(np.rint((point[0]-self.x[0])/self.resolution), 0, len(self.x)-1))
        iy = int(np.clip(np.rint((point[1]-self.y[0])/self.resolution), 0, len(self.y)-1))
        return iy, ix

    def visible(self, start, goal, clearance=None):
        clearance = self.inflation if clearance is None else clearance
        return all(segment_rectangle_distance(start, goal, obstacle) > clearance
                   for obstacle in self.world.obstacles)

    def direction(self, point, role):
        iy, ix = self.cell(point)
        choices = []
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                jy, jx = iy+dy, ix+dx
                if (dx == 0 and dy == 0) or not (0 <= jy < len(self.y) and 0 <= jx < len(self.x)):
                    continue
                value = self.fields[role][jy, jx]
                if not np.isfinite(value):
                    continue
                waypoint = np.array([self.x[jx], self.y[jy]])
                choices.append((value, waypoint))
        # Safety can steer into the conservative planning-inflation shell.
        # Permit an outward segment that never decreases actual clearance;
        # otherwise every segment would be rejected simply because its start
        # lies inside the planning shell, creating an artificial deadlock.
        start_clearance = min((o.distance_geometry(point)[0] for o in self.world.obstacles), default=np.inf)
        escape_clearance = min(self.inflation, start_clearance-1e-7)
        for _, waypoint in sorted(choices, key=lambda item: item[0]):
            if self.visible(point, waypoint, escape_clearance):
                relative = waypoint-point
                return relative/max(np.linalg.norm(relative), 1e-12)
        return np.zeros(2)

    @property
    def inflation(self):
        return self._inflation

    @inflation.setter
    def inflation(self, value):
        self._inflation = value

