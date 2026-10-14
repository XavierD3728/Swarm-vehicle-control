"""Robot-owned grid routes built from the static prior and local observations."""
from dataclasses import replace
import numpy as np
from .communication import Observation
from .dynamics import RobotState
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class LocalRoutePlanner:
    def __init__(self, navigator, config):
        self.navigator, self.config = navigator, config
        self.shape = navigator.occupied.shape
        self.coordinates = np.stack(np.meshgrid(navigator.x, navigator.y), axis=-1).reshape(-1, 2)
        ny, nx = self.shape
        source, destination, length, corner_a, corner_b = [], [], [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            yy, xx = np.mgrid[max(0, -dy):min(ny, ny-dy), max(0, -dx):min(nx, nx-dx)]
            first = (yy*nx+xx).ravel()
            second = ((yy+dy)*nx+xx+dx).ravel()
            source.extend(first)
            destination.extend(second)
            length.extend(np.full(len(first), np.hypot(dx, dy)*navigator.resolution))
            corner_a.extend((yy*nx+xx+dx).ravel())
            corner_b.extend(((yy+dy)*nx+xx).ravel())
        self.source, self.destination = np.array(source), np.array(destination)
        self.length = np.array(length)
        self.corner_a, self.corner_b = np.array(corner_a), np.array(corner_b)
        self.path = np.empty((0, 2))
        self.target = None
        self.last_plan = -np.inf
        self.replans = 0
        self.failures = 0
        self.yield_until = -np.inf
        self.yield_target = None
        self.yield_count = 0
        self.blocked = False
        self.obstacle_memory = {}
        self.planning_neighbors = ()
        self.yield_requester = None

    def planning_view(self, view, now):
        """Local observation memory prevents out-of-range stationary robots
        from disappearing and repeatedly reopening a known blocked passage.
        Memory is used by navigation only; CBF always uses fresh sensing.
        """
        for observation in view.observed:
            self.obstacle_memory[observation.robot_id] = observation
        estimates = []
        expired = []
        for identity, observation in self.obstacle_memory.items():
            age = max(0., now-observation.timestamp)
            stationary = observation.state.speed < .08
            if age > (30. if stationary else .5):
                expired.append(identity)
                continue
            state = observation.state
            if not stationary and age > 0:
                state = RobotState(state.position+age*state.velocity, state.heading, state.speed)
            estimates.append(Observation(identity, state, observation.timestamp))
        for identity in expired:
            del self.obstacle_memory[identity]
        return replace(view, observed=tuple(estimates))

    def clear_segment(self, point, endpoint, observations):
        start_distance = min((o.distance_geometry(point)[0] for o in self.navigator.world.obstacles),
                             default=np.inf)
        if not self.navigator.visible(point, endpoint, min(self.navigator.inflation, start_distance-1e-7)):
            return False
        if observations:
            relative = np.array([o.state.position-point for o in observations])
            segment = endpoint-point
            fraction = np.clip(relative@segment/max(float(segment@segment), 1e-12), 0., 1.)
            distances = np.linalg.norm(relative-fraction[:, None]*segment, axis=1)
            needed = np.minimum(self.config.pair_clearance+.07, np.linalg.norm(relative, axis=1)-1e-6)
            if np.any(distances < needed):
                return False
        xmin, xmax, ymin, ymax = self.navigator.world.bounds
        return xmin+self.config.obstacle_clearance <= endpoint[0] <= xmax-self.config.obstacle_clearance and ymin+self.config.obstacle_clearance <= endpoint[1] <= ymax-self.config.obstacle_clearance

    def plan(self, view, target, now):
        self.replans += 1
        point = view.own.position
        occupied = self.navigator.occupied.ravel().copy()
        if view.observed:
            centers = np.array([o.state.position for o in view.observed])
            distances = np.linalg.norm(self.coordinates[:, None]-centers[None], axis=-1)
            occupied |= np.any(distances < self.config.pair_clearance+.07, axis=1)
        # Connect the continuous target/start to free grid nodes without moving
        # either the physical robot or its target.
        free = np.flatnonzero(~occupied)
        if not len(free):
            self.path = np.empty((0, 2))
            self.failures += 1
            return
        target_order = free[np.argsort(np.linalg.norm(self.coordinates[free]-target, axis=1))]
        goal = target_order[0]
        for candidate in target_order[:30]:
            if self.clear_segment(self.coordinates[candidate], target, view.observed):
                goal = candidate
                break
        valid = ~(occupied[self.source] | occupied[self.destination]
                  | occupied[self.corner_a] | occupied[self.corner_b])
        graph = coo_matrix((self.length[valid], (self.source[valid], self.destination[valid])),
                           shape=(len(occupied), len(occupied))).tocsr()
        costs, predecessor = dijkstra(graph, directed=False, indices=int(goal), return_predecessors=True)
        near = free[np.argsort(np.linalg.norm(self.coordinates[free]-point, axis=1))[:48]]
        candidates = []
        best_candidate = (float("inf"), len(occupied))
        for node in near:
            distance = np.linalg.norm(self.coordinates[node]-point)
            candidate = (costs[node]+distance, node)
            # Exact branch-and-bound: a worse cost cannot win, regardless of visibility.
            if distance <= .85 and np.isfinite(costs[node]) and candidate < best_candidate and self.clear_segment(point, self.coordinates[node], view.observed):
                candidates.append(candidate)
                best_candidate = candidate
        if not candidates:
            self.path = np.empty((0, 2))
            self.failures += 1
        else:
            node = min(candidates)[1]
            route = [self.coordinates[node]]
            for _ in range(len(occupied)):
                if node == goal:
                    break
                node = int(predecessor[node])
                if node < 0:
                    break
                route.append(self.coordinates[node])
            self.path = np.asarray(route)
        self.target = target.copy()
        self.last_plan = now

    def yielding(self, now):
        return now < self.yield_until

    def yielding_waypoint(self, view, target, now):
        """Arrived robots temporarily clear a peer's approach corridor.

        A yielding flag prevents other robots from treating that temporary
        displacement as a new request. This uses received neighbor packets,
        not global robot state or centralized trajectory planning.
        """
        if self.yielding(now):
            return self.yield_target
        if np.linalg.norm(view.own.position-target) > .42:
            return None
        point = view.own.position
        requests = []
        for packet in view.messages:
            if packet.yielding or now-packet.timestamp > self.config.message_ttl:
                continue
            pending = np.linalg.norm(packet.position-packet.target)
            if pending < .55 or np.linalg.norm(packet.position-point) > 2.6:
                continue
            segment = packet.target-packet.position
            fraction = float((point-packet.position)@segment)/max(float(segment@segment), 1e-12)
            if not packet.blocked and not .05 < fraction < 1.15:
                continue
            distance = np.linalg.norm(point-packet.position-np.clip(fraction, 0., 1.)*segment)
            if packet.blocked or distance < self.config.pair_clearance+.14:
                requests.append((packet.robot_id, packet))
        if not requests:
            return None
        _, request = min(requests, key=lambda item: item[0])
        corridor = request.target-request.position
        choices = []
        for radius in (.65, .85, 1.05):
            for angle in np.arange(24)*2*np.pi/24:
                candidate = point+radius*np.array([np.cos(angle), np.sin(angle)])
                if not self.clear_segment(point, candidate, view.observed):
                    continue
                iy, ix = self.navigator.cell(candidate)
                if self.navigator.occupied[iy, ix]:
                    continue
                along = np.clip(float((candidate-request.position)@corridor)/max(float(corridor@corridor), 1e-12), 0., 1.)
                clearance = np.linalg.norm(candidate-request.position-along*corridor)
                if clearance < self.config.pair_clearance+.18:
                    continue
                choices.append((np.linalg.norm(candidate-target), candidate))
        if not choices:
            return None
        self.yield_requester = request.robot_id
        self.yield_target = min(choices, key=lambda item: item[0])[1]
        self.yield_until = now+3.5
        self.yield_count += 1
        return self.yield_target

    def velocity(self, view, target, now):
        view = self.planning_view(view, now)
        self.planning_neighbors = tuple(o.robot_id for o in view.observed)
        temporary = self.yielding_waypoint(view, target, now)
        if temporary is not None:
            target = temporary
        point = view.own.position
        error = target-point
        if self.clear_segment(point, target, view.observed):
            self.path = np.empty((0, 2))
            self.blocked = False
            return self.config.target_weight*error
        changed = self.target is None or np.linalg.norm(target-self.target) > 1e-8
        if changed or now-self.last_plan >= .5 or not len(self.path):
            self.plan(view, target, now)
        if len(self.path):
            nearest = int(np.argmin(np.linalg.norm(self.path-point, axis=1)))
            # Furthest visible point within a short lookahead follows the full
            # grid route without cutting through locally observed robots.
            for index in range(min(nearest+8, len(self.path)-1), nearest-1, -1):
                endpoint = self.path[index]
                if self.clear_segment(point, endpoint, view.observed):
                    direction = endpoint-point
                    norm = np.linalg.norm(direction)
                    if norm > .025:
                        self.blocked = False
                        return self.config.flow_speed*direction/norm
        self.blocked = True
        return np.zeros(2)

