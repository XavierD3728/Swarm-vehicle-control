"""Simulation adapters own ground truth; robot controllers receive only local packets."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from .dynamics import RobotState


@dataclass(frozen=True)
class Observation:
    robot_id: int
    state: RobotState
    timestamp: float


@dataclass(frozen=True)
class Packet:
    robot_id: int
    position: np.ndarray
    velocity: np.ndarray
    target: np.ndarray
    role: int
    capability: np.ndarray
    timestamp: float
    assignment_version: int
    yielding: bool = False
    blocked: bool = False


@dataclass(frozen=True)
class LocalView:
    robot_id: int
    own: RobotState
    observed: tuple[Observation, ...]
    messages: tuple[Packet, ...]


class LocalSensor:
    """Ideal fresh range-limited sensor; no network loss or hidden global state.

    Sensor error, occlusion and sensor dropout are NOT modeled as guaranteed safe.
    The world adapter performs range queries; consumers only see returned objects.
    """
    def __init__(self, radius):
        self.radius = radius

    def observe(self, states, timestamp):
        tree = cKDTree(np.array([s.position for s in states]))
        neighbors = tree.query_ball_point(tree.data, self.radius)
        return [tuple(Observation(j, states[j], timestamp) for j in indices if j != i)
                for i, indices in enumerate(neighbors)]


class MessageBus:
    def __init__(self, config, seed):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.pending = []
        self.cache = [dict() for _ in range(config.robots)]
        self.sent = self.delivered = self.dropped = self.expired = 0

    def exchange(self, step, states, scenario, assignment, version, yielding=None, blocked=None):
        now = step*self.config.dt
        tree = cKDTree(np.array([s.position for s in states]))
        for i, recipients in enumerate(tree.query_ball_point(tree.data, self.config.communication_radius)):
            packet = Packet(i, states[i].position.copy(), states[i].velocity.copy(),
                            scenario.targets[assignment[i]].copy(), int(scenario.roles[i]),
                            scenario.capabilities[i].copy(), now, version, False if yielding is None else bool(yielding[i]),
                            False if blocked is None else bool(blocked[i]))
            for j in recipients:
                if i == j:
                    continue
                self.sent += 1
                if self.rng.random() < self.config.packet_loss:
                    self.dropped += 1
                else:
                    self.pending.append((step+self.config.latency_steps, j, packet))
        remaining = []
        for due, receiver, packet in self.pending:
            if due <= step:
                self.cache[receiver][packet.robot_id] = packet
                self.delivered += 1
            else:
                remaining.append((due, receiver, packet))
        self.pending = remaining
        output = []
        for cache in self.cache:
            stale = [j for j, p in cache.items() if now-p.timestamp > self.config.message_ttl+1e-12]
            for j in stale:
                del cache[j]
            self.expired += len(stale)
            output.append(tuple(cache.values()))
        return output

