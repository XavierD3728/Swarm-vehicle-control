from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    robots: int = 24
    horizon: float = 60.0
    dt: float = 0.05
    seed: int = 20260803
    radius: float = 0.18
    pair_clearance: float = 0.56
    obstacle_clearance: float = 0.30
    sensing_radius: float = 3.2
    communication_radius: float = 3.2
    congestion_radius: float = 1.6
    neighbors: int = 6
    assignment_period: int = 3
    switch_penalty: float = 0.025
    packet_loss: float = 0.10
    latency_steps: int = 1
    message_ttl: float = 0.25
    target_weight: float = 1.25
    graphon_weight: float = 0.22
    congestion_weight: float = 0.12
    flow_speed: float = 1.1
    tracking_time: float = 0.55
    heading_gain: float = 3.0
    turn_limit: float = 2.5
    wheel_radius: float = 0.09
    axle_length: float = 0.42
    wheel_rate_limit: float = 22.0
    lambda1: float = 1.5
    lambda2: float = 2.0
    cbf_margin: float = 0.06
    shared_braking: float = 0.75
    audit_substeps: int = 5
    grid_resolution: float = 0.16
    navigation_inflation: float = 0.48
    slot_tolerance: float = 0.35

    def __post_init__(self):
        if self.robots < 6 or self.robots > 32:
            raise ValueError("warehouse layout supports 6..32 robots")
        if self.dt <= 0 or self.horizon <= 0:
            raise ValueError("dt and horizon must be positive")
        if abs(self.horizon / self.dt - round(self.horizon / self.dt)) > 1e-8:
            raise ValueError("horizon must be an integer number of steps")
        if not 0 <= self.packet_loss <= 1 or self.latency_steps < 0:
            raise ValueError("invalid communication settings")
        if min(self.lambda1, self.lambda2, self.tracking_time, self.shared_braking) <= 0:
            raise ValueError("positive controller constants required")
        if self.pair_clearance <= 2 * self.radius or self.obstacle_clearance <= self.radius:
            raise ValueError("CBF clearances must exceed physical collision limits")
        if self.sensing_radius < self.congestion_radius:
            raise ValueError("sensing must cover congestion support")
        if self.assignment_period < 1 or self.audit_substeps < 1 or self.neighbors < 1:
            raise ValueError("periods, substeps and neighbors must be positive")

    @property
    def steps(self):
        return round(self.horizon / self.dt)
