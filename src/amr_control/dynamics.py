"""Actuator-level model; controls are executed without post-filter clipping."""
from dataclasses import dataclass
import numpy as np

from .config import Config


def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


@dataclass(frozen=True)
class RobotState:
    position: np.ndarray
    heading: float
    speed: float

    @property
    def tangent(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self):
        return self.speed * self.tangent

    @property
    def input_matrix(self):
        t = self.tangent
        return np.column_stack((t, self.speed*np.array([-t[1], t[0]])))


def input_halfspaces(state, max_speed, max_acceleration, config):
    """A q >= lower, q=(longitudinal acceleration, yaw rate).

    Bounds apply throughout the zero-order hold: speed is affine in time;
    wheel speeds are affine in time, so endpoints suffice.
    """
    dt, half_axle = config.dt, config.axle_length/2
    b_low = max(-max_acceleration, -state.speed/dt)
    b_high = min(max_acceleration, (max_speed-state.speed)/dt)
    rows = [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]
    lower = [b_low, -b_high, -config.turn_limit, -config.turn_limit]
    wheel_linear_limit = config.wheel_radius*config.wheel_rate_limit
    for duration in (0., dt):
        for sign in (-1., 1.):
            row = np.array([duration, sign*half_axle])
            rows.extend([row.tolist(), (-row).tolist()])
            lower.extend([-wheel_linear_limit-state.speed,
                          state.speed-wheel_linear_limit])
    return np.array(rows), np.array(lower)


def propagate(state, control, duration):
    """Exact constant b, omega solution (series near omega=0 for stability)."""
    b, omega = control
    x = omega * duration
    if abs(x) < 1e-3:
        # Integrals of exp(i omega t) and t exp(i omega t).
        import math
        j0 = duration * sum((1j*x)**k/(math.factorial(k)*(k+1)) for k in range(8))
        j1 = duration**2 * sum((1j*x)**k/(math.factorial(k)*(k+2)) for k in range(8))
    else:
        phase = np.exp(1j*x)
        j0 = (phase-1)/(1j*omega)
        j1 = phase*duration/(1j*omega)+(phase-1)/(omega*omega)
    displacement = np.exp(1j*state.heading)*(state.speed*j0+b*j1)
    return RobotState(state.position+np.array([displacement.real, displacement.imag]),
                      float(wrap_angle(state.heading+x)), float(state.speed+b*duration))


def nominal_control(state, desired_velocity, max_speed, max_acceleration, config):
    desired_speed = min(float(np.linalg.norm(desired_velocity)), max_speed)
    desired_heading = np.arctan2(desired_velocity[1], desired_velocity[0]) if desired_speed > 1e-8 else state.heading
    error = float(wrap_angle(desired_heading-state.heading))
    longitudinal = (desired_speed*max(0., np.cos(error))-state.speed)/config.tracking_time
    yaw = config.heading_gain*error
    # This is a reference only; actuator limits are enforced inside the safety QP.
    return np.array([longitudinal, yaw])


def wheel_rates(state, control, config):
    return np.array([state.speed-config.axle_length*control[1]/2,
                     state.speed+config.axle_length*control[1]/2])/config.wheel_radius

