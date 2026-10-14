"""Executable-input HOCBF constraints and exact two-dimensional convex QP.

The reciprocal split assumes fresh reciprocal range observations. Delayed
communication is used only by guidance, never as safety ground truth.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .dynamics import input_halfspaces


@dataclass(frozen=True)
class SafetyResult:
    control: np.ndarray
    feasible: bool
    minimum_residual: float
    constraints: int
    fallback: bool
    reason: str


def solve_qp_2d(reference, metric, rows, lower, tolerance=1e-8):
    """Enumerate interior, edge stationary points and vertices of a 2D QP.

    A strictly convex quadratic over a bounded polygon attains its optimum
    among these candidates. Unlike cyclic projections this returns the
    optimum or reports infeasibility, rather than trusting a sweep count.
    """
    rows, lower = np.asarray(rows), np.asarray(lower)
    if not np.all(np.isfinite(reference)):
        return None
    if np.all(rows@reference >= lower-tolerance):
        return reference.copy()
    inverse = np.linalg.inv(metric)
    directions = rows@inverse
    denom = np.einsum("ij,ij->i", rows, directions)
    valid = denom > 1e-14
    edges = reference+((lower[valid]-rows[valid]@reference)/denom[valid])[:, None]*directions[valid]
    i, j = np.triu_indices(len(rows), 1)
    determinant = rows[i, 0]*rows[j, 1]-rows[i, 1]*rows[j, 0]
    valid_pairs = np.abs(determinant) > 1e-12
    i, j, determinant = i[valid_pairs], j[valid_pairs], determinant[valid_pairs]
    vertices = np.column_stack(((lower[i]*rows[j, 1]-rows[i, 1]*lower[j])/determinant,
                                (rows[i, 0]*lower[j]-lower[i]*rows[j, 0])/determinant))
    candidates = np.vstack((edges, vertices))
    feasible = np.all(candidates@rows.T >= lower[None]-tolerance, axis=1)
    candidates = candidates[feasible]
    if len(candidates) == 0:
        return None
    delta = candidates-reference
    cost = np.einsum("ni,ij,nj->n", delta, metric, delta)
    return candidates[int(np.argmin(cost))]


def pair_constraint(first, second, config):
    relative = first.position-second.position
    velocity = first.velocity-second.velocity
    h = float(relative@relative)-config.pair_clearance**2
    h_dot = 2*float(relative@velocity)
    bound = (-2*float(velocity@velocity)
             -(config.lambda1+config.lambda2)*h_dot
             -config.lambda1*config.lambda2*h+config.cbf_margin)
    return 2*relative@first.input_matrix, -2*relative@second.input_matrix, bound


def static_constraints(state, max_speed, max_acceleration, world, config):
    rows, lower = input_halfspaces(state, max_speed, max_acceleration, config)
    rows, lower = list(rows), list(lower)
    rate = config.lambda1+config.lambda2
    product = config.lambda1*config.lambda2
    # Convex distance has nonnegative curvature outside. Using zero curvature
    # is a conservative lower bound, including the flat-edge case.
    for obstacle in world.obstacles:
        distance, normal, _ = obstacle.distance_geometry(state.position)
        if distance > config.sensing_radius:
            continue
        rows.append(normal@state.input_matrix)
        lower.append(-rate*float(normal@state.velocity)
                     -product*(distance-config.obstacle_clearance)+config.cbf_margin)
    xmin, xmax, ymin, ymax = world.bounds
    for distance, normal in (
        (state.position[0]-xmin, np.array([1., 0.])),
        (xmax-state.position[0], np.array([-1., 0.])),
        (state.position[1]-ymin, np.array([0., 1.])),
        (ymax-state.position[1], np.array([0., -1.])),
    ):
        rows.append(normal@state.input_matrix)
        lower.append(-rate*float(normal@state.velocity)
                     -product*(distance-config.obstacle_clearance)+config.cbf_margin)
    return np.asarray(rows), np.asarray(lower)


def control_metric(state):
    return state.input_matrix.T@state.input_matrix+np.diag([.05, .12])


def brake(state, max_acceleration, config):
    # Explicit emergency action; NOT certified safe when the CBF set is empty.
    return np.array([-min(max_acceleration, max(state.speed, 0.)/config.dt), 0.])


def reciprocal_allocation(first, second, config):
    """Exact-sum split, with no positive extra responsibility assigned to an
    endpoint whose conservative actuator box cannot improve its braking anchor.
    Input box support is used only to allocate responsibility; each QP still
    enforces all actual actuator and wheel constraints.
    """
    row, other_row, bound = pair_constraint(first, second, config)
    first_anchor = brake(first, config.shared_braking, config)
    second_anchor = brake(second, config.shared_braking, config)
    residual = bound-float(row@first_anchor)-float(other_row@second_anchor)
    share = .5
    if residual > 0:
        def capacity(state, coefficients, anchor):
            low = -min(config.shared_braking, max(state.speed, 0.)/config.dt)
            # Every robot in this warehouse has a speed limit at least .85.
            # At higher current speed the conservative reference does not accelerate.
            high = min(config.shared_braking, max(.85-state.speed, 0.)/config.dt)
            support = max(coefficients[0]*low, coefficients[0]*high)+abs(coefficients[1])*config.turn_limit
            return max(0., support-float(coefficients@anchor))
        own_capacity = capacity(first, row, first_anchor)
        other_capacity = capacity(second, other_row, second_anchor)
        if own_capacity+other_capacity > 1e-12:
            share = own_capacity/(own_capacity+other_capacity)
    return row, float(row@first_anchor)+share*residual


class LocalSafetyController:
    def __init__(self, config, world):
        self.config, self.world = config, world

    def filter(self, view, reference, max_speed, max_acceleration, force_failure=False):
        rows, lower = static_constraints(view.own, max_speed, max_acceleration,
                                         self.world, self.config)
        for observation in view.observed:
            row, allocated = reciprocal_allocation(view.own, observation.state, self.config)
            rows = np.vstack((rows, row))
            lower = np.append(lower, allocated)
        solution = None if force_failure else solve_qp_2d(reference, control_metric(view.own), rows, lower)
        fallback = solution is None
        if fallback:
            solution = brake(view.own, max_acceleration, self.config)
        residual = float(np.min(rows@solution-lower))
        feasible = not fallback and np.isfinite(solution).all() and residual >= -2e-7
        return SafetyResult(solution, bool(feasible), residual, len(rows), fallback,
                            "forced_failure" if force_failure else ("infeasible_brake" if fallback else "optimal"))


class CentralSafetyController:
    def __init__(self, config, world):
        self.config, self.world = config, world

    def filter(self, views, references, max_speed, max_acceleration):
        count = len(views)
        rows, lower = [], []
        metrics = np.array([control_metric(v.own) for v in views])
        for i, view in enumerate(views):
            local_rows, local_lower = static_constraints(view.own, max_speed[i], max_acceleration[i],
                                                         self.world, self.config)
            block = np.zeros((len(local_rows), 2*count))
            block[:, 2*i:2*i+2] = local_rows
            rows.extend(block)
            lower.extend(local_lower)
            for observation in view.observed:
                j = observation.robot_id
                if j <= i:
                    continue
                ri, rj, bound = pair_constraint(view.own, observation.state, self.config)
                row = np.zeros(2*count)
                row[2*i:2*i+2], row[2*j:2*j+2] = ri, rj
                rows.append(row)
                lower.append(bound)
        rows, lower = np.asarray(rows), np.asarray(lower)
        def objective(value):
            delta = value.reshape(count, 2)-references
            return .5*np.einsum("ni,nij,nj->", delta, metrics, delta)
        def jacobian(value):
            return np.einsum("nij,nj->ni", metrics, value.reshape(count, 2)-references).ravel()
        initial = np.array([brake(v.own, max_acceleration[i], self.config) for i, v in enumerate(views)])
        result = minimize(objective, initial.ravel(), jac=jacobian, method="SLSQP",
                          constraints={"type": "ineq", "fun": lambda q: rows@q-lower,
                                       "jac": lambda q: rows},
                          options={"ftol": 1e-8, "maxiter": 80})
        residual = float(np.min(rows@result.x-lower)) if np.isfinite(result.x).all() else -np.inf
        success = bool(result.success and residual >= -2e-7)
        controls = result.x.reshape(count, 2) if success else initial
        actual_residual = float(np.min(rows@controls.ravel()-lower))
        return [SafetyResult(controls[i], success, actual_residual, len(rows),
                             not success, "optimal" if success else "infeasible_brake")
                for i in range(count)]

