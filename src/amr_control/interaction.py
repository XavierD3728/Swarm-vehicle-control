"""Read-only telemetry: adjacency[receiver, source], never feeds the controller.

Planning edges mean consumed observations/memory, not proven causal detours.
Binding CBF edges have absolute constraint slack <= 1e-5, not KKT multipliers.
"""
import numpy as np
from .safety import pair_constraint, reciprocal_allocation


def cross_role_networks(views, planners, roles, controls, config, mode, now):
    n = len(views)
    result = {name: np.zeros((n, n), dtype=bool) for name in
              ('cross_congestion', 'cross_planning', 'cross_yield', 'cross_cbf', 'cross_cbf_binding')}
    for i, (view, planner) in enumerate(zip(views, planners)):
        for j in planner.planning_neighbors:
            if roles[i] != roles[j]:
                result['cross_planning'][i, j] = True
        j = planner.yield_requester
        if j is not None and planner.yielding(now) and roles[i] != roles[j]:
            result['cross_yield'][i, j] = True
        for observation in view.observed:
            j = observation.robot_id
            if roles[i] == roles[j]:
                continue
            distance = np.linalg.norm(view.own.position-observation.state.position)
            result['cross_congestion'][i, j] = bool(config.congestion_weight != 0 and 0 < distance < config.congestion_radius)
            result['cross_cbf'][i, j] = True
            if mode == 'hybrid':
                row, bound = reciprocal_allocation(view.own, observation.state, config)
                slack = float(row@controls[i])-bound
            else:
                ri, rj, bound = pair_constraint(view.own, observation.state, config)
                slack = float(ri@controls[i]+rj@controls[j])-bound
            result['cross_cbf_binding'][i, j] = abs(slack) <= 1e-5
    return result
