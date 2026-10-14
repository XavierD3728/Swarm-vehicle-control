"""Synchronous simulation harness. Robot logic never receives global state arrays."""
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .assignment import assign, transport
from .communication import LocalSensor, LocalView, MessageBus
from .dynamics import RobotState, nominal_control, propagate, wheel_rates
from .geometry import WorldMap, default_map_path
from .guidance import desired_velocity
from .navigation import NavigationField
from .local_planner import LocalRoutePlanner
from .safety import CentralSafetyController, LocalSafetyController
from .interaction import cross_role_networks
from .scenario import build_scenario, CENTERS, REGION_RADII


def pair_minimum(position):
    relative = position[:, None]-position[None]
    distances = np.linalg.norm(relative, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def audit_hold(states, controls, world, config):
    """Lipschitz lower bounds between exact constant-input trajectory samples.

    Distance-to-closed-set is 1-Lipschitz. Every time is within delta/2 of
    a sample; bounded speed covers the unsampled interval without teleporting.
    """
    samples = []
    obstacle = boundary = pair = float("inf")
    end_speeds = np.array([s.speed+q[0]*config.dt for s, q in zip(states, controls)])
    max_speeds = np.maximum([s.speed for s in states], end_speeds)
    gap = config.dt/config.audit_substeps/2
    for duration in np.linspace(0., config.dt, config.audit_substeps+1):
        current = [propagate(s, q, duration) for s, q in zip(states, controls)]
        positions = np.array([s.position for s in current])
        distances = np.linalg.norm(positions[:, None]-positions[None], axis=-1)
        lower = distances-gap*(max_speeds[:, None]+max_speeds[None])
        np.fill_diagonal(lower, np.inf)
        pair = min(pair, float(lower.min()))
        for i, point in enumerate(positions):
            for item in world.obstacles:
                obstacle = min(obstacle, item.distance_geometry(point)[0]-gap*max_speeds[i])
        boundary = min(boundary, world.boundary_clearance(positions)-gap*max_speeds.max())
        samples.append(current)
    return samples[-1], pair, obstacle, boundary


@threadpool_limits.wrap(limits=1, user_api="blas")
def run(config, mode="hybrid", output=None, progress=True, fault_step=None):
    if mode not in ("hybrid", "centralized"):
        raise ValueError("mode must be hybrid or centralized")
    world = WorldMap.load(default_map_path())
    scenario = build_scenario(config)
    if config.shared_braking > scenario.max_acceleration.min():
        raise ValueError("shared braking reference exceeds fleet capability")
    navigator = NavigationField(world, scenario.targets, scenario.target_roles,
                                config.grid_resolution, config.navigation_inflation, config.obstacle_clearance)
    planners = [LocalRoutePlanner(navigator, config) for _ in range(config.robots)]
    states = [RobotState(p.copy(), float(h), 0.) for p, h in zip(scenario.position, scenario.heading)]
    if pair_minimum(scenario.position) < config.pair_clearance or world.clearance(scenario.position) < config.obstacle_clearance:
        raise ValueError("initial geometry violates configured safety set")
    sensor = LocalSensor(config.sensing_radius)
    bus_config = config if mode == "hybrid" else replace(config, communication_radius=100.,
                                                        latency_steps=0, packet_loss=0.)
    bus = MessageBus(bus_config, config.seed+90001)
    local_safety = LocalSafetyController(config, world)
    central_safety = CentralSafetyController(config, world)
    guidance_config = config if mode == "hybrid" else replace(config, neighbors=config.robots)
    assignment = None
    version = -1
    central_messages = switches = fallbacks = infeasible = 0
    position = np.empty((config.steps+1, config.robots, 2))
    heading = np.empty((config.steps+1, config.robots))
    speed = np.empty((config.steps+1, config.robots))
    assignments = np.empty((config.steps+1, config.robots), dtype=int)
    control_history = np.empty((config.steps, config.robots, 2))
    adjacency = np.zeros((config.steps, config.robots, config.robots), dtype=bool)
    cross_networks = {name: np.zeros_like(adjacency) for name in ("cross_congestion", "cross_planning", "cross_yield", "cross_cbf", "cross_cbf_binding")}
    density = np.zeros((config.steps, config.robots))
    metrics = []
    events = []
    position[0], heading[0], speed[0] = scenario.position, scenario.heading, 0.
    started = time.perf_counter()
    for step in range(config.steps):
        now = step*config.dt
        if assignment is None or mode == "centralized" or step % config.assignment_period == 0:
            updated = assign(position[step], scenario.roles, scenario.targets, scenario.target_roles,
                             assignment, config.switch_penalty)
            changed = np.flatnonzero(updated != assignment) if assignment is not None else np.array([], dtype=int)
            switches += len(changed)
            # Version changes only if assignment changes; identical refreshes do not discard delayed packets.
            if assignment is None or len(changed):
                version += 1
            assignment = updated
            central_messages += 2*config.robots  # individual telemetry upload + target download
        assignments[step] = assignment
        observations = sensor.observe(states, now)
        messages = bus.exchange(step, states, scenario, assignment, version,
                                yielding=[p.yielding(now) for p in planners],
                                blocked=[p.blocked for p in planners])
        views = [LocalView(i, states[i], observations[i], messages[i]) for i in range(config.robots)]
        references = []
        for i, view in enumerate(views):
            velocity, neighbors, intensity = desired_velocity(
                view, scenario.targets[assignment[i]], int(scenario.roles[i]),
                scenario.capabilities[i], config.robots, now, version, navigator, guidance_config, planners[i])
            density[step, i] = intensity
            adjacency[step, i, neighbors] = True
            references.append(nominal_control(states[i], velocity, scenario.max_speed[i],
                                              scenario.max_acceleration[i], config))
        references = np.asarray(references)
        if mode == "centralized":
            results = central_safety.filter(views, references, scenario.max_speed, scenario.max_acceleration)
        else:
            results = [local_safety.filter(view, references[i], scenario.max_speed[i],
                                           scenario.max_acceleration[i],
                                           force_failure=(step == fault_step and i == 0))
                       for i, view in enumerate(views)]
        for i, result in enumerate(results):
            if not result.feasible:
                infeasible += 1
                events.append({"step": step, "robot": i, "reason": result.reason,
                               "residual": result.minimum_residual})
            fallbacks += int(result.fallback)
        controls = np.array([r.control for r in results])
        networks = cross_role_networks(views, planners, scenario.roles, controls, config, mode, now)
        for name, network in networks.items():
            cross_networks[name][step] = network
        states, pair_bound, obstacle_bound, boundary_bound = audit_hold(states, controls, world, config)
        control_history[step] = controls
        position[step+1] = np.array([s.position for s in states])
        heading[step+1] = [s.heading for s in states]
        speed[step+1] = [s.speed for s in states]
        wheel_max = max(float(np.max(np.abs(wheel_rates(s, q, config)))) for s, q in zip(states, controls))
        metrics.append([now+config.dt, pair_minimum(position[step+1]), world.clearance(position[step+1]),
                        pair_bound, obstacle_bound, boundary_bound,
                        min(r.minimum_residual for r in results),
                        sum(r.fallback for r in results), max(r.constraints for r in results), wheel_max])
        if pair_bound <= 2*config.radius or obstacle_bound <= config.radius or boundary_bound <= config.radius:
            # Fail visibly; never project positions or silently discard an unsafe segment.
            if output is not None:
                Path(output).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(output)/f"{mode}_failed.npz",
                                    position=position[:step+2], control=control_history[:step+1],
                                    metrics=np.asarray(metrics))
            raise RuntimeError(f"uncertified physical clearance at step {step}: pair={pair_bound:.4f}, obstacle={obstacle_bound:.4f}")
        if progress and (step % 100 == 0 or step == config.steps-1):
            current_errors = np.linalg.norm(position[step+1]-scenario.targets[assignment], axis=1)
            completed = np.count_nonzero(current_errors <= config.slot_tolerance)
            print(f"{mode}: {step+1}/{config.steps}, slots={completed}/{config.robots}, pair>={pair_bound:.3f}, obstacle>={obstacle_bound:.3f}, fallback={fallbacks}", flush=True)
    assignments[-1] = assignment
    elapsed = time.perf_counter()-started
    table = np.asarray(metrics)
    final_matching = assign(position[-1], scenario.roles, scenario.targets, scenario.target_roles)
    slot_error = np.linalg.norm(position[-1]-scenario.targets[final_matching], axis=1)
    region_distance = np.linalg.norm(position[-1]-CENTERS[scenario.roles], axis=1)
    summary = dict(mode=mode, seed=config.seed, robots=config.robots, horizon=config.horizon,
                   transport=transport(position[-1], scenario.roles, scenario.targets, scenario.target_roles),
                   region_completion=float(np.mean(region_distance <= REGION_RADII[scenario.roles])),
                   slot_completion=float(np.mean(slot_error <= config.slot_tolerance)),
                   mean_slot_error=float(slot_error.mean()),
                   min_pair=float(min(pair_minimum(position[0]), table[:, 1].min())),
                   min_obstacle=float(min(world.clearance(position[0]), table[:, 2].min())),
                   pair_interval_lower=float(table[:, 3].min()),
                   obstacle_interval_lower=float(table[:, 4].min()),
                   boundary_interval_lower=float(table[:, 5].min()),
                   min_cbf_residual=float(table[:, 6].min()),
                   fallback_count=fallbacks, infeasible_count=infeasible,
                   max_constraints=int(table[:, 8].max()), max_wheel_rate=float(table[:, 9].max()),
                   assignment_changes=int(switches), central_messages=central_messages,
                   local_yield_events=sum(p.yield_count for p in planners),
                   local_replans=sum(p.replans for p in planners),
                   local_route_failures=sum(p.failures for p in planners),
                   neighbor_sent=bus.sent, neighbor_delivered=bus.delivered, neighbor_dropped=bus.dropped,
                   neighbor_expired=bus.expired, total_messages=central_messages+bus.sent,
                   runtime_seconds=elapsed, seconds_per_step=elapsed/config.steps,
                   configured_pair_margin_met=bool(table[:, 3].min() >= config.pair_clearance),
                   configured_obstacle_margin_met=bool(table[:, 4].min() >= config.obstacle_clearance))
    result = dict(summary=summary, config=asdict(config), events=events,
                  position=position, heading=heading, speed=speed,
                  assignment=assignments, control=control_history, adjacency=adjacency,
                  density=density, metrics=table, targets=scenario.targets, roles=scenario.roles,
                  target_roles=scenario.target_roles, final_matching=final_matching, **cross_networks)
    if output is not None:
        save_result(result, output)
    return result


def save_result(result, output):
    import json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mode = result["summary"]["mode"]
    np.savez_compressed(output/f"{mode}_trajectory.npz",
                        **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    (output/f"{mode}_summary.json").write_text(
        json.dumps({k: result[k] for k in ("summary", "config", "events")}, indent=2), encoding="utf-8")
    np.savetxt(output/f"{mode}_metrics.csv", result["metrics"], delimiter=",", comments="",
               header="time,min_pair,min_obstacle,pair_interval_lower,obstacle_interval_lower,boundary_interval_lower,min_cbf_residual,fallback_robots,max_constraints,max_wheel_rate")

