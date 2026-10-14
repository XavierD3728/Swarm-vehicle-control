from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from amr_control.assignment import assign
from amr_control.communication import LocalSensor, LocalView, MessageBus
from amr_control.config import Config
from amr_control.dynamics import RobotState, propagate, input_halfspaces
from amr_control.experiment import audit_hold, run
from amr_control.geometry import Rectangle, WorldMap
from amr_control.guidance import desired_velocity
from amr_control.safety import solve_qp_2d, LocalSafetyController, pair_constraint
from amr_control.scenario import build_scenario


@pytest.mark.parametrize("omega", [0., 1e-7, .8, -2.5])
def test_exact_actuator_solution_matches_independent_ode(omega):
    state = RobotState(np.array([2., 3.]), .7, .8)
    control = np.array([-.4, omega])
    duration = .3
    reference = solve_ivp(lambda t, z: [z[3]*np.cos(z[2]), z[3]*np.sin(z[2]), omega, -.4],
                          [0., duration], [2., 3., .7, .8], rtol=1e-11, atol=1e-12).y[:, -1]
    result = propagate(state, control, duration)
    np.testing.assert_allclose(result.position, reference[:2], atol=1e-10)
    assert result.speed == pytest.approx(reference[3])


def test_rectangle_tangential_velocity_has_zero_flat_edge_curvature():
    rectangle = Rectangle(1., 2., 1., 2.)
    point = np.array([2.5, 1.5])
    distance, normal, hessian = rectangle.distance_geometry(point)
    assert distance == .5
    np.testing.assert_allclose(normal, [1., 0.])
    assert np.array([0., 4.])@hessian@np.array([0., 4.]) == 0.
    for offset in (-1e-4, 1e-4):
        assert rectangle.distance_geometry(point+[0., offset])[0] == .5


def test_rectangle_corner_hessian_finite_difference():
    rectangle = Rectangle(1., 2., 1., 2.)
    point, direction, delta = np.array([2.4, 2.3]), np.array([.3, -.7]), 1e-4
    distance, normal, hessian = rectangle.distance_geometry(point)
    plus = rectangle.distance_geometry(point+delta*direction)[0]
    minus = rectangle.distance_geometry(point-delta*direction)[0]
    assert (plus-minus)/(2*delta) == pytest.approx(normal@direction, abs=1e-7)
    assert (plus-2*distance+minus)/delta**2 == pytest.approx(direction@hessian@direction, abs=1e-6)


def test_qp_matches_independent_slsqp_on_random_polygons():
    rng = np.random.default_rng(25)
    for _ in range(20):
        rows = np.vstack((np.eye(2), -np.eye(2), rng.normal(size=(8, 2))))
        lower = -rng.uniform(.1, 1.5, len(rows))  # origin feasible
        reference = rng.normal(size=2)*3
        matrix = rng.normal(size=(2, 2))
        metric = matrix.T@matrix+.1*np.eye(2)
        actual = solve_qp_2d(reference, metric, rows, lower)
        expected = minimize(lambda q: .5*(q-reference)@metric@(q-reference), np.zeros(2),
                            jac=lambda q: metric@(q-reference), method="SLSQP",
                            constraints={"type": "ineq", "fun": lambda q: rows@q-lower, "jac": lambda q: rows},
                            options={"ftol": 1e-9, "maxiter": 200})
        assert expected.success
        np.testing.assert_allclose(actual, expected.x, atol=2e-5)


def test_qp_reports_empty_set():
    assert solve_qp_2d(np.zeros(2), np.eye(2), np.array([[1., 0.], [-1., 0.]]), np.ones(2)) is None


def test_range_sensor_excludes_remote_robot_and_preserves_other_roles():
    states = [RobotState(np.array([x, 0.]), 0., 0.) for x in (0., .8, 10.)]
    observations = LocalSensor(1.).observe(states, 0.)
    assert [o.robot_id for o in observations[0]] == [1]


def test_message_latency_loss_and_expiry():
    config = Config(robots=6, latency_steps=2, packet_loss=0., communication_radius=100., message_ttl=.1)
    scenario = build_scenario(config)
    states = [RobotState(p, 0., 0.) for p in scenario.position]
    bus = MessageBus(config, 2)
    assignment = assign(scenario.position, scenario.roles, scenario.targets, scenario.target_roles)
    assert all(not items for items in bus.exchange(0, states, scenario, assignment, 0))
    assert all(not items for items in bus.exchange(1, states, scenario, assignment, 0))
    assert all(len(items) == 5 for items in bus.exchange(2, states, scenario, assignment, 0))
    bus.config = replace(config, packet_loss=1.)
    bus.exchange(3, states, scenario, assignment, 0)
    bus.exchange(4, states, scenario, assignment, 0)
    for step in range(5, 9):
        output = bus.exchange(step, states, scenario, assignment, 0)
    assert all(not items for items in output)
    assert bus.dropped > 0 and bus.expired > 0


def test_assignment_switches_only_target_identity_and_penalty_can_hold():
    positions = np.array([[1., 0.], [0., 0.]])
    targets = np.array([[0., 0.], [1., 0.]])
    roles = np.zeros(2, dtype=int)
    np.testing.assert_array_equal(assign(positions, roles, targets, roles), [1, 0])
    np.testing.assert_array_equal(assign(positions, roles, targets, roles, np.array([0, 1]), 3.), [0, 1])


def test_forced_solver_failure_is_explicit_and_brakes():
    config = Config(robots=6)
    state = RobotState(np.array([5., 4.]), 0., .4)
    view = LocalView(0, state, (), ())
    result = LocalSafetyController(config, WorldMap(())).filter(view, np.array([.5, 1.]), 1., .75, True)
    assert result.fallback and not result.feasible and result.reason == "forced_failure"
    np.testing.assert_allclose(result.control, [-.75, 0.])


def test_zero_speed_safety_contains_no_division_by_speed():
    config = Config(robots=6)
    state = RobotState(np.array([2., 2.]), 0., 0.)
    result = LocalSafetyController(config, WorldMap(())).filter(LocalView(0, state, (), ()),
                                                              np.array([1., 2.]), 1., .75)
    assert result.feasible and np.isfinite(result.control).all()


def test_reciprocal_rows_sum_to_joint_hocbf():
    config = Config()
    first = RobotState(np.array([0., 0.]), .2, .3)
    second = RobotState(np.array([.8, .1]), 2.8, .2)
    ri, rj, bound = pair_constraint(first, second, config)
    qi, qj = np.array([-.2, .7]), np.array([-.3, -.5])
    r = first.position-second.position
    np.testing.assert_allclose(ri@qi+rj@qj, 2*r@(first.input_matrix@qi-second.input_matrix@qj))


def test_interval_auditor_catches_between_endpoint_encounter():
    config = Config(robots=6, dt=.5, horizon=.5)
    states = [RobotState(np.array([4., 4.]), 0., 2.), RobotState(np.array([5., 4.]), np.pi, 2.)]
    _, pair, _, _ = audit_hold(states, np.zeros((2, 2)), WorldMap(()), config)
    assert pair <= 0.  # both endpoints alone would miss the encounter


def test_control_limits_hold_at_intermediate_times():
    config = Config()
    state = RobotState(np.array([3., 3.]), 0., .5)
    rows, lower = input_halfspaces(state, 1., .75, config)
    q = solve_qp_2d(np.array([2., 5.]), np.eye(2), rows, lower)
    assert np.min(rows@q-lower) >= -1e-8
    for t in np.linspace(0, config.dt, 20):
        current = propagate(state, q, t)
        assert 0 <= current.speed <= 1.
        wheels = np.array([current.speed-.21*q[1], current.speed+.21*q[1]])/.09
        assert max(abs(wheels)) <= config.wheel_rate_limit+1e-8


def test_hybrid_smoke_records_physical_audit_and_actual_message_counts():
    result = run(Config(robots=6, horizon=.2), progress=False)
    assert result["position"].shape == (5, 6, 2)
    summary = result["summary"]
    assert summary["fallback_count"] == 0
    assert summary["pair_interval_lower"] > .36
    assert summary["obstacle_interval_lower"] > .18
    assert summary["neighbor_sent"] >= summary["neighbor_delivered"]



def test_navigation_can_exit_conservative_inflation_shell():
    from amr_control.navigation import NavigationField
    from amr_control.geometry import segment_rectangle_distance
    obstacle = Rectangle(4.35, 7.55, 2., 2.72)
    world = WorldMap((obstacle,))
    field = NavigationField(world, np.array([[10., 4.]]), np.array([0]), inflation=.48)
    point = np.array([4., 2.3])  # safe actual clearance .35, inside planning shell
    assert not field.visible(point, np.array([10., 4.]))
    direction = field.direction(point, 0)
    assert np.linalg.norm(direction) > .99
    assert segment_rectangle_distance(point, point+.1*direction, obstacle) >= .35-1e-7


def test_braking_anchored_reciprocity_preserves_joint_constraint():
    from amr_control.safety import brake
    config = Config()
    first = RobotState(np.array([2., 2.]), .3, .8)
    second = RobotState(np.array([2.8, 2.2]), 2.8, .2)
    ri, rj, bound = pair_constraint(first, second, config)
    qi = brake(first, config.shared_braking, config)
    qj = brake(second, config.shared_braking, config)
    leftover = bound-ri@qi-rj@qj
    bi, bj = ri@qi+leftover/2, rj@qj+leftover/2
    assert bi+bj == pytest.approx(bound)
    if ri@qi+rj@qj >= bound:
        assert ri@qi >= bi and rj@qj >= bj


def test_failure_injection_is_visible_in_experiment_metrics():
    result = run(Config(robots=6, horizon=.1), progress=False, fault_step=0)
    assert result['summary']['fallback_count'] >= 1
    assert any(event['reason'] == 'forced_failure' for event in result['events'])


def test_local_guidance_does_not_need_a_global_population_array():
    from amr_control.navigation import NavigationField
    from amr_control.communication import Observation
    config = Config(robots=6)
    own = RobotState(np.array([2., 3.]), 0., .1)
    other = RobotState(np.array([2.6, 3.]), 0., .1)
    view = LocalView(0, own, (Observation(1, other, 0.),), ())
    field = NavigationField(WorldMap(()), np.array([[8., 3.]]), np.array([0]))
    velocity, neighbors, intensity = desired_velocity(view, np.array([8., 3.]), 0,
                                                       np.ones(4), 6, 0., 0, field, config)
    assert np.isfinite(velocity).all() and neighbors == [] and intensity > 0


@pytest.mark.parametrize("count", [6, 12, 24, 32])
def test_constructed_slots_are_physically_compatible(count):
    config = Config(robots=count)
    scenario = build_scenario(config)
    distances = np.linalg.norm(scenario.targets[:, None]-scenario.targets[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= config.pair_clearance+.12-1e-9
    from amr_control.geometry import default_map_path
    world = WorldMap.load(default_map_path())
    assert world.clearance(scenario.targets) > config.obstacle_clearance
    assert world.boundary_clearance(scenario.targets) > config.obstacle_clearance


def test_local_qp_has_joint_pair_residual_on_executed_controls():
    result = run(Config(robots=6, horizon=.5), progress=False)
    config = Config(robots=6, horizon=.5)
    for step, controls in enumerate(result['control']):
        states = [RobotState(result['position'][step, i], result['heading'][step, i], result['speed'][step, i]) for i in range(6)]
        for i in range(6):
            for j in range(i+1, 6):
                if np.linalg.norm(states[i].position-states[j].position) <= config.sensing_radius:
                    ri, rj, bound = pair_constraint(states[i], states[j], config)
                    assert ri@controls[i]+rj@controls[j]-bound >= -3e-7


def test_actual_adaptive_reciprocal_allocations_sum_exactly():
    from amr_control.safety import reciprocal_allocation
    config = Config()
    rng = np.random.default_rng(4)
    for _ in range(30):
        first = RobotState(rng.normal(size=2), rng.uniform(-3, 3), rng.uniform(0, 1.3))
        second = RobotState(rng.normal(size=2), rng.uniform(-3, 3), rng.uniform(0, 1.3))
        _, left = reciprocal_allocation(first, second, config)
        _, right = reciprocal_allocation(second, first, config)
        _, _, joint = pair_constraint(first, second, config)
        assert left+right == pytest.approx(joint, abs=1e-10)


def test_local_route_planner_goes_around_observed_blocker():
    from amr_control.local_planner import LocalRoutePlanner
    from amr_control.navigation import NavigationField
    from amr_control.communication import Observation
    config = Config(robots=6)
    field = NavigationField(WorldMap(()), np.array([[8., 4.]]), np.array([0]))
    planner = LocalRoutePlanner(field, config)
    own = RobotState(np.array([3., 4.]), 0., 0.)
    blocker = RobotState(np.array([4., 4.]), 0., 0.)
    view = LocalView(0, own, (Observation(1, blocker, 0.),), ())
    velocity = planner.velocity(view, np.array([8., 4.]), 0.)
    assert np.linalg.norm(velocity) > .5
    assert abs(velocity[1]) > .1
    assert planner.replans == 1
    assert len(planner.path) > 2
    for start, end in zip(planner.path[:-1], planner.path[1:]):
        segment = end-start
        fraction = np.clip(float((blocker.position-start)@segment)/float(segment@segment), 0, 1)
        assert np.linalg.norm(start+fraction*segment-blocker.position) >= config.pair_clearance-.02


def test_local_route_replans_after_target_swap():
    from amr_control.local_planner import LocalRoutePlanner
    from amr_control.navigation import NavigationField
    from amr_control.communication import Observation
    config = Config(robots=6)
    field = NavigationField(WorldMap(()), np.array([[8., 4.]]), np.array([0]))
    planner = LocalRoutePlanner(field, config)
    own = RobotState(np.array([3., 4.]), 0., 0.)
    blocker = RobotState(np.array([4., 4.]), 0., 0.)
    view = LocalView(0, own, (Observation(1, blocker, 0.),), ())
    planner.velocity(view, np.array([8., 4.]), 0.)
    planner.velocity(view, np.array([8., 4.1]), .05)
    assert planner.replans == 2
    np.testing.assert_allclose(planner.target, [8., 4.1])


def test_arrived_robot_yields_to_local_blocked_request_without_reassigning_goal():
    from amr_control.local_planner import LocalRoutePlanner
    from amr_control.navigation import NavigationField
    from amr_control.communication import Observation, Packet
    config = Config(robots=6)
    goal = np.array([4.2, 4.])
    own = RobotState(goal.copy(), 0., 0.)
    peer = RobotState(np.array([3., 4.]), 0., 0.)
    request = Packet(1, peer.position, peer.velocity, np.array([8., 4.]), 0,
                     np.ones(4), 0., 0, blocked=True)
    view = LocalView(0, own, (Observation(1, peer, 0.),), (request,))
    planner = LocalRoutePlanner(NavigationField(WorldMap(()), goal[None], np.array([0])), config)
    temporary = planner.yielding_waypoint(view, goal, 0.)
    assert temporary is not None and np.linalg.norm(temporary-goal) > .5
    assert planner.yielding(1.) and not planner.yielding(4.)
    np.testing.assert_array_equal(goal, [4.2, 4.])
    yielding_request = replace(request, yielding=True)
    fresh = LocalRoutePlanner(planner.navigator, config)
    assert fresh.yielding_waypoint(replace(view, messages=(yielding_request,)), goal, 0.) is None


def test_message_bus_transmits_local_blocked_and_yielding_flags():
    config = Config(robots=6, packet_loss=0., latency_steps=0, communication_radius=100.)
    scenario = build_scenario(config)
    states = [RobotState(p, 0., 0.) for p in scenario.position]
    assignment = assign(scenario.position, scenario.roles, scenario.targets, scenario.target_roles)
    packets = MessageBus(config, 1).exchange(0, states, scenario, assignment, 0,
                  yielding=[True,False,False,False,False,False], blocked=[False,True,False,False,False,False])
    assert next(p for p in packets[2] if p.robot_id == 0).yielding
    assert next(p for p in packets[2] if p.robot_id == 1).blocked


def test_local_memory_keeps_observed_stationary_blockers_out_of_range():
    from amr_control.local_planner import LocalRoutePlanner
    from amr_control.navigation import NavigationField
    from amr_control.communication import Observation
    config = Config(robots=6)
    planner = LocalRoutePlanner(NavigationField(WorldMap(()), np.array([[8., 4.]]), np.array([0])), config)
    own = RobotState(np.array([2., 4.]), 0., 0.)
    parked = Observation(1, RobotState(np.array([4., 4.]), 0., 0.), 0.)
    planner.planning_view(LocalView(0, own, (parked,), ()), 0.)
    unseen = LocalView(0, own, (), ())
    assert [o.robot_id for o in planner.planning_view(unseen, 5.).observed] == [1]
    assert planner.planning_view(unseen, 31.).observed == ()
    assert unseen.observed == ()  # the safety input is not mutated


def test_moving_observation_memory_predicts_then_expires():
    from amr_control.local_planner import LocalRoutePlanner
    from amr_control.navigation import NavigationField
    from amr_control.communication import Observation
    config = Config(robots=6)
    planner = LocalRoutePlanner(NavigationField(WorldMap(()), np.array([[8., 4.]]), np.array([0])), config)
    own = RobotState(np.array([2., 4.]), 0., 0.)
    moving = Observation(1, RobotState(np.array([4., 4.]), 0., 1.), 0.)
    planner.planning_view(LocalView(0, own, (moving,), ()), 0.)
    unseen = LocalView(0, own, (), ())
    estimate = planner.planning_view(unseen, .2).observed[0]
    np.testing.assert_allclose(estimate.state.position, [4.2, 4.])
    assert planner.planning_view(unseen, .6).observed == ()
