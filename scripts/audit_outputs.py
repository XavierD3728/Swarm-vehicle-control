"""Independent consistency audit of stored V6 trajectories and metrics."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

from amr_control.scenario import build_scenario
from amr_control.config import Config


def audit(directory):
    directory = Path(directory)
    results = []
    for mode in ("centralized", "hybrid"):
        metadata = json.loads((directory/f"{mode}_summary.json").read_text(encoding="utf-8"))
        summary, config = metadata["summary"], Config(**metadata["config"])
        scenario = build_scenario(config)
        with np.load(directory/f"{mode}_trajectory.npz") as source:
            arrays = {k: source[k] for k in source.files}
        p, theta, speed, q = (arrays[k] for k in ("position", "heading", "speed", "control"))
        assignment = arrays["assignment"]
        assert p.shape == (config.steps+1, config.robots, 2)
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert np.all(np.sort(assignment, axis=1) == np.arange(config.robots))
        assert np.all(arrays["target_roles"][assignment] == arrays["roles"][None])
        assert np.max(np.abs(q[:, :, 0])-scenario.max_acceleration[None]) <= 2e-7
        assert np.max(np.abs(q[:, :, 1])) <= config.turn_limit+2e-7
        assert speed.min() >= -2e-7
        assert np.max(speed-scenario.max_speed[None]) <= 2e-7
        np.testing.assert_allclose(np.diff(speed, axis=0), config.dt*q[:, :, 0], atol=1e-10)
        for endpoint_speed in (speed[:-1], speed[1:]):
            for sign in (-1, 1):
                wheels = (endpoint_speed+sign*config.axle_length*q[:, :, 1]/2)/config.wheel_radius
                assert np.max(abs(wheels)) <= config.wheel_rate_limit+2e-7
        # Verify selected segments against an independent ODE integrator.
        rng = np.random.default_rng(2026)
        for step, i in zip(rng.integers(0, config.steps, 20), rng.integers(0, config.robots, 20)):
            b, omega = q[step, i]
            start = [*p[step, i], theta[step, i], speed[step, i]]
            solution = solve_ivp(lambda t, z: [z[3]*np.cos(z[2]), z[3]*np.sin(z[2]), omega, b],
                                 [0, config.dt], start, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(solution.y[:2, -1], p[step+1, i], atol=2e-9)
        metrics = arrays["metrics"]
        assert metrics[:, 3].min() > 2*config.radius
        assert metrics[:, 4].min() > config.radius
        assert metrics[:, 5].min() > config.radius
        assert summary["pair_interval_lower"] == float(metrics[:, 3].min())
        assert summary["obstacle_interval_lower"] == float(metrics[:, 4].min())
        assert summary["fallback_count"] == int(metrics[:, 7].sum())
        assert summary["fallback_count"] == 0
        assert summary["configured_pair_margin_met"] and summary["configured_obstacle_margin_met"]
        final = arrays["final_matching"]
        errors = np.linalg.norm(p[-1]-arrays["targets"][final], axis=1)
        assert np.count_nonzero(errors <= config.slot_tolerance) == config.robots
        assert summary["slot_completion"] == 1.
        # Report when everyone was first, and then continuously, within tolerance.
        history_errors = np.linalg.norm(p-arrays["targets"][assignment], axis=2)
        complete = np.all(history_errors <= config.slot_tolerance, axis=1)
        first = float(np.flatnonzero(complete)[0]*config.dt) if complete.any() else None
        incomplete = np.flatnonzero(~complete)
        sustained = float((incomplete[-1]+1)*config.dt) if len(incomplete) else 0.
        results.append(dict(mode=mode, passed=True, final_slots=config.robots,
                            first_all_slots_time=first, sustained_all_slots_time=sustained,
                            max_final_slot_error=float(errors.max()),
                            checked_ode_segments=20, checked_control_steps=config.steps,
                            pair_interval_lower=summary["pair_interval_lower"],
                            obstacle_interval_lower=summary["obstacle_interval_lower"],
                            fallback_count=summary["fallback_count"]))
    (directory/"AUDIT.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    audit(sys.argv[1] if len(sys.argv) > 1 else "output/validation")

