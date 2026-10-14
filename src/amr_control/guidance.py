"""Local feedback accepts only self state, observations, packets and map products."""
import numpy as np


def capability_weight(first, second):
    return .15+.85*np.exp(-np.sum((first-second)**2)/.38)


def local_density(point, observations, bandwidth=.7):
    """Local KDE intensity, not an estimate of the full global probability measure."""
    return sum(np.exp(-np.sum((point-o.state.position)**2)/(2*bandwidth**2))
               for o in observations)/(2*np.pi*bandwidth**2)


def desired_velocity(view, target, role, capability, population_count, now, version,
                     navigator, config, planner=None):
    point = view.own.position
    error = point-target
    visible = navigator.visible(point, target)
    nav = -config.target_weight*error if visible else config.flow_speed*navigator.direction(point, role)
    if planner is not None:
        nav = planner.velocity(view, target, now)
    messages = [p for p in view.messages if p.role == role
                and p.assignment_version == version
                and now-p.timestamp <= config.message_ttl+1e-12]
    messages.sort(key=lambda p: np.linalg.norm(p.position-point))
    selected = messages[:config.neighbors]
    graphon = np.zeros(2)
    for packet in selected:
        predicted = packet.position+(now-packet.timestamp)*packet.velocity
        other_error = predicted-packet.target
        graphon += capability_weight(capability, packet.capability)*(error-other_error)
    graphon *= -config.graphon_weight/population_count
    congestion = np.zeros(2)
    # psi(r)=(1-r^2/R^2)^3 on r<R: compact support, C2 at the cutoff.
    radius = config.congestion_radius
    for obs in view.observed:
        relative = point-obs.state.position
        squared = float(relative@relative)
        if squared < radius**2:
            congestion += 6*(1-squared/radius**2)**2*relative/radius**2
    congestion *= config.congestion_weight
    # A local geometric lookahead selects a clear passing direction around
    # observed robots; CBF remains responsible for dynamic execution safety.
    detour = np.zeros(2)
    nav_norm = np.linalg.norm(nav)
    if planner is None and np.linalg.norm(error) > .4 and nav_norm > 1e-8 and view.observed:
        horizon = .85
        offsets = np.array([o.state.position-point for o in view.observed])
        distance = np.linalg.norm(offsets, axis=1)
        clearance = np.minimum(config.pair_clearance+.10, distance-1e-6)
        forward = nav/nav_norm
        fraction = np.clip(offsets@forward, 0., horizon)
        nominal_distance = np.linalg.norm(offsets-fraction[:, None]*forward, axis=1)
        if np.any(nominal_distance < clearance):
            angles = np.arange(24)*2*np.pi/24+np.arctan2(forward[1], forward[0])
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
            along = np.clip(directions@offsets.T, 0., horizon)
            separation = np.linalg.norm(offsets[None]-along[:, :, None]*directions[:, None], axis=-1)
            candidates = np.flatnonzero(np.all(separation >= clearance[None], axis=1))
            start_clearance = min((o.distance_geometry(point)[0] for o in navigator.world.obstacles), default=np.inf)
            map_clearance = min(navigator.inflation, start_clearance-1e-7)
            ranked = []
            for index in candidates:
                endpoint = point+horizon*directions[index]
                iy, ix = navigator.cell(endpoint)
                if navigator.occupied[iy, ix] or not navigator.visible(point, endpoint, map_clearance):
                    continue
                cost = np.linalg.norm(endpoint-target) if visible else navigator.fields[role][iy, ix]
                if np.isfinite(cost):
                    # Small consistent chirality term resolves exact symmetry.
                    cost += .03*(1-float(directions[index]@forward))+.005*np.sin(angles[index]-angles[0])
                    ranked.append((cost, index))
            if ranked:
                index = min(ranked)[1]
                detour = min(nav_norm, config.flow_speed)*directions[index]-nav
    obstacle_push = np.zeros(2)
    for obstacle in navigator.world.obstacles:
        distance, normal, _ = obstacle.distance_geometry(point)
        if distance < 1.0:
            obstacle_push += .12*(1/max(distance, .08)-1)*normal
    velocity = nav+graphon+congestion+obstacle_push+detour
    norm = np.linalg.norm(velocity)
    if norm > config.flow_speed:
        velocity *= config.flow_speed/norm
    return velocity, [p.robot_id for p in selected], local_density(point, view.observed)

