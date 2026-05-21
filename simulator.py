"""
Three-body problem simulator.

State convention:
    The system state is a 1D NumPy array of length 18, ordered as:
    [x1, y1, z1, x2, y2, z2, x3, y3, z3,    <-- positions (9 components)
     vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]  <-- velocities (9)

    Internally, the integrator works with this flat array.
    Physics code uses the helper functions below to access (3,3)
    structured arrays where row i = body i, columns = (x, y, z).
"""

import numpy as np


def pack(positions, velocities):
    """
    Pack (3,3) position and velocity arrays into the flat 18-element state.

    Parameters
    ----------
    positions : ndarray, shape (3, 3)
        positions[i] = position vector of body i, with 3 spatial components.
    velocities : ndarray, shape (3, 3)
        velocities[i] = velocity vector of body i.

    Returns
    -------
    state : ndarray, shape (18,)
    """
    return np.concatenate([positions.flatten(), velocities.flatten()])


def unpack(state):
    """
    Unpack the flat 18-element state into (3,3) position and velocity arrays.

    Parameters
    ----------
    state : ndarray, shape (18,)

    Returns
    -------
    positions : ndarray, shape (3, 3)
    velocities : ndarray, shape (3, 3)
    """
    positions = state[:9].reshape(3, 3)
    velocities = state[9:].reshape(3, 3)
    return positions, velocities

    # Gravitational constant in natural units.
# We work in dimensionless units where G = 1. Real SI value is
# 6.674e-11 m^3/(kg*s^2) but using natural units avoids floating-point
# precision issues with the huge numbers in real astrophysics.
G = 1.0


def compute_accelerations(positions, masses, softening=0.0):
    """
    Compute gravitational acceleration on each body.

    Implements Newton's law of gravitation with optional Plummer softening:

        a_i = sum over j != i of  G * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

    With softening = 0, this reduces to standard 1/r^2 gravity.

    Parameters
    ----------
    positions : ndarray, shape (3, 3)
        positions[i] = (x, y, z) of body i.
    masses : ndarray, shape (3,)
        masses[i] = mass of body i. We assume m_i > 0.
    softening : float
        Softening length epsilon. Default 0 = exact Newtonian gravity.
        Nonzero values prevent the 1/r^2 singularity at close approach.

    Returns
    -------
    accelerations : ndarray, shape (3, 3)
        accelerations[i] = (ax, ay, az) of body i.
    """
    n_bodies = positions.shape[0]
    accelerations = np.zeros_like(positions)

    for i in range(n_bodies):
        for j in range(n_bodies):
            if i == j:
                continue
            # Displacement from body i to body j
            displacement = positions[j] - positions[i]
            # Squared distance, with softening
            distance_squared = np.sum(displacement**2) + softening**2
            # 1 / r^3 = 1 / (r^2)^(3/2)
            inv_dist_cubed = distance_squared**(-1.5)
            # Acceleration contribution from body j
            accelerations[i] += G * masses[j] * displacement * inv_dist_cubed

    return accelerations

def compute_energy(positions, velocities, masses, softening=0.0):
    """
    Compute total mechanical energy E = T + U.

    Kinetic energy:
        T = sum_i (1/2) * m_i * |v_i|^2

    Potential energy (with optional Plummer softening):
        U = -sum_{i<j} G * m_i * m_j / sqrt(|r_j - r_i|^2 + eps^2)

    The softened potential matches the softened force in compute_accelerations,
    so this energy is the conserved quantity of that dynamical system.

    Parameters
    ----------
    positions : ndarray, shape (3, 3)
    velocities : ndarray, shape (3, 3)
    masses : ndarray, shape (3,)
    softening : float
        Same softening parameter used in compute_accelerations.

    Returns
    -------
    total_energy : float
    kinetic : float
    potential : float
    """
    n_bodies = positions.shape[0]

    # Kinetic energy: (1/2) m v^2, summed over bodies
    speeds_squared = np.sum(velocities**2, axis=1)  # shape (3,)
    kinetic = 0.5 * np.sum(masses * speeds_squared)

    # Potential energy: -G m_i m_j / r, summed over unique pairs
    potential = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):  # j > i avoids double-counting
            displacement = positions[j] - positions[i]
            distance = np.sqrt(np.sum(displacement**2) + softening**2)
            potential -= G * masses[i] * masses[j] / distance

    total_energy = kinetic + potential
    return total_energy, kinetic, potential

def state_derivative(state, masses, softening=0.0):
    """
    Compute the time derivative of the state vector.

    For an 18-dimensional state (positions, velocities), the time derivative is:
        d(positions)/dt = velocities
        d(velocities)/dt = accelerations

    This is the function f(y) that integrators repeatedly call.

    Parameters
    ----------
    state : ndarray, shape (18,)
        Flat state vector, see pack/unpack convention.
    masses : ndarray, shape (3,)
    softening : float

    Returns
    -------
    derivative : ndarray, shape (18,)
        Time derivative of the state, same shape and ordering.
    """
    positions, velocities = unpack(state)
    accelerations = compute_accelerations(positions, masses, softening)
    return pack(velocities, accelerations)


def rk4_step(state, dt, masses, softening=0.0):
    """
    Advance the state by one RK4 step of size dt.

    Implements classical 4th-order Runge-Kutta:
        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        k3 = f(y_n + dt/2 * k2)
        k4 = f(y_n + dt   * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Parameters
    ----------
    state : ndarray, shape (18,)
    dt : float
        Timestep.
    masses : ndarray, shape (3,)
    softening : float

    Returns
    -------
    new_state : ndarray, shape (18,)
        State advanced by dt.
    """
    k1 = state_derivative(state,                   masses, softening)
    k2 = state_derivative(state + 0.5 * dt * k1,   masses, softening)
    k3 = state_derivative(state + 0.5 * dt * k2,   masses, softening)
    k4 = state_derivative(state + dt * k3,         masses, softening)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4(initial_state, masses, t_span, dt, softening=0.0):
    """
    Integrate the system from t_span[0] to t_span[1] using RK4 with fixed dt.

    Parameters
    ----------
    initial_state : ndarray, shape (18,)
        Initial state at t = t_span[0].
    masses : ndarray, shape (3,)
    t_span : tuple of (t_start, t_end)
    dt : float
        Fixed timestep.
    softening : float

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
        Time at each output step.
    states : ndarray, shape (n_steps + 1, 18)
        State vector at each output time. states[k] = state at times[k].
    """
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))

    times = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, 18))

    times[0] = t_start
    states[0] = initial_state

    for k in range(n_steps):
        states[k + 1] = rk4_step(states[k], dt, masses, softening)
        times[k + 1] = times[k] + dt

    return times, states

def verlet_step(state, dt, masses, softening=0.0):
    """
    Advance the state by one velocity Verlet step.

    Implements the velocity Verlet algorithm:
        r(t+h) = r(t) + v(t)*h + (1/2)*a(t)*h^2
        a(t+h) = F(r(t+h)) / m
        v(t+h) = v(t) + (1/2)*[a(t) + a(t+h)]*h

    Velocity Verlet is symplectic: it preserves a modified Hamiltonian
    (shadow energy) exactly, so true energy oscillates near a constant
    value instead of drifting. This makes it the integrator of choice
    for long-time integration of gravitational systems.

    Parameters
    ----------
    state : ndarray, shape (18,)
        Flat state vector.
    dt : float
        Timestep.
    masses : ndarray, shape (3,)
    softening : float

    Returns
    -------
    new_state : ndarray, shape (18,)
    """
    positions, velocities = unpack(state)

    # Compute acceleration at current position
    accelerations_old = compute_accelerations(positions, masses, softening)

    # Update positions using current velocity and acceleration
    new_positions = positions + velocities * dt + 0.5 * accelerations_old * dt**2

    # Compute acceleration at new position
    accelerations_new = compute_accelerations(new_positions, masses, softening)

    # Update velocities using average of old and new acceleration
    new_velocities = velocities + 0.5 * (accelerations_old + accelerations_new) * dt

    return pack(new_positions, new_velocities)


def integrate_verlet(initial_state, masses, t_span, dt, softening=0.0):
    """
    Integrate using velocity Verlet with fixed timestep dt.

    See integrate_rk4 for parameter semantics. Output format is identical
    so the two integrators are interchangeable from the notebook's
    perspective.
    """
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))

    times = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, 18))
    times[0] = t_start
    states[0] = initial_state

    for k in range(n_steps):
        states[k + 1] = verlet_step(states[k], dt, masses, softening)
        times[k + 1] = times[k] + dt

    return times, states

def figure_8_initial_conditions():
    """
    Initial conditions for the Chenciner-Montgomery figure-8 orbit.

    Three equal masses (m=1) in a periodic orbit where all three bodies
    chase each other along a single figure-8 curve.

    Reference:
        Chenciner, A. & Montgomery, R. (2000). "A remarkable periodic
        solution of the three-body problem in the case of equal masses."
        Annals of Mathematics, 152(3), 881-901.

    Returns
    -------
    positions : ndarray, shape (3, 3)
    velocities : ndarray, shape (3, 3)
    masses : ndarray, shape (3,)
    period : float
        Approximate orbital period (one full traversal of the figure-8).
    """
    positions = np.array([
        [-0.97000436,  0.24308753, 0.0],
        [ 0.97000436, -0.24308753, 0.0],
        [ 0.0,         0.0,        0.0]
    ])

    v0 = np.array([0.4662036850, 0.4323657300, 0.0])
    velocities = np.array([
        v0,
        v0,
        -2.0 * v0
    ])

    masses = np.array([1.0, 1.0, 1.0])
    period = 6.32591398

    return positions, velocities, masses, period

from scipy.integrate import solve_ivp

def pythagorean_initial_conditions():
    """
    Initial conditions for the Pythagorean three-body problem.

    Three bodies of masses 3, 4, 5 placed at the corners of a 3-4-5 right
    triangle, all at rest. The mass at each corner is opposite the side of
    corresponding length:
        Body 0 (mass 3) is opposite the side of length 3
        Body 1 (mass 4) is opposite the side of length 4
        Body 2 (mass 5) is opposite the side of length 5

    First posed by Burrau (1913), numerically solved by Szebehely & Peters
    (1967). The system has multiple close approaches, eventually ejecting
    body 0 (the lightest) while bodies 1 and 2 form a tight binary.

    Reference:
        Burrau, C. (1913). Numerische Berechnung eines Spezialfalles des
        Dreikoerperproblems. Astron. Nachr., 195, 113.
        Szebehely, V. & Peters, C. F. (1967). Complete solution of a
        general problem of three bodies. Astron. J., 72, 876-883.

    Returns
    -------
    positions : ndarray, shape (3, 3)
    velocities : ndarray, shape (3, 3)
    masses : ndarray, shape (3,)
    """
    # Standard Pythagorean setup: corners of a 3-4-5 right triangle.
    # Place the right angle at the origin, leg of length 4 along +x,
    # leg of length 3 along +y. The hypotenuse (length 5) connects them.
    # Body i sits at the corner opposite side i.
    positions = np.array([
        [ 1.0,  3.0, 0.0],   # body 0 (mass 3), opposite the side of length 3
        [-2.0, -1.0, 0.0],   # body 1 (mass 4), opposite the side of length 4
        [ 1.0, -1.0, 0.0],   # body 2 (mass 5), opposite the side of length 5
    ])
    velocities = np.zeros((3, 3))  # all bodies start at rest
    masses = np.array([3.0, 4.0, 5.0])
    return positions, velocities, masses

def integrate_rk45(initial_state, masses, t_span, dt_output=None,
                   softening=0.0, rtol=1e-8, atol=1e-10):
    """
    Integrate using adaptive RK4(5) Dormand-Prince via scipy.solve_ivp.

    Unlike RK4 and Verlet, RK45 uses *adaptive* timesteps internally — it
    chooses its own step size based on a local error estimate, taking tiny
    steps where physics is hard (close approaches) and large steps where
    physics is easy. The dt_output parameter only controls how densely the
    *output* is sampled; the internal stepper is independent.

    The local error is estimated by comparing 4th-order and 5th-order
    embedded solutions. Steps with estimated error above tolerance are
    rejected and retried with smaller dt.

    Parameters
    ----------
    initial_state : ndarray, shape (18,)
    masses : ndarray, shape (3,)
    t_span : tuple (t_start, t_end)
    dt_output : float or None
        Spacing between output time points. If None, uses scipy's internal
        choice (one output per accepted step).
    softening : float
        Softening parameter passed to the force calculation.
    rtol, atol : float
        Relative and absolute error tolerances. Smaller = more accurate but
        slower. Defaults give ~8 digit accuracy.

    Returns
    -------
    times : ndarray, shape (n_outputs,)
    states : ndarray, shape (n_outputs, 18)
    """
    t_start, t_end = t_span

    # scipy.solve_ivp wants a function f(t, y) — t comes first, even
    # though our state_derivative doesn't depend on t. We wrap to fit.
    def rhs(t, y):
        return state_derivative(y, masses, softening)

    # Output time points (if specified)
    if dt_output is not None:
        n_outputs = int(np.ceil((t_end - t_start) / dt_output)) + 1
        t_eval = np.linspace(t_start, t_end, n_outputs)
    else:
        t_eval = None

    result = solve_ivp(
        rhs,
        t_span=(t_start, t_end),
        y0=initial_state,
        method='RK45',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=False
    )

    if not result.success:
        raise RuntimeError(f"RK45 integration failed: {result.message}")

    # solve_ivp returns states shape (18, n_times) — we want (n_times, 18)
    return result.t, result.y.T

def recommend_softening(states, percentile_above=5.0, safety_factor=1.5):
    """
    Recommend a softening parameter ε based on close-approach statistics.

    Heuristic: ε is set to `safety_factor` times the `percentile_above`-th
    percentile of minimum pairwise distances over the trajectory. By default,
    ε is 1.5× the 5th percentile of close-approach distances — i.e., softening
    will engage on roughly the closest 5% of encounters.

    Lower `percentile_above` → softening engages on more encounters (more
    aggressive). Higher `safety_factor` → softening engages farther from
    bodies (more aggressive).

    Workflow:
        1. Run with an integrator that handles close approaches well (RK45)
           and no softening, to get a ground-truth trajectory.
        2. Pass the resulting states to this function.
        3. Use the returned ε with fixed-step integrators.

    Parameters
    ----------
    states : ndarray, shape (n_outputs, 18)
        State history from an un-softened ground-truth run.
    percentile_above : float
        Percentile of min-distances used as the cutoff scale. Default 5.0.
    safety_factor : float
        Multiplier above the chosen percentile. Default 1.5.

    Returns
    -------
    eps_recommended : float
    diagnostics : dict
    """
    n = states.shape[0]
    min_dists = np.zeros(n)

    for k in range(n):
        positions, _ = unpack(states[k])
        d01 = np.linalg.norm(positions[1] - positions[0])
        d02 = np.linalg.norm(positions[2] - positions[0])
        d12 = np.linalg.norm(positions[2] - positions[1])
        min_dists[k] = min(d01, d02, d12)

    cutoff = np.percentile(min_dists, percentile_above)
    eps = safety_factor * cutoff

    diagnostics = {
        "min_distance":                       np.min(min_dists),
        "mean_distance":                      np.mean(min_dists),
        "median_distance":                    np.median(min_dists),
        f"p{percentile_above}_distance":      cutoff,
        "safety_factor":                      safety_factor,
        "n_below_0.1":                        int(np.sum(min_dists < 0.1)),
        "n_below_0.01":                       int(np.sum(min_dists < 0.01)),
    }

    return eps, diagnostics

# ============================================================
# Collision detection and inelastic merging
# ============================================================

def detect_collision(positions, masses, radii):
    """
    Find the first colliding pair of bodies, if any.

    Two bodies are considered colliding when their centers are within the
    sum of their radii. Inactive bodies (mass = 0) are skipped.

    Parameters
    ----------
    positions : ndarray, shape (3, 3)
    masses : ndarray, shape (3,)
        Mass of zero indicates an inactive (ghost) body.
    radii : ndarray, shape (3,)

    Returns
    -------
    pair : tuple (i, j) or None
        Indices of the first colliding pair found, with i < j.
        None if no collision is detected.
    """
    for i in range(3):
        if masses[i] == 0:
            continue
        for j in range(i + 1, 3):
            if masses[j] == 0:
                continue
            distance = np.linalg.norm(positions[j] - positions[i])
            if distance < radii[i] + radii[j]:
                return (i, j)
    return None


def merge_bodies(positions, velocities, masses, radii, i, j):
    """
    Perform a perfectly inelastic merger of bodies i and j.

    Conservation:
        Mass:     m_new = m_i + m_j  (exact)
        Momentum: p_new = p_i + p_j  (exact)
        Position: r_new = (m_i*r_i + m_j*r_j) / m_new  (center of mass)
        Radius:   r_new = (r_i^3 + r_j^3)^(1/3)  (volume-preserving)

    Kinetic energy is NOT conserved — the relative KE between the two bodies
    is converted to internal energy (heat in reality, simply lost from our
    model). This is the perfectly-inelastic limit of collision physics.

    Convention: the merged body takes the index of the MORE MASSIVE parent.
    The less-massive parent becomes a "ghost" body with mass=0, parked far
    from the system so it does not interfere with diagnostics or plots.

    Parameters
    ----------
    positions, velocities : ndarray, shape (3, 3)
    masses, radii : ndarray, shape (3,)
    i, j : int
        Indices of the two bodies to merge (any order).

    Returns
    -------
    new_positions, new_velocities, new_masses, new_radii : ndarray
        Updated arrays. One slot now holds the merged body; the other slot
        is a ghost.
    """
    # Determine which slot keeps the merged body (the more massive parent).
    if masses[i] >= masses[j]:
        keep, drop = i, j
    else:
        keep, drop = j, i

    m_total = masses[i] + masses[j]
    com_pos = (masses[i] * positions[i] + masses[j] * positions[j]) / m_total
    com_vel = (masses[i] * velocities[i] + masses[j] * velocities[j]) / m_total
    new_radius = (radii[i]**3 + radii[j]**3) ** (1.0 / 3.0)

    new_positions  = positions.copy()
    new_velocities = velocities.copy()
    new_masses     = masses.copy()
    new_radii      = radii.copy()

    # Place merged body in `keep` slot
    new_positions[keep]  = com_pos
    new_velocities[keep] = com_vel
    new_masses[keep]     = m_total
    new_radii[keep]      = new_radius

    # Zero out the `drop` slot — park ghost far from the system at rest
    new_positions[drop]  = np.array([1e6, 1e6, 1e6])
    new_velocities[drop] = np.zeros(3)
    new_masses[drop]     = 0.0
    new_radii[drop]      = 0.0

    return new_positions, new_velocities, new_masses, new_radii


def integrate_with_collisions(initial_state, masses, radii, t_span, dt,
                                integrator="rk4", softening=0.0):
    """
    Integrate with collision detection. Stops the underlying integrator after
    each step to check for body-body contact; merges on collision.

    Parameters
    ----------
    initial_state : ndarray, shape (18,)
    masses : ndarray, shape (3,)
        Initial masses. Will be modified by mergers.
    radii : ndarray, shape (3,)
        Body radii for collision detection. Use radii = 0 to disable.
    t_span : tuple (t_start, t_end)
    dt : float
        Timestep.
    integrator : {'rk4', 'verlet'}
        Underlying step function. RK45 not supported here because adaptive
        stepping doesn't naturally integrate with per-step collision checks.
    softening : float
        Passed through to the force calculation.

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
    states : ndarray, shape (n_steps + 1, 18)
    masses_history : ndarray, shape (n_steps + 1, 3)
        Masses at each timestep (after any mergers).
    radii_history : ndarray, shape (n_steps + 1, 3)
        Radii at each timestep.
    collision_events : list of dict
        Each entry: {"time": t, "merged": (i, j), "kept": index, "ke_lost": float}
    """
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))

    times          = np.zeros(n_steps + 1)
    states         = np.zeros((n_steps + 1, 18))
    masses_history = np.zeros((n_steps + 1, 3))
    radii_history  = np.zeros((n_steps + 1, 3))

    times[0]            = t_start
    states[0]           = initial_state
    masses_history[0]   = masses.copy()
    radii_history[0]    = radii.copy()

    current_masses = masses.copy()
    current_radii  = radii.copy()

    step_fn = {"rk4": rk4_step, "verlet": verlet_step}[integrator]
    collision_events = []

    for k in range(n_steps):
        # Advance one step using the current masses
        states[k + 1] = step_fn(states[k], dt, current_masses, softening)
        times[k + 1]  = times[k] + dt

        # Check for collisions in the new state
        positions, velocities = unpack(states[k + 1])
        pair = detect_collision(positions, current_masses, current_radii)

        if pair is not None:
            i, j = pair
            # Track kinetic energy before merger (for diagnostic)
            ke_before = 0.5 * current_masses[i] * np.sum(velocities[i]**2) \
                      + 0.5 * current_masses[j] * np.sum(velocities[j]**2)

            new_pos, new_vel, new_masses, new_radii = merge_bodies(
                positions, velocities, current_masses, current_radii, i, j
            )

            # Determine which slot kept the merger (more massive parent)
            kept = i if current_masses[i] >= current_masses[j] else j
            ke_after = 0.5 * new_masses[kept] * np.sum(new_vel[kept]**2)

            collision_events.append({
                "time": times[k + 1],
                "merged": (i, j),
                "kept": kept,
                "ke_lost": ke_before - ke_after,
            })

            states[k + 1]  = pack(new_pos, new_vel)
            current_masses = new_masses
            current_radii  = new_radii

        masses_history[k + 1] = current_masses
        radii_history[k + 1]  = current_radii

    return times, states, masses_history, radii_history, collision_events