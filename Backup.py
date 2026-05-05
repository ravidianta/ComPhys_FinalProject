import numpy as np
import copy
from vpython import *

# =========================
# DUMMY INPUT HANDLER
# =========================
def dummy(evt):
    pass

# =========================
# PHYSICS
# =========================
def compute_acceleration(p, m, G=1.0):
    n = len(m)
    acc = np.zeros_like(p)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = p[j] - p[i]
            dist = np.linalg.norm(r) + 1e-4
            acc[i] += G * m[j] * r / dist**3

    return acc


def compute_energy(p, v, m, G=1.0):
    kinetic = sum(0.5 * m[i] * np.linalg.norm(v[i])**2 for i in range(len(m)))

    potential = 0
    for i in range(len(m)):
        for j in range(i+1, len(m)):
            r = np.linalg.norm(p[i] - p[j]) + 1e-4
            potential -= G * m[i] * m[j] / r

    return kinetic + potential


# =========================
# INTEGRATORS
# =========================
def euler_step(state, m, dt):
    p, v = state["positions"], state["velocities"]
    a = compute_acceleration(p, m)
    state["positions"] = p + v * dt
    state["velocities"] = v + a * dt


def verlet_step(state, m, dt):
    p, v = state["positions"], state["velocities"]
    a = compute_acceleration(p, m)

    v_half = v + 0.5 * a * dt
    new_p = p + v_half * dt

    new_a = compute_acceleration(new_p, m)
    new_v = v_half + 0.5 * new_a * dt

    state["positions"], state["velocities"] = new_p, new_v


def rk4_step(state, m, dt):
    p, v = state["positions"], state["velocities"]

    def acc(x): return compute_acceleration(x, m)

    k1_v, k1_p = acc(p), v
    k2_v, k2_p = acc(p + 0.5*k1_p*dt), v + 0.5*k1_v*dt
    k3_v, k3_p = acc(p + 0.5*k2_p*dt), v + 0.5*k2_v*dt
    k4_v, k4_p = acc(p + k3_p*dt), v + k3_v*dt

    state["positions"] = p + (dt/6)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
    state["velocities"] = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)


# =========================
# DEFAULT FIGURE-8
# =========================
positions = np.array([
    [-0.97000436,  0.24308753, 0],
    [ 0.97000436, -0.24308753, 0],
    [ 0,           0,          0]
])

velocities = np.array([
    [ 0.466203685,  0.43236573, 0],
    [ 0.466203685,  0.43236573, 0],
    [-0.93240737,  -0.86473146, 0]
])

masses = np.array([1.0, 1.0, 1.0])

state_euler = {"positions": positions.copy(), "velocities": velocities.copy()}
state_verlet = copy.deepcopy(state_euler)
state_rk4 = copy.deepcopy(state_euler)

# =========================
# SCENES
# =========================
scene_euler = canvas(title="Euler", width=400, height=300)
scene_verlet = canvas(title="Verlet", width=400, height=300)
scene_rk4 = canvas(title="RK4", width=400, height=300)

# =========================
# VISUALS
# =========================
body_colors = [color.yellow, color.cyan, color.magenta]
visuals = {"euler": [], "verlet": [], "rk4": []}

for method, state, scene in zip(
    ["euler", "verlet", "rk4"],
    [state_euler, state_verlet, state_rk4],
    [scene_euler, scene_verlet, scene_rk4]
):
    scene.select()
    for i in range(3):
        s = sphere(
            pos=vector(*state["positions"][i]),
            radius=0.08,
            color=body_colors[i],
            make_trail=True,
            trail_color=body_colors[i],
            retain=2000
        )
        visuals[method].append(s)

# =========================
# GRAPH (TIME-BASED)
# =========================
g = graph(title="Energy Error (%)", xtitle="Time", ytitle="Error %")

curve_euler = gcurve(color=color.red)
curve_verlet = gcurve(color=color.green)
curve_rk4 = gcurve(color=color.blue)

# =========================
# INPUT UI
# =========================
scene_euler.append_to_caption("\n\n===== INPUT PANEL =====\n")

mass_inputs, pos_inputs, vel_inputs = [], [], []

for i in range(3):
    scene_euler.append_to_caption(f"\n--- Body {i} ---\n")

    scene_euler.append_to_caption("Mass: ")
    mass_inputs.append(winput(text="1.0", bind=dummy))

    scene_euler.append_to_caption("\nPos (x y z): ")
    pos_inputs.append([
        winput(text=str(positions[i][0]), bind=dummy),
        winput(text=str(positions[i][1]), bind=dummy),
        winput(text="0", bind=dummy)
    ])

    scene_euler.append_to_caption("\nVel (x y z): ")
    vel_inputs.append([
        winput(text=str(velocities[i][0]), bind=dummy),
        winput(text=str(velocities[i][1]), bind=dummy),
        winput(text="0", bind=dummy)
    ])

    scene_euler.append_to_caption("\n")


def read_inputs():
    m = np.array([float(x.text) for x in mass_inputs])
    p = np.array([[float(x.text) for x in row] for row in pos_inputs])
    v = np.array([[float(x.text) for x in row] for row in vel_inputs])
    return p, v, m


def reset_sim():
    global state_euler, state_verlet, state_rk4, masses
    global E0_euler, E0_verlet, E0_rk4

    p, v, m = read_inputs()
    masses = m

    state_euler = {"positions": p.copy(), "velocities": v.copy()}
    state_verlet = copy.deepcopy(state_euler)
    state_rk4 = copy.deepcopy(state_euler)

    for method in visuals:
        for i in range(3):
            visuals[method][i].pos = vector(*p[i])
            visuals[method][i].clear_trail()

    E0_euler = compute_energy(p, v, m)
    E0_verlet = E0_euler
    E0_rk4 = E0_euler

button(text="RESET", bind=lambda _: reset_sim())

# =========================
# INITIAL ENERGY
# =========================
E0_euler = compute_energy(state_euler["positions"], state_euler["velocities"], masses)
E0_verlet = E0_euler
E0_rk4 = E0_euler

# =========================
# MAIN LOOP
# =========================
dt = 0.0005

for step in range(5000):

    rate(200)

    euler_step(state_euler, masses, dt)
    verlet_step(state_verlet, masses, dt)
    rk4_step(state_rk4, masses, dt)

    for method, state in zip(
        ["euler", "verlet", "rk4"],
        [state_euler, state_verlet, state_rk4]
    ):
        for i in range(3):
            visuals[method][i].pos = vector(*state["positions"][i])

    # TIME INSTEAD OF STEP
    t = step * dt

    # ENERGY
    E_e = compute_energy(state_euler["positions"], state_euler["velocities"], masses)
    E_v = compute_energy(state_verlet["positions"], state_verlet["velocities"], masses)
    E_r = compute_energy(state_rk4["positions"], state_rk4["velocities"], masses)

    err_e = abs((E_e - E0_euler) / E0_euler) * 100
    err_v = abs((E_v - E0_verlet) / E0_verlet) * 100
    err_r = abs((E_r - E0_rk4) / E0_rk4) * 100

    if step % 5 == 0:
        curve_euler.plot(t, err_e)
        curve_verlet.plot(t, err_v)
        curve_rk4.plot(t, err_r)