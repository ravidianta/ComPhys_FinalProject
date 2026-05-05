import numpy as np
import copy
from vpython import *

from physics import compute_energy
from integrators import euler, verlet, rk4
from config import positions, velocities, masses

# =========================
# SINGLE CANVAS (IMPORTANT)
# =========================
scene = canvas(title="3 Body Problem - Euler vs Verlet vs RK4",
                width=1000, height=600, background=color.black)

# =========================
# OFFSETS (SIDE-BY-SIDE EFFECT)
# =========================
offset_euler = vector(-3, 0, 0)
offset_verlet = vector(0, 0, 0)
offset_rk4 = vector(3, 0, 0)
# Vertical dividers
curve(pos=[vector(-1.5, -5, 0), vector(-1.5, 5, 0)], color=color.white)
curve(pos=[vector(1.5, -5, 0), vector(1.5, 5, 0)], color=color.white)
# =========================
# STATES
# =========================
state_euler = {"p": positions.copy(), "v": velocities.copy()}
state_verlet = copy.deepcopy(state_euler)
state_rk4 = copy.deepcopy(state_euler)

# =========================
# COLORS
# =========================
cols = [color.yellow, color.cyan, color.magenta]

# =========================
# VISUAL OBJECTS
# =========================
vis = {"euler": [], "verlet": [], "rk4": []}

for method, state in zip(["euler", "verlet", "rk4"],
                          [state_euler, state_verlet, state_rk4]):

    off = {"euler": offset_euler,
           "verlet": offset_verlet,
           "rk4": offset_rk4}[method]

    for i in range(3):
        s = sphere(pos=vector(*state["p"][i]) + off,
                   radius=0.08,
                   color=cols[i],
                   make_trail=True,
                   retain=1500)
        vis[method].append(s)

# =========================
# GRAPH
# =========================
g = graph(title="Energy Error (%)", xtitle="Time", ytitle="Error %")
c1 = gcurve(color=color.red)
c2 = gcurve(color=color.green)
c3 = gcurve(color=color.blue)

E0 = compute_energy(positions, velocities, masses)

# =========================
# SIMULATION
# =========================
dt = 0.0005

for step in range(5000):

    rate(200)

    euler(state_euler, masses, dt)
    verlet(state_verlet, masses, dt)
    rk4(state_rk4, masses, dt)

    # update visuals with offsets
    for i in range(3):
        vis["euler"][i].pos = vector(*state_euler["p"][i]) + offset_euler
        vis["verlet"][i].pos = vector(*state_verlet["p"][i]) + offset_verlet
        vis["rk4"][i].pos = vector(*state_rk4["p"][i]) + offset_rk4

    t = step * dt

    Ee = compute_energy(state_euler["p"], state_euler["v"], masses)
    Ev = compute_energy(state_verlet["p"], state_verlet["v"], masses)
    Er = compute_energy(state_rk4["p"], state_rk4["v"], masses)

    e1 = abs((Ee - E0) / E0) * 100
    e2 = abs((Ev - E0) / E0) * 100
    e3 = abs((Er - E0) / E0) * 100

    if step % 5 == 0:
        c1.plot(t, e1)
        c2.plot(t, e2)
        c3.plot(t, e3)

label(pos=vector(-3, 3, 0), text="EULER", height=16, box=False, color=color.red)
label(pos=vector(0, 3, 0), text="VERLET", height=16, box=False, color=color.green)
label(pos=vector(3, 3, 0), text="RK4", height=16, box=False, color=color.blue)