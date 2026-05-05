from physics import compute_acceleration

def euler(state, m, dt):
    p, v = state["p"], state["v"]
    a = compute_acceleration(p, m)

    state["p"] = p + v * dt
    state["v"] = v + a * dt


def verlet(state, m, dt):
    p, v = state["p"], state["v"]
    a = compute_acceleration(p, m)

    v_half = v + 0.5 * a * dt
    new_p = p + v_half * dt

    new_a = compute_acceleration(new_p, m)
    new_v = v_half + 0.5 * new_a * dt

    state["p"], state["v"] = new_p, new_v


def rk4(state, m, dt):
    import numpy as np
    from physics import compute_acceleration

    p, v = state["p"], state["v"]

    k1_v = compute_acceleration(p, m)
    k1_p = v

    k2_v = compute_acceleration(p + 0.5*k1_p*dt, m)
    k2_p = v + 0.5*k1_v*dt

    k3_v = compute_acceleration(p + 0.5*k2_p*dt, m)
    k3_p = v + 0.5*k2_v*dt

    k4_v = compute_acceleration(p + k3_p*dt, m)
    k4_p = v + k3_v*dt

    state["p"] = p + (dt/6)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
    state["v"] = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)