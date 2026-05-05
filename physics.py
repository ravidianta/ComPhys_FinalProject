import numpy as np

def compute_acceleration(p, m, G=1.0):
    n = len(m)
    a = np.zeros_like(p)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = p[j] - p[i]
            dist = np.linalg.norm(r) + 1e-4
            a[i] += G * m[j] * r / dist**3

    return a


def compute_energy(p, v, m, G=1.0):
    kinetic = sum(0.5 * m[i] * np.linalg.norm(v[i])**2 for i in range(len(m)))

    potential = 0
    for i in range(len(m)):
        for j in range(i+1, len(m)):
            r = np.linalg.norm(p[i] - p[j]) + 1e-4
            potential -= G * m[i] * m[j] / r

    return kinetic + potential