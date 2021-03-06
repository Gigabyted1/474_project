import sys

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

G_CONST = 6.67408e-11
SPEED_FACTOR = 1
G_EFF = G_CONST * SPEED_FACTOR
TIME_STEP = 1000
NO_STEPS = 15000
VISUALIZE = False

# data for our solar system
sun = {"pos": (0, 0, 0), "m": 2e30, "v": (0, 0, 0)}
mercury = {"pos": (0, 5.7e10, 0), "m": 3.285e23, "v": (47000, 0, 0)}
venus = {"pos": (0, 1.1e11, 0), "m": 4.8e24, "v": (35000, 0, 0)}
earth = {"pos": (0, 1.5e11, 0), "m": 6e24, "v": (30000, 0, 0)}
mars = {"pos": (0, 2.2e11, 0), "m": 2.4e24, "v": (24000, 0, 0)}
jupiter = {"pos": (0, 7.7e11, 0), "m": 1e28, "v": (13000, 0, 0)}
saturn = {"pos": (0, 1.4e12, 0), "m": 5.7e26, "v": (9000, 0, 0)}
uranus = {"pos": (0, 2.8e12, 0), "m": 8.7e25, "v": (6835, 0, 0)}
neptune = {"pos": (0, 4.5e12, 0), "m": 1e26, "v": (5477, 0, 0)}
pluto = {"pos": (0, 3.7e12, 0), "m": 1.3e22, "v": (4748, 0, 0)}

# planet presets
terrestrial_planets = [mercury, venus, earth, mars]
jovian_planets = [jupiter, saturn, uranus, neptune]
all_planets = terrestrial_planets + jovian_planets

# change this to change the utilized bodies
BODIES = [sun] + all_planets

# load body data
positions = np.zeros((len(BODIES), 3))
velocities = np.zeros((len(BODIES), 3))
masses = np.zeros(len(BODIES))
for i in range(len(BODIES)):
    positions[i] = BODIES[i]["pos"]
    velocities[i] = BODIES[i]["v"]
    masses[i] = BODIES[i]["m"]

# figure
fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')


# functions used to parallelize
@njit(parallel=True, fastmath=True)
def calc_dist_jit(x):
    x_dist = np.zeros((x.shape[0], x.shape[0]))
    for i in prange(x.shape[0]):
        for j in prange(x.shape[0]):
            x_dist[j, i] = x[i] - x[j]
    return x_dist.astype(np.float64)


def calc_dist(x):
    x_dist = np.zeros((x.shape[0], x.shape[0]))
    for i in prange(x.shape[0]):
        for j in prange(x.shape[0]):
            x_dist[j, i] = x[i] - x[j]
    return x_dist.astype(np.float64)


@njit(parallel=True, fastmath=True)
def calc_r_jit(x_dist, y_dist, z_dist):
    r = np.zeros(x_dist.shape)
    for i in prange(r.shape[0]):
        for j in prange(r.shape[1]):
            r_t = (x_dist[i, j] ** 2 + y_dist[i, j] ** 2 + z_dist[i, j] ** 2) ** 0.5
            if r_t > 0:
                r[i, j] = r_t ** -3
    return r.astype(np.float64)


def calc_r(x_dist, y_dist, z_dist):
    r = np.zeros(x_dist.shape)
    for i in prange(r.shape[0]):
        for j in prange(r.shape[1]):
            r_t = (x_dist[i, j] ** 2 + y_dist[i, j] ** 2 + z_dist[i, j] ** 2) ** 0.5
            if r_t > 0:
                r[i, j] = r_t ** -3
    return r.astype(np.float64)


@njit(parallel=True, fastmath=True)
def calc_real_dist_jit(r, x_dist):
    dx = np.zeros(r.shape)
    for i in prange(dx.shape[0]):
        for j in prange(dx.shape[1]):
            dx[i, j] += x_dist[i, j] * r[i, j]
    return dx.astype(np.float64)


def calc_real_dist(r, x_dist):
    dx = np.zeros(r.shape)
    for i in prange(dx.shape[0]):
        for j in prange(dx.shape[1]):
            dx[i, j] += x_dist[i, j] * r[i, j]
    return dx.astype(np.float64)


@njit(parallel=True, fastmath=True)
def find_acc_jit(dx, dy, dz, m, pos):
    acceleration = np.zeros(pos.shape)
    for i in prange(dx.shape[0]):
        for j in prange(dx.shape[1]):
            acceleration[i, 0] += dx[i, j] * m[j]
            acceleration[i, 1] += dy[i, j] * m[j]
            acceleration[i, 2] += dz[i, j] * m[j]
        for j in prange(pos.shape[1]):
            acceleration[i, j] *= G_EFF
    return acceleration


def find_acc(dx, dy, dz, m, pos):
    acceleration = np.zeros(pos.shape)
    for i in prange(dx.shape[0]):
        for j in prange(dx.shape[1]):
            acceleration[i, 0] += dx[i, j] * m[j]
            acceleration[i, 1] += dy[i, j] * m[j]
            acceleration[i, 2] += dz[i, j] * m[j]
        for j in prange(pos.shape[1]):
            acceleration[i, j] *= G_EFF
    return acceleration


@njit(parallel=True, fastmath=True)
def time_step_jit(pos, vel, acc, ts=TIME_STEP):
    vel += acc * ts
    pos += vel * ts


def time_step(pos, vel, acc, ts=TIME_STEP):
    vel += acc * ts
    pos += vel * ts


def plot_iteration(iteration, pos):
    plt.cla()
    plt.title('Iteration: {}'.format(iteration))
    max_dist = np.max(abs(pos))
    axis.set(xlim=(-max_dist, max_dist), ylim=(-max_dist, max_dist), zlim=(-max_dist, max_dist), xlabel='x', ylabel='y',
             zlabel='z')

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    c = np.arange(len(x))  # to create distinct colors for each body

    axis.scatter(x, y, z, s=5, cmap='tab10', c=c)
    plt.pause(0.0001)


def run_sim_ser(pos, m, v):
    t = 0
    for i in range(NO_STEPS):
        time_start = time.time()

        x_test = positions[:, 0]
        y_test = positions[:, 1]
        z_test = positions[:, 2]

        x_dist_test = calc_dist(x_test)
        y_dist_test = calc_dist(y_test)
        z_dist_test = calc_dist(z_test)

        r_test = calc_r(x_dist_test, y_dist_test, z_dist_test)

        dx_test = calc_real_dist(r_test, x_dist_test)
        dy_test = calc_real_dist(r_test, y_dist_test)
        dz_test = calc_real_dist(r_test, z_dist_test)

        acc_test = find_acc(dx_test, dy_test, dz_test, m, pos)

        time_step(pos, v, acc_test)

        time_end = time.time()
        t += time_end - time_start
    print('[SER] Average runtime per iter: {:.2e}s'.format(t / NO_STEPS))
    if t > .01:
        print('[SER] Total runtime: {:.2f}s'.format(t))
    else:
        print('[SER] Total runtime: {:.2e}s'.format(t))
    return t


def run_sim_par(pos, m, v):
    t = 0
    numba_t = 0
    for i in range(1000):
        time_start = time.time()

        x_test = positions[:, 0]
        y_test = positions[:, 1]
        z_test = positions[:, 2]

        x_dist_test = calc_dist_jit(x_test)
        y_dist_test = calc_dist_jit(y_test)
        z_dist_test = calc_dist_jit(z_test)

        r_test = calc_r_jit(x_dist_test, y_dist_test, z_dist_test)

        dx_test = calc_real_dist_jit(r_test, x_dist_test)
        dy_test = calc_real_dist_jit(r_test, y_dist_test)
        dz_test = calc_real_dist_jit(r_test, z_dist_test)

        acc_test = find_acc_jit(dx_test, dy_test, dz_test, m, pos)

        time_step_jit(pos, v, acc_test)

        time_end = time.time()
        t += time_end - time_start
        if i == 0:
            numba_t = t
    print('[PAR] Average runtime per iter: {:.2e}s'.format(t / NO_STEPS))
    if t > .01:
        print('[PAR] Total runtime: {:.2f}s'.format(t))
    else:
        print('[PAR] Total runtime: {:.2e}s'.format(t))
    print('[PAR] Time spent on first iter: {:.2f}s.'.format(numba_t))
    print('[PAR] Average runtime per iter, minus first iter: {:.2e}s'.format((t - numba_t) / (NO_STEPS - 1)))
    return t


def run_sim(pos, m, v):
    for i in range(sys.maxsize):
        x_test = positions[:, 0]
        y_test = positions[:, 1]
        z_test = positions[:, 2]

        x_dist_test = calc_dist_jit(x_test)
        y_dist_test = calc_dist_jit(y_test)
        z_dist_test = calc_dist_jit(z_test)

        r_test = calc_r_jit(x_dist_test, y_dist_test, z_dist_test)

        dx_test = calc_real_dist_jit(r_test, x_dist_test)
        dy_test = calc_real_dist_jit(r_test, y_dist_test)
        dz_test = calc_real_dist_jit(r_test, z_dist_test)

        acc_test = find_acc_jit(dx_test, dy_test, dz_test, m, pos)

        time_step_jit(pos, v, acc_test)

        plot_iteration(i, pos)


if __name__ == '__main__':
    # run sim with visualization
    if VISUALIZE:
        run_sim(positions, masses, velocities)
    # run sim in serial, then parallel, then compare
    else:
        print('Running sim with {} iterations'.format(NO_STEPS))
        time_ser = run_sim_ser(positions, masses, velocities)
        time_par = run_sim_par(positions, masses, velocities)
        print('Parallel speedup: {:.3f}'.format(time_ser / time_par))
