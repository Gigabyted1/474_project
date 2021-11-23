import numpy as np
import matplotlib.pyplot as plt
import time
import concurrent.futures as cf
from numba import njit, prange, jit

G_CONST = 6.67408e-11
SPEED_FACTOR = 1
G_EFF = G_CONST * SPEED_FACTOR
TIME_STEP = 1
NO_STEPS = 1200
run_parallel = False

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

# matrices storing the positions, masses, and velocities of the bodies
pos = np.array([sun['pos'], mercury['pos'], venus['pos'], earth['pos'],
                mars['pos']])
m = np.array([sun['m'], mercury['m'], venus['m'], earth['m'],
              mars['m']])
v = np.array([sun['v'], mercury['v'], venus['v'], earth['v'],
              mars['v']]).astype(float)

# figure
fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')


def calc_acc_np():
    # positions for each dimension
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrices to store distances between bodies on each dimension
    x_dist = x.T - x
    y_dist = y.T - y
    z_dist = z.T - z

    # using the L2 norm to store real distances
    r = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)

    # combining all r terms into one
    # using only values were r > 0, as the ones where r = 0 cause a division by 0
    r[r > 0] = r[r > 0] ** -3

    # calculating the acceleration
    ax = G_EFF * (x_dist * r) @ m
    ay = G_EFF * (y_dist * r) @ m
    az = G_EFF * (z_dist * r) @ m

    return np.matrix([ax, ay, az]).T


@njit(parallel=run_parallel)
def calc_acc_par():
    # positions for each dimension
    x = np.zeros((pos.shape[0]))
    for i in prange(pos.shape[0]):
        x[i] = pos[i, 0]

    y = np.zeros((pos.shape[0]))
    for i in prange(pos.shape[0]):
        y[i] = pos[i, 1]

    z = np.zeros((pos.shape[0]))
    for i in prange(pos.shape[0]):
        z[i] = pos[i, 2]

    # matrices to store distances between bodies on each dimension
    x_dist = np.zeros((x.shape[0], x.shape[0]))
    for i in prange(x.shape[0]):
        for j in prange(x.shape[0]):
            x_dist[j, i] = x[i] - x[j]

    y_dist = np.zeros((y.shape[0], y.shape[0]))
    for i in prange(y.shape[0]):
        for j in prange(y.shape[0]):
            y_dist[j, i] = y[i] - y[j]

    z_dist = np.zeros((z.shape[0], z.shape[0]))
    for i in prange(z.shape[0]):
        for j in prange(z.shape[0]):
            z_dist[j, i] = z[i] - z[j]

    # producing a factor to use for L2 normalized distances
    r = np.zeros(x_dist.shape)
    for i in prange(r.shape[0]):
        for j in prange(r.shape[1]):
            r_t = (x_dist[i, j] ** 2 + y_dist[i, j] ** 2 + z_dist[i, j] ** 2) ** 0.5
            if r_t > 0:
                r[i, j] = r_t ** -3

    # finding real Euclidean distances
    dx = np.zeros(r.shape)
    dy = np.zeros(r.shape)
    dz = np.zeros(r.shape)
    for i in prange(dx.shape[0]):
        for j in prange(dx.shape[1]):
            dx[i, j] += x_dist[i, j] * r[i, j]
            dy[i, j] += y_dist[i, j] * r[i, j]
            dz[i, j] += z_dist[i, j] * r[i, j]

    acc = np.zeros(pos.shape)
    for i in prange(dx.shape[0]):
        for j in prange(dx.shape[1]):
            acc[i, 0] += dx[i, j] * m[j]
            acc[i, 1] += dy[i, j] * m[j]
            acc[i, 2] += dz[i, j] * m[j]
        for j in prange(pos.shape[1]):
            acc[i, j] *= G_EFF

    return acc


def calc_acc_ser():
    # positions for each dimension
    x = np.zeros((pos.shape[0]))
    for i in range(pos.shape[0]):
        x[i] = pos[i, 0]

    y = np.zeros((pos.shape[0]))
    for i in range(pos.shape[0]):
        y[i] = pos[i, 1]

    z = np.zeros((pos.shape[0]))
    for i in range(pos.shape[0]):
        z[i] = pos[i, 2]

    # matrices to store distances between bodies on each dimension
    x_dist = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            x_dist[j, i] = x[i] - x[j]

    y_dist = np.zeros((y.shape[0], y.shape[0]))
    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            y_dist[j, i] = y[i] - y[j]

    z_dist = np.zeros((z.shape[0], z.shape[0]))
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            z_dist[j, i] = z[i] - z[j]

    # producing a factor to use for L2 normalized distances
    r = np.zeros(x_dist.shape)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r_t = (x_dist[i, j] ** 2 + y_dist[i, j] ** 2 + z_dist[i, j] ** 2) ** 0.5
            if r_t > 0:
                r[i, j] = r_t ** -3

    # finding real Euclidean distances
    dx = np.zeros(r.shape)
    dy = np.zeros(r.shape)
    dz = np.zeros(r.shape)
    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            dx[i, j] += x_dist[i, j] * r[i, j]
            dy[i, j] += y_dist[i, j] * r[i, j]
            dz[i, j] += z_dist[i, j] * r[i, j]

    acc = np.zeros(pos.shape)
    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            acc[i, 0] += dx[i, j] * m[j]
            acc[i, 1] += dy[i, j] * m[j]
            acc[i, 2] += dz[i, j] * m[j]
        for j in range(pos.shape[1]):
            acc[i, j] *= G_EFF

    return acc


def plot_iteration(iteration):
    plt.title('Iteration: {}'.format(iteration))

    axis.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=5, color='red')

    max_dist = np.max(abs(pos))
    axis.set(xlim=(-max_dist, max_dist), ylim=(-max_dist, max_dist), zlim=(-max_dist, max_dist))


def run_sim():
    global v
    global pos

    print('Simulation start, parallel = {}'.format(run_parallel))
    time_start = time.time()

    for step in range(0, NO_STEPS):
        plt.cla()
        # a = calc_acc_ser()
        a = calc_acc_par()
        v += a * TIME_STEP
        pos += v * TIME_STEP

        plot_iteration(step)

    time_end = time.time()
    print('Simulation end, {} elapsed'.format(time_end - time_start, 2))


if __name__ == '__main__':
    test = calc_acc_par()

    # run_sim()
    # run_parallel = True
    # run_sim()

    print('Simulation start, serial')
    ts = time.time()

    for step in range(0, NO_STEPS):
        plt.cla()
        a = calc_acc_ser()
        v += a * TIME_STEP
        pos += v * TIME_STEP

        plot_iteration(step)

    te = time.time()
    print('Simulation end, {} elapsed'.format(te - ts, 2))

    print('Simulation start, parallel')
    ts = time.time()

    for step in range(0, NO_STEPS):
        plt.cla()
        a = calc_acc_par()
        v += a * TIME_STEP
        pos += v * TIME_STEP

        plot_iteration(step)

    te = time.time()
    print('Simulation end, {} elapsed'.format(te - ts, 2))

    print('Simulation start, numpy')
    ts = time.time()

    for step in range(0, NO_STEPS):
        plt.cla()
        a = calc_acc_np()
        v += a * TIME_STEP
        pos += v * TIME_STEP

        plot_iteration(step)

    te = time.time()
    print('Simulation end, {} elapsed'.format(te - ts, 2))
