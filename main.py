import numpy as np
import matplotlib.pyplot as plt
import time
import concurrent.futures as cf
from numba import njit, prange

G_CONST = 6.67408e-11
SPEED_FACTOR = 1
G_EFF = G_CONST * SPEED_FACTOR
# G = .01
TIME_STEP = 1000000
NO_STEPS = 1000

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

# matrices storing the positions, masses, velocities, and accelerations of the bodies
pos = np.array([sun['pos'], mercury['pos'], venus['pos'], earth['pos'],
                mars['pos']])
m = np.array([sun['m'], mercury['m'], venus['m'], earth['m'],
              mars['m']])
v = np.array([sun['v'], mercury['v'], venus['v'], earth['v'],
              mars['v']])

# figure
fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')


def calc_acc_ser():
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


@njit(parallel=True)
def calc_r(x_dist, y_dist, z_dist):
    r = np.zeros(x_dist.shape)
    for i in prange(x_dist.shape[0] - 1):
        for j in prange(x_dist.shape[1] - 1):
            r[i, j] = (x_dist[i, j] ** 2 + y_dist[i, j] ** 2 + z_dist[i, j] ** 2) ** 0.5

    return r


def calc_acc_par():
    # positions for each dimension
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrices to store distances between bodies on each dimension
    x_dist = x.T - x
    y_dist = y.T - y
    z_dist = z.T - z

    r = calc_r(x_dist, y_dist, z_dist)
    # r = np.zeros(x_dist.shape)
    #
    # print(r)
    #
    # for i in prange(x_dist.shape[0] - 1):
    #     for j in prange(x_dist.shape[1] - 1):
    #         r[i, j] = (x_dist[i, j] ** 2 + y_dist[i, j] ** 2 + z_dist[i, j] ** 2) ** 0.5
    #
    # print(r)

    r[r > 0] = r[r > 0] ** -3

    # calculating the acceleration
    ax = G_EFF * (x_dist * r) @ m
    ay = G_EFF * (y_dist * r) @ m
    az = G_EFF * (z_dist * r) @ m

    return np.matrix([ax, ay, az]).T


def plot_iteration(iteration):
    plt.title('Iteration: {}'.format(iteration))

    axis.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=5, color='red')

    max_dist = np.max(abs(pos))
    axis.set(xlim=(-max_dist, max_dist), ylim=(-max_dist, max_dist), zlim=(-max_dist, max_dist))


if __name__ == '__main__':
    v = v.astype(float)
    time_start = time.time()
    for i in range(0, NO_STEPS):
        plt.cla()
        a = calc_acc_ser()
        # a = calc_acc_par()
        v += a * TIME_STEP
        pos += v * TIME_STEP

        plot_iteration(i)
        plt.pause(0.001)

    time_end = time.time()
    print('Time taken: {}'.format(time_end - time_start, 2))
