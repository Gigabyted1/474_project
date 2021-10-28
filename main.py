import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multiprocessing import Pool

G = 6.67408e-11
ITER = 200


class Particle:
    def __init__(self, pos, m, v, a):
        self.pos = pos
        self.m = m
        self.v = v
        self.a = a


def calc_force(p1, p2):
    d = np.subtract(p2.pos, p1.pos)
    return ((G * p1.m * p2.m) / (abs(d) ** 2)) * (d / abs(d))


def calc_acceleration(p1, p2):
    return calc_force(p1, p2) / p1.m


def plt_test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-1000, 1000)

    particles = []
    for i in range(0, 2):
        x = np.random.randint(-10, 10)
        y = np.random.randint(-10, 10)
        z = np.random.randint(-10, 10)
        vx = np.random.randint(-1000, 1000) / 1000
        vy = np.random.randint(-1000, 1000) / 1000
        vz = np.random.randint(-1000, 1000) / 1000
        if i == 0:
            particles.append(Particle([x, y, z], 99999999, [vx, vy, vz], [0, 0, 0]))
        else:
            particles.append(Particle([x, y, z], 99, [vx, vy, vz], [0, 0, 0]))

    for i in range(0, ITER):
        for j in particles:
            for k in particles:
                if j != k:
                    j.a += calc_acceleration(j, k)
            j.v += j.a
            j.pos += j.v

        for j in particles:
            ax.scatter(j.pos[0], j.pos[1], j.pos[2])

        plt.pause(0.001)


if __name__ == '__main__':
    plt_test()
