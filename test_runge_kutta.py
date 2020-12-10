from runge_kutta import *
import numpy as np
import matplotlib.pyplot as plt
import pdb

def test_rk4_scalar():
    def dydt(t, y):
        return np.cos(t)

    yi = 0.

    times, y_out = rk4(dydt, 0., 0., 10., step=10**(-4))

    plt.plot(times, y_out)
    plt.show()

    print(y_out[-1])
    print(np.sin(times[-1]))
    assert np.abs(times[-1] - 10.) < 10**(-12)
    assert np.abs(y_out[-1] - np.sin(times[-1])) < 10**(-2)


def test_rk4_vector():
    def dydt(t, y):
        A = np.array([[-0.5, 1.1],[0.9, -1.]])
        return np.dot(y, A)

    yi = np.array([1., 1.]).reshape(1, -1)
    print(yi)
    print(yi.shape)

    times, y_out = rk4(dydt, yi, 0., 10., step=10**(-4))

    plt.plot(times, y_out[:,0], times, y_out[:,1])
    plt.show()

    print(y_out[-1])
    assert np.abs(times[-1] - 10.) < 10**(-12)

