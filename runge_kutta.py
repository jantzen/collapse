# file: runge_kutta.py
import numpy as np

def rk4(f, yi, ti, tf, args=None, step=10**(-3)):
    """ Implements 4th-order Runge-Kutta ODE solver with fixed step size.

    f : callable function returning the derivative at t as a function of t and y
    yi : initial value
    ti : initial time
    tf : upper bound of the integration range
    step : fraction of (xf - xi) to step at every stage of integration
    args: optional arguments to pass to the inegrand, f
    """
    # if multidimensional, make yi a row-vector
    if type(yi) == np.ndarray:
        yi.reshape(1, -1)
    elif type(yi) == float:
        yi = np.array(yi).reshape(1, -1)
    t = ti
    y = yi
    y_out = [yi]
    times = [t]
    h = (tf - ti) * step
    if args == None:
        for ii in range(int(1 / step)):
            t = t + h
            k1 = h * f(t, y)
            k2 = h * f(t + h/2., y + k1/2.)
            k3 = h * f(t + h/2., y + k2/2.)
            k4 = h * f(t + h, y + k3)
            y = y + k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.
            times.append(t)
            y_out.append(y)
    else:
        for ii in range(int(1 / step)):
            t = t + h
            k1 = h * f(t, y, args)
            k2 = h * f(t + h/2., y + k1/2., args)
            k3 = h * f(t + h/2., y + k2/2., args)
            k4 = h * f(t + h, y + k3, args)
            y = y + k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.
            times.append(t)
            y_out.append(y)
    y_out = np.concatenate(y_out, axis=0)
    times = np.array(times)

    return times, y_out

