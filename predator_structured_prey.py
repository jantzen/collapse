# file: predator_structured_prey.py

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from runge_kutta import rk4
from scipy.stats import variation 

def integrand(t, X, args):
    J, A, P = X.flatten().tolist()

    mu_J0, mu_A0, mu_P0 = args
#    mu_P0 = 0.553

    def f(x):
        b = 1.
        return b * x

    def g(x):
        try:
            return x / (1. + x**2)
        except:
            print(x)
            raise Exception('g(x) caused an exception')
    
    def h(x, y):
        return x * y

    c = 1.

    mu_J = mu_J0 + np.random.normal(loc=0., scale=0.005)
    mu_A = mu_A0 + np.random.normal(loc=0., scale=0.005)
    mu_P = mu_P0 + np.random.normal(loc=0., scale=0.005)

    dJ = f(A) - g(J) - mu_J * J
    dA = g(J) - h(A,P) - mu_A * A
    dP = h(A,P) * c - mu_P * P

    dXdt = np.array([dJ, dA, dP]).reshape(1,-1)

    return dXdt


#ic = np.array([0.01, 0.5, 0.5]).reshape(1,-1)
ic = np.random.rand(1,3)

mu_J = 0.05
mu_A = 0.1
mu_P = 0.4

t, y = rk4(integrand, ic, 0., 6. * 10**4, args=(mu_J, mu_A, mu_P), step=1./(6.*10**4))

cov = variation(y[10000:,:], axis=0)
print("Coefficients of variation: {}".format(cov))

plt.plot(t, y[:,0], t, y[:,1], t, y[:,2])
plt.legend(('J','A','P'))
plt.title('mu_P = 0.4')

ic = np.random.rand(1,3)

mu_J = 0.05
mu_A = 0.1
mu_P = 0.553

t, y = rk4(integrand, ic, 0., 6. * 10**4, args=(mu_J, mu_A, mu_P), step=1./(6.*10**4))

cov = variation(y[10000:,:], axis=0)
print("Coefficients of variation: {}".format(cov))

plt.figure()
plt.plot(t, y[:,0], t, y[:,1], t, y[:,2])
plt.legend(('J','A','P'))
plt.title('mu_P = 0.553')

cov = []
mu = []
for mu_P in np.arange(0.4, 0.555, 0.001):
    ic = np.random.rand(1,3)

    mu_J = 0.05
    mu_A = 0.1

    t, y = rk4(integrand, ic, 0., 6. * 10**4, args=(mu_J, mu_A, mu_P), step=1./(6.*10**4))

    cov.append(variation(y[10000:,:], axis=0).reshape(1,-1))
    mu.append(mu_P)
    print("progress (%): {:.3}".format((mu_P - 0.4)/(0.555 - 0.4) * 100), end='\r')
cov = np.concatenate(cov, axis=0)
mu = np.array(mu).reshape(-1, 1)

plt.figure()
plt.plot(mu, cov[:,0], mu, cov[:,1], mu, cov[:,2])
plt.legend(('J','A','P'))
plt.title('Coeffiecients of varitation vs. mu_P')

plt.show()
