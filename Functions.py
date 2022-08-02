import matplotlib.pyplot as plt
import numpy as np
from scipy.special import roots_chebyt

from Classes import DressingFunctions, Kernel

def bisection (f, a, b, maxit=100, eps=1e-5):
    a_n = a
    b_n = b

    for n in range(maxit):
        m_n = (a_n + b_n)/2
        f_a_n, f_m_n, f_b_n = f(a_n), f(m_n), f(b_n)

        if f_m_n == 0:
            print("Found exact solution.")
            return m_n

        elif f_a_n*f_m_n < 0:
            b_n = m_n

        else:
            a_n = m_n

        # If desired precision is reached
        if np.abs(a_n-b_n) < eps:
            break

    print(f'Bisection iterations: {n}')

    return (a_n + b_n)/2

def power_method (M, v0=None, maxit=1000, tol=1e-10, res=1e-10):

    if v0 is None:
        d = np.random.rand(M.shape[1],1)
    else:
        d = v0

    d_new = M@d

    for i in range(maxit):
        # To avoid dividing by tiny numbers we strip away everything below some
        # threshold tol. Afterwards we calculate an approximation for the
        # eigenvalue
        L = d > tol
        lambda_1 = np.mean(d_new[L]/d[L])

        # Normalize vector
        d = d_new/np.linalg.norm(d_new)

        d_new = M@d

        if np.abs(d.T@d_new - lambda_1) < res:
            break

    print(f'Power method iterations: {i}')
    return lambda_1, d

def init_lambdaP_s (quarkArgs, solverArgs):
    # Setup q integration
    u = np.zeros(solverArgs['nq'], dtype=np.complex_) # q integration nodes
    w = np.zeros(solverArgs['nq'], dtype=np.complex_) # q integration weights
    u[:], w[:] = roots_chebyt(solverArgs['nq'])

    # Variable transformation
    var_trafo = lambda u: (1+u)/(1-u)

    # Internal and external momenta
    q_s = var_trafo(u)
    p_s = var_trafo(u)

    K = Kernel(quarkArgs)
    K.setup(p_s, q_s, u, w)

    # Function where everything is included and only has P^2 as input variable to call for bisector
    def f (P_s):
        K.set_P_s(P_s)

        val, vec = power_method(K.get_Kij(),
                            maxit=solverArgs['maxit'],
                            tol=solverArgs['tol'],
                            res=solverArgs['res'])

        if np.abs(val.imag) > 1e-15: print('Eigenvalue complex!')

        return val.real, vec

    # Interval to search for bound state mass
    interval = (-K.B(0),0)

    return f, interval, K

def plot_dressing_contour (f, eta, M, xDom, yDom, axs, n_mesh=100, abs_max=None, levels=None):
    a = 1/(4*eta**2*M**2)
    c = -eta**2*M**2
    parabola = lambda x: np.sqrt((x-c)/a)
    parab_x_max = a*np.max(np.abs(yDom))**2 + c

    xl = np.linspace(xDom[0], xDom[1], n_mesh)
    xr = np.logspace(np.log10(xDom[1]), np.log10(xDom[2]), n_mesh)
    xl_parab = np.linspace(c,parab_x_max,1000)
    xr_parab = np.logspace(np.log10(xDom[1]),np.log10(parab_x_max),1000)
    axs[1].set_xscale('log')

    x = np.concatenate((xl,xr))
    y = np.linspace(*yDom, n_mesh)

    X, Y = np.meshgrid(x, y)
    Z = np.abs(f(X+1j*Y))
    Z[Z>abs_max] = np.nan

    CSl = axs[0].contourf(X, Y, Z, 15, levels=levels, cmap=plt.cm.terrain)
    CSr = axs[1].contourf(X, Y, Z, 15, levels=levels, cmap=plt.cm.terrain)

    axs[0].plot(xl_parab, parabola(xl_parab),'k--')
    axs[0].plot(xl_parab, -parabola(xl_parab),'k--')
    axs[1].plot(xr_parab, parabola(xr_parab),'k--')
    axs[1].plot(xr_parab, -parabola(xr_parab),'k--')

    axs[0].set_xlim((xDom[0],xDom[1]))
    axs[1].set_xlim((xDom[1],xDom[2]))
    axs[0].set_ylim(yDom)
    axs[1].set_ylim(yDom)

    return CSl
