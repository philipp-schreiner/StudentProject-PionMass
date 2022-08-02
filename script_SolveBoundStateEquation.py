import numpy as np
import matplotlib.pyplot as plt

from Classes import Kernel
from Functions import init_lambdaP_s, bisection, power_method



quarkArgs = {'N': 150,        # integration nodes
             'D': 16,         # interaction strength
             'omega': 0.5,    # energy scale
             'm': 5e-3,       # bare quark mass
             'res': 1e-15,    # residuum
             'M': 7}          # extrapolation polynoms

solverArgs = {'nq': 150,       # number of q^2 nodes
              'nt': 10,       # number of theta_2 nodes
              'res': 1e-15,   # Residuum for power method
              'maxit': 1000,  # Maxit for power method
              'tol': 1e-10}   # Tolerance for eigenvalues in power method

# Toggle to perform different sub-tasks
find_boundState_mass = False
check_spectral_radius = False
m_dependence_of_mB = False

if find_boundState_mass:
    # Returns function which takes P^2 as an input and returns the eigenvalue and
    # eigenvector
    lamb_P_s, interval, _ = init_lambdaP_s(quarkArgs, solverArgs)

    # Perform bisection to find bound state mass mB = sqrt(-P^2)
    mB_s = -bisection(lambda x: lamb_P_s(x)[0]-1,*interval,maxit=100,eps=1e-5)

    mB = np.sqrt(mB_s)
    print(f'Pion mass: {mB:.3f} GeV')

    # plot P^2 dependency of lambda
    N = 25
    x = np.linspace(0,0.2,N)
    y = np.zeros_like(x)

    for i in range(N):
        y[i] = lamb_P_s(-x[i]**2)[0]

    fig, ax = plt.subplots(1,1)

    ax.plot(x*1000,y,'o--')
    ax.plot([0,200],[1,1],'k--')
    ax.plot([mB*1000, mB*1000],[np.min(y),np.max(y)],'k--')
    ax.set_xlabel('$m_B~/~MeV$')
    ax.set_ylabel('$\lambda(m_B)$')
    ax.set_title(f'$\omega={quarkArgs["omega"]},~D={quarkArgs["D"]},m={quarkArgs["m"]*1000:.0f}~MeV,~m_B={mB*1000:.0f}~MeV$')
    ax.set_xlim((0,200))
    ax.set_ylim((np.min(y),np.max(y)))

    #plt.savefig(f'graphics/boundStateMass.pdf', bbox_inches='tight')
    plt.show()

if check_spectral_radius:
    _, _, kernel = init_lambdaP_s(quarkArgs, solverArgs)

    N = 25
    pow = 21
    P_s = np.linspace(0,0.2,N)
    bound_fro = np.zeros((pow,N))

    for p in range(2,pow,5):
        for i in range(N):
            kernel.set_P_s(P_s[i])
            bound_fro[p,i] = np.linalg.norm(np.linalg.matrix_power(kernel.get_Kij(),p+1), ord='fro')

        plt.plot(P_s,np.power(bound_fro[p,:],1/(p+1)),
                 'o--', label=f'n={p+1}')

    #plt.plot([np.min(P_s),np.max(P_s)],[1,1],'k--')
    plt.xlabel('$P^2~/~MeV$')
    plt.ylabel('$||\mathcal{K}^n(P^2)||^{1/n}_F$')
    plt.xlim((np.min(P_s),np.max(P_s)))
    plt.ylim((0.8,1.8))
    plt.grid(True)
    plt.legend()
    #plt.savefig(f'graphics/spectralRadius.pdf', bbox_inches='tight')
    plt.show()

if m_dependence_of_mB:
    # plot m dependency of mB
    N = 20
    x = np.linspace(0,0.1,N)
    y = np.zeros_like(x)

    for i in range(N):
        quarkArgs['m'] = x[i]
        lamb_P_s, interval, _ = init_lambdaP_s(quarkArgs, solverArgs)
        mB_s = -bisection(lambda x: lamb_P_s(x)[0]-1,*(-1,0),maxit=100,eps=1e-5)
        y[i] = np.sqrt(mB_s)

    fig, ax = plt.subplots(1,1)

    ax.plot(x,y,'o--')
    ax.set_xlim((0,0.1))
    ax.set_ylim((0,0.6))
    ax.grid(True)

    #plt.savefig(f'graphics/boundStateMass.pdf', bbox_inches='tight')
    plt.show()
