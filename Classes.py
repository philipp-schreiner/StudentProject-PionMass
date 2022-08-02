import numpy as np
from scipy.special import roots_laguerre, roots_chebyt, roots_chebyu, eval_chebyt, ive

class DressingFunctions ():
    """Class for calculating the dressing functions A and B of the quark propagator."""

    def __init__ (self, A_init, B_init, D, omega, m):
        # Initial function values for the dressing functions
        self.A = A_init
        self.B = B_init

        # Model parameters
        self.D = D
        self.omega = omega
        self.m = m

        # Used later to calculate values off of the positive real axis
        self.var_trafo = None
        self.L = None
        self.E = None

    def solve_laguerre (self, n, res=1e-10, max_iter=100):
        # Solves the integral equation using Gauss-Laguerre quadrature:
        # Calculate nodes and weights
        yi, wi = roots_laguerre(n)
        yi, wi = yi[np.newaxis,:], wi[np.newaxis,:]

        # Nodes and weights are saved as class attributes
        self.nodes, self.weights = yi, wi

        # For better readibility we will work with xi and yi instead of self.nodes all the time. It is important that at this point x and y are the same because we want to plug the resulting approximation for A, B into the integral equation again without interpolation
        xi = yi

        # Gauss-Laguerre treats integrals with domain (0,inf) so no transformation is needed (this is just here for consistency with other integration methods)
        self.var_trafo = lambda x: x

        # Part of the integrant with exponentials
        self.E = lambda x,y: np.exp(-x.T + 2*np.sqrt(x.T@y))

        # Initialize iteration
        A_old, B_old = self.A, self.B

        for iter in range(max_iter):
            # Integral measure and denominator part of the integrant
            L = yi/(yi*self.A**2 + self.B**2/self.omega**2)

            # Calling integrator
            self.A, self.B = self.__solve(L, xi)

            # Check for convergence
            if np.max(np.abs(self.A-A_old))<res and np.max(np.abs(self.B-B_old))<res:
                break
            else:
                A_old, B_old = self.A, self.B

        print(f'Integral equation iterations: {iter}')

        # Save L because we can use it to calculate A,B for every x later on. Note that the final A and B are already saved in self.A and self.B
        self.L = L

        return self.A, self.B, yi*self.omega**2

    def solve_chebyshev (self, n, type='1st', res=1e-10, max_iter=100):
        # Solves the integral equation using Gauss-Chebyshev quadrature:
        # Calculate nodes and weights (for Chebychev polynomials of first or second kind)
        if type=='1st': ui, wi = roots_chebyt(n)
        elif type=='2nd': ui, wi = roots_chebyu(n)
        else: print('Unknown type.') # TO DO: error handeling

        ui, wi = ui[np.newaxis,:], wi[np.newaxis,:]

        # Nodes and weights are saved as class attributes
        self.nodes, self.weights = ui, wi

        # Transformation from u to y (to get appropriate integration domain for Gauss-Chebychev (-1,1))
        self.var_trafo = lambda u: (1+u)/(1-u)

        # Part of the integrant with exponentials
        self.E = lambda x,y: np.exp(-x.T -y + 2*np.sqrt(x.T@y))

        # For better readibility we will work with xi and yi instead of self.var_trafo(self.nodes).
        xi, yi = self.var_trafo(ui), self.var_trafo(ui)

        # Initialize iteration
        A_old, B_old = self.A, self.B

        for iter in range(max_iter):
            # Integral measure and denominator part of the integrant (also includes the compensating sqrt(1-u^2) from Chebyshev integration)
            if type=='1st':
                L = 2*np.power(1+ui,3/2)/np.power(1-ui,5/2)/(yi*self.A**2 + self.B**2/self.omega**2)
            else:
                L = 2*np.sqrt(1-ui**2)/np.power(ui-1,4)/(yi*self.A**2 + self.B**2/self.omega**2)

            # Calling integrator
            self.A, self.B = self.__solve(L, xi)

            # Check for convergence
            if np.max(np.abs(self.A-A_old))<res and np.max(np.abs(self.B-B_old))<res:
                break
            else:
                A_old, B_old = self.A, self.B

        print(f'Integral equation iterations: {iter}')

        # Save L because we can use it to calculate A,B for every x later on. Note that the final A and B are already saved in self.A and self.B
        self.L = L

        return self.A, self.B, self.var_trafo(ui)*self.omega**2

    def solve_chebyshev_log (self, n, type='1st', res=1e-10, max_iter=100):
        # Solves the integral equation using Gauss-Chebyshev quadrature:
        # Calculate nodes and weights (for Chebychev polynomials of first or second kind)
        if type=='1st': ui, wi = roots_chebyt(n)
        elif type=='2nd': ui, wi = roots_chebyu(n)
        else: print('Unknown type.') # TO DO: error handeling

        ui, wi = ui[np.newaxis,:], wi[np.newaxis,:]

        # Nodes and weights are saved as class attributes
        self.nodes, self.weights = ui, wi

        # Transformation from u to y (to get appropriate integration domain for Gauss-Chebychev (-1,1))
        #self.var_trafo = lambda u: (1+u)/(1-u)
        self.var_trafo = lambda u: -np.log((1-u)/2)

        # Part of the integrant with exponentials
        self.E = lambda x,y: np.exp(-x.T -y + 2*np.sqrt(x.T@y))

        # For better readibility we will work with xi and yi instead of self.var_trafo(self.nodes).
        xi, yi = self.var_trafo(ui), self.var_trafo(ui)

        # Initialize iteration
        A_old, B_old = self.A, self.B

        for iter in range(max_iter):
            # Integral measure and denominator part of the integrant (also includes the compensating sqrt(1-u^2) from Chebyshev integration)
            if type=='1st':
                #L = 2*np.power(1+ui,3/2)/np.power(1-ui,5/2)/(yi*self.A**2 + self.B**2/self.omega**2)
                L = yi*np.sqrt(1-ui**2)/(1-ui)/(yi*self.A**2 + self.B**2/self.omega**2)
            else:
                #L = 2*np.sqrt(1-ui**2)/np.power(ui-1,4)/(yi*self.A**2 + self.B**2/self.omega**2)
                L = yi/np.sqrt(1-ui**2)/(1-ui)/(yi*self.A**2 + self.B**2/self.omega**2)

            # Calling integrator
            self.A, self.B = self.__solve(L, xi)

            # Check for convergence
            if np.max(np.abs(self.A-A_old))<res and np.max(np.abs(self.B-B_old))<res:
                break
            else:
                A_old, B_old = self.A, self.B

        print(f'Integral equation iterations: {iter}')

        # Save L because we can use it to calculate A,B for every x later on. Note that the final A and B are already saved in self.A and self.B
        self.L = L

        return self.A, self.B, self.var_trafo(ui)*self.omega**2

    def __solve (self, L, xi):
        # Calculates new approximation for A, B from previous ones
        # Modified, exponentially rescaled Bessel functions
        I1 = ive(1, 2*np.sqrt(xi.T@self.var_trafo(self.nodes)))
        I2 = ive(2, 2*np.sqrt(xi.T@self.var_trafo(self.nodes)))

        # This exponential was previously saved as a lambda function such that we can use it here independently from the method used
        E = self.E(xi,self.var_trafo(self.nodes))

        # Putting everything under the integral together
        kernel1 = L*self.A*E*( -2*np.sqrt(self.var_trafo(self.nodes)/xi.T)*I1 + (1+(2+self.var_trafo(self.nodes))/xi.T)*I2 )
        kernel2 = L*self.B*E*( (xi.T + self.var_trafo(self.nodes))/np.sqrt(xi.T@self.var_trafo(self.nodes))*I1 - 2*I2 )

        # Approximating integral by sum
        RA = np.sum(kernel1*self.weights, axis=1, keepdims=True).T
        RB = np.sum(kernel2*self.weights, axis=1, keepdims=True).T

        # Returning new A,B
        return 1 + self.D*self.omega**2*RA, self.m + self.D*self.omega**2*RB

    def AB_fun (self, p_squared):
        # To be called only after the integral equations were solved. Uses the obtained A,B and integration nodes/weights to calculate A,B for any given p_squared
        if p_squared.shape[0] != 1: p_squared = p_squared[np.newaxis,:]

        # Transform p_squared to dimensionless x
        x = p_squared/self.omega**2

        return self.__solve(self.L, x)

    def AB_fun_int (self, M):
        # Interpolates existing data for A,B with Chebyshev polynomials. This is possible of integration was performed via Gauss-Chebyshev. TO DO: WRITE A CHECK FOR THAT
        # If Gauss-Chebyshev was used, we already know A,B at the integration nodes which are the roots of the N-th Chebyshev polynomial. This is all we need to calculate the expansion coefficients.
        # M is the maximal degree of polynomial to be used in the final approximation
        N = self.nodes.shape[1]
        diA, diB = np.zeros((1,M)), np.zeros((1,M))

        # Calculate expansion coefficients
        diA[0,0] = 1/N*np.sum(self.A*eval_chebyt(0,self.nodes))
        diB[0,0] = 1/N*np.sum(self.B*eval_chebyt(0,self.nodes))
        for i in range(1,M):
            diA[0,i] = 2/N*np.sum(self.A*eval_chebyt(i,self.nodes))
            diB[0,i] = 2/N*np.sum(self.B*eval_chebyt(i,self.nodes))

        def fA (p_squared):
            x = p_squared/self.omega**2
            # Transform x to u in (-1,1)
            u = -(1-x)/(1+x)
            return(np.polynomial.chebyshev.chebval(u,np.squeeze(diA)))

        def fB (p_squared):
            x = p_squared/self.omega**2
            # Transform x to u in (-1,1)
            u = -(1-x)/(1+x)
            return(np.polynomial.chebyshev.chebval(u,np.squeeze(diB)))

        return fA, fB

class Kernel ():
    """Class to calculate the kernel of the integral equation, i.e. the matrix that we need to solve the eigenvalue problem for."""

    def __init__(self, quarkArgs):
        # Dictionary of quark propagator arguments
        self.quark = quarkArgs

        # Later used to allocate memory
        self.cont2d = None
        self.cont3d = None

        self.p_s, self.q_s, self.uj, self.wj = None, None, None, None
        self.c2, self.w2 = None, None
        self.k_s = None

        # Quark propagator dressing functions
        self.A = None
        self.B = None

        # Solve for them
        self.__initQuarkPropagator()

    def __initQuarkPropagator (self):
        # Calculates the quark propagator function and saves them to be
        # evaluated later
        A_init = np.ones((1,self.quark['N']))
        B_init = 0.01*np.ones((1,self.quark['N']))

        dressings = DressingFunctions(A_init, B_init, self.quark['D'],
                                      self.quark['omega'],
                                      self.quark['m'])
        dressings.solve_chebyshev(n=self.quark['N'], type='1st',
                                  res=self.quark['res'])
        self.A, self.B = dressings.AB_fun_int(M=self.quark['M'])

    def setup (self, p_s, q_s, uj, wj, n_2=5):
        # Reshape vectors to use broadcasting. The final matrix should
        # have q as lines and p as columns. As an intermediate step, we
        # also have the nodes of the angular integration lie along the
        # third axis to avoid loops
        self.p_s = p_s.reshape(p_s.size,1,1) # row vector
        self.q_s = q_s.reshape(1,q_s.size,1) # column vector
        self.uj = uj.reshape(1,uj.size,1)    # column vector
        self.wj = wj.reshape(1,wj.size,1)    # column vector

        c2, w2 = roots_chebyt(n_2)
        self.c2 = c2[np.newaxis,np.newaxis,:] # vector along 3rd axis
        self.w2 = w2[np.newaxis,np.newaxis,:] # vector along 3rd axis

        self.k_s = self.q_s + self.p_s - 2*np.sqrt(self.p_s*self.q_s)*self.c2
        self.qp_s = np.zeros((1,self.q_s.size, self.c2.size), dtype=np.complex_)
        self.qm_s = np.zeros_like(self.qp_s, dtype=np.complex_)

        # Allocate memory
        self.cont2d = np.zeros((self.p_s.size,self.q_s.size), dtype=np.complex_)
        self.cont3d = np.zeros_like(self.k_s, dtype=np.complex_)

    def __mathcalK (self, P_s):
        # assuming q_s is a row vector, p_s a column vector and c2 either
        # a scalar or a vector pointing along the third direction
        self.qp_s[:,:,:] = self.q_s + P_s/4 + 1j*np.sqrt(-P_s*self.q_s)*self.c2
        self.qm_s[:,:,:] = self.q_s + P_s/4 - 1j*np.sqrt(-P_s*self.q_s)*self.c2

        # Precalculating dressing functions
        Ap, Am = self.A(self.qp_s), self.A(self.qm_s)
        Bp, Bm = self.B(self.qp_s), self.B(self.qm_s)

        self.cont3d[:,:,:] = self.quark['D']/(np.pi**2*self.quark['omega']**2)*(
                    self.k_s*np.exp(-self.k_s/self.quark['omega']**2))
        self.cont3d[:,:,:] *= ((self.q_s-P_s/4)*Ap*Am + Bp*Bm)/((self.qp_s*Ap**2 + Bp**2)*(self.qm_s*Am**2 + Bm**2))

        # Everything inside the sum part of the matrix. The result is of
        # size (n_p,n_q,n_t) and summed over theta to leave an array of
        # size (n_p,n_q,1).
        self.cont2d[:,:] = 4*np.pi*self.wj*np.power(1+self.uj,3/2)*np.power(1-self.uj,-5/2)*np.sum(self.w2*(1-self.c2**2)*self.cont3d, axis=2)

    def set_P_s (self, P_s):
        self.__mathcalK(P_s)

    def get_Kij (self):
        return self.cont2d
