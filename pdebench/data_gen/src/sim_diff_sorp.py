import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import logging

class Simulator:
    
    def __init__(self,
                 D: float = 5E-4,
                 por: float = 0.29,
                 rho_s: float = 2880,
                 k_f: float = 3.5E-4,
                 n_f: float = 0.874,
                 sol: float = 1.0,
                 t: float = 2500,
                 tdim: int = 501,
                 x_left: float = 0.0,
                 x_right: float = 1.0,
                 xdim: int = 50,
                 n: int = 1,
                 seed: int = 0):
    
        """
        Constructor method initializing the parameters for the diffusion
        sorption problem.
        :param D: The diffusion coefficient
        :param por: The porosity
        :param rho_s: The dry bulk density
        :param k_f: The Freundlich parameter
        :param n_f: The Freundlich exponent
        :param sol: The solubility of the contaminant
        :param t: Stop time of the simulation
        :param tdim: Number of simulation steps
        :param x_left: Left end of the 2D simulation field
        :param x_right: Right end of the 2D simulation field
        :param xdim: Number of spatial steps between x_left and x_right
        :param n: Number of batches
        """

        # Set class parameters
        self.D = D
        self.por = por
        self.rho_s = rho_s
        self.k_f = k_f
        self.n_f = n_f
        self.sol = sol

        self.T = t
        self.X0 = x_left
        self.X1 = x_right
        
        self.Nx = xdim
        self.Nt = tdim
        
        # Calculate grid size and generate grid        
        self.dx = (self.X1 - self.X0)/(self.Nx)
        self.x = np.linspace(self.X0 + self.dx/2, self.X1 - self.dx/2, self.Nx)
        
        # Time steps to store the simulation results
        self.t = np.linspace(0, self.T, self.Nt)
        
        # Initialize the logger
        self.log = logging.getLogger(__name__)
        
        self.seed = seed
        
    def generate_sample(self):
        """
        Single sample generation using the parameters of this simulator.
        :return: The generated sample as numpy array(t, x, y, num_features)
        """
        np.random.seed(self.seed)
        
        # Generate initial condition
        u0 = np.ones(self.Nx) * np.random.uniform(0,0.2)

        # Generate arrays as diagonal inputs to the Laplacian matrix
        main_diag = -2*np.ones(self.Nx)/self.dx**2
        
        left_diag = np.ones(self.Nx-1)/self.dx**2
        
        right_diag = np.ones(self.Nx-1)/self.dx**2
        
        # Generate the sparse Laplacian matrix
        diagonals = [main_diag, left_diag, right_diag]
        offsets = [0, -1, 1]
        self.lap = diags(diagonals, offsets)
        
        # Initialize the right hand side to account for the boundary condition
        self.rhs = np.zeros(self.Nx)

        # Solve the diffusion reaction problem
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        ode_data = prob.y
        
        sample_c = np.transpose(ode_data)
        
        return np.expand_dims(sample_c, axis=-1)

    def rc_ode(self, t, y):
        """
        Solves a given equation for a particular time step.
        :param t: The current time step
        :param y: The equation values to solve
        :return: A finite volume solution
        """
        
        c = y
        
        # Define left and right boundary conditions
        left_BC = self.sol
        right_BC = (c[-2]-c[-1])/self.dx * self.D
       
        # Calculate the Freundlich retardation factor
        retardation = 1 + ((1 - self.por)/self.por)*self.rho_s\
                       *self.k_f*self.n_f*(c + 1e-6)**(self.n_f-1)
                       
        # Calculate the right hand side
        self.rhs[0] = self.D/retardation[0]/(self.dx**2)*left_BC
        self.rhs[-1] = self.D/retardation[-1]/(self.dx**2)*right_BC
       
        # Calculate time derivative
        c_t = self.D/retardation * (self.lap @ c) + self.rhs
        y_t = c_t
        
        # Log the simulation progress
        # self.log.info('t = ' + str(t))
       
        return y_t
