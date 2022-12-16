import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import logging

class Simulator:
    
    def __init__(self,
                 Du: float = 1E-3,
                 Dv: float = 5E-3,
                 k: float = 5E-3,
                 t: float = 50,
                 tdim: int = 501,
                 x_left: float = -1.0,
                 x_right: float = 1.0,
                 xdim: int = 50,
                 y_bottom: float = -1.0,
                 y_top: float = 1.0,
                 ydim: int = 50,
                 n: int = 1,
                 seed: int = 0):
    
        """
        Constructor method initializing the parameters for the diffusion
        sorption problem.
        :param Du: The diffusion coefficient of u
        :param Dv: The diffusion coefficient of v
        :param k: The reaction parameter
        :param t: Stop time of the simulation
        :param tdim: Number of simulation steps
        :param x_left: Left end of the 2D simulation field
        :param x_right: Right end of the 2D simulation field
        :param xdim: Number of spatial steps between x_left and x_right
        :param y_bottom: bottom end of the 2D simulation field
        :param y_top: top end of the 2D simulation field
        :param ydim: Number of spatial steps between y_bottom and y_top
        :param n: Number of batches
        """

        # Set class parameters
        self.Du = Du
        self.Dv = Dv
        self.k = k

        self.T = t
        self.X0 = x_left
        self.X1 = x_right
        self.Y0 = y_bottom
        self.Y1 = y_top
        
        self.Nx = xdim
        self.Ny = ydim
        self.Nt = tdim
        
        # Calculate grid size and generate grid        
        self.dx = (self.X1 - self.X0)/(self.Nx)
        self.dy = (self.Y1 - self.Y0)/(self.Ny)
        
        self.x = np.linspace(self.X0 + self.dx/2, self.X1 - self.dx/2, self.Nx)
        self.y = np.linspace(self.Y0 + self.dy/2, self.Y1 - self.dy/2, self.Ny)
        
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
        
        u0 = np.random.randn(self.Nx*self.Ny)
        v0 = np.random.randn(self.Nx*self.Ny)
        
        u0 = u0.reshape(self.Nx*self.Ny)
        v0 = v0.reshape(self.Nx*self.Ny)
        u0 = np.concatenate((u0,v0))
        
        # # Normalize u0
        # u0 = 2 * (u0 - u0.min()) / (u0.max() - u0.min()) - 1

        # Generate arrays as diagonal inputs to the Laplacian matrix
        main_diag = -2*np.ones(self.Nx)/self.dx**2 -2*np.ones(self.Nx)/self.dy**2
        main_diag[0] = -1/self.dx**2 -2/self.dy**2
        main_diag[-1] = -1/self.dx**2 -2/self.dy**2
        main_diag = np.tile(main_diag, self.Ny)
        main_diag[:self.Nx] = -2/self.dx**2 -1/self.dy**2
        main_diag[self.Nx*(self.Ny-1):] = -2/self.dx**2 -1/self.dy**2
        main_diag[0] = -1/self.dx**2 -1/self.dy**2
        main_diag[self.Nx-1] = -1/self.dx**2 -1/self.dy**2
        main_diag[self.Nx*(self.Ny-1)] = -1/self.dx**2 -1/self.dy**2
        main_diag[-1] = -1/self.dx**2 -1/self.dy**2
        
        left_diag = np.ones(self.Nx)
        left_diag[0] = 0
        left_diag = np.tile(left_diag, self.Ny)
        left_diag = left_diag[1:]/self.dx**2
        
        right_diag = np.ones(self.Nx)
        right_diag[-1] = 0
        right_diag = np.tile(right_diag, self.Ny)
        right_diag = right_diag[:-1]/self.dx**2
        
        bottom_diag = np.ones(self.Nx*(self.Ny-1))/self.dy**2
        
        top_diag = np.ones(self.Nx*(self.Ny-1))/self.dy**2
        
        # Generate the sparse Laplacian matrix
        diagonals = [main_diag, left_diag, right_diag, bottom_diag, top_diag]
        offsets = [0, -1, 1, -self.Nx, self.Nx]
        self.lap = diags(diagonals, offsets)

        # Solve the diffusion reaction problem
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        ode_data = prob.y

        sample_u = np.transpose(ode_data[:self.Nx*self.Ny]).reshape(-1,self.Ny,self.Nx)
        sample_v = np.transpose(ode_data[self.Nx*self.Ny:]).reshape(-1,self.Ny,self.Nx)

        return np.stack((sample_u, sample_v),axis=-1)

    def rc_ode(self, t, y):
        """
        Solves a given equation for a particular time step.
        :param t: The current time step
        :param y: The equation values to solve
        :return: A finite volume solution
        """
        
        # Separate y into u and v
        u = y[:self.Nx*self.Ny]
        v = y[self.Nx*self.Ny:]
       
        # Calculate reaction function for each unknown
        react_u = u - u**3 - self.k - v
        react_v = u - v
       
        # Calculate time derivative for each unknown
        u_t = react_u + self.Du * (self.lap @ u)
        v_t = react_v + self.Dv * (self.lap @ v)
        
        # Stack the time derivative into a single array y_t
        y_t = np.concatenate((u_t,v_t))
        
        # Log the simulation progress
        # self.log.info('t = ' + str(t))
       
        return y_t
