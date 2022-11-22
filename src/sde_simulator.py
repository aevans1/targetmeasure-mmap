import numpy as np


class ItoDiffusion(object):
    def __init__(self, drift, sqrt_diffusion=None, potential=None, beta_inv=1.0, dim=2):
        """ Initializes ItoDiffusion  simulator
        
        Corresponds to SDE 
        dX = drift(X)dt + sqrt(2 beta_inv) sqrt_diffusion(X) dW_t
        
        Parameters
        ----------
        drift : function handle
            needs to take vector input, and output vector of same size
        sqrt_diffusion: function handle
            needs to take vector input, and output square matrix  with same col size
        beta_inv : float, optional
            inverse temperature, controls noise level, by default 1.0
        dim : int, optional
            input dimension of SDE, by default 2
        """
        self.drift = drift
        if sqrt_diffusion is None:
            def sqrt_diffusion(x): return np.eye(dim)
        self.sqrt_diffusion = sqrt_diffusion
        self.beta_inv = beta_inv
        self.dim = dim
        self.potential = potential

    def simulate(self, X_init, N, dt, subsample=None, delay_steps=None):
        """Simulates SDE for given time length and intial data

        Parameters
        ----------
        X_init : vector
            initial condition for simulator
        N : integer
            number of timesteps to simulate
        dt : positive scalar
            size of a given timestep


        Returns
        -------
        X : vector
            vector output of SDE after simulation
        """
        
        if delay_steps is not None:
            # Compute delay steps and do not store
            X_curr = X_init
            for n in range(0, delay_steps):
                X_curr = self._step(X_curr, dt)

            # Store all steps only after delay  
            print(f"Did a delay of {delay_steps} steps")
            X_init = X_curr

        if subsample is not None: 
            # Store all steps
            Nsub = N // subsample
            X = np.zeros((self.dim, Nsub))
            print(f"Size of Trajectory: {Nsub}") 
            X[:, 0] = X_init
            X_curr = X[:, 0]
            m = 1
            step = 1
            while step < N:
                X_curr = self._step(X_curr, dt)
                if step % subsample == 0:
                    X[:, m] = X_curr
                    m += 1
                step +=1
        else:
            print(f"Size of Trajectory: {N}") 
            X = np.zeros((self.dim, N))
            X[:, 0] = X_init
            for n in range(1, N):
                X[:, n] = self._step(X[:, n-1], dt)
        return X
    
    def burst_simulate(self, burst_size, burst_time, X_init, dt):
        """Simulates SDE for ensemble of short ('burst') simulations

        All short simulations have same initial condition and timestep

        Parameters
        ----------
        burst_size : integer
            specifies the number of short simulations
        burst_time : scalar
            specifies the length of each short simulation
        X_init : vector
            initial condition for each short simulation
        dt : scalar
            timestep for each short simulation

        Returns
        -------
        cloud : array
            dim by burst_size array, ith column is end of ith burst simulation
        """
        cloud = np.zeros((self.dim, burst_size))
        m = int(np.floor(burst_time / dt))
        if burst_time < dt:
            dt = burst_time
            m = 1
        for n in range(burst_size):
            # Run simulation from X until T = burst_time 
            X = X_init
            for i in range(m):
                X = self._step(X, dt)
            cloud[:, n] = X
        return cloud

    def _step(self, X_init, dt):
        """ Base method to simulate one time-step of SDE

        Parameters
        ----------
        X_init : vector
            Initial condition for time-step
        dt : scalar
            time-step length

        Returns
        -------
        X_new : vector
            Output of simulation one time step from X_init
        """
        brownian = (np.sqrt(2*self.beta_inv*dt)
                    * np.random.normal(0, 1, (self.dim)))
        X_new = (X_init + self.drift(X_init)*dt 
                 + np.dot(brownian, self.sqrt_diffusion(X_init)))
        return X_new 
    

