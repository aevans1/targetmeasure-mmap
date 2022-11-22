import numpy as np 
import matplotlib.pyplot as plt 
import scipy.special as sc
import datetime

import src.helpers as helpers
import src.model_systems as model_systems

def main():

    # Use timestamp for saving
    time = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    SAVE = False

    beta_inv = 1.0
    seed = 1000
    np.random.seed(seed)
   
    # Create the simulator
    def my_sqrt_diffusion(x): 
        return model_systems.morocardin_sqrt_diffusion(x)

    def my_potential(x): 
        return model_systems.morocardin_potential(x)

    def my_drift(x):
        return model_systems.morocardin_drift(x, beta_inv=beta_inv)
        
    ### Set up and run a simulation
    dt = 0.01
    num_steps = 100000
    sub_steps = 1
    num_points = num_steps // sub_steps
    x_init = np.array([-1, 0])
    X = np.zeros((2, num_points))
    X[:, 0] = x_init

    step = 0
    n = 0
    x = x_init 
    while n < num_points - 1:
        # Run a simulator step 
        x = x + my_drift(x)*dt + np.sqrt(2*beta_inv*dt)*my_sqrt_diffusion(x).dot(np.random.normal(size=2))
        step +=1 
        
        # Save new timestep if applicable, update subsampled count 
        if step % sub_steps == 0:
            X[:, n + 1] = x
            n +=1

    diffusion_list = np.zeros((X.shape[1], 2, 2))
    for n in range(X.shape[1]):
        x = X[:, n]
        sqrt_diffusion = my_sqrt_diffusion(x)
        diffusion_list[n, :, :] = sqrt_diffusion.dot(sqrt_diffusion.T)

    if SAVE:
        time = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
        fname = "systems/MoroCardin/data/trajectory" + time
        np.savez(fname, data=X, diffusion_list=diffusion_list, dt=dt, num_steps=num_steps, sub_steps=sub_steps, beta_inv=beta_inv)


    # Plot
    plt.figure()
    plt.scatter(X[0, :], X[1, :], s=1.0)

    # Plot potential contours
    nx = 128
    ny = nx
    xmax = 4
    xmin = -4
    ymax = 4
    ymin = -4

    plot_params = [nx, ny, xmin, xmax, ymin, ymax]
    [pot, xx, yy] = helpers.gen_plot_data(my_potential, plot_params)

    contour_levels = np.linspace(np.min(pot), 7, 15) 
    plt.contour(xx, yy, pot, levels=contour_levels, cmap='plasma')
    plt.colorbar()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.show()

if __name__ == '__main__':
    main()
