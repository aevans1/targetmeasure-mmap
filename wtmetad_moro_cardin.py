import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.model_systems as model_systems

def main():
    ##########################################################################
    # Initialize Euler-Maruyama parameters
    ##########################################################################
    seed = 1
    np.random.seed(seed)
    
    beta_inv = 1.0
    
    def my_sqrt_diffusion(x): 
            return model_systems.morocardin_sqrt_diffusion(x)
    
    def my_potential(x): 
        return model_systems.morocardin_potential(x)

    def my_drift(x): 
            return model_systems.morocardin_drift(x, beta_inv=beta_inv)

    N = 500000   # Number of steps
    subsample = 10
    Nsub = N // subsample   # Trajectory size after subsampling
    dt = 0.01   # timestep
    X = np.zeros((2, Nsub))        # initialize trajectory
    X[:, 0] = np.array([-1, 0])    # Set initial condition

    # Set parameters for diffusion / noise
    beta_inv = 1.0
        
    ##########################################################################
    # Initialize Metadynamics Parameters
    ##########################################################################
    height = 1.0              # height of gaussian bias
    stride_steps = 100         # Number of simulations steps per deposition
    width = 0.01                 # Widths / Covar matrix for collective vars
    gamma = 10.0              # fluctuation parameter for WTmetad
    welltemp = 1/(beta_inv*(gamma - 1))
    total_deps = int(np.floor(N/stride_steps))

    m = 0           # Initialize number of gaussians
    scales = np.zeros(total_deps) # Initialize list of gaussian centers/means
    centers = np.zeros((2, total_deps))# Initialize list of exp. scaling factors
    ##########################################################################
    # Simulate Euler-Maruyama with Metadynamics biased potential
    ##########################################################################
    i = 0
    step = 1  
    xval = X[:, i]
    while step < N:
        ##### Update Trajectory #####
        # Update bias if we hit another stride step
        if (step + 1) % stride_steps == 0:
            new_center = xval
            Vbias = 0
            k = centers.shape[1]
            diffs = new_center.reshape(2,1) - centers[:, :m]
            sqdists = np.sum(diffs**2, axis=0)
            gaussian_vec = height*np.exp(-sqdists/(2.*width))
            wt_scales_vec = np.exp(-welltemp*scales[:m])
            scales[m] = np.sum(gaussian_vec*wt_scales_vec) 
            centers[:, m] = new_center
            m += 1
            #print("adding bias number %d " % m)

        # Update bias gradient
        if m > 0:
            bias_grad = np.zeros(2)
            diffs = xval.reshape(2,1) - centers[:, :m] 
            sqdists = np.sum(diffs**2, axis=0)
            gaussian_vec = height*np.exp(-sqdists/(2.*width))
            wt_scales_vec = (np.exp(-welltemp*scales[:m]))
            bias_grad = np.sum(-gaussian_vec*(wt_scales_vec/width)*diffs, axis=1)
        else:
            bias_grad = np.zeros(2)
        # Update Drift 
        diffusion = my_sqrt_diffusion(xval)**2
        mod_drift = my_drift(xval) - np.dot(diffusion, bias_grad.flatten())     # drift = - grad
        diffusion = np.sqrt(2.*beta_inv*dt)*np.dot(diffusion, np.random.normal(0, 1, (2)))
        xval = xval + mod_drift*dt + diffusion
        
        if step % subsample == 0:  
            X[:, i + 1] = xval
            if i % 1000 == 0:
                print(i)
            i += 1
        step +=1

    diffusion_list = np.zeros((X.shape[1], 2, 2))
    for n in range(X.shape[1]):
        x = X[:, n]
        sqrt_diffusion = my_sqrt_diffusion(x)
        diffusion_list[n, :, :] = sqrt_diffusion.dot(sqrt_diffusion.T)

    time = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    fname = "data/trajectory_metad" + time
    np.savez(fname, data=X, diffusion_list=diffusion_list, dt=dt, centers=centers, beta_inv=beta_inv)

    plt.figure()
    plt.scatter(X[0, :], X[1, :], s=1.0)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4]) 
    plt.figure()
    
    ##########################################################################
    # Evaluate potential and bias on a grid 
    ##########################################################################
    #nx, ny = 128, 128
    #xmin, xmax = -4, 4
    #ymin, ymax = -4, 4
    #plot_params = [nx, ny, xmin, xmax, ymin, ymax]
    #metad_params = [welltemp, height, centers, width]
    #[xx, yy, cutoff, plot_data] = gen_data(plot_params, metad_params, scales, my_potential)
    #pot, bias_pot, mod_pot = plot_data

    ##########################################################################
    # Error Computations
    ##########################################################################
    #dx = (xmax - xmin) / nx
    #dy = (ymax - ymin) / ny

    ## The constant C is chosen to translate the free energy estimate to the free
    ## energy at the origin

    #scaled_bias = (gamma / (gamma - 1.0))*bias_pot
    #C = np.max(scaled_bias)
    #approx_free_energy = C - scaled_bias
    #approx_free_energy[cutoff] = 0
    #error = np.sqrt(dx * dy * np.sum((pot - approx_free_energy)**2))
    #rel_error = error / np.sqrt(dx * dy * np.sum(pot**2))

    #print("potential energy error: %f \n" % error)
    #print("potential energy relative error: %f \n" % rel_error)
    # Save data for later
    #np.savez('test_file', xx=xx, yy=yy, pot=pot, mod_pot=mod_pot,
    #         bias_pot=bias_pot, error=error, rel_error=rel_error)

    fig = plt.figure()
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    # Add relevant plots
    plt.scatter(X[0, :], X[1, :], s=1, c='r')
    plt.scatter(centers[0, :], centers[1, :], c='k', s=20,
                marker='^', zorder=10)

    # Add title and legends, aspect-ratio
    plt.legend(['trajectory', 'biases'])
    plt.title(' MetaD trajectory plot')

    plt.tight_layout()
    plt.savefig("pot_traj.png")



    ##########################################################################
    # Plot potential and bias 
    ##########################################################################
    #fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)

    # ax1: Set up sub-plot and color-bar
    #ax1.set_xlim([-4, 4])
    #ax1.set_ylim([-4, 4])
    #pos = ax1.imshow(pot)
    #ax1.cla()
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(pos, cax=cax)

    ## Add relevant plots
    #ax1.contourf(xx, yy, pot, levels=50)

    ## Add title and legends, aspect-ratio
    #ax1.set_title('Original Potential(Truncated)')
    #ax1.set_aspect('equal', 'box')

    # ax2:  Set up sub-plot and color-bar
    #ax2.set_xlim([-4, 4])
    #ax2.set_ylim([-4, 4])
    #pos = ax2.imshow(pot)
    #ax2.cla()
    #divider = make_axes_locatable(ax2)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(pos, cax=cax)

    # Add relevant plots
    #ax2.contour(xx, yy, pot, levels=10)
    #ax2.scatter(X[0, :], X[1, :], s=1, c='r')
    #centers_np = np.array(centers).T
    #ax2.scatter(centers_np[0, :], centers_np[1, :], c='k', s=20,
    #            marker='^', zorder=10)

    # Add title and legends, aspect-ratio
#    ax2.legend(['trajectory', 'biases'])
#    ax2.set_title(' MetaD trajectory plot')
#    ax2.set_aspect('equal', 'box')
#
#    plt.tight_layout()
#    plt.savefig("pot_traj.png")
#
#    fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)
#
#    # ax1: Set up sub-plot and color-bar
#    ax1.set_xlim([-4, 4])
#    ax1.set_ylim([-4, 4])
#    pos = ax1.imshow(mod_pot)
#
#    ax1.cla()
#    divider = make_axes_locatable(ax1)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    fig.colorbar(pos, cax=cax)
#
    # Add relevant plots
    #ax1.contourf(xx, yy, mod_pot, levels=50)     # plot modified potential

    # Add title and legends, aspect-ratio
    #title = 'MetaD Modified Potential (contours)'
    #ax1.set_title(title)
    #ax1.set_aspect('equal', 'box')

    # ax2:  Set up sub-plot and color-bar
    #ax2.set_xlim([-4, 4])
    #ax2.set_ylim([-4, 4])
    #pos = ax2.imshow(bias_pot)
    #ax2.cla()
    #divider = make_axes_locatable(ax2)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(pos, cax=cax)

    # Add relevant plots
    #   Plot approximated potential
    #ax2.contourf(xx, yy, bias_pot, levels=50)

    # Add title and legends, aspect-ratio
    #title = 'MetaD Bias Potential'
    #ax2.set_title(title)
    #ax2.set_aspect('equal', 'box')

    #plt.tight_layout()
    #plt.savefig("metad.png")


    # Plot approximate free energy
    #plt.figure()

    # ax: Set up sub-plot and color-bar
    #plt.xlim([-4, 4])
    #plt.ylim([-4, 4])

    # Plot data and add colorbar
    #plt.contourf(xx, yy, approx_free_energy, levels=50)
    #plt.colorbar()
    
    # Add title and legends, aspect-ratio
    #plt.title('Approximate Free Energy')

    #plt.tight_layout()
    #plt.savefig("fre_energy.png")

    plt.show()


def bias_potential(x, centers, scales, welltemp, height, width):
    r""" Computes bias potential from Well-Tempered Metadynamics.

    Computes sum of gaussian biases at centers, along with Well-tempered 
    Metadynamics exponential bias

    Parameters
    ----------
    centers : array-like
        Python list, each entry is a 2-dim vector
        each entry corresponds to a bias center
    scales:   array-like
        Python list, each entry is a 2-dim vector
        each entry corresponds an exponential
        potential factor in WT-MetaD
    well-temp:  scalar
        scaling constant for WT-metadynamics weight term,
        well-temp = 1 /(temp * (gamma - 1))
    height : scalar
        Height of gaussian kernel.
    inv_cov : array-like
        Two by Two array, inverse of covariance matrix

    Returns
    -------
    V_modified : scalar
        MetaD modifed potential V + V_bias

    """
    Vbias = 0
    k = centers.shape[1]
    for i in range(0, k):
        gaussian = height*np.exp(-np.sum((x - centers[:, i])**2)/(2.*width))
        Vbias += gaussian*np.exp(-welltemp * scales[i])
    return Vbias


def gen_data(plot_params, metad_params, scales, my_potential):
    r"""Evaluates potential and bias potential from metad on a grid for plotting

    Parameters
    ----------
    plot_params: array-like,
        Python list of values for grid,
        plot_params must be of form [nx, ny, xmin, xmax, ymin, ymax]
        as defined below

        nx, ny : scalar
            Number of steps in x and y direction respectively.
        xmin, xmax : scalar
            Interval bounds [xmin, xmax] for the x-bounds of the grid
        xmin, xmax : scalar
            Interval bounds [ymin, ymax] for the y-bounds of the grid

    metad_params: array-like,
        Python list of metadynamics parameters,
        metad_params must be of form [height, centers, inv_cov] as defined below

        height : scalar
            Height of gaussian kernel.
        centers : array-like
            Python list, each entry isa 2-dim vector
            each entry corresponds to a bias center
        inv_cov : array-like
            Two by Two array, inverse of covariance matrix

    Returns
    -------
    xx, yy : array-like
        meshgrid coordinates the potential and bias were evaluated on
    cutoff : array-like (boolean)  
        boolean array marking the plotting cutoff for the potential function  
    plot_data : array-like,
        Python list of cartesian grids for plotting metad data,
        of form [pot, bias_pot, metad_pot]

        pot : array-like
            nx by ny array, cartesian grid of potential function
        bias_pot : array-like
            nx by ny array, cartesian grid of bias potential function
        metad_pot : array-like
            nx by ny array, cartesian grid of full modified potential function
    
    """

    # Read in input lists
    #nx, ny, xmin, xmax, ymin, ymax = plot_params
    welltemp, height, centers, inv_widths = metad_params
    ##########################################################################
    # Compute potential energy contours on grid
    ##########################################################################
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    pot = np.zeros((nx, ny))
    for j in range(nx):
        for i in range(ny):
            a = xx[i, j]
            b = yy[i, j]
            v = np.array([a, b])
            pot[i, j] = my_potential(v)
    cutoff = np.nonzero(pot > 4)
    pot[cutoff] = 0   # Zero out all potential energy too far past the 3 wells
    #######################################################################
    # Calculate Modified potential and Bias on grid
    #######################################################################
    mod_pot = np.zeros((nx, nx))   # Modified potential values on grid
    bias_pot = np.zeros((nx, nx))     # Bias values on grid
    for j in range(nx):
        for i in range(nx):
            a = xx[i, j]
            b = yy[i, j]
            v = np.array([a, b])

            bias_pot[i, j] = bias_potential(v, centers, scales, welltemp, height, inv_widths)
            mod_pot[i, j] = bias_pot[i, j] + my_potential(v)

    # Only showing modified potential with respect to corrseponding level
    # sets of potential
    mod_pot[cutoff] = 0
    bias_pot[cutoff] = 0

    plot_data = [pot, bias_pot, mod_pot]
    return xx, yy, cutoff, plot_data


if __name__ == '__main__':
    main()
