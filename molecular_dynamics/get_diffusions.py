import numpy as np
import datetime
import sys

def get_diffusions(data_type):
    """Collects diffusion tensors along a trajectory or grid 
    and stores in a numpy array
    """
    time = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
    
    if data_type == 'traj':
        traj_size = 10001
        d = 4 # number of CVs
        # Compute diffusion coefficients over the trajectory
        diffusions = np.zeros((traj_size, d, d))
        start_idx = 0 # start collecting where simulations has static restraint
        for i in range(traj_size):
            idx = i + 1
            fname='traj/restrain_deriv{}'.format(idx)
            deriv_data = np.loadtxt(fname)
            deriv_data = deriv_data[start_idx:, 1:]
            diffusion = avg_diffusion_mat(deriv_data)
            diffusions[i, :, :] = diffusion
            if (i % 5000) == 0:
        	    print("finished with {}".format(i))
        fname = 'data/diffusions_traj' + time 
    
    elif data_type == 'grid':
        # Define grid dimensions
        num_steps = 128
        xmax = 6.28318
        dx = (xmax - (xmax/num_steps))/(num_steps-1.0)    
        dy = dx 

        # Compute diffusion coefficients over the grid
        start_idx = 0 # start collecting where simulation has static restraint
        diffusions = np.zeros((num_steps, num_steps, 2, 2))
        for j in range(num_steps):
            for i in range(num_steps):    
                fname='grid/restrain_deriv_phi{}psi{}'.format(i, j)
                deriv_data = np.loadtxt(fname)
                N = deriv_data.shape[0]
                deriv_data = deriv_data[start_idx:, 1:]
                diffusion = avg_diffusion_mat(deriv_data)
                diffusions[i, j, :, :] = diffusion
            print("done with column {}".format(j))

        fname = 'data/diffusions_grid' + time

    # Save
    np.savez(fname, diffusions=diffusions)

def avg_diffusion_mat(deriv_data):
    N = deriv_data.shape[0]
    deriv_sum = np.sum(deriv_data, axis=0)
    mat_dim = int(np.sqrt(deriv_sum.shape[0]))
    diffusion_mat = np.reshape(deriv_sum, (mat_dim, mat_dim)) 
    diffusion_mat *= 1.0/N
    return diffusion_mat

def main():
    #get_diffusions(data_type='grid')
    get_diffusions(data_type='traj')

if __name__ == '__main__':
    main()
