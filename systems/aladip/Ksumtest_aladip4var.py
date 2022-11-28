import os, sys
sys.path.append('../..')
sys.path.append('.')


# Regular Modules
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.integrate as scint
from numpy.random import default_rng
import numpy.ma as ma
import matplotlib.tri as tri
import scipy.special as sc
import scipy.sparse as sps
import datetime
import itertools
import time

# My Modules
import src.helpers as helpers
import src.model_systems as model_systems
import src.diffusion_map as dmap

def main():

    # Setting inverse temperature for plotting
    xmin, xmax = -np.pi, np.pi
    ymin, ymax = -np.pi, np.pi

    kb = 0.0083144621
    T = 300
    beta = 1.0 / (kb * T)

    dataset = '4var'

    fname = 'data/TRAJ_COLVAR_METAD_2VAR_GRID_STATIC'
    colvar_data = np.loadtxt(fname)
    fname = 'data/diffusions_traj_static_4var.npz'
    inData = np.load(fname)
    diffusions = inData["diffusions"]

    end = colvar_data.shape[0]
    start = 0
    step = 5   # Subsampling Rate
    t = colvar_data[start:end:step,0]  # timestep
    phi = colvar_data[start:end:step,1] # phi dihedral angle coordinates
    psi = colvar_data[start:end:step,2] # psi dihedral angle coordinates
    theta = colvar_data[start:end:step,3] # theta dihedral angle coordinates
    xi = colvar_data[start:end:step,4] # xi dihedral angle coordinates
    bias = colvar_data[start:end:step,5] 
    rct = colvar_data[start:end:step,6] 
    rbias = colvar_data[start:end:step,7] 

    data = np.vstack([phi, psi, theta, xi])
    diffusions = diffusions[start:end, :, :]
    N = data.shape[1]
    print(f"number of data points: {N}")
    
    kernels = ['regular', 'mahal']
    pbc_dims = np.array([2*np.pi])
    eps_vals = 2.0**np.arange(-11, 3, 0.5)
    Ksums = np.zeros((2, len(eps_vals)))
    chi_logs = np.zeros_like(Ksums) 
    optimal_eps_vals = np.zeros(2)
    optimal_errors = np.zeros(2)
    errors = np.zeros_like(Ksums)
    for i in range(2):
        kernel = kernels[i] 

        if kernel == 'mahal':
            metric = 'mahalanobis'
        if kernel == 'regular':
            metric = 'regular'
        [Ksums[i,:], chi_logs[i,:], errors[i, :], optimal_eps_vals[i], optimal_errors[i]] = Ksum_test_new(diffusions, pbc_dims, eps_vals, data, metric)

        fname = f"data/Ktest_{dataset}_{kernel}.npz"
        print(f"optimal kde eps for {dataset}: {optimal_eps_vals[0]}")
        print(f"optimal mahal eps for {dataset}: {optimal_eps_vals[1]}")
        np.savez(fname, eps_vals=eps_vals, Ksum=Ksums[i,:], chi_log=chi_logs[i,:], optimal_eps = optimal_eps_vals[i])

    plt.figure()
    plt.plot(eps_vals, Ksums[0,:])
    plt.plot(eps_vals, Ksums[1,:])
    plt.legend(['regular kernel', 'mahal kernel']) 
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.axvline(x=optimal_eps_vals[0], ls='--', c='C0')
    plt.axvline(x=optimal_eps_vals[1], ls='--', c='C1')
    plt.axvline(x=optimal_errors[0], ls='--', c='C2')
    plt.axvline(x=optimal_errors[1], ls='--', c='C3')
    fname = f"figures/Ksums/logKsums_{dataset}"
    plt.savefig(fname, dpi=300)

    plt.figure()
    plt.plot(eps_vals, chi_logs[0, :])
    plt.plot(eps_vals, chi_logs[1, :])
    plt.legend(['regular kernel', 'mahal kernel']) 
    plt.title("log Ksums")
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.title("dlog_Sum/dlog_eps")
    fname = f"figures/Ksums/dlogKsums_{dataset}"
    plt.axvline(x=optimal_eps_vals[0], ls='--', c='C0')
    plt.axvline(x=optimal_eps_vals[1], ls='--', c='C1')
    plt.axvline(x=optimal_errors[0], ls='--', c='C2')
    plt.axvline(x=optimal_errors[1], ls='--', c='C3')
    plt.savefig(fname, dpi=300)
    
    plt.figure()
    plt.plot(eps_vals, errors[0, :])
    plt.plot(eps_vals, errors[1, :])
    plt.legend(['regular kernel', 'mahal kernel']) 
    plt.title("log Ksums")
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.title("log Ksum errors")
    fname = f"figures/Ksums/kernelsum_errors_{dataset}"
    plt.axvline(x=optimal_eps_vals[0], ls='--', c='C0')
    plt.axvline(x=optimal_eps_vals[1], ls='--', c='C1')
    plt.axvline(x=optimal_errors[0], ls='--', c='C2')
    plt.axvline(x=optimal_errors[1], ls='--', c='C3')
    plt.savefig(fname, dpi=300)
    plt.show()

def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        print("elapsed time: %f" % (end_ts - beg_ts))
        return retval
    return wrapper

@time_usage
def Ksum_test_new(diffusion_list, pbc_dims, eps_vals, data, metric):

     # Compute target measure 
    N = data.shape[1]
    d = data.shape[0]

    # Compute a kernel density estimate 
    num_idx = eps_vals.shape[0]
    Ksum = np.zeros(num_idx)

    chi_log_analytical = np.zeros(num_idx)
    chi_log = np.zeros(num_idx)
    errors = np.zeros(num_idx)    
    for i in range(num_idx):
        
        # Construct sparsified sqdists, kernel and generator 
        eps = eps_vals[i]
        radius = None
        if eps > 1:
            radius = 3*np.sqrt(1)
        if metric == 'regular':
            my_dmap = dmap.TargetMeasureDiffusionMap(epsilon=eps, pbc_dims=pbc_dims, radius=radius)
        elif metric == 'mahalanobis':
            my_dmap = dmap.TargetMeasureMahalanobisDiffusionMap(epsilon=eps, diffusion_list=diffusion_list, pbc_dims=pbc_dims, radius=radius)
        my_dmap.construct_generator(data)

        K = my_dmap.get_kernel()
        sqdists = my_dmap.sqdists
        Ksum[i] = K.sum(axis=None)
        if sps.issparse(sqdists) and sps.issparse(K):
            mat = K.multiply(sqdists)
        else:
            mat = sqdists*K
        chi_log_analytical[i] = mat.sum(axis=None)  / (Ksum[i])
        chi_log_analytical[i] *= (2/d)/(2*eps)
        print(f"chi log: {chi_log_analytical[i]}")
        
        subgraph = my_dmap.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        
        if np.mod(i, 5) == 0:
            print(i)
            print(Ksum[i])

        K1 = np.array(K.sum(axis=1)).ravel()
        K1_sqdist = np.array(mat.sum(axis=1)).ravel()
        K1_sqdist *= (2/d)/(2*eps) 
        error_vec = np.abs(K1 - K1_sqdist)
        errors[i] = np.sum(error_vec) / np.sum(np.abs(K1))
        print(f"errors: {errors[i]}")
    
    optimal_eps = eps_vals[np.nanargmax(chi_log_analytical)]
    optimal_error_eps = eps_vals[np.nanargmin(errors)]
    #optimal_error = np.nanargmin(errors) 
    #optimal_eps = eps_vals[np.nanargmin(errors)]
    return [Ksum, chi_log_analytical, errors, optimal_eps, optimal_error_eps]

if __name__ == "__main__":
    main()