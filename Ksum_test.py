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
    beta = 1.0

    def potential(x): return model_systems.morocardin_potential(x)

    xmin, xmax = -2, 2
    ymin, ymax = -2, 2

    #datasets = ['betainv3', 'metad', 'deltanet']
    datasets = ['metad']
    kernels = ['regular', 'mahalanobis']
    eps_vals = 2.0**np.arange(-20, 10, 0.5)
    Ksums = np.zeros((2, len(eps_vals)))
    chi_logs = np.zeros_like(Ksums) 
    optimal_eps_vals = np.zeros(2)
    for dataset in datasets:
        fname = f"systems/MoroCardin/data/data_solution_{dataset}.npz"
        inData = np.load(fname)
        data = inData['data']
        diffusion_list = inData['diffusion_list']
        N = data.shape[1]
        
        # Build Target Measure
        target_beta = 1. 
        target_measure = np.zeros(N)
        for i in range(N):
            target_measure[i] = np.exp(-target_beta*potential(data[:, i]))
        
        for i in range(2):
            kernel = kernels[i] 
            [Ksums[i,:], chi_logs[i,:], optimal_eps_vals[i]] = Ksum_test(diffusion_list, eps_vals, data, kernel, target_measure)
            fname = f"data/Ktest_{dataset}_{kernel}.npz"
            #np.savez(fname, eps_vals=eps_vals, Ksum=Ksums[i,:], chi_log=chi_logs[i,:], errors=errors[i, :], optimal_eps = optimal_eps_vals[i])
        print(f"optimal kde eps for {dataset}: {optimal_eps_vals[0]}")
        print(f"optimal mahal eps for {dataset}: {optimal_eps_vals[1]}")

        plt.figure()
        plt.plot(eps_vals, Ksums[0,:])
        plt.plot(eps_vals, Ksums[1,:])
        plt.legend(['regular kernel', 'mahal kernel']) 
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.axvline(x=optimal_eps_vals[0], ls='--', c='C0')
        plt.axvline(x=optimal_eps_vals[1], ls='--', c='C1')
        fname = f"figures/logKsums_{dataset}"
        #plt.savefig(fname, dpi=300)

        plt.figure()
        plt.plot(eps_vals, chi_logs[0, :])
        plt.plot(eps_vals, chi_logs[1, :])
        plt.legend(['regular kernel', 'mahal kernel']) 
        plt.title("log Ksums")
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.title("dlog_Sum/dlog_eps")
        fname = f"figures/dlogKsums_{dataset}"
        plt.axvline(x=optimal_eps_vals[0], ls='--', c='C0')
        plt.axvline(x=optimal_eps_vals[1], ls='--', c='C1')
        #plt.savefig(fname, dpi=300)
    
    plt.show()
        
def Ksum_test(diffusion_list, eps_vals, data, kernel, target_measure):

    num_idx = eps_vals.shape[0]
    Ksum = np.zeros(num_idx)
    chi_log_analytical = np.zeros(num_idx)
    
    for i in range(num_idx):
        # Construct sparsified sqdists, kernel and generator with radius nearest neighbors 
        eps = eps_vals[i]
        radius = None

        # put a maximum for the radius in radius-nearest neighbors
        if eps > 1:
            radius = 3*np.sqrt(1)
        
        if kernel == 'regular':
            my_dmap = dmap.TargetMeasureDiffusionMap(epsilon=eps, pbc_dims=None, radius=radius, target_measure = target_measure)
        elif kernel == 'mahalanobis':
            my_dmap = dmap.TargetMeasureMahalanobisDiffusionMap(epsilon=eps, diffusion_list=diffusion_list, pbc_dims=None, radius=radius, target_measure = target_measure)
        my_dmap.construct_generator(data)

        Ksymm = my_dmap.get_kernel_reweight_symmetric()
        sqdists = my_dmap.get_sqdists()
        Ksum[i] = Ksymm.sum(axis=None)

        # Compute deriv of log Ksum w.r.t log epsilon ('chi log')
        if sps.issparse(sqdists) and sps.issparse(Ksymm):
            mat = Ksymm.multiply(sqdists)
        else:
            mat = sqdists*Ksymm
        chi_log_analytical[i] = mat.sum(axis=None)  / ((2*eps)*Ksum[i])
        print(f"epsilon: {eps}")
        print(f"chi log: {chi_log_analytical[i]}")
        print("\n") 
        optimal_eps = eps_vals[np.nanargmax(chi_log_analytical)]
    return [Ksum, chi_log_analytical, optimal_eps]


if __name__ == "__main__":
    main()