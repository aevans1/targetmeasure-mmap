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
    
    #datasets = ['subsampled', 'deltanet']
    datasets = ['subsampled']
    kernels = ['regular', 'mahalanobis']
    pbc_dims = np.array([2*np.pi])
    eps_vals = 2.0**np.arange(-11, 3, 0.5)
    #eps_vals = 2.0**np.arange(-11, 3, 2.0)
    Ksums = np.zeros((2, len(eps_vals)))
    chi_logs = np.zeros_like(Ksums) 
    optimal_eps_vals = np.zeros(2)
    for dataset in datasets:
        ###########################################################################
        # Load Trajectory data
        ###########################################################################
        if dataset == 'deltanet':
            fname = f"systems/aladip/data/wtmetad_phipsi_deltanet_validation.npz"
            inData = np.load(fname)
            data = inData['data']
            diffusion_list = inData['diffusion_list']
            free_energy = inData['free_energy'].flatten()

        if dataset == 'subsampled':
            # Trajectory data
            fname = f"systems/aladip/data/wtmetad_phipsi_long_validation.npz"
            inData = np.load(fname)
            data = inData['data']
            diffusion_list = inData['diffusion_list']
            free_energy = inData['free_energy'].flatten()
            free_energy = free_energy[::5]
            data = data[:, ::5]
            diffusion_list = diffusion_list[::5, :, :]

        N = data.shape[1]
        print("number of data points: %d" % N)

        target_measure = np.exp(-beta*free_energy).flatten()

        for i in range(2):
            kernel = kernels[i] 
            [Ksums[i,:], chi_logs[i,:], optimal_eps_vals[i]] = Ksum_test(diffusion_list, pbc_dims, eps_vals, data, kernel, target_measure)

        fname = f"data/Ktest_{dataset}_{kernel}_regular.npz"
        print(f"optimal kde eps for {dataset}: {optimal_eps_vals[0]}")
        print(f"optimal mahal eps for {dataset}: {optimal_eps_vals[1]}")
        #np.savez(fname, eps_vals=eps_vals, Ksum=Ksums[i,:], chi_log=chi_logs[i,:], optimal_eps = optimal_eps_vals[i])

        plt.figure()
        plt.plot(eps_vals, Ksums[0,:])
        plt.plot(eps_vals, Ksums[1,:])
        plt.legend(['regular kernel', 'mahal kernel']) 
        plt.xscale("log", base=10)
        plt.yscale("log", base=10)
        plt.axvline(x=optimal_eps_vals[0], ls='--', c='C0')
        plt.axvline(x=optimal_eps_vals[1], ls='--', c='C1')

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

    plt.show()

def Ksum_test(diffusion_list, pbc_dims, eps_vals, data, kernel, target_measure):

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
            my_dmap = dmap.TargetMeasureDiffusionMap(epsilon=eps, pbc_dims=pbc_dims, radius=radius, target_measure=target_measure)
        elif kernel == 'mahalanobis':
            my_dmap = dmap.TargetMeasureMahalanobisDiffusionMap(epsilon=eps, diffusion_list=diffusion_list, pbc_dims=pbc_dims, radius=radius, target_measure=target_measure)
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