import numpy as np 
import scipy.sparse as sps
import scipy.spatial
import scipy.linalg.lapack as lapack
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.linalg.lapack import dpstrf
from scipy.spatial.distance import cdist

import src.helpers as helpers
import src.model_systems as model_systems

def main():

    # Load data
    fname = "systems/MoroCardin/data/data_solution_deltanet.npz"
    inData = np.load(fname)
    print("Keys in data:")
    print(list(inData.keys()))

    data = inData["data"]
    print("Data shape from trajectory:")
    print(data.shape)
    num_samples = data.shape[1]
    num_features = data.shape[0]

    # Define Target Measure
       # Setting inverse temperature for plotting
    beta = 1.0
    def potential(x): return model_systems.morocardin_potential(x)
    target_measure = np.zeros(num_samples) 
    for i in range(num_samples):
        target_measure[i] = np.exp(-beta*potential(data[:, i]))
    
    #[Ksum, chi_log_analytical, eps_kde, effective_dim] = Ksum_test_unweighted(eps_vals, data)
    #print(f"eps_kde = {eps_kde}")
    #plt.figure()
    #plt.plot(eps_vals, Ksum)
    #plt.xscale("log", base=10)
    #plt.yscale("log", base=10)
    #plt.axvline(x=eps_kde, ls='--')

    #plt.figure()
    #plt.plot(eps_vals, chi_log_analytical)
    #plt.title("log Ksums")
    #plt.xscale("log", base=10)
    #plt.yscale("log", base=10)
    #plt.title("dlog_Sum/dlog_eps")
    #plt.axvline(x=eps_kde, ls='--')
    ##plt.savefig(fname, dpi=300)
    #[Ksum, chi_log_analytical, optimal_eps] = Ksum_test_weighted(eps_vals, eps_kde, data, target_measure, d=effective_dim)

    #print(f"optimal_eps = {optimal_eps}")
    #plt.figure()
    #plt.plot(eps_vals, Ksum)
    #plt.xscale("log", base=10)
    #plt.yscale("log", base=10)
    #plt.axvline(x=optimal_eps, ls='--')

    #plt.figure()
    #plt.plot(eps_vals, chi_log_analytical)
    #plt.title("log Ksums")
    #plt.xscale("log", base=10)
    #plt.yscale("log", base=10)
    #plt.title("dlog_Sum/dlog_eps")
    #plt.axvline(x=optimal_eps, ls='--')
    ##plt.savefig(fname, dpi=300)

    # For setting, A,B based on circles
    center_A = np.array([-1, 0])
    center_B = np.array([1, -0])
    radius_A = 0.2
    radius_B = 0.2

    dist_A = np.sqrt(np.sum((center_A[..., np.newaxis] - data)**2, axis=0))
    dist_B = np.sqrt(np.sum((center_B[..., np.newaxis] - data)**2, axis=0))
    A = dist_A < radius_A
    B = dist_B < radius_B 
    C = ~np.logical_or(A, B)

    # Run diffusion map
    #epsilon  = optimal_eps
    epsilon = 0.01

    [_, L] = create_laplacian(data, target_measure, epsilon=epsilon)
    q = solve_committor(L, B, C, num_samples)
    plt.figure()
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.ylabel("committor")
    plt.scatter(data[0, :], data[1, :], c=q)

    #eigvecs, _ = compute_spectrum(L, epsilon, num_eigvecs=4)
    #plt.figure()
    #plt.scatter(data[0, :], data[1, :], c=eigvecs[:, 0])
    #plt.title(f"eigvec1")
    #plt.figure()
    #plt.scatter(data[0, :], data[1, :], c=eigvecs[:, 1])
    #plt.title(f"eigvec2")
    #plt.figure()
    #plt.scatter(data[0, :], data[1, :], c=eigvecs[:, 2])
    #plt.title(f"eigvec3")


    plt.show() 
    return None

def create_laplacian(data, target_measure, epsilon):

    num_features = data.shape[0]
    num_samples = data.shape[1]

    ### Create distance matrix
    neigh = NearestNeighbors(n_neighbors=512, metric='sqeuclidean')
    neigh.fit(data.T)
    sqdists = neigh.kneighbors_graph(data.T, mode="distance") 
   
    ### Create Graph Laplacian
    K = sqdists.copy()
    K.data = np.exp(-K.data / (2*epsilon))
    print(f"Data type of kernel: {type(K)}")
    K = 0.5*(K + K.T)
 
    kde = np.asarray(K.sum(axis=1)).ravel()
 
    u = (target_measure**(0.5)) / kde
    U = sps.spdiags(u, 0, num_samples, num_samples) 
    W = U @ K @ U
    stationary = np.asarray(W.sum(axis=1)).ravel()
    inv_stationary = stationary**(-1)
    P = sps.spdiags(inv_stationary, 0, num_samples, num_samples) @ W 
    L = (P - sps.eye(num_samples, num_samples))/epsilon

    return [K, L]

def solve_committor(L, B, C, num_samples):

    ### Solve Committor
    Lcb = L[C, :][:, B]
    Lcc = L[C, :][:, C]

    q = np.zeros(num_samples)
    q[B] = 1
    #row_sum = np.asarray(Lcb.sum(axis=1))
    row_sum = Lcb.sum(axis=1)
    q[C] = sps.linalg.spsolve(Lcc, -row_sum)
    return q

def create_laplacian_dense(data, target_measure, epsilon):

    num_features = data.shape[1]
    num_samples = data.shape[0]

    print(epsilon)
    ### Create distance matrix
    sqdists = cdist(data.T, data.T, 'sqeuclidean') 

    ### Create Kernel
    K = np.exp(-sqdists / (2*epsilon))
    

    ### Create Graph Laplacian
    kde = K.sum(axis=1)
    u = (target_measure**(0.5)) / kde
    U = np.diag(u)
    W = U @ K @ U
    stationary = W.sum(axis=1)
    inv_stationary = np.power(stationary, -1)
    P = np.diag(inv_stationary) @ W 
    L = (P - np.eye(num_samples))/epsilon

    return [K, L]

def solve_committor_dense(L, B, C, num_samples):

    ### Solve Committor
    Lcb = L[C, :]
    Lcb = Lcb[:, B]
    Lcc = L[C, :]
    Lcc = Lcc[:, C]

    q = np.zeros(num_samples)
    q[B] = 1
    row_sum = Lcb.sum(axis=1)
    q[C] = np.linalg.solve(Lcc, -row_sum)
    return q


def compute_spectrum(L, eps, num_eigvecs):
    # Symmetrize the generator 
    num_samples = L.shape[0]
    P = eps*L + sps.eye(num_samples, num_samples)
    d = np.asarray(P.sum(axis=1)).ravel()
    Dinv_onehalf =  sps.spdiags(np.power(d,-0.5), 0, num_samples, num_samples)
    D_onehalf =  sps.spdiags(np.power(d,0.5), 0, num_samples, num_samples)
    Lsymm = D_onehalf @ L @ Dinv_onehalf

    # Compute eigvals, eigvecs 
    evals, evecs = sps.linalg.eigsh(Lsymm, k=num_eigvecs, which='SM')

    # Convert back to L^2 norm-1 eigvecs of L 
    evecs = (Dinv_onehalf) @ evecs
    evecs /= (np.sum(evecs**2, axis=0))**(0.5)
    
    idx = evals.argsort()[::-1][1:]     # Ignore first eigval / eigfunc
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    return evecs, evals

def Ksum_test_unweighted(eps_vals, data):

    num_idx = eps_vals.shape[0]
    Ksum = np.zeros(num_idx)
    chi_log_analytical = np.zeros(num_idx)
       
    num_features = data.shape[1]
    num_samples = data.shape[0]
    
    ### Create distance matrix
    neigh = NearestNeighbors(n_neighbors=64, metric='sqeuclidean')
    neigh.fit(data)
    sqdists = neigh.kneighbors_graph(data, mode="distance") 

    for i in range(num_idx):
        # Construct sparsified sqdists, kernel and generator with radius nearest neighbors 
        epsilon = eps_vals[i]
        print(f"doing epsilon {i}")
        
        ### Create Kernel
        K = sqdists.copy()
        K.data = np.exp(-K.data / (2*epsilon))
        K = 0.5*(K + K.T) # symmetrize kernel

        ### Create Graph Laplacian
        Ksum[i] = K.sum(axis=None)

        # Compute deriv of log Ksum w.r.t log epsilon ('chi log')
        mat = K.multiply(sqdists)
        chi_log_analytical[i] = mat.sum(axis=None)  / ((2*epsilon)*Ksum[i])
        print(f"epsilon: {epsilon}")
        print(f"chi log: {chi_log_analytical[i]}")
        print("\n") 
    optimal_eps = eps_vals[np.nanargmax(chi_log_analytical)]
    effective_dim = (2*np.amax(chi_log_analytical))
    print(f"effective dim: {effective_dim}")
    return [Ksum, chi_log_analytical, optimal_eps, effective_dim]


def Ksum_test_weighted(eps_vals, eps_kde, data, target_measure, d=None):

    num_idx = eps_vals.shape[0]
    Ksum = np.zeros(num_idx)
    chi_log_analytical = np.zeros(num_idx)

    num_features = data.shape[1]
    num_samples = data.shape[0]
    
    ### Create distance matrix
    neigh = NearestNeighbors(n_neighbors=64, metric='sqeuclidean')
    neigh.fit(data)
    sqdists = neigh.kneighbors_graph(data, mode="distance") 
    print(f"Data type of squared distance matrix: {type(sqdists)}")

    ### Create KDE
    K = sqdists.copy()
    K.data = np.exp(-K.data / (2*eps_kde))
    K = 0.5*(K + K.T) # symmetrize kernel

    ### Create Graph Laplacian
    kde = np.asarray(K.sum(axis=1)).squeeze()
    if d == None:
        kde *=  (1.0/num_samples)*(2*np.pi*eps_kde)**(-num_features/2) 
    else:
        kde *=  (1.0/num_samples)*(2*np.pi*eps_kde)**(-d/2) 

    ### Create reweighting vector
    u = (target_measure**(0.5)) / kde
    U = sps.spdiags(u, 0, num_samples, num_samples) 

    for i in range(num_idx):
        # Construct sparsified sqdists, kernel and generator with radius nearest neighbors 
        epsilon = eps_vals[i]
        print(f"doing epsilon {i}")

        K = sqdists.copy()
        K.data = np.exp(-K.data / (2*epsilon))
        K = 0.5*(K + K.T) # symmetrize kernel
        W = U @ K @ U
        Ksum[i] = W.sum(axis=None)

        # Compute deriv of log Ksum w.r.t log epsilon ('chi log')
        mat = W.multiply(sqdists)
        chi_log_analytical[i] = mat.sum(axis=None)  / ((2*epsilon)*Ksum[i])
        print(f"epsilon: {epsilon}")
        print(f"chi log: {chi_log_analytical[i]}")
        print("\n") 
        optimal_eps = eps_vals[np.nanargmax(chi_log_analytical)]
    return [Ksum, chi_log_analytical, optimal_eps]

if __name__ == '__main__':
    main()