import numpy as np 
import scipy.sparse as sps
import scipy.spatial
import scipy.linalg.lapack as lapack
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def main():

    # Load data
    fname = "systems/butane/data/butane_hightemp.npz"
    inData = np.load(fname)
    print("Keys in data:")
    print(list(inData.keys()))

    data = inData["data"]
    print("Data shape from trajectory:")
    print(data.shape)
    dihedrals = inData["dihedrals"]
    potential = inData["potential"]
    kbT = inData["kbT"]
    print(f"kbT for data:{kbT}")
    kbT_roomtemp = inData["kbT_roomtemp"]
    print(f"kbT for room temperature:{kbT_roomtemp}")

    # Adjust dihedral angles from [0, pi] for convenience
    dihedrals_shift = dihedrals.copy()
    dihedrals_shift[dihedrals < 0] = dihedrals_shift[dihedrals < 0] + 2*np.pi 

    # Define Target Measure
    target_measure = np.exp(-potential/(kbT_roomtemp))

    # Subsample dataset
    sub = 10
    new_data = data[::sub, :]
    target_measure = np.exp(-potential[::sub]/(kbT_roomtemp))
    num_features = new_data.shape[1]
    num_samples = new_data.shape[0]

    eps_vals = 2.0**np.arange(-12, -4, 1)
    [Ksum, chi_log_analytical, optimal_eps] = Ksum_test(eps_vals, new_data, target_measure)
    print(f"optimal_eps = {optimal_eps}")

    plt.figure()
    plt.plot(eps_vals, Ksum)
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.axvline(x=optimal_eps, ls='--')

    plt.figure()
    plt.plot(eps_vals, chi_log_analytical)
    plt.title("log Ksums")
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.title("dlog_Sum/dlog_eps")
    plt.axvline(x=optimal_eps, ls='--')
    #plt.savefig(fname, dpi=300)
    plt.show() 


    # Define A,B sets, based on [0, pi] shifted dihedrals
    radius = 0.2
    Acenter = np.pi
    Bcenter = np.pi/3

    A = np.abs(dihedrals[::sub] - Acenter) < radius
    B = np.abs(dihedrals[::sub] - Bcenter) < radius

    C = np.ones(num_samples, dtype=bool)
    C[A] = False
    C[B] = False

    # Run diffusion map
    epsilon  = optimal_eps

    [_, L, nonisolated] = create_laplacian(new_data, target_measure, epsilon)
    q = solve_committor(L, B, C, nonisolated, num_samples)
    plt.scatter(dihedrals_shift[::sub], q)
    plt.show()
    return None

def create_laplacian(data, target_measure, epsilon):

    num_features = data.shape[1]
    num_samples = data.shape[0]

    ### Create distance matrix
    eps_radius = np.sqrt(epsilon)
    neigh = NearestNeighbors(radius = eps_radius)
    neigh.fit(data)
    sqdists = neigh.radius_neighbors_graph(data, mode="distance") 
    print(f"Data type of squared distance matrix: {type(sqdists)}")

    # Find isolated points
    isolated = np.asarray(sqdists.sum(axis=1)).squeeze() == 0
    nonisolated = ~isolated

    ### Create Kernel
    # Kernel should only look at nonisolated points
    K = sqdists.copy()[nonisolated, :]
    K = K[:, nonisolated]
    num_nodes = K.shape[0]

    K.data = np.exp(-K.data**2 / (2*epsilon))
    print(f"Data type of kernel: {type(K)}")

    # Check sparsity of kernel
    num_entries = K.shape[0]**2
    nonzeros_ratio = K.nnz / (num_entries)
    print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")

    ### Create Graph Laplacian
    kde = np.asarray(K.sum(axis=1)).squeeze()
    #kde *=  (1.0/num_nodes)*(2*np.pi*epsilon)**(-num_features/2) 
    u = (target_measure[nonisolated]**(0.5)) / kde
    U = sps.spdiags(u, 0, num_nodes, num_nodes) 
    W = U @ K @ U
    stationary = np.asarray(W.sum(axis=1)).squeeze()
    P = sps.spdiags(1.0/stationary, 0, num_nodes, num_nodes) @ W 
    L = (P - sps.eye(num_nodes, num_nodes))/epsilon

    return [K, L, nonisolated]

def solve_committor(L, B, C, nonisolated, num_samples):

    B_nonisolated = B[nonisolated]
    C_nonisolated = C[nonisolated]

    ### Solve Committor
    Lcb = L[C_nonisolated, :]
    Lcb = Lcb[:, B_nonisolated]
    Lcc = L[C_nonisolated, :]
    Lcc = Lcc[:, C_nonisolated]
    q = np.zeros(num_samples)
    q[B] = 1
    row_sum = np.asarray(Lcb.sum(axis=1)).squeeze()
    q[np.logical_and(C,nonisolated)] = sps.linalg.spsolve(Lcc, -row_sum)
    q[~nonisolated] = np.nan
    return q

def Ksum_test(eps_vals, data, target_measure):

    num_idx = eps_vals.shape[0]
    Ksum = np.zeros(num_idx)
    chi_log_analytical = np.zeros(num_idx)
    
    for i in range(num_idx):
        # Construct sparsified sqdists, kernel and generator with radius nearest neighbors 
        epsilon = eps_vals[i]
        print(f"doing epsilon {i}")
        radius = None

        # put a maximum for the radius in radius-nearest neighbors
        if epsilon > 0.5:
            eps_radius = (3*np.sqrt(0.5))**(0.5)
        
        num_features = data.shape[1]
        num_samples = data.shape[0]

        ### Create distance matrix
        eps_radius = np.sqrt(epsilon)
        neigh = NearestNeighbors(radius = eps_radius)
        neigh.fit(data)
        sqdists = neigh.radius_neighbors_graph(data, mode="distance") 
    
        # Find isolated points, adjust distance matrix for nonisolated points
        isolated = np.asarray(sqdists.sum(axis=1)).squeeze() == 0
        nonisolated = ~isolated
        sqdists = sqdists[nonisolated, :]
        sqdists = sqdists[:, nonisolated] 

        ### Create Kernel
        # Kernel should only look at nonisolated points
        K = sqdists.copy()
        K.data = np.exp(-K.data**2 / (2*epsilon))
        num_nodes = K.shape[0]
        
        num_entries = K.shape[0]**2
        nonzeros_ratio = K.nnz / (num_entries)
        print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")

        ### Create target measure symmetric kernel
        kde = np.asarray(K.sum(axis=1)).squeeze()
        kde *=  (1.0/num_samples)*(2*np.pi*epsilon)**(-num_features/2) 
        u = (target_measure[nonisolated]**(0.5)) / kde
        U = sps.spdiags(u, 0, num_nodes, num_nodes) 
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