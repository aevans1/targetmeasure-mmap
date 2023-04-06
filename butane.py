import numpy as np 
import scipy.sparse as sps
import scipy.spatial
import scipy.linalg.lapack as lapack
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def main():

    # Load data
    fname = "systems/butane/data/butane_metad_alt.npz"
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
    new_data = data[-10000:, :]
    target_measure = np.exp(-potential[-10000:]/(kbT_roomtemp))
    num_features = new_data.shape[1]
    num_samples = new_data.shape[0]

    #eps_vals = 2.0**np.arange(-20, 0, 1)
    #[Ksum, chi_log_analytical, optimal_eps, effective_dim] = Ksum_test_unweighted(eps_vals, new_data)
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

    # Define A,B sets, based on [0, pi] shifted dihedrals
    radius = 0.3
    Acenter = -np.pi/3
    #Bcenter = np.pi/3
    B = np.logical_or(np.abs(dihedrals[-10000:] - np.pi) < radius, np.abs(dihedrals[-10000:] + np.pi) < radius)
    A = np.abs(dihedrals[-10000:] - Acenter) < radius
    #B = np.abs(dihedrals[-10000:] - Bcenter) < radius
    C = np.ones(num_samples, dtype=bool)
    C[A] = False
    C[B] = False

    # Run diffusion map
    #epsilon  = optimal_eps
    epsilon = 8E-5

    [_, L] = create_laplacian(new_data, target_measure, epsilon)
    q = solve_committor(L, B, C, num_samples)
    plt.figure()
    plt.scatter(dihedrals[-10000:], q, s=0.1)
    plt.show()
    return None

def create_laplacian(data, target_measure, epsilon):

    num_features = data.shape[1]
    num_samples = data.shape[0]

    ### Create distance matrix
    neigh = NearestNeighbors(n_neighbors=512, metric='sqeuclidean')
    neigh.fit(data)
    sqdists = neigh.kneighbors_graph(data, mode="distance") 
    print(f"Data type of squared distance matrix: {type(sqdists)}")

    ### Create Kernel
    K = sqdists.copy()
    K.data = np.exp(-K.data / (2*epsilon))
    print(f"Data type of kernel: {type(K)}")
    K = K.minimum(K.T) # symmetrize kernel

    # Check sparsity of kernel
    num_entries = K.shape[0]**2
    nonzeros_ratio = K.nnz / (num_entries)
    print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")

    ### Create Graph Laplacian
    kde = np.asarray(K.sum(axis=1)).ravel()
    #kde *=  (1.0/num_samples)*(2*np.pi*epsilon)**(-num_features/2) 
    u = (target_measure**(0.5)) / kde
    U = sps.spdiags(u, 0, num_samples, num_samples) 
    W = U @ K @ U
    stationary = np.asarray(W.sum(axis=1)).ravel()
    inv_stationary = np.power(stationary, -1)
    P = sps.spdiags(inv_stationary, 0, num_samples, num_samples) @ W 
    L = (P - sps.eye(num_samples, num_samples))/epsilon

    return [K, L]

def solve_committor(L, B, C, num_samples):

    ### Solve Committor
    Lcb = L[C, :]
    Lcb = Lcb[:, B]
    Lcc = L[C, :]
    Lcc = Lcc[:, C]
    q = np.zeros(num_samples)
    q[B] = 1
    row_sum = np.asarray(Lcb.sum(axis=1)).ravel()
    q[C] = sps.linalg.spsolve(Lcc, -row_sum)
    return q


def Ksum_test_unweighted(eps_vals, data):

    num_idx = eps_vals.shape[0]
    Ksum = np.zeros(num_idx)
    chi_log_analytical = np.zeros(num_idx)
    
    for i in range(num_idx):
        # Construct sparsified sqdists, kernel and generator with radius nearest neighbors 
        epsilon = eps_vals[i]
        print(f"doing epsilon {i}")
       
        num_features = data.shape[1]
        num_samples = data.shape[0]
        
        ### Create distance matrix
        neigh = NearestNeighbors(n_neighbors=512, metric='sqeuclidean')
        neigh.fit(data)
        sqdists = neigh.kneighbors_graph(data, mode="distance") 

        ### Create Kernel
        K = sqdists.copy()
        K.data = np.exp(-K.data / (2*epsilon))
        #K = K.minimum(K.T) # symmetrize kernel
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


def Ksum_test(eps_vals, data, target_measure, d=None):

    num_idx = eps_vals.shape[0]
    Ksum = np.zeros(num_idx)
    chi_log_analytical = np.zeros(num_idx)
    
    for i in range(num_idx):
        # Construct sparsified sqdists, kernel and generator with radius nearest neighbors 
        epsilon = eps_vals[i]
        print(f"doing epsilon {i}")
       
        num_features = data.shape[1]
        num_samples = data.shape[0]
        
        ### Create distance matrix
        neigh = NearestNeighbors(n_neighbors=512, metric='sqeuclidean')
        neigh.fit(data)
        sqdists = neigh.kneighbors_graph(data, mode="distance") 
        print(f"Data type of squared distance matrix: {type(sqdists)}")

        ### Create Kernel
        K = sqdists.copy()
        K.data = np.exp(-K.data / (2*epsilon))
        print(f"Data type of kernel: {type(K)}")
        K = K.minimum(K.T) # symmetrize kernel

        # Check sparsity of kernel
        num_entries = K.shape[0]**2
        nonzeros_ratio = K.nnz / (num_entries)
        print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")

        ### Create Graph Laplacian
        kde = np.asarray(K.sum(axis=1)).squeeze()
        #if d == None:
        #    kde *=  (1.0/num_samples)*(2*np.pi*epsilon)**(-num_features/2) 
        #else:
        #    kde *=  (1.0/num_samples)*(2*np.pi*epsilon)**(-d/2) 
        u = (target_measure**(0.5)) / kde
        U = sps.spdiags(u, 0, num_samples, num_samples) 
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