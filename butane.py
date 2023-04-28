import numpy as np 
import scipy.sparse as sps
import scipy.spatial
import scipy.linalg.lapack as lapack
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.linalg.lapack import dpstrf
from scipy.spatial.distance import cdist

def main():

    # Load data
    fname = "systems/butane/data/butane_300K.npz"
    inData = np.load(fname)
    print("Keys in data:")
    print(list(inData.keys()))

    data = inData["data"]
    #data = inData["data_all_atom"]
    print("Data shape from trajectory:")
    print(data.shape)
    dihedrals = inData["dihedrals"]
    potential = inData["potential"]
    kbT = inData["kbT"]
    print(f"kbT for data:{kbT}")
    #kbT_roomtemp = inData["kbT_roomtemp"]
    kbT_roomtemp = kbT

    print(f"kbT for room temperature:{kbT_roomtemp}")

    # Load up delta net indices
    #fname = "systems/butane/data/butane_metad_deltanet.npz"
    #delta_idx = np.load(fname)["delta_idx"]
    print(data.shape)
    # Adjust dihedral angles from [0, pi] for convenience
    dihedrals_shift = dihedrals.copy()
    dihedrals_shift[dihedrals < 0] = dihedrals_shift[dihedrals < 0] + 2*np.pi 

    # Define Target Measure
    target_measure = np.exp(-potential/(kbT_roomtemp))
    
    # Subsample dataset
    indices = np.arange(data.shape[0])
    sub_indices = indices[::2]
    #sub_indices = delta_idx
    
    new_data = data[sub_indices, :]
    target_measure = np.exp(-potential[sub_indices]/(kbT_roomtemp))
    num_samples = new_data.shape[0]
    num_features = new_data.shape[1]
    print(f"number of samples for subsampled data:{num_samples}") 
    #eps_vals = 2.0**np.arange(-20, 0, 1)
    #[Ksum, chi_log_analytical, eps_kde, effective_dim] = Ksum_test_unweighted(eps_vals, new_data)
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
    #[Ksum, chi_log_analytical, optimal_eps] = Ksum_test_weighted(eps_vals, eps_kde, new_data, target_measure, d=effective_dim)

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

    radius = 0.1

    OPTION = 1
    
    if OPTION == 0: 
        Acenter = -np.pi/3
        A = np.abs(dihedrals[sub_indices] - Acenter) < radius
        B = np.logical_or(np.abs(dihedrals[sub_indices] - np.pi) < radius, np.abs(dihedrals[sub_indices] + np.pi) < radius)
    elif OPTION == 1:
        Acenter = -np.pi/3
        Bcenter = np.pi/3
        A = np.abs(dihedrals[sub_indices] - Acenter) < radius
        B = np.abs(dihedrals[sub_indices] - Bcenter) < radius
    elif OPTION == 2:
        Acenter = -np.pi/3
        A = np.abs(dihedrals[sub_indices] - Acenter) < radius
        B = np.logical_or(np.abs(dihedrals[sub_indices] - (2/3)*np.pi) < np.pi/3 + radius , np.abs(dihedrals[sub_indices] + np.pi) < radius)
    elif OPTION == 3:
        A = np.abs(dihedrals[sub_indices]) < np.pi/3 + radius
        B = np.logical_or(np.abs(dihedrals[sub_indices] - np.pi) < radius, np.abs(dihedrals[sub_indices] + np.pi) < radius)
    
    elif OPTION == 4:
        A = np.logical_or(np.abs(dihedrals[sub_indices] - np.pi/3) < radius,np.abs(dihedrals[sub_indices] + np.pi/3) < radius)
        B = np.logical_or(np.abs(dihedrals[sub_indices] - np.pi) < radius, np.abs(dihedrals[sub_indices] + np.pi) < radius)
    

    print(f"samples in A:{new_data[A, :].shape[0]}")
    print(f"samples in B:{new_data[B, :].shape[0]}")
    C = np.ones(num_samples, dtype=bool)
    C[A] = False
    C[B] = False

    # Run diffusion map
    #epsilon  = optimal_eps
    epsilon = 0.01

    [stationary, _, L] = create_laplacian_sparse(new_data, target_measure, epsilon=epsilon, n_neighbors=64)
    q = solve_committor_sparse(L, B, C, num_samples)
    plt.figure()
    plt.xlabel("dihedral")
    plt.ylabel("committor")
    plt.scatter(dihedrals[sub_indices], q, s=0.1)
    plt.title(f"committor, option={OPTION}, eps={epsilon}")

    eigvecs, _ = compute_spectrum_sparse(L, stationary, num_eigvecs=4)
    plt.figure()
    plt.xlabel("dihedral")
    plt.ylabel("eigvec 1")
    plt.scatter(dihedrals[sub_indices], eigvecs[:, 0], s=0.1)
    plt.title(f"eigvec1")

    plt.figure()
    plt.xlabel("dihedral")
    plt.ylabel("eigvec 2")
    plt.scatter(dihedrals[sub_indices], eigvecs[:, 1], s=0.1)
    plt.title(f"eigvec2")

    plt.figure()
    plt.xlabel("dihedral")
    plt.ylabel("eigvec 3")
    plt.scatter(dihedrals[sub_indices], eigvecs[:, 2], s=0.1)
    plt.title(f"eigvec3")


    plt.show() 
    return None

def create_laplacian_sparse(data, target_measure, epsilon, n_neighbors):

    num_features = data.shape[1]
    num_samples = data.shape[0]

    ### Create distance matrix
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='sqeuclidean')
    neigh.fit(data)
    sqdists = neigh.kneighbors_graph(data, mode="distance") 
    print(f"Data type of squared distance matrix: {type(sqdists)}")

    ### Create Kernel
    K = sqdists.copy()
    K.data = np.exp(-K.data / (2*epsilon))
    K = 0.5*(K + K.T)
 
    kde = np.asarray(K.sum(axis=1)).ravel()
    #kde *=  (1.0/num_samples)*(2*np.pi*epsilon)**(-num_features/2) 
    
    # Check sparsity of kernel
    num_entries = K.shape[0]**2
    nonzeros_ratio = K.nnz / (num_entries)
    print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")

    ### Create Graph Laplacian
    u = (target_measure**(0.5)) / kde
    U = sps.spdiags(u, 0, num_samples, num_samples) 
    W = U @ K @ U
    stationary = np.asarray(W.sum(axis=1)).ravel()
    inv_stationary = np.power(stationary, -1)
    P = sps.spdiags(inv_stationary, 0, num_samples, num_samples) @ W 
    L = (P - sps.eye(num_samples, num_samples))/epsilon

    return [stationary, K, L]

def solve_committor_sparse(L, B, C, num_samples):

    ### Solve Committor
    Lcb = L[C, :][:, B]
    Lcc = L[C, :][:, C]

    q = np.zeros(num_samples)
    q[B] = 1
    row_sum = np.asarray(Lcb.sum(axis=1)).ravel()
    q[C] = sps.linalg.spsolve(Lcc, -row_sum)
    return q

def compute_spectrum_sparse(L, stationary, num_eigvecs):
    # Symmetrize the generator 
    num_samples = L.shape[0]
    Dinv_onehalf =  sps.spdiags(stationary**(-0.5), 0, num_samples, num_samples)
    D_onehalf =  sps.spdiags(stationary**(0.5), 0, num_samples, num_samples)
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

def create_laplacian_dense(data, target_measure, epsilon):

    num_features = data.shape[1]
    num_samples = data.shape[0]

    ### Create distance matrix
    sqdists = cdist(data, data, 'sqeuclidean') 

    ### Create Kernel
    K = np.exp(-sqdists / (2.0*epsilon))

    ### Create Graph Laplacian
    kde = K.sum(axis=1)
    u = (target_measure**(0.5)) / kde
    U = np.diag(u)
    W = U @ K @ U
    stationary = W.sum(axis=1)
    P = np.diag(stationary**(-1)) @ W 
    L = (P - np.eye(num_samples))/epsilon

    return [stationary, K, L]

def solve_committor_dense(L, B, C, num_samples):

    ### Solve Committor
    Lcb = L[C, :][:, B]
    Lcc = L[C, :][:, C]

    q = np.zeros(num_samples)
    q[B] = 1
    row_sum = Lcb.sum(axis=1)
    q[C] = np.linalg.solve(Lcc, -row_sum)
    return q

def compute_spectrum_dense(L, stationary, num_eigvecs):
    # Symmetrize the generator 
    Dinv_onehalf =  np.diags(stationary**(-0.5))
    D_onehalf =  np.diag(stationary**(0.5))
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