import numpy as np 
import scipy.sparse as sps
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def main():

    # Load data
    fname = "systems/butane/data/butane_metad.npz"
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
    kbT_roomtemp = inData["kbT_roomtemp"]

    print(f"kbT for room temperature:{kbT_roomtemp}")

    # Load up delta net indices
    fname = "systems/butane/data/butane_metad_deltanet.npz"
    delta_idx = np.load(fname)["delta_idx"]

    # Define Target Measure
    target_measure = np.exp(-potential/(kbT_roomtemp))
    
    # Subsample dataset (in time or in space(deltanet) )
    indices = np.arange(data.shape[0])
    #sub_indices = indices[::20]
    sub_indices = delta_idx
    
    new_data = data[sub_indices, :]
    target_measure = np.exp(-potential[sub_indices]/(kbT_roomtemp))
    num_samples = new_data.shape[0]
    num_features = new_data.shape[1]
    print(f"number of samples for subsampled data:{num_samples}") 

    # Try out Ksum test, likely to no avail...
    #eps_vals = 2.0**np.arange(-20, 0, 1)
    #[Ksum, chi_log_analytical, optimal_eps, effective_dim] = Ksum_test_unweighted(eps_vals, new_data, n_neighs=64)
    #print(f"eps_kde = {optimal_eps}")
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
    #plt.axvline(x=eps_kde, ls='--')
    ##plt.savefig(fname, dpi=300)
    
    # Try out different A, B sets for committor 
    radius = 0.1
    OPTION = 2
    if OPTION == 0: 
        Acenter = np.pi
        Bcenter = 5*(np.pi/3)
        A = np.abs(dihedrals[sub_indices] - Acenter) < radius
        B = np.abs(dihedrals[sub_indices] - Bcenter) < radius
    elif OPTION == 1:
        Acenter = np.pi/3
        Bcenter = 5*(np.pi/3)
        A = np.abs(dihedrals[sub_indices] - Acenter) < radius
        B = np.abs(dihedrals[sub_indices] - Bcenter) < radius
    elif OPTION == 2:
        Acenter = np.pi
        A = np.abs(dihedrals[sub_indices] - Acenter) < radius
        B = np.logical_or(np.abs(dihedrals[sub_indices] - np.pi/3) < radius,np.abs(dihedrals[sub_indices] - 5*(np.pi/3)) < radius)
    print(f"samples in A:{new_data[A, :].shape[0]}")
    print(f"samples in B:{new_data[B, :].shape[0]}")
    C = np.ones(num_samples, dtype=bool)
    C[A] = False
    C[B] = False

    # Run diffusion map
    epsilon = 0.1
    DENSE = True

    if DENSE:
        print("dense dmaps!")
        [stationary, K, L] = create_laplacian_dense(new_data, target_measure, epsilon=epsilon)
        q = solve_committor_dense(L, B, C, num_samples)
        diffcoords, eigvecs, eigvals = compute_spectrum_dense(L, stationary, num_eigvecs=4)
        rate = compute_rate_dense(L, K, target_measure, q, beta=(1/kbT), effective_dim=12, epsilon=epsilon, C=C)
        print(rate)

    else:
        n_neighbors = 512
        print("sparse dmaps!")
        [stationary, K, L] = create_laplacian_sparse(new_data, target_measure, epsilon=epsilon, n_neighbors=n_neighbors)
        q = solve_committor_sparse(L, B, C, num_samples)
        diffcoords, eigvecs, eigvals = compute_spectrum_sparse(L, stationary, num_eigvecs=4)
        #fric = 0.01 #friction in femtoseconds
        #mass = 12.01 #mass in amus
        #rate = (1/fric)*mass*compute_rate_sparse(L, K, target_measure, q, beta=(1/kbT), effective_dim=12, epsilon=epsilon)
        rate = compute_rate_sparse(L, K, target_measure, q, beta=(1/kbT), effective_dim=12, epsilon=epsilon, C=C)
        print(rate)

    plt.figure()
    plt.xlabel("dihedral")
    plt.ylabel("committor")
    plt.scatter(dihedrals[sub_indices], q, s=0.1)
    plt.title(f"committor, option={OPTION}, eps={epsilon}")

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

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    s=ax.scatter(diffcoords[:, 0], diffcoords[:, 1], diffcoords[:, 2], c=dihedrals[sub_indices], cmap='hsv', s=0.5)
    ax.set_xlabel('eigvec1')
    ax.set_ylabel('eigvec2')
    ax.set_zlabel('eigvec3')
    fig.colorbar(s)

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
    dmap = np.dot(evecs, np.diag(np.sqrt(-1./evals)))
    return dmap, evecs, evals

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

    Lcb = L[C, :][:, B]
    Lcc = L[C, :][:, C]

    q = np.zeros(num_samples)
    q[B] = 1
    row_sum = Lcb.sum(axis=1).ravel()
    q[C] = np.linalg.solve(Lcc, -row_sum)
    return q

def compute_spectrum_dense(L, stationary, num_eigvecs):
    # Symmetrize the generator 
    Dinv_onehalf =  np.diag(stationary**(-0.5))
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
    dmap = np.dot(evecs, np.diag(np.sqrt(-1./evals)))
    return dmap, evecs, evals


def Ksum_test_unweighted(eps_vals, data, n_neighs):

    num_idx = eps_vals.shape[0]
    Ksum = np.zeros(num_idx)
    chi_log_analytical = np.zeros(num_idx)
       
    num_features = data.shape[1]
    num_samples = data.shape[0]
    
    ### Create distance matrix
    neigh = NearestNeighbors(n_neighbors=n_neighs, metric='sqeuclidean')
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
    neigh = NearestNeighbors(n_neighbors=512, metric='sqeuclidean')
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

def compute_rate_sparse(L, K, target_measure, q, beta, effective_dim, epsilon, C):
    N = L.shape[0]
    kde = np.asarray(K.sum(axis=1)).ravel()
    #kde *=  (1.0/N)*(2*np.pi*epsilon)**(-effective_dim/2) 
    kde *= (1./np.sum(kde))
    Z_dmap = (1.0/N)*np.sum(target_measure / kde)
    weight_Zdmap = (target_measure/(kde*Z_dmap)).flatten()
    
    # Note: for i in C, both sum_j L_ij = 0 and sum_j Lij qj = 0, so sum_ij L_ij (q_j - q_i)**2 = sum_ij L_ij q_j*2 
    rate = (1/beta)*(1/np.count_nonzero(C))*np.sum(weight_Zdmap[C]*L[C, :].toarray().dot(q**2))
    return rate

def compute_rate_dense(L, K, target_measure, q, beta, effective_dim, epsilon, C):
    N = L.shape[0]
    kde = np.asarray(K.sum(axis=1)).ravel()
    #kde *=  (1.0/N)*(2*np.pi*epsilon)**(-effective_dim/2) 
    kde *= (1./np.sum(kde))
    Z_dmap = (1.0/N)*np.sum(target_measure / kde)
    weight_Zdmap = (target_measure/(kde*Z_dmap)).flatten()
    
    # Note: for i in C, both sum_j L_ij = 0 and sum_j Lij qj = 0, so sum_ij L_ij (q_j - q_i)**2 = sum_ij L_ij q_j*2 
    rate = (1/beta)*(1/np.count_nonzero(C))*np.sum(weight_Zdmap[C]*L[C, :].dot(q**2))
    return rate

if __name__ == '__main__':
    main()