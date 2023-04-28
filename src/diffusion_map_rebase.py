import numpy as np 
import scipy.sparse as sps
import scipy.spatial
import scipy.linalg.lapack as lapack
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def main():

    np.random.seed(30)
    ### Set up data
    num_features = 2
    num_samples = 10000
    data = np.random.randn(num_samples, num_features) 
    #data = np.random.uniform(-2, 2, size=(num_samples, num_features)) 

    plt.scatter(data[:, 0], data[:, 1], s=0.1)

    ### Set parameters
    # For now, sparse only, and euclidean distances only
    # radius nearest neighbors only
    epsilon = 0.02
    eps_radius = 3*np.sqrt(epsilon)
    #target_measure = None 
    radius = 0.1

    ### Set up A, B
    center_a = np.array([-1, -1])
    center_b = np.array([1, 1])
    dist_a = ((data[:, 0] - center_a[0])**2 + (data[:, 1] - center_a[1])**2)**(0.5)
    dist_b = ((data[:, 0] - center_b[0])**2 + (data[:, 1] - center_b[1])**2)**(0.5)
    A = dist_a < radius
    B = dist_b < radius

    norms = (data[:, 0]**2  + data[:, 1]**2)**(0.5)
    A = norms < 0.1
    B = norms > 2
    
    C = np.ones(num_samples, dtype=bool)
    C[A] = False
    C[B] = False

    ### Create distance matrix
    neigh = NearestNeighbors(radius = eps_radius)
    neigh.fit(data)
    sqdists = neigh.radius_neighbors_graph(data, mode="distance") 
    #kdtree = scipy.spatial.KDTree(data)
    #sqdists = kdtree.sparse_distance_matrix(kdtree, eps_radius, output_type='coo_matrix')
    print(f"Data type of squared distance matrix: {type(sqdists)}")

    # Find isolated points
    isolated = np.asarray(sqdists.sum(axis=1)).squeeze() == 0
    print(isolated)
    nonisolated = ~isolated

    A_nonisolated = np.logical_and(A, nonisolated) 
    B_nonisolated = np.logical_and(B, nonisolated)
    C_nonisolated = np.logical_and(C, nonisolated)

    ### Create Kernel
    K = sqdists.copy()
    K.data = np.exp(-K.data**2 / (2*epsilon))
    print(f"Data type of kernel: {type(K)}")

    # Check sparsity of kernel
    num_entries = K.shape[0]**2
    nonzeros_ratio = K.nnz / (num_entries)
    print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")

    ### Create Graph Laplacian
    kde = np.asarray(K.sum(axis=1)).squeeze()
    kde *=  (1.0/num_samples)*(2*np.pi*epsilon)**(-num_features/2) 
    u = kde**(-1/2)
    U = sps.spdiags(u, 0, num_samples, num_samples) 
    W = U @ K @ U
    stationary = np.asarray(W.sum(axis=1)).squeeze()
    P = sps.spdiags(1.0/stationary, 0, num_samples, num_samples) @ W 
    L = (P - sps.eye(num_samples, num_samples))/epsilon

    ### Solve Committor
    Lcb = L[C_nonisolated, :]
    Lcb = Lcb[:, B_nonisolated]
    Lcc = L[C_nonisolated, :]
    Lcc = Lcc[:, C_nonisolated]
    q = np.zeros(num_samples)
    q[B] = 1
    row_sum = np.asarray(Lcb.sum(axis=1)).squeeze()
    q[C_nonisolated] = sps.linalg.spsolve(Lcc, -row_sum)
    q[isolated] = np.nan
    
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=1, c=q)

    ### Gradients
    Q = sps.spdiags(q, 0, num_samples, num_samples)
    F = sps.spdiags(data[:,0], 0, num_samples, num_samples)
    G = sps.spdiags(data[:,1], 0, num_samples, num_samples)
    H = sps.spdiags(kde, 0, num_samples, num_samples)
    curr = np.zeros((num_samples, 2))
    curr[:, 0]  = np.asarray((H @ (L@Q@F - F@L@Q - Q@L@F + Q@F@L)).sum(axis=1)).squeeze()
    curr[:, 1]  = np.asarray((H @ (L@Q@G - G@L@Q - Q@L@G + Q@G@L)).sum(axis=1)).squeeze()

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1],c=((curr[:, 0]**2 + curr[:, 1]**2)**(0.5)), cmap='turbo', s=10)
    plt.quiver(data[:, 0], data[:, 1], curr[:, 0], curr[:, 1], angles='xy', scale_units='xy', headwidth=2, minlength=0)
    plt.show()

    return None

# sparse code
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
 
    #kde = np.asarray(K.sum(axis=1)).ravel()
    kde = K.sum(axis=1)
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
    Dinv_onehalf =  sps.spdiags(np.power(stationary, -0.5), 0, num_samples, num_samples)
    D_onehalf =  sps.spdiags(np.power(stationary,0.5), 0, num_samples, num_samples)
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



if __name__ == '__main__':
    main()