import numpy as np 
import scipy.sparse as sps
import scipy.spatial
import scipy.linalg.lapack as lapack
from sklearn.neighbors import NearestNeighbors
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
    center_a = np.array([-1, -1])
    center_b = np.array([1, 1])
    dist_a = ((data[:, 0] - center_a[0])**2 + (data[:, 1] - center_a[1])**2)**(0.5)
    dist_b = ((data[:, 0] - center_b[0])**2 + (data[:, 1] - center_b[1])**2)**(0.5)
    A = dist_a < radius
    B = dist_b < radius
    C = np.ones(num_samples, dtype=bool)
    C[A] = False
    C[B] = False

    ### Create distance matrix
    neigh = NearestNeighbors(radius = eps_radius)
    neigh.fit(data)
    #sqdists = neigh.radius_neighbors_graph(data, mode="distance") 
    kdtree = scipy.spatial.KDTree(data)
    sqdists = kdtree.sparse_distance_matrix(kdtree, eps_radius, output_type='coo_matrix')
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
    K.data = np.exp(-K.data / (2*epsilon))
    print(f"Data type of kernel: {type(K)}")

    # Check sparsity of kernel
    num_entries = K.shape[0]**2
    nonzeros_ratio = K.nnz / (num_entries)
    print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")

    ### Create Graph Laplacian
    kde = np.asarray(K.sum(axis=1)).squeeze()
    kde *=  (1.0/num_samples)*(2*np.pi*epsilon)**(-num_features/2) 
    u = kde**(-1/2)
    u[np.isnan(u)] = 0.0
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
    print(q[C_nonisolated].shape)
    print(Lcc.shape)
    print(row_sum.shape)
    q[C_nonisolated] = sps.linalg.spsolve(Lcc, -row_sum)
    q[isolated] = np.nan
    
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=1, c=q)
    plt.show()
    
    return None

if __name__ == '__main__':
    main()