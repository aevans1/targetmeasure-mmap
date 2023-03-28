import numpy as np 
import scipy.sparse as sps
import scipy.linalg.lapack as lapack
from sklearn.neighbors import NearestNeighbors

def main():
    num_features = 5
    num_samples = 10
    data = np.random.rand(num_samples, num_features)

    # For now, sparse only, and euclidean distances only
    # radius nearest neighbors only
    epsilon = 0.1
    eps_radius = 3*np.sqrt(epsilon)
    #target_measure = None 

    neigh = NearestNeighbors(radius = eps_radius)
    neigh.fit(data)
    sqdists = neigh.radius_neighbors_graph(data, mode="distance") 
    print(sqdists.toarray())


    K = sqdists.copy()
    K.data = np.exp(-K.data / (2*epsilon))

    # Check sparsity of kernel
    num_entries = K.shape[0]**2
    nonzeros_ratio = K.nnz / (num_entries)
    print(f"Ratio of nonzeros to zeros in kernel matrix: {nonzeros_ratio}")
    print(K.toarray())
    return None

if __name__ == '__main__':
    main()