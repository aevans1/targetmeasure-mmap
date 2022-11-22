from os import remove
import numpy as np 
import scipy.sparse as sps
from scipy.linalg.lapack import clapack as cla
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance as sp_dist

import src.helpers as helpers

import matplotlib.pyplot as plt
def main():
    return None

class TargetMeasureDiffusionMap(object):
    r"""
    Class for computing the diffusion map of a given data set. 
    """

    def __init__(self, epsilon, radius=None, n_neigh=None, neigh_mode='RNN',
                 num_evecs=1, rho=None, target_measure=None, 
                 remove_isolated=True, pbc_dims=None):
        r""" Initialize diffusion map object with basic hyperparameters."""    
        self.epsilon = epsilon
        self.radius = radius
        self.n_neigh = n_neigh
        self.neigh_mode = neigh_mode
        self.num_evecs = num_evecs
        self.rho = rho
        self.target_measure = target_measure
        self.remove_isolated = remove_isolated
        self.pbc_dims = pbc_dims

    def fit_transform(self, data):
        r""" Fits the data as in fit() method, and returns diffusion map 

        Parameters
        ----------
        data: array (num features, num samples)

        """  
        
        self.fit(data) 

        return self.dmap

    def fit(self, data):
        r""" Computes the generator and diffusion map for input data

        Parameters
        ----------
        data: array (num features, num samples)

        """  
        self.construct_generator(data) 
        dmap, evecs, evals = self._construct_diffusion_coords()

        self.dmap = dmap
        self.evecs = evecs
        self.evals = evals

        return self

    def _construct_diffusion_coords(self):
        r""" Description Here

        Parameters
        ----------
        Returns
        -------
        dmap : array, (num features, desired number of evecs)
            ith column is the ith `diffusion coordinate'
        evecs : array,  (num features, desired number of evecs)
            ith column is the ith eigenvector of generator
        evals : list, (1, desired number of evecs)
            ith entry is the ith eigval of the generator
        """    
        # Compute eigvals, eigvecs 
        print("computing eigvec matrix") 
        evals, evecs = sps.linalg.eigs(self.L, self.num_evecs + 1, which='LR')
        idx = evals.argsort()[::-1][1:]     # Ignore first eigval / eigfunc
        evals = np.real(evals[idx])
        evecs = np.real(evecs[:, idx])
        dmap = np.dot(evecs, np.diag(np.sqrt(-1./evals)))

        return dmap, evecs, evals

    def construct_generator(self, data):
        K = self._construct_kernel(data)
        N = K.shape[-1]     # Number of data points
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]

        if self.rho is None:
            self.rho = np.array(K.sum(axis=1)).ravel()
        
        if self.target_measure is None:
            self.target_measure = self.rho

        #########################################################
        # Make sure we are using correct indices of the subgraph
        if len(self.rho) > N: 
            self.rho = self.rho[nonisolated_bool] 
        if len(self.target_measure) > N: 
            self.target_measure = self.target_measure[nonisolated_bool]
        #########################################################

        if sps.issparse(K):
            # Make right normalizing vector
            right_normalizer = (sps.spdiags(self.rho**(-1), 0, N, N) 
                                @ sps.spdiags(self.target_measure**(0.5), 0 , N, N))
            
            K_reweight = right_normalizer @ K @ right_normalizer

            # Make left normalizing vector
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()

            rowsums_inv = np.power(rowsums, -1) 
            left_normalizer = sps.spdiags(rowsums_inv, 0, N, N)
            
            P = left_normalizer @ K_reweight

            L = (P - sps.eye(N, N))/self.epsilon

            self.stationary_measure = rowsums
            self.right_normalizer = right_normalizer 

        else:
            print("Doing dense matrix calculations")
            # Make right normalizing vector
            right_normalizer = np.diag(self.rho**(-1)).dot(np.diag(self.target_measure**(0.5)))
            
            K_reweight = right_normalizer.dot(K.dot(right_normalizer))

            # Make left normalizing vector
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            
            left_normalizer = np.diag(rowsums**(-1))
            
            #P = left_normalizer.dot(K_rnorm)
            P = left_normalizer.dot(K_reweight) 

            L = (P - np.eye(N))/self.epsilon
            self.stationary_measure = rowsums
            self.right_normalizer = right_normalizer 

        self.L = L
        return L

    def _construct_kernel(self, data, eps=None):
        r"""Construct kernel matrix of a given data set

        Takes an input data set with structure num_features x
        num_observations, constructs squared distance
        matrix and gaussian kernel

        Parameters
        ----------
        data: array (num features, num samples)
       
        Returns
        -------
        K : array (num samples, num samples)
          pair-wise kernel matrix K(i, j) is kernel
          applied to data points i, j

        """  
        if eps == None:
            eps = self.epsilon

        sqdists = self._compute_sqdists(data)
        sqdists = self._compute_nearest_neigh_graph(sqdists)
        
        # Symmetrizing square dists!
        #sqdists = 0.5*(sqdists + sqdists.T) 
        
        if sps.issparse(sqdists):
            K = sqdists.copy()
            K.data = np.exp(-K.data / (2*eps))
    
            # Check sparsity of kernel
            num_entries = K.shape[0]**2
            nonzeros_ratio = K.nnz / (num_entries)
            print(f"ratio of nonzeros: {nonzeros_ratio}")
            if nonzeros_ratio > 0.5:
                # Convert to dense matrix
                print("shifting to dense")
                #self.dense = True
                K = K.toarray()

        else:
            K = np.exp(-sqdists / (2*eps))

        # Symmetrize kernel 
        #K = 0.5*(K + K.T)
        print("NOTE: trying a new symmetrizaitons!")
        K = K.minimum(K.T)
        self.K = K

        return K

    


    def _compute_nearest_neigh_graph(self, sqdists):
        r""" Given dataset data, computes matrix of pairwise squared distances and stores sparsely based on k - nearest neighbors

        Parameters
        ----------
        data : array, (num features, num samples)
            data matrix
    
        Returns
        -------
        sqdists : csr matrix
                      sparse matrix of squared distances
        """
        if self.neigh_mode == 'KNN':
            print("computing KNN kernel")
            neigh = NearestNeighbors(n_neighbors = self.n_neigh,
                                     metric='precomputed')
            neigh.fit(sqdists)
            sqdists = neigh.kneighbors_graph(sqdists, mode='distance')
        elif self.neigh_mode == 'RNN':
            print("computing RNN kernel")
            #float_machine_eps = np.finfo(float).eps
            #eps_radius = -2*self.epsilon * np.log(float_machine_eps)  # neighbor radius for machine epsilon in kernel similarity        
            eps_radius = 3*np.sqrt(self.epsilon)
            if self.radius == None:
                self.radius = eps_radius

            neigh = NearestNeighbors(radius = self.radius,
                                     metric='precomputed')
            neigh.fit(sqdists)
            sqdists = neigh.radius_neighbors_graph(sqdists, mode='distance')
        # Find isolated indices
        if self.remove_isolated:
            row_sums = np.array(sqdists.sum(axis=1)).ravel()
            nonisolated_bool =  row_sums > 0
        else:
            nonisolated_bool = True*np.ones(sqdists.shape[0], dtype=bool) 
            print("Not leaving out any nodes") 

        # Remove isolated indices from the graph
        sqdists = sqdists[nonisolated_bool, :]
        sqdists = sqdists[:, nonisolated_bool]
        
        self.sqdists = sqdists 

        # Store subgraph of nodes which we use for the algorithm 
        subgraph = {}
        subgraph["nonisolated_bool"] = nonisolated_bool
        print(f"nodes left after removing isolated: {sqdists.shape[0]}")
        self.subgraph = subgraph
        return sqdists

    def _compute_sqdists(self, data):
        # only do vectorized computation if data is small
        # diffs tensor is shape num features**2 * num_samples
        #if (np.amax(data.shape)**2)*np.amin(data.shape) < 2E8:
        if data.shape[1] < 5000:
            diffs = data.T[np.newaxis, ...] - data.T[:, np.newaxis, ...]
            if self.pbc_dims is not None:
               # Use input pbc_dimensions for distance calculations
               diffs = helpers.periodic_restrict(diffs, self.pbc_dims)
        
            ## Construct nearest neighbors graph, sparsify square distances
            sqdists = np.sum(diffs**2, axis=-1)
        
        else:
            sqdists = np.zeros((data.shape[1], data.shape[1]))
            for i in range(sqdists.shape[0]):
                diffs_row = data[:, i, np.newaxis] - data[:, i:]
                if self.pbc_dims is not None: 
                    diffs_row = helpers.periodic_restrict(diffs_row, self.pbc_dims) 
                sqdists[i, i:] = np.sum(diffs_row**2, axis=0)
                sqdists[i, 0:i] = sqdists[0:i, i]

        return sqdists

    def _compute_kde(self, data):
        print("computing kde")
        d = data.shape[0]
        N = data.shape[1]
        kde = np.array(self.K.sum(axis=1)).ravel()
        kde *= (N*(2*np.pi*self.epsilon)**(d/2))**(-1) 
        return kde

    def construct_committor(self, B_bool, C_bool):
        r"""Constructs the committor function w.r.t to product set A, reactant set B, C = domain \ (A U B) using the generator L

        Applies boundary conditions and restricts L to solve 
        solve Lq = 0, with q(A) = 0, q(B) = 1

        Parameters
        ----------

        L : sparse array, num data points x num data points
            generator matrix corresponding to a data set, generally the L
                matrix from diffusion maps
        B_bool : boolean vector
            indicates data indices corresponding to reactant B, same length
                as number of data points
        C_bool : boolean vector
            indicates data indices corresponding to transition region domain
                \ (A U B), same length as number of data points

        Returns
        ---------
        q : vector
            Committor function with respect to sets defined by B_bool, C_bool
        """

        # Restrict B, C to subgraph from radius sparsity
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        C_bool = C_bool[nonisolated_bool]
        B_bool = B_bool[nonisolated_bool]

        L = self.L
        Lcb = L[C_bool, :]
        Lcb = Lcb[:, B_bool]
        Lcc = L[C_bool, :]
        Lcc = Lcc[:, C_bool]

        # Assign boundary conditions for q, then solve L(C,C)q(C) = L(C,B)1
        q = np.zeros(L.shape[1])
        q[B_bool] = 1
        row_sum = np.array(np.sum(Lcb, axis=1)).ravel()

        if sps.issparse(L):
            q[C_bool] = sps.linalg.spsolve(Lcc, -row_sum)
        else:
            q[C_bool] = np.linalg.solve(Lcc, -row_sum)
        return q, self.subgraph

    def construct_committor_symmetric(self, B_bool, C_bool):
        r"""Constructs the committor function w.r.t to product set A, reactant set B, C = domain \ (A U B) using the generator L

        Applies boundary conditions and restricts L to solve 
        solve Lq = 0, with q(A) = 0, q(B) = 1

        Parameters
        ----------

        L : sparse array, num data points x num data points
            generator matrix corresponding to a data set, generally the L
                matrix from diffusion maps
        B_bool : boolean vector
            indicates data indices corresponding to reactant B, same length
                as number of data points
        C_bool : boolean vector
            indicates data indices corresponding to transition region domain
                \ (A U B), same length as number of data points

        Returns
        ---------
        q : vector
            Committor function with respect to sets defined by B_bool, C_bool
        """

        # Restrict B, C to subgraph from radius sparsity
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        C_bool = C_bool[nonisolated_bool]
        B_bool = B_bool[nonisolated_bool]

        # ONLY NEW LINE
        L = self.get_generator_symmetric()

        Lcb = L[C_bool, :]
        Lcb = Lcb[:, B_bool]
        Lcc = L[C_bool, :]
        Lcc = Lcc[:, C_bool]

        # Assign boundary conditions for q, then solve L(C,C)q(C) = L(C,B)1
        q = np.zeros(L.shape[1])
        q[B_bool] = 1
        row_sum = np.array(np.sum(Lcb, axis=1)).ravel()

        if sps.issparse(L):
            q[C_bool] = sps.linalg.spsolve(Lcc, -row_sum)
        else:
            q[C_bool] = np.linalg.solve(Lcc, -row_sum)
        return q, self.subgraph

    def construct_MFPT(self, B_bool, C_bool):
        r"""Constructs the mean first passage time w.r.t to set B, C = domain \ (B) using the generator L

        Applies boundary conditions and restricts L to solve 
        solve [Lm](C) = -1 and m(A) = 0 
        Parameters
        ----------

        L : sparse array, num data points x num data points
            generator matrix corresponding to a data set, generally the L
                matrix from diffusion maps
        B_bool : boolean vector
            indicates data indices corresponding to reactant B, same length
                as number of data points
        C_bool : boolean vector
            indicates data indices corresponding to complement
              domain \ (B), same length as number of data points

        Returns
        ---------
        m : vector
            mean first passage time with respect to sets defined by B_bool, C_bool
        """

        # Restrict B, C to subgraph from radius sparsity
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        C_bool = C_bool[nonisolated_bool]
        B_bool = B_bool[nonisolated_bool]

        L = self.L
        Lcc = L[C_bool, :]
        Lcc = Lcc[:, C_bool]

        # Assign boundary conditions for q, then solve L(C,C)q(C) = L(C,B)1
        m = np.zeros(L.shape[1])
        m[B_bool] = 0
        mc = m[C_bool]

        if not self.dense:
            m[C_bool] = sps.linalg.spsolve(Lcc, -np.ones_like(mc))
        else:
            m[C_bool] = np.linalg.solve(Lcc, -np.ones_like(mc))
        return m, self.subgraph

    def get_kernel_reweight(self):
        K_reweight = self.right_normalizer @ self.K @ self.right_normalizer
        return K_reweight
    
    def get_generator_symmetric(self):
        N = self.L.shape[0]
        if sps.issparse(self.L):
            Lsymm = sps.spdiags(self.stationary_measure, 0, N, N) @ self.L
        else:
            Lsymm = np.diags(self.stationary_measure) @ self.L
        return Lsymm
    
    def get_stationary_measure(self):
        return self.stationary_measure
 
    def get_epsilon(self):
        return self.epsilon

    def get_kernel(self):
        return self.K

    def get_subgraph(self):
        return self.subgraph

    def get_generator(self):
        return self.L
    
    def get_evecs(self):
        return self.evecs

class TargetMeasureMahalanobisDiffusionMap(TargetMeasureDiffusionMap):
    r""" 
    Class for implementing Mahalonobis diffusion maps, replacing the square distance of usual diffusion maps 
    """

    def __init__(self, epsilon, diffusion_list, load_drifts=None, kde_epsilon=None, density_mode=None, 
                 radius=None, n_neigh=None, neigh_mode='RNN',
                 num_evecs=1, rho=None, target_measure=None, 
                 remove_isolated=True, pbc_dims=None, SYMMETRIZE=True, local_eps=None):
        r""" Initialize diffusion map object with basic hyperparameters."""    
        super().__init__(epsilon=epsilon, radius=radius, n_neigh=n_neigh, neigh_mode=neigh_mode,
                         num_evecs=num_evecs, rho=rho, target_measure=target_measure, 
                         remove_isolated=remove_isolated, pbc_dims=pbc_dims)
        self.diffusion_list = diffusion_list 
        self.kde_epsilon = kde_epsilon
        self.density_mode = density_mode
        self.load_drifts = load_drifts
        self.SYMMETRIZE = SYMMETRIZE
        self.local_eps = local_eps

    def construct_generator(self, data):
        
        K = self._construct_kernel(data, eps=self.local_eps)
        N = K.shape[-1]     # Number of data points
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        #########################################################
        self.rho = self.construct_kde_kernel(data)
        #########################################################

        #########################################################
        # Make sure we are using correct indices of the subgraph
        #if len(self.rho) > N: 
        #    self.rho = self.rho[nonisolated_bool] 
        #if self.diffusion_list.shape[0] > N: 
        #    self.diffusion_list = self.diffusion_list[nonisolated_bool, :, :]
        #########################################################

        # Use determinant in normalizing if we are doing kde normalization or targetMMAP 
        if sps.issparse(K):
            # Make right normalizing vector
            rho_inv = np.power(self.rho, -1)
            right_normalizer = sps.spdiags(rho_inv, 0, N, N) 

            K_rnorm = K @ right_normalizer 
            # Make left normalizing vector
            
            rowsums = np.array(K_rnorm.sum(axis=1)).ravel()
            rowsums_inv = np.power(rowsums, -1)
            left_normalizer = sps.spdiags(rowsums_inv, 0, N, N)

            P = left_normalizer @ K_rnorm
            #P = left_normalizer @ K_reweight

            L = (P - sps.eye(N, N))/self.local_eps

            self.stationary_measure = rowsums
            self.right_normalizer = right_normalizer 
        
        else:
            print("Doing dense matrix calculations")
            # Make right normalizing vector
            right_normalizer = np.diag(self.rho**(-1))
            
            K_rnorm = K.dot(right_normalizer)

            # Make left normalizing vector
            rowsums = np.array(K_rnorm.sum(axis=1)).ravel()
            left_normalizer = np.diag(rowsums**(-1))

            P = left_normalizer.dot(K_rnorm) 
            L = (P - np.eye(N))/self.local_eps
            self.stationary_measure = rowsums
            self.right_normalizer = right_normalizer 

        self.L = L
        return L


    def _compute_sqdists(self, data, metric='mahalanobis'):
        r""" Computes matrix of pairwise mahalanobis squared distances 

        Parameters
        ----------
        data : array, (num features, num samples)
            data matrix
    
        Returns
        -------
        mahal_sq_dists : csr matrix
                       knn-sparse matrix of squared distances

        """

        dim = data.shape[0]
        N = data.shape[1]

        if data.shape[1] < 5000:
            # Create block matrix of pairwise differences
            diffs = data.T[:, np.newaxis, ...] - data.T[np.newaxis, ...]

            if self.pbc_dims is not None: 
                diffs = helpers.periodic_restrict(diffs, self.pbc_dims)

            if metric=='mahalanobis':
                print("mahalanobis computing")
                ################################################################### 
                # Create tensor copies of inverse cholesky matrices:
                ################################################################### 
                #    1) move axis 0,1,2 of inv_chol_covs to axis 1,2,3
                #    2) Copies inv_chol_covs N times along axis 0, then shift to axis 1:
                #          bigL[i, j, k, l] = inv_chol covs(i, k, l) for j=1,...N

                self._compute_inv_chol_covs(N, dim)
                bigL = np.broadcast_to(self.inv_chol_covs, (N, N, dim, dim))
                bigL = np.swapaxes(bigL, 0, 1)
        

                if self.load_drifts is not None:
                    # Add in drifts
                    drift_factor = self.local_eps*self.load_drifts
                    diffs += drift_factor.T[:, np.newaxis, ...]
        
                # Multiply each inverse cholseky matrix by each pairwise difference
                Ldiffs = np.einsum('ijkl,ijl->ijk', bigL, diffs)
                sqdists = np.sum(Ldiffs**2, axis=-1)
                if self.SYMMETRIZE == True: 
                    print("symmetrizing")
                    sqdists += sqdists.T 
                    sqdists *= 0.5 
            else:
                sqdists = np.sum(diffs**2, axis=-1)

        else:
            sqdists = np.zeros((N, N))
            if metric == 'mahalanobis':
                print("Computing Mahalanobis distance")
                self._compute_inv_chol_covs(N, dim)
                for n in range(N):
                    diffs_row = data[:, n, np.newaxis] - data
                    if self.pbc_dims is not None: 
                        diffs_row = helpers.periodic_restrict(diffs_row, self.pbc_dims)
                    if self.load_drifts is not None: 
                        diffs_row += self.local_eps*self.load_drifts[:, n, np.newaxis]
                    L_row = self.inv_chol_covs[n, :, :]

                    Ldiffs_row = L_row.dot(diffs_row)
                    sqdists[n, :] = np.sum(Ldiffs_row**2, axis=0)
                if self.SYMMETRIZE == True:
                    print("symmetrizing!")
                    sqdists += sqdists.T
                    sqdists *= 0.5
            else: 
                for i in range(sqdists.shape[0]):
                    diffs_row = data[:, i, np.newaxis] - data[:, i:]
                    if self.pbc_dims is not None: 
                        diffs_row = helpers.periodic_restrict(diffs_row, self.pbc_dims) 
                    sqdists[i, i:] = np.sum(diffs_row**2, axis=0)
                    sqdists[i, 0:i] = sqdists[0:i, i]
        return sqdists


    def construct_kde_kernel(self, data):
        r"""Construct kernel matrix of a given data set

        Takes an input data set with structure num_features x
        num_observations, constructs squared distance
        matrix and gaussian kernel

        Parameters
        ----------
        data: array (num features, num samples)
       
        Returns
        -------
        K : array (num samples, num samples)
          pair-wise kernel matrix K(i, j) is kernel
          applied to data points i, j

        """  
        d = data.shape[0]
        N = data.shape[1]

        sqdists = self._compute_sqdists(data, metric=None)
        #sqdists = self._compute_nearest_neigh_graph(sqdists)
        
        if sps.issparse(sqdists):
            K = sqdists.copy()
            K.data = np.exp(-K.data / (2*self.epsilon))
    
            # Check sparsity of kernel
            num_entries = K.shape[0]**2
            nonzeros_ratio = K.nnz / (num_entries)
            print(f"ratio of nonzeros: {nonzeros_ratio}")
            if nonzeros_ratio > 0.5:
                # Convert to dense matrix
                print("shifting to dense")
                #self.dense = True
                K = K.toarray()

        else:
            K = np.exp(-sqdists / (2*self.epsilon))

        # Symmetrize kernel 
        #K = 0.5*(K + K.T)
        print("NOTE: trying a new symmetrizaitons!")

        if sps.issparse(sqdists):
            K = K.minimum(K.T)
        else:
            K = np.minimum(K, K.T)

        print("computing kde")
        kde = np.array(K.sum(axis=1)).ravel()
        kde *= (N*(2*np.pi*self.epsilon)**(d/2))**(-1)
        print(kde) 
        return kde


    def _compute_inv_chol_covs(self, N, dim, data=None):        
        r""" Compute inverse cholesky factorization of input diffusion matrices

        """
        inv_chol_covs = np.zeros((N, dim, dim))
        if self.diffusion_list is not None:
            for n in range(N):
                chol = self.compute_cholesky(self.diffusion_list[n, :, :], n)
                inv_chol_covs[n, :, :] = cla.dtrtri(chol, lower=1)[0]
            self.inv_chol_covs = inv_chol_covs
        else: 
            print("No capacity to compute covariances right now! Please upload some, this is defaulting to regular dmaps")
            if data is not none:
                # Make a list of identity matrices
                self.inv_chol_covs = np.ones((N,1,1)) * np.eye(dim)[np.newaxis, :] 
        return self
        
    @staticmethod
    def compute_cholesky(M, n=-1):
        # Error handling block of code for cholesky decomp
        try:
            chol = np.linalg.cholesky(M)
        except np.linalg.LinAlgError as err:
            if 'positive definite' in str(err):
                print(f"index {n} covar is NOT positive definite, using cholesky hack")
                chol = helpers.cholesky_hack(M)
            else:
                raise
        return chol



if __name__ == '__main__':
    main()