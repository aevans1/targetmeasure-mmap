"""
Diffusion map class, closely following ``pydiffmap'' library 
by Erik Thiede, Zofia Trstanova and Ralf Banisch, 
Github: https://github.com/DiffusionMapsAcademics/pyDiffMap/blob/master/docs/usage.rst
"""
import numpy as np 
import scipy.sparse as sps
from scipy.linalg.lapack import clapack as cla
from sklearn.neighbors import NearestNeighbors

import src.helpers as helpers

def main():

    return None

class TargetMeasureDiffusionMap(object):
    r"""
    Class for computing the diffusion map of a given data set. 
    """

    def __init__(self, epsilon, radius=None, n_neigh=None, neigh_mode='RNN',
                 num_evecs=1, target_measure=None, 
                 remove_isolated=True, pbc_dims=None):
        r""" Initialize diffusion map object with basic hyperparameters."""    
        self.epsilon = epsilon
        self.radius = radius
        self.n_neigh = n_neigh
        self.neigh_mode = neigh_mode
        self.num_evecs = num_evecs
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
        r""" Computes eigenvectors, eigenvalues, and diffusion coordinates of generator 

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
        # Symmetrize the generator 
        d = self.stationary_measure
        N = len(d)
        Dinv_onehalf =  sps.spdiags(np.power(d,-0.5), 0, N, N)
        D_onehalf =  sps.spdiags(np.power(d,0.5), 0, N, N)
        Lsymm = D_onehalf @ self.L @ Dinv_onehalf

        # Compute eigvals, eigvecs 
        evals, evecs = sps.linalg.eigsh(Lsymm, k=self.num_evecs + 1, which='SM')

        # Convert back to L^2 norm-1 eigvecs of L 
        evecs = (Dinv_onehalf.toarray()).dot(evecs)
        evecs /= (np.sum(evecs**2, axis=0))**(0.5)
        
        idx = evals.argsort()[::-1][1:]     # Ignore first eigval / eigfunc
        evals = np.real(evals[idx])
        evecs = np.real(evecs[:, idx])
        dmap = np.dot(evecs, np.diag(np.sqrt(-1./evals)))
        return dmap, evecs, evals

    def construct_generator(self, data):
        r"""
        
        Parameters
        ----------
        Returns
        -------
        """
        K = self._construct_kernel(data)
        N = K.shape[-1]                     # Number of data points
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]

        self.rho = self._compute_kde(data)

        # Use kde as target measure if none provided 
        if self.target_measure is None:
            self.target_measure = self.rho

        # Make sure we are using correct indices of the subgraph
        if len(self.target_measure) > N: 
            self.target_measure = self.target_measure[nonisolated_bool]

        if sps.issparse(K):
            # Right Normalize
            rho_inv = np.power(self.rho,-1)
            sqrt_pi = np.power(self.target_measure, 0.5)
            right_normalizer = (sps.spdiags(rho_inv, 0, N, N) 
                                @ sps.spdiags(sqrt_pi, 0 , N, N))
            K_reweight = K @ right_normalizer

            # Left Normalize
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            rowsums_inv = np.power(rowsums, -1) 
            left_normalizer = sps.spdiags(rowsums_inv, 0, N, N)
            P = left_normalizer @ K_reweight

            L = (P - sps.eye(N, N))/self.epsilon

            self.stationary_measure = rowsums * rho_inv * sqrt_pi
            self.right_normalizer = right_normalizer 

        else:
            print("Doing dense matrix calculations")
            rho_inv = np.power(self.rho,-1)
            sqrt_pi = np.power(self.target_measure, 0.5)

            # Right Normalize 
            right_normalizer = np.diag(np.power(self.rho,-1)).dot(np.diag(self.target_measure**(0.5)))
            K_reweight = right_normalizer.dot(K.dot(right_normalizer))

            # Left Normalize
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            left_normalizer = np.diag(rowsums**(-1))
            P = left_normalizer.dot(K_reweight) 

            L = (P - np.eye(N))/self.epsilon
            self.stationary_measure = rowsums * rho_inv * sqrt_pi
            self.right_normalizer = right_normalizer 

        self.L = L
        return L

    def _construct_kernel(self, data):
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
        sqdists = self._compute_sqdists(data)
        sqdists = self._compute_nearest_neigh_graph(sqdists)
        
        if sps.issparse(sqdists):
            K = sqdists.copy()
            K.data = np.exp(-K.data / (2*self.epsilon))

            # symmetrize kernel
            K = K.minimum(K.T)

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
            
            # symmetrize kernel
            K = np.minimum(K,K.T)

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
        N = self.K.shape[1]
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

    def get_kernel_reweight_symmetric(self):
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
    
    def get_sqdists(self):
        return self.sqdists

    def get_subgraph(self):
        return self.subgraph

    def get_generator(self):
        return self.L
    
    def get_evecs(self):
        return self.evecs
    
    def get_evals(self):
        return self.evals

class TargetMeasureMahalanobisDiffusionMap(TargetMeasureDiffusionMap):
    r""" 
    Class for implementing Mahalonobis diffusion maps, replacing the square distance of usual diffusion maps 
    """

    def __init__(self, epsilon, diffusion_list, radius=None, n_neigh=None, neigh_mode='RNN',
                 num_evecs=1, target_measure=None, 
                 remove_isolated=True, pbc_dims=None):
        
        # Initialize diffusion map object with basic hyperparameters
        super().__init__(epsilon=epsilon, radius=radius, n_neigh=n_neigh, neigh_mode=neigh_mode,
                         num_evecs=num_evecs, target_measure=target_measure, 
                         remove_isolated=remove_isolated, pbc_dims=pbc_dims)
        self.diffusion_list = diffusion_list 

    def construct_generator(self, data):
        
        K = self._construct_kernel(data)
        N = K.shape[-1]     # Number of data points
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        pi = np.zeros(N)    # initialize right normalization

        a = -1
        self.rho = self._compute_kde(data)

        # Use kde as target measure if none provided
        if self.target_measure is None:
            print("Doing regular MMAP")
            self.target_measure = self.rho
            a = 0

        # Make sure we are using correct indices of the subgraph
        if len(self.target_measure) > N: 
            self.target_measure = self.target_measure[nonisolated_bool]
        if self.diffusion_list.shape[0] > N: 
            self.diffusion_list = self.diffusion_list[nonisolated_bool, :, :]

        # Use determinant in normalizing if we are doing kde normalization or targetMMAP 
        for n in range(N):
            M = self.diffusion_list[n, :, :]
            pi[n] = self.target_measure[n]*((np.linalg.det(M))**(-a/2))

        if sps.issparse(K):
            # Right Normalize
            rho_inv = np.power(self.rho, -1)
            sqrt_pi = np.power(pi, 0.5)
            right_normalizer = (sps.spdiags(rho_inv, 0, N, N) 
                                  @ sps.spdiags(sqrt_pi, 0 , N, N))
            K_reweight = K @ right_normalizer
           
            # Left Normalize
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            rowsums_inv = np.power(rowsums, -1)
            left_normalizer = sps.spdiags(rowsums_inv, 0, N, N)
            P = left_normalizer @ K_reweight

            L = (P - sps.eye(N, N))/self.epsilon
            
            
            self.stationary_measure = rowsums * rho_inv * sqrt_pi
            self.right_normalizer = right_normalizer 

        else:
            print("Doing dense matrix calculations")
            rho_inv = np.power(self.rho, -1)
            sqrt_pi = np.power(pi, 0.5)
            # Right Normalize
            right_normalizer = np.diag(rho_inv * sqrt_pi)
            K_reweight = right_normalizer.dot(K.dot(right_normalizer))

            # Left Normalize
            rowsums = np.array(K_reweight.sum(axis=1)).ravel()
            left_normalizer = np.diag(rowsums**(-1))
            P = left_normalizer.dot(K_reweight) 
            
            L = (P - np.eye(N))/self.epsilon
            self.stationary_measure = rowsums * rho_inv * sqrt_pi
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

        sqdists = np.zeros((N, N))
        if metric == 'mahalanobis':
            print("Computing Mahalanobis distance")
            self._compute_inv_chol_covs(N, dim)
            for n in range(N):
                diffs_row = data[:, n, np.newaxis] - data
                if self.pbc_dims is not None: 
                    diffs_row = helpers.periodic_restrict(diffs_row, self.pbc_dims)
                L_row = self.inv_chol_covs[n, :, :]
                Ldiffs_row = L_row.dot(diffs_row)
                sqdists[n, :] = np.sum(Ldiffs_row**2, axis=0)
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

    def _compute_kde(self, data):
        subgraph = self.get_subgraph()
        nonisolated_bool = subgraph["nonisolated_bool"]
        diffusion_list_nonisolated = self.diffusion_list[nonisolated_bool, :, :]
        N = self.K.shape[1]
        d = data.shape[0]

        kde = np.array(self.K.sum(axis=1)).ravel()
        kde *= (N*(2*np.pi*self.epsilon)**(d/2))**(-1) 
        det_list = np.zeros(N)
        for n in range(N):
            det_list[n] = np.linalg.det(diffusion_list_nonisolated[n, :, :])**(-1/2)
        kde *= det_list
        return kde

    def _compute_inv_chol_covs(self, N, dim):        
        r""" Compute inverse cholesky factorization of input diffusion matrices

        """
        inv_chol_covs = np.zeros((N, dim, dim))
        if self.diffusion_list is not None:
            for n in range(N):
                chol = self.compute_cholesky(self.diffusion_list[n, :, :], n)
                inv_chol_covs[n, :, :] = cla.dtrtri(chol, lower=1)[0]
            self.inv_chol_covs = inv_chol_covs
        else: 
            print("Defaulting to regular dmaps, no diffusion matrices provided")
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

    # NOTE: This is a test function to try out making a `bistochastic` kernel with respect to pi / self.rho
    #def construct_bistochastic(self, data):
    #    
    #    K = self._construct_kernel(data)
    #    N = K.shape[-1]     # Number of data points
    #    subgraph = self.get_subgraph()
    #    nonisolated_bool = subgraph["nonisolated_bool"]
    #    pi = np.zeros(N)    # initialize right normalization
    #    
    #    # Adjust right normalization for whether we are unbiasing with a 
    #    #   1) KDE estimate or  2) mahalanobis kernel sum
    #    if self.density_mode == 'KDE':
    #        print("Doing KDE normalization")
    #        a = -1
    #        if self.rho is None:
    #            print("computing rho from kernel")
    #            self.rho = self._compute_kde(data)
    #    else:
    #        print("Doing MMAP normalization")
    #        a = 1
    #        self.rho = np.array(K.sum(axis=1)).ravel()

    #    if self.target_measure is None:
    #        print("Doing regular MMAP")
    #        self.target_measure = self.rho
    #        a = 0
    #    
    #    # Make sure we are using correct indices of the subgraph
    #    if len(self.rho) > N: 
    #        self.rho = self.rho[nonisolated_bool] 
    #    if len(self.target_measure) > N: 
    #        self.target_measure = self.target_measure[nonisolated_bool]
    #    if self.diffusion_list.shape[0] > N: 
    #        self.diffusion_list = self.diffusion_list[nonisolated_bool, :, :]
    #    #########################################################

    #    # Use determinant in normalizing if we are doing kde normalization or targetMMAP 
    #    print(f"Computing determinants with power: {a}" )
    #    a = 1
    #    for n in range(N):
    #        M = self.diffusion_list[n, :, :]
    #        pi[n] = self.target_measure[n]*((np.linalg.det(M))**(a/2))

    #    if sps.issparse(K):
    #        # Make right normalizing vector
    #        winv = pi/self.rho
    #        Winv = sps.spdiags(winv, 0, N, N) 
    #        Dnew = sps.eye(N, N)
    #        Dprimenew = sps.eye(N, N)
    #        for i in range(100):
    #            D = Dnew
    #            Dprime = Dprimenew

    #            Dinv = sps.spdiags((np.array(D.diagonal()).ravel())**(-1), 0, N , N)
    #            Dprimeinv = sps.spdiags((np.array(Dprime.diagonal()).ravel())**(-1), 0, N , N)

    #            Dnew = np.array((K @ Dinv @ Dprimeinv.power(0.5) @ Winv).sum(axis=1).ravel())
    #            Dnew = sps.spdiags(Dnew, 0, N, N)
    #            
    #            Dprime = np.array((Dinv @ K @ Dinv @ Winv).sum(axis=1).ravel())
    #            Dprime = sps.spdiags(Dprime, 0, N, N)

    #            if i % 100 == 0:
    #                print(i)
    #                Dtest = (Dnew.power(0.5))*(D.power(0.5))
    #                aux = np.array(Dtest.diagonal()).ravel()
    #                Dtest_inv = sps.spdiags(aux**-1, 0, N, N)
    #                print(np.array(np.sum(Dtest_inv.dot(K.dot(Dtest_inv.dot(Winv))), axis=1)).ravel())

    #        D = Dnew.power(0.5)*D.power(0.5)
    #        Dinv = sps.spdiags((np.array(D.diagonal()).ravel())**(-1), 0, N , N)
    #        self.K_reweight = Dinv @ K @ Dinv
    #        P = self.K_reweight @ Winv
    #        self.P = P
    #        self.winv = winv
    #        L = (P - sps.eye(N, N))/self.epsilon

    #        self.Winv = Winv
    #    
    #    else:
    #        print("Doing dense matrix calculations")
    #        # Make right normalizing vector
    #        right_normalizer = np.diag(self.rho**(-1)).dot(np.diag(self.target_measure**(0.5)))
    #        
    #        #K_rnorm = K.dot(right_normalizer)
    #        K_reweight = right_normalizer.dot(K.dot(right_normalizer))

    #        # Make left normalizing vector
    #        rowsums = np.array(K_reweight.sum(axis=1)).ravel()
    #        
    #        left_normalizer = np.diag(rowsums**(-1))
    #        
    #        P = left_normalizer.dot(K_reweight) 

    #        L = (P - np.eye(N))/self.epsilon
    #        self.stationary_measure = rowsums
    #        self.right_normalizer = right_normalizer 

    #    self.L = L

    #    return L

if __name__ == '__main__':
    main()