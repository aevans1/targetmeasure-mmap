import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp_linalg
import scipy.sparse as sps
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.interpolate as scinterp
import scipy.spatial

###############################################################################
# For doing discrete carre du champ or dirichlet form 
###############################################################################
def duchamp(A, f, g, pbc_dims=None, PBC_ARG1=False, PBC_ARG2=False):
    """ Given square matrix A and vectors f,g, 
    computes sum_j A_ij (f_i - f_j)(g_i - g_j) for all i

    Args:
        A (numpy array): num data points x num data points 
        f (numpy array): num data points x 1
        g (numpy array): num data points x 1
        pbc_dims (numpy array, optional): Array specifying periodic dimensions. Defaults to None.
        PBC_ARG1 (bool, optional): Specifies if f needs periodic boundary . Defaults to False.
        PBC_ARG1 (bool, optional): Specifies if g needs periodic boundary . Defaults to False.

    Returns:
        out: num_data points x 1 vector, index i is sum_j A_ij (f_i - f_j)(g_i - g_j) 
    """    
    # NOTE: A should be a square matrix with each row summing to 0
    # F, G below are pairwise diff matrices for vectors f,g:
    # F[i, j] = f[j] - f[i]

    # Convert csr matrices to dense numpy arrays
    if sps.issparse(A):
        A = A.toarray()

    # Use vectorized computation for smaller datasets 
    if A.shape[0] < 5000: 
        F = f[np.newaxis, ...] - f[:, np.newaxis, ...]
        G = g[np.newaxis, ...] - g[:, np.newaxis, ...]
        if pbc_dims is not None:
            if PBC_ARG1:
                F = periodic_restrict(F, pbc_dims)
            if PBC_ARG2:
                G = periodic_restrict(G, pbc_dims)
        out = np.sum(A*F*G, axis=1).ravel()
        # If A has row sums 0 and there's no periodic restrictions, above is equivalent to
        # out = np.dot(A, f*g) - f*np.dot(A, g) - g*np.dot(A ,f)

    # If a larger dataset, go row-by-row and don't make F,G matrices
    else: 
        out = np.zeros(A.shape[0])
        if pbc_dims is not None:
            if PBC_ARG1 and PBC_ARG2:
                for i in range(A.shape[0]):
                    F_row = periodic_restrict(f - f[i], pbc_dims)
                    G_row = periodic_restrict(g - g[i], pbc_dims)
                    out[i] = np.sum(A[i, :]*F_row*G_row)
            elif PBC_ARG1:
                #pbc for just first arg
                for i in range(A.shape[0]):
                    F_row = periodic_restrict(f - f[i], pbc_dims)
                    G_row = g - g[i]
                    out[i] = np.sum(A[i, :]*F_row*G_row)
            elif PBC_ARG2:
                # pbc for just second arg
                for i in range(A.shape[0]):
                    F_row = f - f[i]
                    G_row = periodic_restrict(g - g[i], pbc_dims)
                    out[i] = np.sum(A[i, :]*F_row*G_row)
        else:
            for i in range(A.shape[0]):
                F_row = f[i] - f
                G_row = g[i] - g
                out[i] = np.sum(A[i, :]*F_row*G_row)
    return out




def dirichlet_form(A, f, g, pbc_dims=None):
    N = f.shape[0]
    return (1/N)*np.sum(duchamp(A, f, g, pbc_dims=pbc_dims)) 

##############################################################
# Functions helping with committor error, plotting committor, etc.
##############################################################
def committor_contours():
    """ Quick function returning useful level sets for committor
    
    Returns:
        my_levels (numpy array)
    """    
    my_levels = np.arange(0.1, 0.6, 0.1)
    my_levels = np.concatenate((my_levels, np.arange(0.6, 1.0, 0.1)))    

    return my_levels 


def is_in_ABC(data, centerx_A, centery_A, rad_A, centerx_B, centery_B, rad_B):
    """_summary_

    Args:
        data (numpy array): 2 x num data points
        centerx_A, centery_A, rad_A (scalars): coordinates of center of A set, radius
        centerx_B, centery_B, rad_B (scalars): coordinates of center of B set, radius
    
    Returns:
        A_bool (boolean array): 1 x num_data points boolean array, True indicates data i is in A
        B_bool (boolean array): similarly with B
        C_bool(boolean array): similarly with complement of (A U B)
    """
    A_bool = is_in_circle(data[0, :], data[1, :], centerx_A, centery_A, rad_A)
    B_bool = is_in_circle(data[0, :], data[1, :], centerx_B, centery_B, rad_B)
    C_bool = np.logical_not(np.logical_or(A_bool, B_bool))
    return A_bool, B_bool, C_bool


def is_in_circle(x, y, centerx=0, centery=0, rad=1.0): 
    return ((x - centerx)**2 + (y-centery)**2 <= rad**2)


def RMSerror(approx, truth, weights = None):
    N = approx.shape[0]
    if weights is not None:
        output = np.sqrt(np.sum(weights*(approx - truth)**2))
    else:
        output = np.sqrt((1./N)*np.sum((approx - truth)**2))
    return output

###############################################################################
# Periodic boundary Condition functions
###############################################################################
def periodic_restrict(x, boundary):
    """Restricts an array x to comply with periodic boundary conditions

    example: x = np.array([100, 700, -500]), boundary = np.array([360])
            will return y = np.array([100, 340, -140 ])

    Args:
        x (array): np array of size (m, n) or (n,)
        boundary (array): np array of size (1, n), (m, 1), (n,) or (1,)

    Returns:
        y (array): numpy array of same size as x
    """

    if (x > 0.5*boundary).any():
        x = np.where(x > 0.5*boundary, x - boundary, x) 
    if (x <= -0.5*boundary).any(): 
        x = np.where(x <= -0.5*boundary, x + boundary, x) 
    return x

def periodic_add(x, y, boundary):
    y = np.where(x - y > 0.5*boundary, y - boundary, y)
    y = np.where(x - y < -0.5*boundary, y + boundary, y)
    return periodic_restrict(x + y, boundary)

################################################################################
# Data Plotting Funtions
################################################################################
def plot_cov_ellipse(cov, x, plot_scale=1, color='k', plot_evecs=False,
                     quiver_scale=1):
    """Plots the ellipse corresponding to a given covariance matrix at x (2D) 

    Args:
        cov (array): 2 by 2 symmetric positive definite array.
        x (vector): 2 by 1 vector.
        scale (float, optional): Scale for ellipse size. Defaults to 1.
    """

    evals, evecs = np.linalg.eig(cov)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    t = np.linspace(0, 2*np.pi)
    val = plot_scale*evecs*evals
    if plot_evecs:
        plt.quiver(*x, *val[:, 0], color=color, angles='xy', scale_units='xy',
                    scale=quiver_scale, width=0.002)
        plt.quiver(*x, *val[:, 1], color=color, angles='xy', scale_units='xy',
                   scale=quiver_scale, width= 0.002)
    else:
        a = np.dot(val, np.vstack([np.cos(t), np.sin(t)]))
    
        plt.plot(a[0, :] + x[0], a[1, :] + x[1], linewidth=0.5, c=color)

def gen_plot_data(potential, plot_params):
    r"""Evaluates potential on a grid for plotting

    Parameters
    ----------
    potential: function,
        Potential function which takes a 2-dim vector as input
        and returns a scalar
    
    plot_params: array-like,
        Python list of values for grid,
        plot_params must be of form [nx, ny, xmin, xmax, ymin, ymax]
        as defined below

        nx, ny : scalar
            Number of steps in x and y direction respectively.
        xmin, xmax : scalar
            Interval bounds [xmin, xmax] for the x-bounds of the grid
        xmin, xmax : scalar
            Interval bounds [ymin, ymax] for the y-bounds of the grid

    Returns
    -------
        plot_data : array-like,
        Python list of cartesian grids for plotting metad data,
        of form [pot, xx, yy]

        pot : array-like
            nx by ny array, cartesian grid of potential function
        xx, yy : array-like
            meshgrid coordinates the potential and bias were evaluated on

    """

    # Read in input lists
    nx, ny, xmin, xmax, ymin, ymax = plot_params
    
    # Compute potential energy contours on grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    pot = np.zeros((nx, ny))
    for j in range(nx):
        for i in range(ny):
            a = xx[i, j]
            b = yy[i, j]
            v = np.array([a, b])
            pot[i, j] = potential(v)
    
    plot_data = [pot, xx, yy]
    
    return plot_data

##############################################################
# Other functions
##############################################################
def cholesky_hack(C):
    #Computes the (not necessarily unique) Cholesky decomp. for a symmetric positive SEMI-definite matrix, C = LL.T, returns L
    # NOTE: this is a bit more expensive than regular cholesky, should only be used if input matrix is likely not positive definite but it is semi-definite

    # C = MM^T, M^T = QR ---> MM^T = R^T R, so L = R^T

    M = sp_linalg.sqrtm(C)
    R = np.real(np.linalg.qr(M.T)[1])
    return R.T


