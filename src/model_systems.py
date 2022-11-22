import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp_linalg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.interpolate as scinterp
import scipy.special as sc

# From eqn 2.2 in Moro/Cardin 1998
def morocardin_potential(x, h=5.0, a=1.0, alpha=140, beta_inv=1.0):
    V = (h*((x[0]/a)**2 - 1)**2 + 2*h*((x[1]/a)*sc.cotdg(alpha/2))**2)
    return V

# Gradient of 'morocardin_potential'
def morocardin_mean_force(x, h=5.0, a=1.0, alpha=140):
    mean_force1 = ((x[0]/a)**2 - 1)*x[0]
    mean_force2 = (sc.cotdg(alpha/2)**2)*x[1]
    mean_force = ((4*h)/a**2)*np.array([mean_force1, mean_force2])
    return mean_force

# Inverse friction From eqn 2.3, 2.6 in Moro/Cardin 1998
def morocardin_sqrt_diffusion(x, lower=1.0, upper=8.0, std=0.2):
    weight = np.exp(-(x[0]**2 + x[1]**2)/(2.0*(std**2)))
    sqrt_friction = np.sqrt(lower + upper*weight)
    sqrt_diffusion = np.eye(2)/sqrt_friction
    return sqrt_diffusion

# Divergence of square of 'morocardin_sqrt_diffusion'
def morocardin_div_diffusion(x, lower=1.0, upper=8.0, std=0.2):
    weight = np.exp(-(x[0]**2 + x[1]**2)/(2.0*std**2))
    div_diff = ((weight*upper) / ((std**2)*(lower + upper*weight)**2)) * x

    return div_diff

def morocardin_drift(x, beta_inv):
    h=5.0
    a=1.0
    alpha=140
    lower=1.0
    upper=8.0
    std=0.2

    sqrt_diff = morocardin_sqrt_diffusion(x, lower=lower, upper=upper, std=std)
    diff = np.dot(sqrt_diff, sqrt_diff.T)
    mf = morocardin_mean_force(x, h=h, a=a, alpha=alpha)
    div_diff = morocardin_div_diffusion(x, lower=lower, upper=upper, std=std)
    return -np.dot(diff, mf) + beta_inv*div_diff

def muller_drift(x):

    # Define list of parameters
    A = np.array([-200., -100., -170., 15.])
    nu = np.array([[1., 0.], 
                   [0., 0.5], 
                   [-0.5, 1.5], 
                   [-1., 1.]])
    sig_inv = np.array([[[1., 0.],
                         [0., 10.]],
                        [[1., 0.],
                         [0., 10.]],
                        [[6.5, -5.5],
                         [-5.5, 6.5]],
                        [[-0.7, -0.3],
                         [-0.3, -0.7]]])
    force = np.array([0., 0.])
    for i in range(4):
        u = x - nu[i, :]
        M = sig_inv[i, :, :]
        force += 2*A[i]*np.exp(-u.dot(M.dot(u)))*M.dot(u)
    return force

def muller_potential(x):

    # Define list of parameters
    A = np.array([-200., -100., -170., 15.])
    nu = np.array([[1., 0.], 
                   [0., 0.5], 
                   [-0.5, 1.5], 
                   [-1., 1.]])
    sig_inv = np.array([[[1., 0.],
                         [0., 10.]],
                        [[1., 0.],
                         [0., 10.]],
                        [[6.5, -5.5],
                         [-5.5, 6.5]],
                        [[-0.7, -0.3],
                         [-0.3, -0.7]]])
    V = 0.
    for i in range(4):
        u = x - nu[i, :]
        M = sig_inv[i, :, :]
        V += A[i]*np.exp(-u.dot(M.dot(u)))
    return V

