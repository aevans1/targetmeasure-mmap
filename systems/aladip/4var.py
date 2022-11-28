import os, sys
sys.path.append('../..')
sys.path.append('.')


# Regular Modules
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.integrate as scint
from numpy.random import default_rng
import numpy.ma as ma
import matplotlib.tri as tri
import scipy.special as sc
import scipy.sparse as sps
import datetime

# My Modules
import src.helpers as helpers
import src.model_systems as model_systems
import src.diffusion_map as dmap

# Use timestamp for saving
time = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")

###############################################################################
# Load up molecular dynamics data
###############################################################################
# Set Gromacs Temperature
kb = 0.0083144621
T = 300
beta = 1.0 / (kb * T)

fname = 'data/TRAJ_COLVAR_METAD_2VAR_GRID_STATIC'
colvar_data = np.loadtxt(fname)
fname = 'data/diffusions_traj_static_4var.npz'
inData = np.load(fname)
diffusions = inData["diffusions"]

end = colvar_data.shape[0]
start = 0
step = 5   # Subsampling Rate
t = colvar_data[start:end:step,0]  # timestep
phi = colvar_data[start:end:step,1] # phi dihedral angle coordinates
psi = colvar_data[start:end:step,2] # psi dihedral angle coordinates
theta = colvar_data[start:end:step,3] # theta dihedral angle coordinates
xi = colvar_data[start:end:step,4] # xi dihedral angle coordinates
bias = colvar_data[start:end:step,5] 
rct = colvar_data[start:end:step,6] 
rbias = colvar_data[start:end:step,7] 

data = np.vstack([phi, psi, theta, xi])
diffusions = diffusions[start:end, :, :]
N = data.shape[1]
print(f"number of data points: {N}")

fname = f"data/Ktest_4var_regular.npz"
inData = np.load(fname)
kde_eps = inData["optimal_eps"]

fname = f"data/Ktest_4var_mahal.npz"
inData = np.load(fname)
mahal_eps = inData["optimal_eps"]

###############################################################################
# Approximate Target Measure
###############################################################################
kde = np.zeros(N)
experiment = np.zeros(N)
pbc_dims = np.array([2*np.pi])
for i in range(N):
    diffs = data[:, i, np.newaxis] - data
    diffs = helpers.periodic_restrict(diffs, pbc_dims) 
    sqdists_row = np.sum(diffs**2, axis=0)
    kernel_sqdists_row = np.exp(-sqdists_row / (2*kde_eps))
    kde[i] = np.sum(kernel_sqdists_row)
d = data.shape[0]
kde *= (N)**(-1)*(2*np.pi*kde_eps)**(-d/2)
target_measure = kde*np.exp(beta*rbias)

###############################################################################
# Define Reactant, Product Sets, Plot
###############################################################################
# Create boolean arrays defining A,B and rest of the domain (C)
A_bool = np.zeros(N, dtype=bool)
B_bool = np.zeros(N, dtype=bool)
a = np.array([-1.44338729, 1.282817, 0.02792527, -0.07504916])
b = np.array([1.23045712, -1.20602251, -0.01396263,  0.09948377])

asums = np.sqrt(np.sum((a.reshape([4, 1]) - data)**2, axis=0))
bsums = np.sqrt(np.sum((b.reshape([4, 1]) - data)**2, axis=0))
# Create boolean arrays defining A,B and rest of the domain (C)
A_bool = np.zeros(N, dtype=bool)
B_bool = np.zeros(N, dtype=bool)
for n in range(N):
    if asums[n] < 0.3:
        A_bool[n] = True
    if bsums[n] < 0.3:
        B_bool[n] = True

C_bool = ~np.logical_or(A_bool, B_bool)

A_test = data[:, A_bool]
B_test= data[:, B_bool]
C_test = data[:, C_bool]

import matplotlib
my_cmap = matplotlib.cm.get_cmap('brg')
A_color= my_cmap(0.0)
B_color = my_cmap(1.0)

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

fig = plt.figure() 
#plt.scatter(data[0, :], data[1, :], s=0.5, c='gray')
plt.scatter(C_test[0, :], C_test[1, :], s=0.2, c='gray')
plt.scatter(A_test[0, :], A_test[1, :], s=5, c=A_color)
plt.scatter(B_test[0, :], B_test[1, :], s=5, c=B_color)
plt.gca().text(-1.5, 1.1, 'C7eq', fontsize=20)
plt.gca().text(1.2, -1.15, 'C7ax', fontsize=20)

xmin, xmax = -np.pi, np.pi
ymin, ymax = -np.pi, np.pi
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Psi$', size=20)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"figures/4var/reactant_product_phipsi.png", dpi=300)

fig = plt.figure() 
#plt.scatter(data[0, :], data[1, :], s=0.5, c='gray')
plt.scatter(C_test[0, :], C_test[2, :], s=0.2, c='gray')
plt.scatter(A_test[0, :], A_test[2, :], s=5, c=A_color)
plt.scatter(B_test[0, :], B_test[2, :], s=5, c=B_color)
#plt.gca().text(-1.5, 1.1, 'C7eq', fontsize=20)
#plt.gca().text(1.2, -1.15, 'C7ax', fontsize=20)

xmin, xmax = -np.pi, np.pi
ymin, ymax = -1, 1
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Theta$', size=20)
plt.tight_layout()
#plt.gca().set_aspect('equal', adjustable='box')
fname = "figures/4var/reactantproduct_phitheta.png"
plt.savefig(fname, dpi=300)


fig = plt.figure() 
#plt.scatter(data[0, :], data[1, :], s=0.5, c='gray')
plt.scatter(C_test[1, :], C_test[2, :], s=0.2, c='gray')
plt.scatter(A_test[1, :], A_test[2, :], s=5, c=A_color)
plt.scatter(B_test[1, :], B_test[2, :], s=5, c=B_color)
#plt.gca().text(-1.5, 1.1, 'C7eq', fontsize=20)
#plt.gca().text(1.2, -1.15, 'C7ax', fontsize=20)

xmin, xmax = -np.pi, np.pi
ymin, ymax = -1, 1
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Xi$', size=20)
plt.tight_layout()
#plt.gca().set_aspect('equal', adjustable='box')

###############################################################################
# Compute a diffusion map
###############################################################################
eps = mahal_eps
pbc_dims = np.array([2*np.pi])
radius = None
n_neigh = 512
density_mode = None
neigh_mode = "RNN"
rho = None

# Compute target measure
method = "targetMMAP"
##### Compute some sort of diffusion map 
if method == 'MMAP' or method == 'targetMMAP':
    if method == 'MMAP':
        rho = None
        target_measure = None
    my_dmap = dmap.TargetMeasureMahalanobisDiffusionMap(epsilon=eps, diffusion_list=diffusions,
                                                       kde_epsilon=kde_eps, density_mode=density_mode, 
                                                       radius=radius, n_neigh=n_neigh, neigh_mode=neigh_mode,
                                                       rho=rho, target_measure=target_measure, 
                                                       remove_isolated=True, pbc_dims=pbc_dims)

if method == 'DMAP' or method =='targetDMAP':
    if method == 'DMAP':
        rho = None
        target_measure = None
    my_dmap = dmap.TargetMeasureDiffusionMap(epsilon=eps, radius=radius, n_neigh=n_neigh, 
                                             neigh_mode=neigh_mode,rho=rho, 
                                             target_measure=target_measure, 
                                             remove_isolated=True, pbc_dims=None)

my_dmap.construct_generator(data)
q, subgraph  = my_dmap.construct_committor(B_bool, C_bool)
L = my_dmap.get_generator()

nonisolated_bool = subgraph["nonisolated_bool"] 
data_nonisolated = data[:, nonisolated_bool]
print("Done!")
################################################################################
# Compute marginalized committors
################################################################################

# distribution of committor value conditioned on coordinates
N_trunc = data_nonisolated.shape[1]
conditional = np.zeros((N_trunc, N_trunc))
i = 0
j = 1

q_phi_psi = np.zeros_like(q)
i = 0
j = 1
val = 0.2

for n in range(len(q_phi_psi)):
    diff = data_nonisolated[[i, j], n].reshape(2,1) - data_nonisolated[[i, j], :]
    diff = helpers.periodic_restrict(diff, boundary=pbc_dims)
    my_dists = np.sqrt(np.sum(diff**2, axis=0))
    neigh_bool = my_dists < val
    q_phi_psi[n]  = np.mean(q[neigh_bool])

q_phi_theta = np.zeros_like(q)
i = 0
j = 2 
val = 0.2
for n in range(len(q_phi_theta)):
    diff = data_nonisolated[[i, j], n].reshape(2,1) - data_nonisolated[[i, j], :]
    diff = helpers.periodic_restrict(diff, boundary=pbc_dims)
    my_dists = np.sqrt(np.sum(diff**2, axis=0))
    neigh_bool = my_dists < val
    q_phi_theta[n]  = np.mean(q[neigh_bool])

################################################################################
## Plot Committors
################################################################################
i = 0
j = 1 


contour_bool = np.ones(data_nonisolated.shape[1], dtype=bool)
for n in range(data_nonisolated.shape[1]):
    #if free_energy_nonisolated[n] > contour_bdry:
    #    contour_bool[n] = False
    if data_nonisolated[j, n] > 3.1:
        contour_bool[n] = False
    if data_nonisolated[j, n] < -3.1:
        contour_bool[n] = False
triang = tri.Triangulation(data_nonisolated[i, :], data_nonisolated[j, : ])
triangle_mask = np.any(np.logical_not(contour_bool)[triang.triangles], axis=1)
triang.set_mask(triangle_mask)
#triang = tri.Triangulation(data_nonisolated[i, :], data_nonisolated[j, : ])
#triangle_mask = np.any(np.logical_not(contour_bool)[triang.triangles], axis=1)
#triang.set_mask(triangle_mask)

cmap = 'brg'
fig = plt.figure()
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=0.1, c='gray')
my_levels = helpers.committor_contours() 
C1=plt.tricontour(triang, q, levels=my_levels, linewidths=2.0, cmap=cmap, linestyles='dashed')
norm= matplotlib.colors.Normalize(vmin=C1.cvalues.min(), vmax=C1.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = C1.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=C1.levels)

my_cmap = matplotlib.cm.get_cmap('brg')
A_color= my_cmap(0.0)
B_color = my_cmap(1.0)
plt.scatter(A_test[i, :], A_test[j, :], s=0.5, c=A_color)
plt.scatter(B_test[i, :], B_test[j, :], s=0.5, c=B_color)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Psi$', size=20)
#plt.tight_layout()
#plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"figures/4var/committor_contours_{method}_phipsi.png", dpi=300)


plt.figure()
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=5, c=q)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Psi$', size=20)
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=0.1, c='gray')
my_levels = helpers.committor_contours() 
C1=plt.tricontour(triang, q, levels=my_levels, linewidths=2.0, cmap=cmap, linestyles='dashed')
norm= matplotlib.colors.Normalize(vmin=C1.cvalues.min(), vmax=C1.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = C1.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=C1.levels)
xmin, xmax = -0.5, 0.5
ymin, ymax = -1, 0
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.tight_layout()
plt.colorbar()
plt.savefig(f"figures/4var/committor_contours_{method}_phipsi_box.png", dpi=300)



plt.figure()
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=5, c=q)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Psi$', size=20)
xmin, xmax = -np.pi, np.pi
ymin, ymax = -np.pi, np.pi
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.tight_layout()
plt.colorbar()
plt.savefig(f"figures/4var/committor_contours_{method}_phipsi_coarse.png", dpi=300)
#plt.gca().set_aspect('equal', adjustable='box')

fig = plt.figure()
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=0.1, c='gray')
my_levels = helpers.committor_contours() 
C1=plt.tricontour(triang, q_phi_psi, levels=my_levels, linewidths=2.0, cmap=cmap, linestyles='dashed')
norm= matplotlib.colors.Normalize(vmin=C1.cvalues.min(), vmax=C1.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = C1.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=C1.levels)

my_cmap = matplotlib.cm.get_cmap('brg')
A_color= my_cmap(0.0)
B_color = my_cmap(1.0)
plt.scatter(A_test[i, :], A_test[j, :], s=0.5, c=A_color)
plt.scatter(B_test[i, :], B_test[j, :], s=0.5, c=B_color)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Psi$', size=20)
#plt.tight_layout()
#plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"figures/4var/committor_contours_{method}_phipsi_avg.png", dpi=300)

plt.figure()
plt.scatter(data_nonisolated[0, :], data_nonisolated[1, :], s=5, c=q_phi_psi)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Psi$', size=20)
xmin, xmax = -np.pi, np.pi
ymin, ymax = -np.pi, np.pi
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.tight_layout()
plt.colorbar()
plt.savefig(f"figures/4var/committor_contours_{method}_phipsi_avg_coarse.png", dpi=300)
#plt.gca().set_aspect('equal', adjustable='box')


######
# Now for phi, theta
xmin, xmax = -np.pi, np.pi
ymin, ymax = -1, 1

i = 0
j = 2 

contour_bool = np.ones(data_nonisolated.shape[1], dtype=bool)
for n in range(data_nonisolated.shape[1]):
    #if free_energy_nonisolated[n] > contour_bdry:
    #    contour_bool[n] = False
    if data_nonisolated[j, n] > 0.5:
        contour_bool[n] = False
    if data_nonisolated[j, n] < -0.5:
        contour_bool[n] = False
triang = tri.Triangulation(data_nonisolated[i, :], data_nonisolated[j, : ])
triangle_mask = np.any(np.logical_not(contour_bool)[triang.triangles], axis=1)
triang.set_mask(triangle_mask)

fig = plt.figure()
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=0.1, c='gray')
my_levels = helpers.committor_contours() 
#my_levels = [0.0001, 0.9999] cmap = 'brg'
#my_levels = 10
C1=plt.tricontour(triang, q, levels=my_levels, linewidths=2.0, cmap=cmap, linestyles='dashed')
norm= matplotlib.colors.Normalize(vmin=C1.cvalues.min(), vmax=C1.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = C1.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=C1.levels)

my_cmap = matplotlib.cm.get_cmap('brg')
A_color= my_cmap(0.0)
B_color = my_cmap(1.0)
plt.scatter(A_test[i, :], A_test[j, :], s=0.5, c=A_color)
plt.scatter(B_test[i, :], B_test[j, :], s=0.5, c=B_color)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Theta$', size=20)
#plt.tight_layout()
#plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"figures/4var/committor_contours_{method}_phitheta.png", dpi=300)

plt.figure()
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=5, c=q)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Theta$', size=20)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.tight_layout()
plt.colorbar()
plt.savefig(f"figures/4var/committor_contours_{method}_phitheta_coarse.png", dpi=300)
#plt.gca().set_aspect('equal', adjustable='box')

fig = plt.figure()
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=0.1, c='gray')
my_levels = helpers.committor_contours() 
C1=plt.tricontour(triang, q_phi_theta, levels=my_levels, linewidths=2.0, cmap=cmap, linestyles='dashed')
norm= matplotlib.colors.Normalize(vmin=C1.cvalues.min(), vmax=C1.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = C1.cmap)
sm.set_array([])
fig.colorbar(sm, ticks=C1.levels)

plt.scatter(A_test[i, :], A_test[j, :], s=0.5, c=A_color)
plt.scatter(B_test[i, :], B_test[j, :], s=0.5, c=B_color)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Theta$', size=20)
#plt.tight_layout()
#plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f"figures/4var/committor_contours_{method}_phitheta_avg.png", dpi=300)

plt.figure()
plt.scatter(data_nonisolated[i, :], data_nonisolated[j, :], s=5, c=q_phi_theta)
plt.xlabel(r'$\Phi$', size=20)
plt.ylabel(r'$\Theta$', size=20)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.tight_layout()
plt.colorbar()
plt.savefig(f"figures/4var/committor_contours_{method}_phitheta_avg_coarse.png", dpi=300)
#plt.gca().set_aspect('equal', adjustable='box')

plt.show()
