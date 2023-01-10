import numpy as np
fname = 'TRAJ_COLVAR_METAD_2VAR_GRID_STATIC'
colvars = np.loadtxt(fname)
phi = colvars[:, 1]
psi = colvars[:, 2]
theta = colvars[:, 3]
xi = colvars[:, 4]

np.savetxt('data/phi_data_static.txt', phi, fmt='%.6f')
np.savetxt('data/psi_data_static.txt', psi, fmt='%.6f')
np.savetxt('data/theta_data_static.txt', theta, fmt='%.6f')
np.savetxt('data/xi_data_static.txt', xi, fmt='%.6f')
