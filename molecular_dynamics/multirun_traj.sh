#!/bin/zsh

# Disable backups for gmx, command below creates too many backups
export GMX_MAXBACKUP=-1

# Set initial conditions for setting restraints
# Example: the following data was collected from a static bias, subsampled metadynamics run
init_phi=$(sed "1q;d" data/phi_data_static_sub.txt)
init_psi=$(sed "1q;d" data/psi_data_static_sub.txt)
init_theta=$(sed "1q;d" data/theta_data_static_sub.txt)
init_xi=$(sed "1q;d" data/xi_data_static_sub.txt)

for ((j=0; j<6416; j++))
do

# Find iteration number, then pull dihedral values from that line
iter=$(echo "$j+1" | bc -l)
curr_phi=$(sed "${iter}q;d" data/phi_data_static_sub.txt)
curr_psi=$(sed "${iter}q;d" data/psi_data_static_sub.txt)
curr_theta=$(sed "${iter}q;d" data/theta_data_static_sub.txt)
curr_xi=$(sed "${iter}q;d" data/xi_data_static_sub.txt)

# Create new plumed file with restraints at CV vals pulled above 
cat >plumed.dat << EOF
UNITS LENGTH=A

phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17
theta: TORSION ATOMS=6,5,7,9
xi: TORSION ATOMS=16,15,17,19

# Step 1: Guide restraint to bring CVs to specified position
# Step 2: Restrain coords at specified CVs
MOVINGRESTRAINT ...
    ARG=phi,psi,theta,xi
    STEP0=0    AT0=$init_phi,$init_psi,$init_theta,$init_xi KAPPA0=0.0,0.0,0.0,0.0
    STEP1=2000 AT1=$init_phi,$init_psi,$init_theta,$init_xi KAPPA1=1000.0,1000.0,1000.0,1000.0
    STEP2=4000 AT2=$curr_phi,$curr_psi,$curr_theta,$curr_xi KAPPA2=1000.0,1000.0,1000.0,1000.0
... MOVINGRESTRAINT

# Now that the simulation has CV values at desired CV value j, we can compute JJ^T for the jacobian of the CVs

# Output matrix of pairwise product of derivatives to file, and trajectory from restrained sim. if desired
#PRINT ARG=phi,psi,theta,xi STRIDE=100 UPDATE_FROM=12 FILE=traj/restrain_colvar$iter
DUMPPROJECTIONS ARG=phi,psi,theta,xi STRIDE=100 UPDATE_FROM=12 FILE=traj/restrain_deriv$iter
EOF

# Run restrained simulation for 13500 steps

# CHARMM
/usr/local/gromacs/bin/gmx mdrun -nb cpu -plumed plumed.dat -s gromacs_files/topol_masterclass21.6.tpr -nsteps 13500

done

