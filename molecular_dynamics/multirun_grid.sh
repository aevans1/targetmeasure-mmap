#!/bin/zsh

# Disable backups for gmx, command below creates too many backups
export GMX_MAXBACKUP=-1

nrep=128

# Set initial conditions for setting restraints
init_phi=$(sed "1q;d" data/phi_data.txt)
init_psi=$(sed "1q;d" data/psi_data.txt)

dx=$(echo "(6.28318 - 6.28318/$nrep) / ( $nrep - 1)" | bc -l)
dy=$(echo "(6.28318 - 6.28318/$nrep) / ( $nrep - 1)" | bc -l)

for ((j=108; j<nrep; j++))
do

for ((i=0; i<nrep; i++))
do

# Find iteration number, then pull phi, psi from that line
curr_phi=$(echo "$i * $dx - 3.14159" | bc -l)
curr_psi=$(echo "$j * $dy - 3.14159" | bc -l)

# Create new plumed with restraints at phi,psi pulled above
cat >plumed.dat << EOF
UNITS LENGTH=A

phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17

# Step 1: Guide restraint to bring CVs to specified position
# Step 2: Restrain coords at specified CVs
MOVINGRESTRAINT ...
    ARG=phi,psi
    STEP0=0    AT0=$init_phi,$init_psi KAPPA0=0.0,0.0
    STEP1=2000 AT1=$init_phi,$init_psi KAPPA1=1000.0,1000.0
    STEP2=4000 AT2=$curr_phi,$curr_psi KAPPA2=1000.0,1000.0
... MOVINGRESTRAINT
  
# Now that the simulation has CV values at desired CV value j, we can compute JJ^T for the jacobian of the CVs
# Output matrix of pairwise product of derivatives to file, and trajectory from restrained sim. if desired
PRINT ARG=phi,psi STRIDE=100 UPDATE_FROM=12 FILE=grid/restrain_colvar_phi${i}psi${j}
DUMPPROJECTIONS ARG=phi,psi STRIDE=100 UPDATE_FROM=12 FILE=grid/restrain_deriv_phi${i}psi${j}
EOF

# Run restrained simulation for 13500 steps

# CHARMM
/usr/local/gromacs/bin/gmx mdrun -plumed plumed.dat -s gromacs_files/topol_charmm.tpr -nsteps 13500

done

echo "Finished with phi${i},psi${j}"

done

