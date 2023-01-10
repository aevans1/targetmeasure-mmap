## Under Construction!

The file for the diffusion map code is src/diffusion_map.py. This uses scikitlearn extensively and is based largely on the ["pydiffmap" library](https://github.com/DiffusionMapsAcademics/pyDiffMap).

The examples available in rough .ipynb form are: 
-  a 2d toy system with position-dependent diffusion, see 2D_example.ipynb.
-  a 2d molecular dynamics example, alanine dipeptide with two dihedral angles, see aladip_2var.ipynb
-  a 4d molecular dynamics example, alanine dipeptide with four dihedral angles, see aladip_4var.ipynb

There are python scripts available for the `Ksum test' on the datasets, Ksumtest_datasetname.py

### In progress:
 - Step-by-step process for alanine dipeptide in vacuum data (at current state these aren't runnable by themselves, they currently serve as a reference):
   - Gromacs simulation files for alanine dipeptide in vacuum (see [Some_Gromacs_Files](molecular_dynamics/Some_Gromacs_Files/))
   - Simulation with a static metadynamics bias (see [Some_Plumed_Files](molecular_dynamics/Some_Plumed_Files))
   - Free energy computation on $\Phi, \Psi$ grid with metadynamics (see [Some_Plumed_Files](molecular_dynamics/Some_Plumed_Files)) 
   - Restrained Simulations for diffusion matrices on grid (for finite element method validation) and on trajectory (see [multirun files](molecular_dynamics/))
- **Note**: The plumed used for this was a custom plumed recompiled particularly with an [adjusted DUMPPROJECTIONS](molecular_dynamics/Some_Plumed_Files/Value.cpp) so that it computes with the mass matrix as in equation (6) of our paper.   
 - Cleaning up the .ipynb for readability