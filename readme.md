## Under Construction!

The file for the diffusion map code is src/diffusion_map.py. This uses scikitlearn extensively and is based largely on the ["pydiffmap" library](https://github.com/DiffusionMapsAcademics/pyDiffMap).

The examples available in rough .ipynb form are: 
-  a 2d toy system with position-dependent diffusion, see 2D_example.ipynb.
-  a 2d molecular dynamics example, alanine dipeptide with two dihedral angles, see aladip_2var.ipynb
-  a 4d molecular dynamics example, alanine dipeptide with four dihedral angles, see aladip_4var.ipynb

There are python scripts available for the `Ksum test' on the datasets, Ksumtest_datasetname.py

### In progress:
 - Step-by-step process for alanine dipeptide in vacuum data (at current state these aren't runnable by themselves, they currently serve as a reference):
   - Gromacs simulation files for alanine dipeptide in vacuum (see [Some_Gromacs_Files](https://github.com/aevans1/targetmeasure-mmap/tree/main/molecular_dynamics/Some_Gromacs_Files)), used from [Plumed Masterclass 21.6](https://www.plumed.org/doc-v2.7/user-doc/html/masterclass-21-6.html)
   - Simulation with a static metadynamics bias (see [Some_Plumed_Files](https://github.com/aevans1/targetmeasure-mmap/tree/main/molecular_dynamics/Some_Plumed_Files))
   - Free energy computation on $\Phi, \Psi$ grid with metadynamics (see [Some_Plumed_Files](molecular_dynamics/Some_Plumed_Files)) 
   - Restrained Simulations for diffusion matrices on grid (for finite element method validation) and on trajectory (see [multirun files](https://github.com/aevans1/targetmeasure-mmap/tree/main/molecular_dynamics))
- **Note**: The plumed used for this was a custom plumed recompiled particularly with an [adjusted DUMPPROJECTIONS](https://github.com/aevans1/targetmeasure-mmap/blob/main/molecular_dynamics/Some_Plumed_Files/Value.cpp) ([plumed documentation](https://www.plumed.org/doc-v2.7/user-doc/html/_d_u_m_p_p_r_o_j_e_c_t_i_o_n_s.html)) in plumed/src/core/value.cpp so that it computes with the mass matrix as in equation (6) of our tm-mmap paper.   
 - Cleaning up the .ipynb for readability
