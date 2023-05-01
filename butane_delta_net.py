import numpy as np 
import scipy.sparse as sps
import scipy.spatial
import scipy.linalg.lapack as lapack
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.linalg.lapack import dpstrf
from scipy.spatial.distance import cdist

def main():

    # Load data
    fname = "systems/butane/data/butane_metad.npz"
    inData = np.load(fname)
    print("Keys in data:")
    print(list(inData.keys()))

    #data = inData["data_all_atom"]
    data = inData["data"]

    print("Data shape from trajectory:")
    print(data.shape)
    dihedrals = inData["dihedrals"]
    potential = inData["potential"]
    kbT = inData["kbT"]
    print(f"kbT for data:{kbT}")
    kbT_roomtemp = inData["kbT_roomtemp"]
    print(f"kbT for room temperature:{kbT_roomtemp}")

    print(data.shape)
    # Adjust dihedral angles from [0, pi] for convenience
    dihedrals_shift = dihedrals.copy()
    dihedrals_shift[dihedrals < 0] = dihedrals_shift[dihedrals < 0] + 2*np.pi 

    #delta = 0.065
    delta = 0.017
    [delta_idx, delta_net_data] = epsilon_net(data, delta)
    print(delta_net_data.shape)
    fname = "systems/butane/data/butane_metad_deltanet.npz" 
    #fname = "systems/butane/data/butane_metad_deltanet_all_atom.npz" 

    np.savez(fname, delta_idx=delta_idx, delta=delta)
def epsilon_net(data, ϵ):

    #initialize the net

    dense = True # parameter that checks whether the net is still dense
    iter = 0 
    ϵ_net = np.array(range(data.shape[0]))
    current_point_index = ϵ_net[0]

    #fill the net

    while dense:
        current_point = data[current_point_index, :]# set current point
        #ϵ_ball = np.where( (np.sum((data.T - current_point)**2, axis=1)) <=ϵ**2) # get indices for ϵ-balli
        dists = cdist(current_point[np.newaxis,:], data, metric="euclidean")[0]
        ϵ_ball = np.where(dists  <=ϵ) # get indices for ϵ-balli
        ϵ_net = np.delete(ϵ_net, np.where(np.isin(ϵ_net, ϵ_ball))) # kill elements from the ϵ-ball from the net
        ϵ_net = np.append(ϵ_net, current_point_index) # add the current point at the BACK OF THE QUEUE. THIS IS KEY
        current_point_index = ϵ_net[0] # set current point for killing an epsilon ball in the next iteration
        if current_point_index == 0: # if the current point is the initial one, we are done! 
            dense = False
    return ϵ_net, data[ϵ_net, :]

if __name__ == '__main__':
    main()