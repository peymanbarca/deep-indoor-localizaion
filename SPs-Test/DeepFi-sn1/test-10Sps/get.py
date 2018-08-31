
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio

def get_csi():
    mat = scipy.io.loadmat('csi2.mat')
    #print(mat.keys() ,"\n **********")

    keys_to_select = ['csi']
    csi_list = [mat[k] for k in mat.keys() \
                       if k in keys_to_select]
    csi = np.asarray(csi_list)

    csi=csi[0,0,:,:]
    #print(csi)

    csi_vector=np.reshape(csi,(1,90))
    #print(csi_vector.shape)
    return csi_vector
