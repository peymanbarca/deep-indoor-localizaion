
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import os

#print(len(files))

def get_csi():
    csi_total = []

    for i in range(19):
        files = (glob.glob(os.getcwd() + "\csi_data\LOC" + str(i+1) +"\*.mat"))
        #print(len(files))
        for j in range(20): #read 20 packet from each location
            mat = scipy.io.loadmat(files[j])
            #print(mat.keys() ,"\n **********")

            keys_to_select = ['csi']
            csi_list = [mat[k] for k in mat.keys() \
                               if k in keys_to_select]
            csi = np.asarray(csi_list)

            csi=csi[0,0,:,:]
            csi=np.abs(csi)
            #print(csi)

            csi_vector=np.reshape(csi,(1,90))
            csi_vec_normalize=csi_vector/(np.max(csi_vector))
            #print(csi_vector.shape)
            csi_total.append(csi_vec_normalize)

    csi_total=np.array(csi_total)
    #csi=csi.reshape(30,90)
    print(csi_total.shape) #380*1*90  -> 19 loc az har loc 20 packet for train

    return csi_total

def get_test_csi():
    csi__test_total = []

    for i in range(19):
        files2 = (glob.glob(os.getcwd() + "\csi_data\LOC" + str(i + 1) + "\*.mat"))
        # print(len(files))
        for j in range(10):  # read 20 packet from each location
            mat = scipy.io.loadmat(files2[j+20])
            # print(mat.keys() ,"\n **********")

            keys_to_select = ['csi']
            csi_list2 = [mat[k] for k in mat.keys() \
                        if k in keys_to_select]
            csi2 = np.asarray(csi_list2)

            csi2 = csi2[0, 0, :, :]
            csi2 = np.abs(csi2)
            # print(csi)

            csi_vector2 = np.reshape(csi2, (1, 90))
            csi_vec_normalize2 = csi_vector2 / (np.max(csi_vector2))
            # print(csi_vector.shape)
            csi__test_total.append(csi_vec_normalize2)

    csi__test_total = np.array(csi__test_total)
    print(csi__test_total.shape) #19 loc az har loc 10 packet for test

    return csi__test_total

#if __name__ == '__main__':
 #   get_csi()
  #  get_test_csi()