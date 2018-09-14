
import numpy as np
import scipy
# import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import os



def get_csi(selected_csi,random_point):
    csi_total = []
    selected_csi=[selected_csi]
    for i in random_point:
        if i in selected_csi:
            continue
        else:
            files = (glob.glob(os.getcwd() + "/csi_data/LOC" + str(i+1) +"/*.mat"))

            for j in range(20): #read 30 packet from each location
                mat = scipy.io.loadmat(files[j])


                keys_to_select = ['csi']
                csi_list = [mat[k] for k in mat.keys() \
                                   if k in keys_to_select]
                csi = np.asarray(csi_list)

                csi=csi[0,0,:,:]
                csi=np.abs(csi)


                csi_vector=np.reshape(csi,(1,90))
                csi_vec_normalize=csi_vector/(np.max(csi_vector))
                csi_total.append(csi_vec_normalize)

    csi_total=np.array(csi_total)

    print(csi_total.shape)

    return csi_total

def get_test_csi(selected_csi):
    csi__test_total = []
    selected_csi=[selected_csi]

    for i in selected_csi:
        files2 = (glob.glob(os.getcwd() + "/csi_data/LOC" + str(i) + "/*.mat"))

        for j in range(10):  # read 20 packet from each location
            mat = scipy.io.loadmat(files2[j+20])
            # print(mat.keys() ,"\n **********")

            keys_to_select = ['csi']
            csi_list2 = [mat[k] for k in mat.keys() \
                        if k in keys_to_select]
            csi2 = np.asarray(csi_list2)

            csi2 = csi2[0, 0, :, :]
            csi2 = np.abs(csi2)


            csi_vector2 = np.reshape(csi2, (1, 90))
            csi_vec_normalize2 = csi_vector2 / (np.max(csi_vector2))

            csi__test_total.append(csi_vec_normalize2)

    csi__test_total = np.array(csi__test_total)
    print(csi__test_total.shape)

    return csi__test_total

# if __name__ == '__main__':
#     get_csi()
#     get_test_csi()
