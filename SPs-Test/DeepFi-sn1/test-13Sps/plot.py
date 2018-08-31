
import numpy
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio





mat = scipy.io.loadmat('csi2.mat')
print(mat.keys() ,"\n **********")

keys_to_select = ['csi']
csi_list = [mat[k] for k in mat.keys() \
                   if k in keys_to_select]
csi = numpy.asarray(csi_list)

csi=csi[0,0,:,:]
print(csi)

t = numpy.arange(1, 31, 1)

from matplotlib import pyplot as PLT

fig = PLT.figure()

ax1 = fig.add_subplot(221)
ax1.plot(t, abs(csi[0, 0:30]))
ax1.set_ylim([0, 30])
plt.title("A")
plt.xlabel('subCarriers')
plt.ylabel('db')

ax2 = fig.add_subplot(222)
ax2.plot(t, abs(csi[ 1, 0:30]))
ax2.set_ylim([0, 30])
plt.title("B")
plt.xlabel('subCarriers')


ax3 = fig.add_subplot(223)
ax3.plot(t, abs(csi[ 2, 0:30]))
ax3.set_ylim([0, 30])
plt.title("C")
plt.xlabel('subCarriers')
plt.ylabel('db')

PLT.show()