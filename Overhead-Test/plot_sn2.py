
import numpy
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

num_of_SPs=[6,10,14,17]
total_training_time_fi=[29.91,52.61,76.31,96.5]
total_training_time_pos=[7.37,9.45,10.78,12.82]

avg_memory_pos=[3.09,3.07,3.09,3.11]
avg_memory_fi=[3.09,3.07,3.09,3.11]


from matplotlib import pyplot as PLT

fig = PLT.figure()


PLT.plot(num_of_SPs, total_training_time_fi,color='b',label='DeepFi', linewidth=2)
PLT.plot(num_of_SPs, total_training_time_fi,'*',color='y')
PLT.plot(num_of_SPs, total_training_time_pos,color='r',label='DeepPos', linewidth=2)
PLT.plot(num_of_SPs, total_training_time_pos,'*',color='g')
PLT.ylim([5, 100])
PLT.xlim([5, 18])
PLT.grid()
plt.xlabel('num of SPs')
plt.ylabel('total training time (s) ')

plt.legend()
PLT.show()


PLT.plot(num_of_SPs, np.array(avg_memory_fi)*np.array(total_training_time_fi),color='b',label='DeepFi', linewidth=2)
PLT.plot(num_of_SPs, np.array(avg_memory_fi)*np.array(total_training_time_fi),'*',color='y')
PLT.plot(num_of_SPs, np.array(avg_memory_pos)*np.array(total_training_time_pos),color='r',label='DeepPos', linewidth=2)
PLT.plot(num_of_SPs, np.array(avg_memory_pos)*np.array(total_training_time_pos),'*',color='g')
#PLT.ylim([5, 90])
PLT.xlim([5, 20])
PLT.grid()
plt.xlabel('num of SPs')
plt.ylabel('total RAM usage  (GB*s) ')
plt.legend()

PLT.show()