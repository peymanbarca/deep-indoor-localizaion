
import numpy
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

num_of_SPs=[6,10,13,16,19]
total_training_time_fi=[17.28,37.28,43.22,62.75,86.58]
total_training_time_pos=[6.06,6.92,9.13,9.45,11.09]

avg_memory_pos=[3.01,3.00,3.01,3.01,3.01]
avg_memory_fi=[3.09,3.07,3.09,3.11,3.14]


from matplotlib import pyplot as PLT

fig = PLT.figure()


PLT.plot(num_of_SPs, total_training_time_fi,color='b',label='DeepFi', linewidth=2)
PLT.plot(num_of_SPs, total_training_time_fi,'*',color='y')
PLT.plot(num_of_SPs, total_training_time_pos,color='r',label='DeepPos', linewidth=2)
PLT.plot(num_of_SPs, total_training_time_pos,'*',color='g')
PLT.ylim([5, 90])
PLT.xlim([5, 20])
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