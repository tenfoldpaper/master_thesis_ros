import numpy as np
import ros_numpy



#90%
#data = np.load('/home/main/Documents/Numpy_play/test.npy')
# arr_width = np.arange(640)
# arr_height = np.arange(480)
# modval = 2 #50%
# arr_width_sample = np.where(arr_width % modval != 0)
# arr_height_sample = np.where(arr_height % modval == 0)
# breakpoint()


import_module = 'import numpy as np'
stmt1 = '''
data = np.load('/home/main/Documents/Numpy_play/test.npy')
arr_width = np.array(range(0, 640))
arr_height = np.array(range(0, 480))
modval = 2 #50%
arr_width_sample = np.where(arr_width % modval != 0)[0]
arr_height_sample = np.where(arr_height % modval != 0)[0]
sdata = data[np.ix_(arr_height_sample, arr_width_sample)]

'''

stmt2 = '''
data = np.load('/home/main/Documents/Numpy_play/test.npy')
sdata = data[::2, ::2]
'''

stmt3 = '''
data = np.load('/home/main/Documents/Numpy_play/test.npy')
mask = np.ones(data.shape, dtype=bool)
mask[::2, ::2] = 0
sdata = data[mask]
'''

import timeit
print(timeit.repeat(stmt=stmt1, number=100, setup=import_module, repeat=1))
print(timeit.repeat(stmt=stmt2, number=100, setup=import_module, repeat=1))
print(timeit.repeat(stmt=stmt3, number=100, setup=import_module, repeat=1))