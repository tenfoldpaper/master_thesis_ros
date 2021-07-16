import numpy as np

mask = np.ones((480, 640), dtype=np.float32)

mask[::2, ::2] = 2
mask[1::2, 1::2] = 3