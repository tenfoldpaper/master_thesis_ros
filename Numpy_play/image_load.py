import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import time


start = time.time()
depth_array = np.load('depthRaw.npy')
rgb_array = np.load('colorRaw.npy')
#rgbC_array = np.load('colorRaw.npy')

norm_array = (depth_array/5)
norm_array = (norm_array*65536).astype(np.uint16)

gm = Image.fromarray(norm_array, mode="I;16")

resize1 = gm.resize((int(depth_array.shape[1]/4), int(depth_array.shape[0]/4)), resample=Image.NEAREST)
resize2 = gm.resize((int(depth_array.shape[1]/2), int(depth_array.shape[0]/2)), resample=Image.NEAREST)

#gm.show()
#resize1.show()

reresize1 = resize1.resize((640, 480), resample=Image.NEAREST)
reresize2 = resize2.resize((640, 480), resample=Image.NEAREST)

reresize1.save("reresize1.tiff", compression="tiff_adobe_deflate")
reresize2.save("reresize2.tiff", compression="tiff_adobe_deflate")
gm.save("fmode.tiff", compression="tiff_adobe_deflate") 
print(f"this took {time.time() - start}")



#plt.imshow(depth_array)
#plt.imshow(scaled_array)
#plt.show()



# im_frame = Image.open('pngfile.png')
# im_frame2 = Image.open('pngfile2.png')
# np_frame = np.array(im_frame)
# np_frame2 = np.array(im_frame2)
