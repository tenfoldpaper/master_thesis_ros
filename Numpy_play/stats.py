import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

orig = Image.open("fmode.tiff")
orig = orig.convert('L')
image = Image.open("cropped.tiff", mode="r") # cropped depth map
image = image.convert('L')

image.show()
data = np.array(orig)
data = data.flatten()

plt.hist(data, density=True, bins=20)
print(np.histogram(data))
#plt.hist(data2, density=True, bins=10)

plt.show()

#Reasoning
'''
Assume here, that when an affordance object is identified, the image is likely to be focused around the center.
Also, that the image occupies a decent chunk of the space in the bounding box suggested.
doing a simple quadrant analysis would yield a range of value (based on how good the depth resolution is) that indicates 
the likely depth from the camera that the object is located in.

then we can use that number plus a couple standard deviations away from the centre of that mass of points as our 3D "fovea".

so basically I need to create a dynamic range of values based on the min-max of the depth values that exist within that image.

given that the resolution is 2048, and the actual kinect's range is 1.2~3.5m, we can use about 10 bins for this division.


Alternative: Do object segmentation so I don't have to come up with a statistical method, but leave it to the ML model to do it for me.
then the foveation can proceed quite easily from there.  --> probably more interesting for the thesis? 


'''