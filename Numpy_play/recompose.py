import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw # Only use this for visualisation!

import skimage.io

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.show()



fovlevel = 5
img_coords = [((184, 189), (414, 398)), ((138, 141), (430, 458)), ((92, 94), (447, 519)), ((46, 47), (463, 579)), ((0, 0), (480, 640))]
img_dict = {}
imgs = []
recomposed_img = np.zeros((480, 640))

for i in range(0, fovlevel):
    print(i)    
    img_bound_x = img_coords[i][1][1] - img_coords[i][0][1]
    img_bound_y = img_coords[i][1][0] - img_coords[i][0][0]
    box = (img_coords[i][0][0], img_coords[i][0][1])

    #img = skimage.io.imread(f'resized{i}.png', plugin='tifffile') # this is a 16-bit image, but only 11 bit of information should be encoded from a kinect.\
    img = skimage.io.imread(f'resized{i}.tiff', plugin='tifffile')
    img = Image.fromarray(img, mode='I;16')
    img = img.resize((img_bound_y, img_bound_x), resample=Image.NEAREST)
    img_dict['img_'+str(i)+'_rescaled'] = img
    print(img.size)
    img = np.array(img).transpose()
    img_dict['img_'+str(i)+'_transposed'] = Image.fromarray(img, mode='I;16')
    curr_coords = img_coords[i]
    curr_ys = (img_coords[0][0], img_coords[1][0])
    curr_xs = (img_coords[0][1], img_coords[1][1])
    a = 0
    ri_slice = recomposed_img[img_coords[i][0][0]:img_coords[i][1][0], img_coords[i][0][1]:img_coords[i][1][1]]
    print(ri_slice.shape)
    print(img.shape)
    for x in range(img_coords[i][0][1], img_coords[i][1][1]):
        b = 0
        for y in range(img_coords[i][0][0], img_coords[i][1][0]):
            if(img[b][a] > recomposed_img[y][x]):
                recomposed_img[y][x] = img[b][a]
            b += 1
        a += 1

    #mask = Image.fromarray(mask, mode='I;16')


    img_dict['pasted_'+str(i)+'_rescaled'] = Image.fromarray(recomposed_img, mode="I;16")

plot_figures(img_dict, fovlevel, 3)