import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image # Only use this for visualisation!

import skimage.io


def identify_center_depth_range(img, bb_origin, bb_size):
    height = img.shape[0]
    width = img.shape[1]
    bb_range = (bb_origin[0] + bb_size[0] if bb_origin[0] + bb_size[0] < height else height, \
        bb_origin[1] + bb_size[1] if bb_origin[1] + bb_size[1] < width else width)

    crop_height = np.array(range(bb_origin[0], bb_range[0]))
    crop_width = np.array(range(bb_origin[1], bb_range[1]))
    crop_img = img[np.ix_(crop_height, crop_width)]
    hist_edges = np.histogram_bin_edges(img, bins=20) # we want to do the histogram partition based on the whole image, not just the cropped image.
    hist = np.histogram(crop_img, bins=hist_edges)

    # the range of depth for the center image lies between two range bins with highest number of pixels assigned to them.

    range_ind = np.argpartition(hist[0], -2)[-2:] # result is ascending-sorted, i.e. argmax is at the last index.
    range_ind_low = range_ind[0] if range_ind[0] < range_ind[1] else range_ind[1]
    range_ind_high = range_ind[1] if range_ind[1] > range_ind[0] else range_ind[0]

    return range_ind_low, range_ind_high, hist[1]

def calculate_fovlevel_bb(bb_origin, bb_size, img_width, img_height, fovlevel):

    # we want a dynamically sized foveation boxes based on the location of the original bounding box and its size.

    # assuming it's (height, width)
    lower_bound = (bb_origin[0], bb_origin[1])
    upper_bound = (bb_origin[0] + bb_size[0], bb_origin[1] + bb_size[1])

    # we could replace np.linspace to some nonlinear space if that improves performance, since this might result in
    # the outermost foveation being too small.
    lower_bounds = (np.linspace(lower_bound[0], 0, num=fovlevel).astype(int), np.linspace(lower_bound[1], 0, num=fovlevel).astype(int))
    upper_bounds = (np.linspace(upper_bound[0], img_height, num=fovlevel).astype(int), np.linspace(upper_bound[1], img_width, num=fovlevel).astype(int))

    lower_bounds = list(zip(lower_bounds[0], lower_bounds[1]))
    upper_bounds = list(zip(upper_bounds[0], upper_bounds[1]))
    

    fovlevel_bb = list(zip(lower_bounds, upper_bounds))

    return fovlevel_bb

def calculate_fovlevel_depth(center_low, center_high, bins, fovlevel):

    fovlevel_depth = []
    
    lower_bounds = np.linspace(center_low, 0, num=fovlevel).astype(int)
    upper_bounds = np.linspace(center_high, len(bins) - 1, num=fovlevel).astype(int)
    

    fovlevel_depth = list(zip(lower_bounds, upper_bounds))
    return fovlevel_depth


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



img = skimage.io.imread('fmode.tiff', plugin='tifffile') # this is a 16-bit image, but only 11 bit of information should be encoded from a kinect.

#centerImg = skimage.io.imread('cropped.tiff', plugin='tifffile')

start = time.time()

bb_origin = (184, 189)
bb_size = (230, 209)

# configuration values
fovlevel = 5
max_scale = 5

center_low, center_high, bins = identify_center_depth_range(img, bb_origin, bb_size)
fovlevel_bb = calculate_fovlevel_bb( bb_origin, bb_size, img.shape[1], img.shape[0], fovlevel)
fovlevel_depth = calculate_fovlevel_depth(center_low, center_high, bins, fovlevel)
scale_values = np.linspace(1, max_scale, num=fovlevel).astype(int)
print(fovlevel_bb)
print(fovlevel_depth)

print('value init done')

img_dict = {}
for i in range(0, fovlevel):
    img_dict['fovlevel_'+str(i)+'_img'] = img.copy()
    bb_origin, bb_end = fovlevel_bb[i]
    depth_lower, depth_upper = fovlevel_depth[i]

    # crop it according to the bounding box calculated from fovlevel_bb
    focus_height = np.array(range(int(bb_origin[0]), int(bb_end[0])))
    focus_width = np.array(range(int(bb_origin[1]), int(bb_end[1])))
    cropped_img = img[np.ix_(focus_height, focus_width)].astype(np.uint16)


    # get rid of out-of-range depth values
    cropped_img[cropped_img < bins[int(depth_lower)]] = 0
    cropped_img[cropped_img > bins[int(depth_upper)]] = 0
    
    # get the cropped image and scale it according to the current foveation level
    gm = Image.fromarray(cropped_img, mode="I;16")
    resized = gm.resize((int(cropped_img.shape[1]/scale_values[i]), int(cropped_img.shape[0]/scale_values[i])), resample=Image.NEAREST)
    resized.save(f"resized{i}.tiff", compression="tiff_adobe_deflate")
    #resized.save(f"resized{i}.png", optimize=True)

    img_dict['fovlevel_'+str(i)+'_cropped'] = cropped_img.copy()

    # obtain a mask based on values that have been retained in the current level
    cropped_mask = (cropped_img == 0)

    img_dict['fovlevel_'+str(i)+'_cropped_mask'] = cropped_mask.copy()

    img_mask = np.ones(img.shape)
    img_mask[bb_origin[0]:bb_end[0], bb_origin[1]:bb_end[1]] = cropped_mask
    img_dict['fovlevel_'+str(i)+'_img_mask'] = img_mask.copy()

    # Apply mask to the original image
    img = (img * img_mask).astype(np.uint16)
    img_dict['fovlevel_'+str(i)+'_masked_img'] = img.copy()

print(f"The operation took {time.time() - start}")
plot_figures(img_dict, fovlevel, 5)

    
