from attention_package.msg import FoveatedImageGroups, FoveatedImage, FoveatedImages, Tuple
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

'''
Topics in: 
- Bounding box of classified objects topic, of format 
    - bounding_box[]: whose len() will reveal the # of detected objects;
        - uintx[] bounding_box_origin
        - uintx[] bounding_box_size
- Color image from the RGBD camera, of format
    - image 
- Depth image from the RGBD camera, of format
    - image

Topic out:
- FoveatedImageGroups

Logic:
Subscribe to the 3 topics above.

Callback should be tied to the image stream.

Bounding box data should be CONSUMED, and not stored in memory once it has been used for foveation.

If bounding box does not exist, then just set the foveation level to 1 by default, and compress the images accordingly.
Otherwise, use the provided foveation level, and 
    Compose the output message in a loop of [for object in detected_objects: for i in foveation_level]

'''

color_sub = None
depth_sub = None
bound_sub = None
curr_bb = None
curr_color = None
curr_depth = None

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

def dataSaverCallback(data, args):
    if(args[0] == 0):
        curr_color = data
    elif(args[0] == 1):
        curr_depth = data
    elif(args[0] == 2):
        curr_bb = data

if __name__ == "__main__":

    rospy.init_node("foveate")
    rospy.Subscriber('/camera/rgb/image_rect_color', Image, dataSaverCallback, [0])
    rospy.Subscriber('/camera/depth_registered/image_raw', Image, dataSaverCallback, [1])
    rospy.Subscriber('/affordance/detection/bounding_box', Tuple, dataSaverCallback, [2])
    
    foveation_publisher = rospy.Publisher('/attention/foveated', FoveatedImageGroups)
    if(curr_color != None and curr_depth != None):
        if(curr_bb != None): # some kinda classification has been detected, so do foveation.
            payload = FoveatedImageGroups()
            
            pass

        else: # no bounding box has been detected. Simply downscale the images in this case.
            pass
    
    rospy.spin()