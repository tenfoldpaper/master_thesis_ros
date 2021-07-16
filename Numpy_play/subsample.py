import numpy as np
import ros_numpy

from sensor_msgs.msg import PointCloud2

empty_var = (np.nan, np.nan, np.nan, 1.6410822e-38)
def crop_foveation(data, bb_origin, bb_size):
    
    # Assuming it's (height, width),
    
    # variable initialisation
    width = 640
    height = 480
    empty_var = data[0][0].copy()
    empty_var[0] = np.nan
    empty_var[1] = np.nan
    empty_var[2] = np.nan
    empty_var[3] = 1.6410822e-38
    
    bb_range = (bb_origin[0] + bb_size[0] if bb_origin[0] + bb_size[0] < height else height, \
        bb_origin[1] + bb_size[1] if bb_origin[1] + bb_size[1] < width else width)
    focus_height = np.array(range(bb_origin[0], bb_range[0]))
    focus_width = np.array(range(bb_origin[1], bb_range[1]))
    focus_data = data[np.ix_(focus_height, focus_width)]
    data[np.ix_(focus_height, focus_width)] = np.tile(empty_var, focus_data.shape) # zero out for compression later; might have to use nan instead?
    peripheral_radius = 0.3
    # Do a second foveation around the edge of the bounding box
    #peripheral area
    pa_origin = ( int(bb_origin[0] - (peripheral_radius*bb_size[0]) if bb_origin[0] - (peripheral_radius*bb_size[0]) > 0 else 0), \
        int(bb_origin[1] - (peripheral_radius*bb_size[1]) if bb_origin[1] - (peripheral_radius*bb_size[1]) > 0 else 0))
    pa_range = ( int(bb_range[0] + (peripheral_radius*bb_size[0]) if bb_range[0] + (peripheral_radius*bb_size[0]) < height else height), \
        int(bb_range[1] + (peripheral_radius*bb_size[1]) if bb_range[1] + (peripheral_radius*bb_size[1]) < width else width))

    pa_height = np.array(range(pa_origin[0], pa_range[0]))
    pa_width = np.array(range(pa_origin[1], pa_range[1]))

    pa_data = data[np.ix_(pa_height,pa_width)]
    data[np.ix_(pa_height,pa_width)] = np.tile(empty_var, pa_data.shape)
    pa_mask = np.ones(pa_data.shape, dtype=bool)
    pa_mask[::3, ::3] = 0
    pa_data = pa_data[pa_mask]

    mask = np.ones(data.shape, dtype=bool)
    mask[np.ix_(pa_height,pa_width)] = 0
    mask[::2, ::2] = 0
    mask[1::2, 1::2] = 0

    out_data = data[mask]
    breakpoint()
    return focus_data, pa_data, out_data


data = np.load('/home/main/Documents/Numpy_play/zero_test.npy')
bb_origin = (100, 100)
bb_size = (50, 60)

focus_data, pa_data, out_data = crop_foveation(data, bb_origin, bb_size)

with open('msg1.npy', 'wb+') as f:
    np.save(f, focus_data)
with open('msg2.npy', 'wb+') as f:
    np.save(f, pa_data)
with open('msg3.npy', 'wb+') as f:
    np.save(f, out_data)
msg1 = ros_numpy.msgify(PointCloud2, focus_data)
msg2 = ros_numpy.msgify(PointCloud2, pa_data)
msg3 = ros_numpy.msgify(PointCloud2, out_data)

breakpoint()
