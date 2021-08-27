import sys
import time
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

import cv2
import cv_bridge
import numpy as np
import time

import rospy
import message_filters

'''
rgb_camera_matrix = np.array([[520.055928,   0.000000, 312.535255], \
                                      [  0.000000, 520.312173, 242.265554], \
                                      [  0.000000,   0.000000,   1.000000]])
        
rgb_distortion = np.array([0.169160,  -0.311272, -0.014471, -0.000973, 0.000000])

rgb_projection = np.array([[532.087158,   0.000000, 311.450990, 0.000000], \
                           [  0.000000, 532.585205, 237.199006, 0.000000], \
                           [  0.000000,   0.000000,   1.000000, 0.000000]])

depth_camera_matrix = np.array([[519.9970609150881,   0.0            , 312.1825832030777], \
                                [  0.0            , 519.9169979264075, 256.9132353905213], \
                                [  0.0            ,   0.0            ,   1.0            ]])

depth_distortion = np.array([0.1309893410538465, -0.2220648862292244, -0.0007491207145344614, -0.001087706204362299, 0.0])

depth_projection = np.array([[529.3626708984375, 0.0,            311.3649876060372, 0.0], \
                             [  0.0,           530.262939453125, 256.6228139598825, 0.0], \
                             [  0.0,             0.0,              1.0,             0.0]])
'''
depth_to_rgb = np.array([[1, 0, 0, -0.0254 ],\
                        [0, 1, 0, -0.00013],\
                        [0, 0, 1, -0.00218]]) # R|T 

rgb_img = cv2.imread('./rgb_image.png', cv2.IMREAD_ANYCOLOR)
depth_img = cv2.imread('./depth_image.png', cv2.IMREAD_ANYDEPTH)

d_h, d_w = depth_img.shape[:2]
c_h, c_w = rgb_img.shape[:2]

d_ctx = [[576.092756, 0.000000, 316.286974], [0.000000, 575.853472, 239.895662], [0.000000, 0.000000, 1.000000]]
r_ctx = [[520.055928, 0.000000, 312.535255], [0.000000, 520.312173, 242.265554], [0.000000, 0.000000, 1.000000]]
d_ptx = [610.716858, 0.000000, 316.006104, 0.000000, 0.000000, 608.990112, 241.483958, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
r_ptx = [532.087158, 0.000000, 311.450990, 0.000000, 0.000000, 532.585205, 237.199006, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
d_dst = [-0.142470, 0.621352, 0.003223, -0.001925, 0.000000]
r_dst = [0.169160, -0.311272, -0.014471, -0.000973, 0.000000]

# fucking datatypes man
# 1 = all the source image pixels are retained
# 0 = all the pixels in the image are valid -- probably go with this
rgb_new_ctx, rgb_roi = cv2.getOptimalNewCameraMatrix(np.array(r_ctx), np.array(r_dst), (c_w, c_h), 0, (c_w, c_h))
depth_new_ctx, depth_roi = cv2.getOptimalNewCameraMatrix(np.array(d_ctx), np.array(d_dst), (d_w, d_h), 0, (d_w, d_h))


d_undist = cv2.undistort(depth_img, np.array(d_ctx), np.array(d_dst), None, depth_new_ctx)
r_undist = cv2.undistort(rgb_img, np.array(r_ctx), np.array(r_dst), None, rgb_new_ctx)
x, y, w, h = depth_roi
d_undist = d_undist[y:y+h, x:x+w]
cv2.imwrite('depth_calib0.png', d_undist)

x, y, w, h = rgb_roi
r_undist = r_undist[y:y+h, x:x+w]
cv2.imwrite('rgb_calib0.png', r_undist)




