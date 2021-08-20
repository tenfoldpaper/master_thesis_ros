import sys
import time
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

import cv2
import cv_bridge
import numpy as np
import time

import rospy
import message_filters

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
depth_to_rgb = np.array([[1, 0, 0, -0.0254 ],\
                        [0, 1, 0, -0.00013],\
                        [0, 0, 1, -0.00218]]) # R|T 

rgb_img = cv2.imread('./rgb_image.png', cv2.IMREAD_ANYCOLOR)
depth_img = cv2.imread('./depth_image.png', cv2.IMREAD_ANYDEPTH)

