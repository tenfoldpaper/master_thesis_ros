import struct
import sys
import time
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, PointCloud2, PointField
#from ros_numpy import point_cloud2
from sensor_msgs import point_cloud2
import std_msgs.msg

import cv2
import cv_bridge
import numpy as np
import time

import rospy
import message_filters

rgb_img = cv2.imread('./rgb_calib0.png', cv2.IMREAD_ANYCOLOR)
depth_img = cv2.imread('./depth_calib0.png', cv2.IMREAD_ANYDEPTH)

d_h, d_w = depth_img.shape[:2]
c_h, c_w = rgb_img.shape[:2]

d_ctx = [[576.092756, 0.000000, 316.286974], [0.000000, 575.853472, 239.895662], [0.000000, 0.000000, 1.000000]]
r_ctx = [[520.055928, 0.000000, 312.535255], [0.000000, 520.312173, 242.265554], [0.000000, 0.000000, 1.000000]]
d_ptx = [610.716858, 0.000000, 316.006104, 0.000000, 0.000000, 608.990112, 241.483958, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
r_ptx = [532.087158, 0.000000, 311.450990, 0.000000, 0.000000, 532.585205, 237.199006, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
d_dst = [-0.142470, 0.621352, 0.003223, -0.001925, 0.000000]
r_dst = [0.169160, -0.311272, -0.014471, -0.000973, 0.000000]
depth_to_rgb = np.array([[1, 0, 0, -0.0254 ],\
                        [0, 1, 0, -0.00013],\
                        [0, 0, 1, -0.00218]]) # R|T 
# fucking datatypes man
# 1 = all the source image pixels are retained
# 0 = all the pixels in the image are valid -- probably go with this
rgb_new_ctx, rgb_roi = cv2.getOptimalNewCameraMatrix(np.array(r_ctx), np.array(r_dst), (c_w, c_h), 1, (c_w, c_h))
depth_new_ctx, depth_roi = cv2.getOptimalNewCameraMatrix(np.array(d_ctx), np.array(d_dst), (d_w, d_h), 1, (d_w, d_h))

P3D = np.zeros((d_h, d_w, 6), dtype=np.float32)

print(rgb_new_ctx)
print(depth_new_ctx)
#depth conversion
depth_img = depth_img.astype(np.float32)
depth_img = depth_img/1000 # conversion from mm to m

print(np.unique(depth_img))

# ideally should be using a vectorized operation for this
for i in range(0, depth_img.shape[0]):
    for j in range(0, depth_img.shape[1]):
        P3D[i, j, 0] = ((j - depth_new_ctx[0, 2]) * depth_img[i, j] / depth_new_ctx[0, 0]) + (-0.0254)  # x 
        P3D[i, j, 1] = ((i - depth_new_ctx[1, 2]) * depth_img[i, j] / depth_new_ctx[1, 1]) + (-0.00013) # y
        P3D[i, j, 2] = (depth_img[i, j]) + (-0.00218) # z



P2Ds_rgb = []
points = []
pc_counter = 0
for i in range(0, depth_img.shape[0]):
    for j in range(0, depth_img.shape[1]):
        P2D_rgb_x = (P3D[i,j,0] * rgb_new_ctx[0, 0] / P3D[i,j,2]) + rgb_new_ctx[0, 2] # x
        P2D_rgb_y = (P3D[i,j,1] * rgb_new_ctx[1, 1] / P3D[i,j,2]) + rgb_new_ctx[1, 2] # y

        try:
            P3D[i,j,3:] = rgb_img[int(P2D_rgb_y), int(P2D_rgb_x)]
            x = P3D[i, j, 0]
            y = P3D[i, j, 1]
            z = P3D[i, j, 2]
            r = int(P3D[i, j, 3])
            g = int(P3D[i, j, 4])
            b = int(P3D[i, j, 5])
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b,g,r,a))[0]
            points.append([x,y,z,rgb])
#            print(hex(rgb))
            pc_counter += 1
        except:
            pass
print(pc_counter)
print(np.unique(P3D[:, :, 2]))

looper = rospy.Publisher('testPointCloud', PointCloud2, queue_size=10)

rospy.init_node('testPointCloud')

fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1)
    ]
header = std_msgs.msg.Header()
header.frame_id = "map"
header.stamp = rospy.Time.now()
pc2 = point_cloud2.create_cloud(header, fields, points)

while not rospy.is_shutdown():
    pc2.header.stamp = rospy.Time.now()
    looper.publish(pc2)
    rospy.sleep(1.0)
