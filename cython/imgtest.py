import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
from numpy.lib.recfunctions import append_fields
image = np.random.randint(0, 1000, (480, 640))
rgb_image = cv2.imread('rgb_img.png', cv2.IMREAD_ANYCOLOR)
#plt.imshow(image, cmap="gray")


start = time.time()
y, x= (image > 10).nonzero()
zipped = zip(y, x)
depth = image[y, x]
print(time.time() - start)
# this takes 20 ms

xyz_points = np.empty((0,3), dtype=np.float32)
rgb_points = np.empty((0), dtype=np.uint32)
# setting up
start = time.time()
rgb_camera_info = {'K':[520.055928, 0.000000, 312.535255, 0.000000, 520.312173, 242.265554, 0.000000, 0.000000, 1.000000]}
depth_camera_info = {'K':[576.092756, 0.000000, 316.286974, 0.000000, 575.853472, 239.895662, 0.000000, 0.000000, 1.000000]}

cx_d = depth_camera_info['K'][2]
cy_d = depth_camera_info['K'][5]
fx_d = depth_camera_info['K'][0]
fy_d = depth_camera_info['K'][4]
cx_r = rgb_camera_info['K'][2]
cy_r = rgb_camera_info['K'][5]
fx_r = rgb_camera_info['K'][0]
fy_r = rgb_camera_info['K'][4]

scale_factor = 1
scaled_cx_d = cx_d * scale_factor
scaled_cy_d = cy_d * scale_factor
scaled_fx_d = fx_d * scale_factor
scaled_fy_d = fy_d * scale_factor

scaled_cx_r = cx_r * scale_factor
scaled_cy_r = cy_r * scale_factor
scaled_fx_r = fx_r * scale_factor
scaled_fy_r = fy_r * scale_factor
T = 1000 * [-0.0254, -0.00013, -0.00218]

P3D_x = ((x - scaled_cx_d) * depth / scaled_fx_d) + T[0]
P3D_y = ((y - scaled_cy_d) * depth / scaled_fy_d) + T[1]
P3D_z = (depth) + T[2]


P2D_x = (P3D_x * scaled_fx_r / P3D_z) + scaled_cx_r
P2D_y = (P3D_y * scaled_fy_r / P3D_z) + scaled_cy_r
P2D_rgba = (np.c_[rgb_image[P2D_y.astype(int), P2D_x.astype(int)], (np.ones(len(P2D_y)) * 255)]).astype(np.uint8)

P3D_xyz = np.column_stack((P3D_x, P3D_y, P3D_z)) / 1000
P3D_xyz32 = P3D_xyz.astype(np.float32)

packed_rgba = np.squeeze(P2D_rgba.view(np.uint32))

print(time.time() - start)
start = time.time()
#P3D_xyzrgb = np.c_[P3D_x, P3D_y, P3D_z, packed_rgba]

#P3D_xyzrgba = np.array([P3D_xyz, packed_rgba], dtype='f8, f8, f8, u4')
xyz_points = np.vstack((xyz_points, P3D_xyz.astype(np.float32)))
rgb_points = np.hstack((rgb_points, packed_rgba))
print(time.time() - start)
start = time.time()
xyz_list = xyz_points.tolist()
rgb_list = rgb_points.tolist()
print(time.time() - start)
start = time.time()
final_points = [(val[0][0], val[0][1], val[0][2], val[1]) for val in zip(xyz_list, rgb_list)]
#final_points = [(val[0][0], val[0][1], val[0][2], val[1]) for val in zip(xyz_points, rgb_points)]

print(time.time() - start)

