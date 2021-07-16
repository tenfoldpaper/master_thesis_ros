import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
def cv2_imshow(img):
    plt.imshow(img)
    plt.show()

blank = np.zeros((480, 640), dtype=np.uint16) * 2**15
cv2_imshow(blank)

img_coords = [((184, 189), (414, 398)), ((138, 141), (430, 458)), ((92, 94), (447, 519)), ((46, 47), (463, 579)), ((0, 0), (480, 640))]
fov0_img = cv2.imread('resized0.tiff', -1)
fov1_img = cv2.imread('resized1.tiff', -1)
fov2_img = cv2.imread('resized2.tiff', -1)
fov3_img = cv2.imread('resized3.tiff', -1)
fov4_img = cv2.imread('resized4.tiff', -1)
#cv2_imshow(fov1_img)
fov1_img = cv2.resize(fov1_img, ((abs(img_coords[1][0][1] - img_coords[1][1][1])),abs(img_coords[1][0][0] - img_coords[1][1][0])), interpolation=cv2.INTER_NEAREST)
fov2_img = cv2.resize(fov2_img, ((abs(img_coords[2][0][1] - img_coords[2][1][1])),abs(img_coords[2][0][0] - img_coords[2][1][0])), interpolation=cv2.INTER_NEAREST)
fov3_img = cv2.resize(fov3_img, ((abs(img_coords[3][0][1] - img_coords[3][1][1])),abs(img_coords[3][0][0] - img_coords[3][1][0])), interpolation=cv2.INTER_NEAREST)
#cv2_imshow(fov3_img)
fov4_img = cv2.resize(fov4_img, ((abs(img_coords[4][0][1] - img_coords[4][1][1])),abs(img_coords[4][0][0] - img_coords[4][1][0])), interpolation=cv2.INTER_NEAREST)
#cv2_imshow(fov1_img)
#fov1_img_mask = cv2.bitwise_not(fov1_img)
#ret, fov1_img_mask = cv2.threshold(fov1_img_mask, 2**16 - 10, 2**16-1, cv2.THRESH_BINARY)
#fov1_img_mask = cv2.bitwise_not(fov1_img_mask)

blank[img_coords[0][0][0] : img_coords[0][1][0],img_coords[0][0][1] : img_coords[0][1][1]] = cv2.bitwise_or(blank[img_coords[0][0][0] : img_coords[0][1][0],img_coords[0][0][1] : img_coords[0][1][1]], fov0_img)
cv2_imshow(blank)
blank[img_coords[1][0][0] : img_coords[1][1][0],img_coords[1][0][1] : img_coords[1][1][1]] = cv2.bitwise_or(blank[img_coords[1][0][0] : img_coords[1][1][0],img_coords[1][0][1] : img_coords[1][1][1]], fov1_img)
print(img_coords[1][0][1], img_coords[1][1][1])
cv2_imshow(blank)
blank[img_coords[2][0][0] : img_coords[2][1][0],img_coords[2][0][1] : img_coords[2][1][1]] = cv2.bitwise_or(blank[img_coords[2][0][0] : img_coords[2][1][0],img_coords[2][0][1] : img_coords[2][1][1]], fov2_img)
print(img_coords[2][0][1], img_coords[2][1][1])
cv2_imshow(blank)
blank[img_coords[3][0][0] : img_coords[3][1][0],img_coords[3][0][1] : img_coords[3][1][1]] = cv2.bitwise_or(blank[img_coords[3][0][0] : img_coords[3][1][0],img_coords[3][0][1] : img_coords[3][1][1]], fov3_img)
print(img_coords[3][0][1], img_coords[3][1][1])
cv2_imshow(blank)
blank[img_coords[4][0][0] : img_coords[4][1][0],img_coords[4][0][1] : img_coords[4][1][1]] = cv2.bitwise_or(blank[img_coords[4][0][0] : img_coords[4][1][0],img_coords[4][0][1] : img_coords[4][1][1]], fov4_img)
print(img_coords[4][0][1], img_coords[4][1][1])

cv2_imshow(blank)

gm = Image.fromarray(blank, mode="I;16")
gm.save(f"stitched.tiff", compression="tiff_adobe_deflate")
#blank = cv2.bitwise_or(blank, fov1_img)
#cv2_imshow(blank)
