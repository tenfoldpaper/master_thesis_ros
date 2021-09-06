import numpy as np
import cv2
import matplotlib.pyplot as plt
#from skimage.segmentation import inverse_gaussian_gradient, chan_vase, watershed, clear_border
import skimage.segmentation

# Node for merging the different affordance segmentations to generate the bounding boxes.


image = np.load('kitchen2_mask.npy')
affseg = np.load('kitchen2_mask.npy') # this would come from the affordance node.

min_threshold = 50

# Threshold it.
affseg_t = cv2.threshold(affseg, min_threshold, 255, cv2.THRESH_BINARY)[1] 

affseg_img = np.zeros_like(affseg_t[:,:,0])
for i in range(0, affseg_t.shape[-1]):
    affseg_img = cv2.bitwise_or(affseg_img, affseg_t[:, :, i])

# What we want to send is the rgb image, depth image, and segmentation mask.
cnts = cv2.findContours(affseg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
detection_count = 0
xywh_tuple = []
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if(w < 15 and h < 15):
        continue
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    xywh_tuple.append([x,y,w,h])
    detection_count += 1
# extend detectionMsg by adding seg_image
# then assign image to seg_image 
# cvbridge gonna be tricky 
# foveation will need segmentation mode


# this node's msg will be sent to foveation.