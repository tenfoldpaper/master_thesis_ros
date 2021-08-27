import sys
import time
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time

import rospy
import message_filters

class img_receiver:
    def __init__(self):
        self.bridge = CvBridge()
        topic = '/camera/depth_registered/hw_registered/image_rect_raw'
        self.subscriber = rospy.Subscriber(topic, Image, queue_size=5, callback=self.img_callback)
        
    def img_callback(self, data):
        data.encoding = "mono16" # if depth
        img = self.bridge.imgmsg_to_cv2(data, 'mono16')
        print(img.shape)
        print(np.max(img))

def main():
    rospy.init_node("imgrec_boilerplate", anonymous=True)
    receiver = img_receiver()#(**vars(opt))
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    #opt = parse_opt()
    #print(opt)
    #main(opt)
    main()
