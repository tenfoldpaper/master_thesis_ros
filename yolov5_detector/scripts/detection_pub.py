#!/usr/bin/python3
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

from yolov5_detector.msg import DetectionMsg, DetectionArray
import numpy as np
import cv2
from sensor_msgs.msg import Image
import message_filters
def subview():
    rospy.init_node('det_pub')
    img_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=5, buff_size=2**25)
    depth_subscriber = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=5, buff_size=2**25)
    ts = message_filters.ApproximateTimeSynchronizer([img_subscriber, depth_subscriber], queue_size=1, slop=0.1)
    ts.registerCallback(callback)
    
    #sub = rospy.Subscriber('/camera/color/image_raw', Image, callback)
    rospy.spin()

def msgview():
    rospy.init_node('det_pub')
    sub = rospy.Subscriber('/yolov5/detection', DetectionMsg, msgcallback)
    rospy.spin()

def msgcallback(data):
    #print("Received detectionmsg")
    print(data.detection_array)
    
def callback(data1, data2):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data1, "bgr8")
    data2.encoding="mono16"
    img2 = bridge.imgmsg_to_cv2(data2, "bgr8")
    
    cv2.imshow("orig img", img)
    cv2.imshow("orig img2", img2)
    cv2.waitKey(1)  # 1 millisecond	
    
def run(message=''):
    pub = rospy.Publisher('detector_coords', DetectionMsg, queue_size = 10)
    rospy.init_node('det_pub')
    print("Init'd detector pub")
    r = rospy.Rate(10)
    msg = DetectionMsg()
    msg.detection_count = 2
    msg.header.seq = 0
    while not rospy.is_shutdown():
        msg.header.seq += 1
        msg.header.frame_id = f"frame{msg.header.seq}"
        elem2 = DetectionArray()
        elem2.detection_info = np.array([0,0,0,0,0]).astype(np.float)
        elem1 = DetectionArray()
        elem1.detection_info = np.array([1,1,1,1,1]).astype(np.float)
        msg.detection_array = []
        msg.detection_array.append(elem1)
        msg.detection_array.append(elem2)

        pub.publish(msg)
        r.sleep()


if __name__ == '__main__':
    msgview()
