#!/usr/bin/env python

import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
print(root_path)
root_path_1 = '/'.join(root_path.split('/')[:-1])
root_path_2 = '/'.join(root_path.split('/')[:-2])
root_path_3 = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path_1)
sys.path.append(root_path_2)
sys.path.append(root_path_3)
# sys.path.append('/home/gongjt4/graspnegt_ws/src/yolodet_pkg/src/yolov7/')

import numpy as np
import rospy
from sensor_msgs.msg import Image
import message_filters
from geometry_msgs.msg import Transform
from cv_bridge import CvBridge
import cv2
from IPython import embed
import demo
# from src import demo
# from src.demo import load_model
from yolov7.detect import load_model
from scipy.spatial.transform import Rotation

from std_srvs.srv import *
from det_pkg.srv import det, detRequest, detResponse

import threading



image_path = '../src/images/'
depth_path ='../src/depths/'
image_topic_name = '/camera/color/image_raw'
depth_topic_name = '/camera/aligned_depth_to_color/image_raw'
pub_topic_name = 'grasp_transform'

image_index = 0


weights = os.path.join(root_path,'best.pt') #'/home/hpf/det_ws/src/det_pkg/best.pt'
imgsz   = 640
device  = 'cpu'
model_info = load_model(weights, imgsz, device)

class MyService:
    def __init__(self, req):
        self.grasp_enable = req.grasp_enable
        self.label = req.label
        self.flag = True
        self.process_done = False
        self.bridge = CvBridge()
        add_thread = threading.Thread(target=self.thread_job)
        add_thread.start()
        while(True):
            if self.process_done == True:
                # add_thread.join()
                break
        # self.color = message_filters.Subscriber(image_topic_name, Image)
        # self.depth = message_filters.Subscriber(depth_topic_name, Image)
        # self.color_depth = message_filters.TimeSynchronizer([self.color, self.depth], 1)
        # self.color_depth.registerCallback(self.callback)
        # rospy.spin()

    def thread_job(self):
        self.color = message_filters.Subscriber(image_topic_name, Image)
        self.depth = message_filters.Subscriber(depth_topic_name, Image)
        self.color_depth = message_filters.TimeSynchronizer([self.color, self.depth], 1)
        self.color_depth.registerCallback(self.callback)
        rospy.spin()

    def callback(self, image, depth):
        if self.flag:
            # self.cv_color = cv2.resize(self.bridge.imgmsg_to_cv2(image, 'bgr8'), (640, 360))
            # self.cv_depth = cv2.resize(self.bridge.imgmsg_to_cv2(depth, '16UC1'), (640, 360))
            self.cv_color = self.bridge.imgmsg_to_cv2(image, 'bgr8')
            self.cv_depth = self.bridge.imgmsg_to_cv2(depth, '16UC1')
            global image_index
            rospy.loginfo('Receive and save images ')
            image_name = image_path + 'image_' + str(image_index)+ ".png"
            depth_name = depth_path + 'depth_' + str(image_index)+ ".png"
            # print('image: ', image_name)
            # print('depth: ', depth_name)
            image_index = image_index + 1
            cv2.imwrite(image_name, self.cv_color)
            cv2.imwrite(depth_name, self.cv_depth)
            self.flag = False
            self.process_images()

    def process_images(self):
        rospy.loginfo('Processing images .....')
        rospy.loginfo('grasp_enable {}'.format(self.grasp_enable))
        # if self.grasp_enable:

        rotat, trans, center_pose, bbox = demo.ros_demo(self.grasp_enable, self.label, 
                                        self.cv_color, self.cv_depth, model_info, save_path='top5.txt')
        # if rotat==None:    ### TODO
        #     return 
        # transform = Transform()
        
        # R = np.array([[rotat[0],rotat[1],rotat[2]],
        #                 [rotat[3],rotat[4],rotat[5]],
        #                 [rotat[6],rotat[8],rotat[8]]])
        # r = Rotation.from_matrix(R)
        # q = r.as_quat()
        # transform.rotation.x = q[0]
        # transform.rotation.y = q[1]
        # transform.rotation.z = q[2]
        # transform.rotation.w = q[3]
        # transform.translation.x = trans[0]
        # transform.translation.y = trans[1]
        # transform.translation.z = trans[2]
        # rospy.loginfo('yolodet output rotation: {}'.format(q))
        # rospy.loginfo('yolodet output translation: {}'.format(trans))
        
        self.output = detResponse()
        # self.output.rotation_w = transform.rotation.w 
        # self.output.rotation_x = transform.rotation.x 
        # self.output.rotation_y = transform.rotation.y 
        # self.output.rotation_z = transform.rotation.z 
        # self.output.translation_x = transform.translation.x
        # self.output.translation_y = transform.translation.y
        # self.output.translation_z = transform.translation.z
        self.output.rotat_0 = rotat[0]
        self.output.rotat_1 = rotat[1]
        self.output.rotat_2 = rotat[2]
        self.output.rotat_3 = rotat[3]
        self.output.rotat_4 = rotat[4]
        self.output.rotat_5 = rotat[5]
        self.output.rotat_6 = rotat[6]
        self.output.rotat_7 = rotat[7]
        self.output.rotat_8 = rotat[8]
        self.output.trans_x = trans[0]
        self.output.trans_y = trans[1]
        self.output.trans_z = trans[2]

        self.output.center_x = center_pose[0]
        self.output.center_y = center_pose[1]
        self.output.center_z = center_pose[2]

        self.output.bbox_height = bbox[1]
        self.output.bbox_width = bbox[0]
    
        # yolodet output is T_cam_grasp
        rospy.loginfo('yolodet output: {}'.format(self.output))
        # rospy.loginfo('yolodet output translation: {}'.format(trans))
        self.process_done = True
    # else:
    #     center_pose = demo.ros_demo(self.grasp_enable, self.label, 
    #                                 self.cv_color, self.cv_depth, model_info, save_path='top5.txt')
    #     rospy.loginfo('center_pose: {}'.format(center_pose))
            
def yolodet(req):
    serviceNode = MyService(req)
    return serviceNode.output

if __name__ == '__main__':
    try:
        rospy.init_node('yolodet_service_node')
        rospy.Service('yolo_det', det, yolodet)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass