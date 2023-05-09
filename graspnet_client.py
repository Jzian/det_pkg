#!/usr/bin/env python

import sys
import rospy

from std_srvs.srv import *
from det_pkg.srv import det, detRequest, detResponse

def get_graspnet(start):
    rospy.wait_for_service('yolo_det')
    try:
        graspnet = rospy.ServiceProxy('yolo_det', det)
        r_t = graspnet(start)
        print(r_t.rotation_w)
        print(r_t.rotation_x)
        print(r_t.rotation_y)
        print(r_t.rotation_z)
        print(r_t.translation_x)
        print(r_t.translation_y)
        print(r_t.translation_z)
    except rospy.ROSInterruptException:
        print("Service call failed") 

if __name__ == '__main__':
    try:
        start = bool(sys.argv[1])
        # print(start)
        get_graspnet(start)
    except rospy.ROSInterruptException:
        pass