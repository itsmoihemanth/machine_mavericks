#!/usr/bin/env python3

import rospy
from stretch_srvs.srv import MoveArm  # Updated service message
from geometry_msgs.msg import Point

def move_arm_client(x, y, z):
    rospy.wait_for_service('/funmap/move_arm')
    try:
        control_service = rospy.ServiceProxy('/funmap/move_arm', MoveArm)
        target = Point(x=x, y=y, z=z)
        response = control_service(target)
        return response.success, response.message
    except rospy.ServiceException as e:
        print("Service call failed:", e)
        return False, "Service call failed"

if __name__ == '__main__':
    rospy.init_node('arm_gripper_control_client')
    # Example usage: Send a target position (1.0, 1.0, 1.0) for the end-effector
    success, message = move_arm_client(1.0, 1.0, 1.0)
    if success:
        print(f"Arm movement successful: {message}")
    else:
        print(f"Failed to move arm: {message}")
