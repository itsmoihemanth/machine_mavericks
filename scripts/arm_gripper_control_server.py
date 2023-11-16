#!/usr/bin/env python3

import rospy
from stretch_mavericks.srv import MoveArm  # Updated service message
from geometry_msgs.msg import PointStamped

def move_arm(request):
    # Extract 3D target for positioning end-effector
    target = request.pt

    goal = PointStamped()
    goal.header.frame_id = "map"
    goal.point.x = target.x
    goal.point.y = target.y
    goal.point.z = target.z
    
    # Publish the goal to /clicked_point
    goal_publisher.publish(goal)

    success = True  # Change this based on the actual success criteria
    if success:
        rospy.loginfo("Arm movement completed successfully")
        return True, "Arm movement completed successfully"
    else:
        rospy.logwarn("Arm movement failed")
        return False, "Arm movement failed"

def handle_move_arm(request):
    return move_arm(request)

def arm_gripper_control_server():
    rospy.init_node('arm_gripper_control_server')

    # Create a publisher for the goal
    global goal_publisher
    goal_publisher = rospy.rospy.ServiceProxy('/clicked_point', PointStamped, queue_size=10)

    service = rospy.Service('move_arm', MoveArm, handle_move_arm)
    rospy.loginfo("Arm control server is ready to receive commands...")
    rospy.spin()

if __name__ == '__main__':
    arm_gripper_control_server()

