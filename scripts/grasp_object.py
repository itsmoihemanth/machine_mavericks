#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped

def move_arm(x, y, z):
    pub = rospy.Publisher('/clicked_point', PointStamped, queue_size=10)
    goal = PointStamped()
    goal.header.frame_id = "map"
    goal.point.x = x
    goal.point.y = y
    goal.point.z = z
    print("Publishing arm goal:", goal)  # Print the data
    pub.publish(goal)

def move_gripper(x, y, z):
    pub_left = rospy.Publisher('/stretch_gripper_controller/follow_joint_trajectory/goal', PointStamped, queue_size=10)
    goal = PointStamped()
    goal.header.frame_id = "map"
    goal.point.x = x
    goal.point.y = y
    goal.point.z = z
    print("Publishing gripper goal:", goal)  # Print the data
    pub_left.publish(goal)

def main():
    rospy.init_node('grasp_object_node')
    
    # Move the arm up and down
    move_arm(1.384, 2.156, 0.5)  # Move up (adjust the position as needed)
    rospy.sleep(30)  # Adjust the duration 
    
    # Open and close the gripper
    move_gripper(1.0, 1.0, 0.0)  # Open the gripper (adjust the positions as needed)
    rospy.sleep(30)  # Adjust the duration
    
    move_gripper(0.0, 0.0, 0.0)  # Close the gripper (adjust the positions as needed)
    
    rospy.spin()

if __name__ == '__main__':
    main()
