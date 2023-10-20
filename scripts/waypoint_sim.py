#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import time
def send_goal():

    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    time.sleep(1)
    waypoints = [
        (4.672, -0.034,0.0),
        (0.0, 0.0, 0.0)
    ]
    for waypoint in waypoints:
        x, y, yaw = waypoint
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = yaw
        goal.pose.orientation.w = 1.0

        rate = rospy.Rate(10)  # Publish at 10 Hz
        timeout = 30  # 30 seconds timeout

        start_time = rospy.get_time()
        while (rospy.get_time() - start_time) < timeout:
            goal_pub.publish(goal)
            rate.sleep()

def main():
    rospy.init_node('waypoint_node', anonymous=True)
    send_goal()
    rospy.spin()
  
if __name__ == '__main__':
    main()
