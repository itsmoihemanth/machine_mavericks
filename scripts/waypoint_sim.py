#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback, MoveBaseResult
import actionlib
from actionlib_msgs.msg import GoalStatus

def move_base_client(waypoints):
    client = actionlib.SimpleActionClient('move_base_simple', MoveBaseAction)
    client.wait_for_server()

    for waypoint in waypoints:
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.pose = waypoint

        client.send_goal(goal, feedback_cb=feedback_callback)
        client.wait_for_result()

        if client.get_state() == GoalStatus.SUCCEEDED:
            rospy.loginfo("Waypoint reached")
        else:
            rospy.logwarn("Failed to reach waypoint")

def feedback_callback(feedback):
    # This callback can be used to monitor the progress of the navigation goal
    rospy.loginfo("Current pose: x=%.2f, y=%.2f", feedback.base_position.pose.position.x, feedback.base_position.pose.position.y)

if __name__ == '__main__':
    try:
        rospy.init_node('waypoint_planner', anonymous=True)

        waypoints = [(2.248, 0.704,0.0), (2.956, 0.638,0.0), (3.604, 2.762,0.0)]

        move_base_client(waypoints)

    except rospy.ROSInterruptException:
        pass
