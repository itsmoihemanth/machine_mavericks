#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import GoalStatus

def move_base_client(waypoints):
    client = actionlib.SimpleActionClient('move_base_simple', MoveBaseAction)
    client.wait_for_server()

    for waypoint in waypoints:
        goal = MoveBaseGoal()
        goal.target_pose = PoseStamped()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.pose = waypoint

        client.send_goal(goal)
        client.wait_for_result()

        if client.get_state() == GoalStatus.SUCCEEDED:
            rospy.loginfo("Waypoint reached")
        else:
            rospy.logwarn("Failed to reach waypoint")

if __name__ == '__main__':
    try:
        rospy.init_node('waypoint_planner', anonymous=True)

        waypoints = [
            PoseStamped(
                pose=Pose(
                    position=Point(1.0, 0.0, 0.0),  # Adjust waypoint coordinates
                    orientation=Quaternion(0.0, 0.0, 0.0, 1.0)
                )
            )
        ]

        move_base_client(waypoints)

    except rospy.ROSInterruptException:
        pass
