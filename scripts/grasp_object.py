#!/usr/bin/env python3

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def move_arm_to_waypoints(waypoints):
    rospy.init_node('stretch_arm_actionlib', anonymous=True)

    client = actionlib.SimpleActionClient('/stretch_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()

    # Define the arm joint names
    arm_joint_names = ["joint_wrist_pitch", "joint_gripper_finger_left", "joint_gripper_finger_right"]

    for waypoint in waypoints:
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = arm_joint_names

        point = JointTrajectoryPoint()
        point.positions = waypoint
        point.time_from_start = rospy.Duration(2.0)  # Adjust the duration as needed

        trajectory_msg.points.append(point)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory_msg

        # Send the trajectory goal to move the arm
        client.send_goal(goal)

        # Wait for the arm to reach the waypoint
        client.wait_for_result()

if __name__ == '__main__':
    try:
        waypoints = [
            [0.1, 0.2, 0.3],  # Joint positions for waypoint 1
            [0.4, 0.5, 0.6],  # Joint positions for waypoint 2
            [0.7, 0.8, 0.9],  # Joint positions for waypoint 3
        ]

        move_arm_to_waypoints(waypoints)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

