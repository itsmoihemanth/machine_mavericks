#!/usr/bin/env python3
import rospy
from std_srvs.srv import Trigger
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import GoalStatus
from skimage.morphology import binary_erosion
import numpy as np
from math import atan2, pi, sin, cos

class MappingNode:
    def __init__(self):
        rospy.init_node('mapping_node')
        self.perform_head_scan_service = rospy.ServiceProxy('/funmap/trigger_head_scan', Trigger)
        self.trigger_drive_to_scan_service = rospy.ServiceProxy('/funmap/trigger_drive_to_scan', Trigger)
        self.occupancy_grid_subscriber = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.rate = rospy.Rate(10)  # Adjust as needed
        self.robot_pose = Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        self.waypoints = []  # List to store waypoints dynamically
        self.map_data = None

    def perform_head_scan_and_update_map(self):
        try:
            scan_result = self.perform_head_scan_service()
            if scan_result.success:
                rospy.loginfo("Scan successful.")
            else:
                rospy.logwarn("Head scan was not successful.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call to perform_head_scan failed: %s" % str(e))

    def map_callback(self, msg):
        self.map_data = msg

    def occupancy_at(self, x, y):
        # Function to get the occupancy value at a given map coordinate
        if self.map_data is not None:
            width = int(self.map_data.info.width)
            index = int(y * width + x)
            if 0 <= index < len(self.map_data.data):
                return self.map_data.data[index]
        return -1

    def find_frontier(self):
        if self.map_data is not None:
            map_array = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))

            # Apply binary erosion to expand occupied regions
            eroded_map = binary_erosion(map_array, np.ones((5, 5)))

            # Identify frontiers as unexplored (erosion) adjacent to explored (original map)
            frontiers = np.logical_and(np.logical_not(eroded_map), map_array == -1)

            # Convert frontiers to coordinates
            frontier_coords = np.argwhere(frontiers)

            if len(frontier_coords) > 0:
                # Choose the closest frontier point as the next waypoint
                closest_frontier = max(frontier_coords, key=lambda coord: ((coord[0] - self.robot_pose.position.y)**2 +
                                                                           (coord[1] - self.robot_pose.position.x)**2))
                x_frontier, y_frontier = closest_frontier[1], closest_frontier[0]
                return Pose(position=Point(x=x_frontier, y=y_frontier, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))

        return None

    def move_robot_to_waypoint(self, waypoint):
        try:
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.pose = waypoint

            client = actionlib.SimpleActionClient('move_base_simple', MoveBaseAction)
            client.wait_for_server()

            client.send_goal(goal)
            client.wait_for_result()

            if client.get_state() == GoalStatus.SUCCEEDED:
                rospy.loginfo("Waypoint reached")
            else:
                rospy.logwarn("Failed to reach waypoint")
        except rospy.ServiceException as e:
            rospy.logerr("Service call to move_base failed: %s" % str(e))

    def map_room(self):
        # Move towards the detected frontier and add a new waypoint
        try:
            drive_result = self.trigger_drive_to_scan_service()
            if drive_result.success:
                rospy.loginfo("Drive successful.")
            else:
                rospy.logwarn("Drive was not successful.")
        except rospy.ServiceException as e:
            rospy.logerr("Service call to Drive to map failed: %s" % str(e))
        #

    def run(self):
        while not rospy.is_shutdown():
            try:
                self.perform_head_scan_and_update_map()
                self.map_room()
                self.rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("Mapping node was interrupted.")
                break

if __name__ == '__main__':
    try:
        node = MappingNode()
        node.run()
    except rospy.ROSInitException:
        rospy.logerr("ROS initialization failed.")

