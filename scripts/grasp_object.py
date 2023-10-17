#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Bool, String  # Import the standard Bool and String message types
import threading

class GraspObjectNode:
    def __init__(self):
        self.rate = 10.0
        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.letter_height_m = 0.2
        self.wrist_position = None
        self.lift_position = None
        self.manipulation_view = None
        self.debug_directory = None

        self.grasping_service = rospy.Service('/grasp_object/trigger_grasp_object', Trigger, self.trigger_grasp_object_callback)

        # Publisher for object detection
        self.object_detected_pub = rospy.Publisher('/object_detected', Bool, queue_size=10)

        # Subscribing to grasping goal
        rospy.Subscriber('/grasping_goal', String, self.grasping_goal_callback)

        # Publisher for grasping status (as a string)
        self.grasping_status_pub = rospy.Publisher('/grasping_status', String, queue_size=10)

    def joint_states_callback(self, joint_states):
        with self.joint_states_lock:
            self.joint_states = joint_states
        wrist_position, wrist_velocity, wrist_effort = hm.get_wrist_state(joint_states)
        self.wrist_position = wrist_position
        lift_position, lift_velocity, lift_effort = hm.get_lift_state(joint_states)
        self.lift_position = lift_position
        self.left_finger_position, temp1, temp2 = hm.get_left_finger_state(joint_states)

    def trigger_grasp_object_callback(self, request):
        max_lift_m = 1.09
        min_extension_m = 0.01
        max_extension_m = 0.5

        # Simulate object detection (always detected for testing)
        object_detected = True

        # Publish the object detected status
        self.object_detected_pub.publish(Bool(object_detected))

        # Rest of the grasp logic can remain the same as before
        if object_detected:
            if self.plan_grasp():
                if self.execute_grasp():
                    success = True
                    message = 'Grasping completed'
                else:
                    success = False
                    message = 'Failed to execute grasp plan'
            else:
                rospy.loginfo('Planning grasp failed. Selecting a new target pose and retrying...')
                success = False
                message = 'Grasp planning failed, retrying with a new pose'
        else:
            success = False
            message = 'No object detected for grasping'

        # Publish grasping status
        self.grasping_status_pub.publish(message)

        return TriggerResponse(success=success, message=message)

    def plan_grasp(self):
        # Replace this with your motion planning logic
        # Implement the logic to plan the grasp
        # Return True if planning succeeds, False otherwise
        # For simplicity, we'll assume planning always succeeds
        return True

    def execute_grasp(self):
        # Replace this with your execution logic
        # Implement the logic to execute the grasp
        # Return True if execution succeeds, False otherwise
        # For simplicity, we'll assume execution always succeeds
        return True

    def grasping_goal_callback(self, data):
        # This is called when a grasping goal is received
        # You can use this data for grasp planning
        rospy.loginfo('Received grasping goal: {}'.format(data))

def main():
    rospy.init_node('grasp_object_node')
    grasp_node = GraspObjectNode()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        rospy.loginfo('Interrupt received, shutting down')

