#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math

def parse_action_sequence(action_sequence):
    return action_sequence.split()

def get_next_pose(current_pose, action):
    if action == "MoveAhead":
        current_pose['x'] += math.cos(current_pose['theta']) * 0.3  # Move forward 30cm
        current_pose['y'] += math.sin(current_pose['theta']) * 0.3
    elif action == "MoveBack":
        current_pose['x'] -= math.cos(current_pose['theta']) * 0.3  # Move back 30cm
        current_pose['y'] -= math.sin(current_pose['theta']) * 0.3
    elif action == "MoveLeft":
        current_pose['x'] -= math.cos(current_pose['theta'] + math.pi/2) * 0.3  # Move left 30cm
        current_pose['y'] -= math.sin(current_pose['theta'] + math.pi/2) * 0.3
    elif action == "MoveRight":
        current_pose['x'] += math.cos(current_pose['theta'] + math.pi/2) * 0.3  # Move right 30cm
        current_pose['y'] += math.sin(current_pose['theta'] + math.pi/2) * 0.3
    elif action == "RotateRight":
        current_pose['theta'] -= math.pi / 3  # Turn right 60 degrees
    elif action == "RotateLeft":
        current_pose['theta'] += math.pi / 3  # Turn left 60 degrees
    elif action == "LookUp":
        current_pose['pitch'] += math.radians(30)  # Look up 30 degrees
    elif action == "LookDown":
        current_pose['pitch'] -= math.radians(30)  # Look down 30 degrees
    return current_pose

def get_arm_positions(action):
    if action == "PickupObject":
        return [1.0, 1.5, 1.0, 0.5, 0.5, 1.0, -0.5]  # Example positions
    elif action == "PutObject":
        return [0.5, 1.0, 1.5, 0.5, 1.0, 0.5, -1.0]  # Example positions
    elif action == "OpenObject":
        return [0.7, 1.2, 1.2, 0.7, 0.8, 0.6, -0.3]  # Example positions
    elif action == "CloseObject":
        return [0.6, 1.0, 1.0, 0.6, 0.7, 0.5, -0.4]  # Example positions
    return None

def move_to_pose(x, y, theta):
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = "map"
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.orientation.z = math.sin(theta / 2.0)
    pose_stamped.pose.orientation.w = math.cos(theta / 2.0)
    goal_pub.publish(pose_stamped)
    rospy.sleep(5)

def move_arm_to_positions(joint_positions):
    trajectory = JointTrajectory()
    trajectory.header.stamp = rospy.Time.now()
    trajectory.header.frame_id = "base_link"
    trajectory.joint_names = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
                              "arm_5_joint", "arm_6_joint", "arm_7_joint"]
    point = JointTrajectoryPoint()
    point.positions = joint_positions
    point.velocities = [0.0] * 7
    point.time_from_start = rospy.Duration(5.0)
    trajectory.points.append(point)
    arm_pub.publish(trajectory)
    rospy.sleep(6)

if __name__ == '__main__':
    rospy.init_node('robot_controller', anonymous=True)
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    arm_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)

    rospy.loginfo("Waiting for publishers to become available...")
    rospy.sleep(2)

    current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'pitch': 0.0}
    action_sequence = "LookDown MoveAhead RotateRight MoveAhead PickupObject RotateLeft PutObject"
    actions = parse_action_sequence(action_sequence)

    for action in actions:
        if action in ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateRight", "RotateLeft", "LookUp", "LookDown"]:
            current_pose = get_next_pose(current_pose, action)
            move_to_pose(current_pose['x'], current_pose['y'], current_pose['theta'])
        elif action in ["PickupObject", "PutObject", "OpenObject", "CloseObject"]:
            joint_positions = get_arm_positions(action)
            if joint_positions:
                move_arm_to_positions(joint_positions)

    rospy.signal_shutdown("Action sequence completed")
