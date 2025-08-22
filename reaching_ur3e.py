import gymnasium as gym
import rtde_control
from rtde_control import RTDEControlInterface as RTDEControl



class UR3eReach(gym.Env):
    def __init__(self, robot_ip="147.175.108.138"):
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip, rtde_control.RTDEControl.FLAG_USE_EXT_UR_CAP)
        self.rtde_r = rtde_control.RTDEReceiveInterface(robot_ip)

        # Observations:  Box(-inf, inf, (25,), float32)
        # Actions:  Box(-1.0, 1.0, (6,), float32)

        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(25,), dtype=float)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=float)

        self.default_joint_position = [0.0, -1.712, 1.712, 0.0, 0.0, 0.0]

    def _get_observations(self):
        try:
            joint_positions = self.rtde_r.get_joint_positions()
            end_effector_pose = self.rtde_r.get_tool_pose()
            print("Joint Positions:", joint_positions)
            print("End Effector Pose:", end_effector_pose)

            return joint_positions + end_effector_pose
        except Exception as e:
            print("Error getting observations:", e)
            return self.default_joint_position + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


 

if __name__ == "__main__":
    env = UR3eReach()
    env._get_observations()